import csv
from matplotlib import pyplot as plt
import torch
from tqdm import tqdm
import os
from data import dataloader 
import torch.nn.functional as F
from torch.amp import GradScaler
from diffusion_utils import load_latest_checkpoint, save_checkpoint
from diff_model import nn_model
import gc
import pandas as pd

from helpers import MonitorParameters

# torch.cuda.empty_cache()
gc.collect()

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
save_dir = "weights/data_context/"

timesteps = 500
n_feat = 64
batch_size = 4
in_channels = 3
height = 128

beta1 = 1e-4
beta2 = 0.02
n_epoch = 32
lrate = 1e-4


# construct DDPM noise schedule
b_t = (beta2 - beta1) * torch.linspace(0, 1, timesteps + 1, device=device) + beta1
a_t = 1 - b_t
ab_t = torch.cumsum(a_t.log(), dim=0).exp()
ab_t[0] = 1

os.makedirs(save_dir, exist_ok=True)

# Training
nn_model.train()
optim = torch.optim.Adam(nn_model.parameters(), lr=lrate, weight_decay=0.0001)
# helper function: perturbs an image to a specified noise level
def perturb_input(x, t, noise):
    t = t.to(torch.long)
    return (
        ab_t.sqrt()[t, None, None, None] * x + (1 - ab_t[t, None, None, None]) * noise
    )


scaler = GradScaler("cuda")
loss_file_path = os.path.join(save_dir, "loss_val.csv")
grad_file = os.path.join(save_dir, 'mean_std.csv')
all_losses = []

if not os.path.exists(loss_file_path):
    with open(loss_file_path, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "epoch_loss"])

if not os.path.exists(grad_file):
        grad_writer = csv.writer(grad_file)
        grad_writer.writerow(['epoch', 'parameter', 'grad_mean', 'grad_std'])  # Header row

start_epoch = 0
nn_model, optim, start_epoch, loss = load_latest_checkpoint(nn_model, optim, save_dir, device=device)

def train_model(nn_model, data_loader, start_epoch, n_epoch):
    # Instantiate the monitor
    monitor = MonitorParameters()
    grad_stats = {}

    # Register hooks to each layer of your model
    for layer in nn_model.children():
        layer.register_forward_hook(monitor)
    
    for ep in range(start_epoch, n_epoch):
        print(f"!!!epoch {ep}!!!")

        # linearly decay learning rate
        optim.param_groups[0]["lr"] = lrate * (1 - ep / n_epoch)
        epoch_loss = 0.0
        accumulation_steps = 4

        pbar = tqdm(data_loader, mininterval=2)
        for i, (images, seg_masks, text_embeds) in enumerate(pbar):
            # print('iamges dataloader', images.shape, seg_masks.shape, text_embeds.shape)
            # Use images, segmentation masks, and text embeddings as inputs
            optim.zero_grad()
            images = images.to(device)
            seg_masks = seg_masks.to(device)
            text_embeds = text_embeds.to(device)

            # perturb data
            noise = torch.randn_like(images)
            t = torch.randint(1, timesteps + 1, (images.shape[0],)).to(torch.float32).to(device)
            # t_long = t.to(torch.long)
            x_pert = perturb_input(images, t, noise)
            
            with torch.autocast(device_type='cuda'):        #adding this memory better performance along with scaler as GradScaler
            # Forward pass
                pred_noise = nn_model(images, t, text_embeds, seg_masks)
                print('pred noise and noise', pred_noise.shape, noise.shape)
                # loss is mean squared error between the predicted and true noise
                loss = F.mse_loss(pred_noise, noise)
                all_losses.append(loss.item())
                pbar.set_postfix(loss=loss.item())
                
                if torch.isnan(loss) or torch.isinf(loss):
                    print(f'!!!!loss-- {loss}, is inf or nan')
                    continue    # skip iteration if loss invalid
            
            scaler.scale(loss).backward()
            # loss.backward()
            
            # Optionally, if you want to check gradients after a backward pass
            # Ensure this is after your loss.backward() call
            for name, param in nn_model.named_parameters():
                if param.requires_grad and param.grad is not None:
                    if name not in grad_stats: 
                        grad_stats[name] = {'mean': [], 'std': []}
                    mean_grad = param.grad.data.mean().item()
                    std_grad = param.grad.data.std().item()
                    grad_stats[name]['mean'].append(mean_grad)
                    grad_stats[name]['std'].append(std_grad)

                    print(f"{name} grad -- mean: {mean_grad:.4f}, std: {std_grad:.4f}")
                else:
                    print(f'layer {name} grad is None', param.grad)

            if (i + 1) % accumulation_steps == 0 or i == len(pbar):
                # Clip gradients because gradients exploding that give Nan
                scaler.unscale_(optim)  # Unscale gradients of model parameters
                torch.nn.utils.clip_grad_norm_(nn_model.parameters(), max_norm=1.0)  
                optim.step()
                scaler.update()
                optim.zero_grad()

        epoch_loss += loss.item()

        del images, seg_masks, text_embeds, loss, x_pert, noise, t
        torch.cuda.empty_cache()
        
        with open(loss_file_path, mode='a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([ep+1, epoch_loss/len(pbar)])
            
        # Save gradient mean and std to the file for each parameter for each epoch
        with open(grad_file, mode='a', newline='') as f:
            grad_writer = csv.writer(f)
            for name, stats in grad_stats.items():
                mean_grad = sum(stats['mean']) / len(stats['mean'])
                std_grad = sum(stats['std']) / len(stats['std'])
                grad_writer.writerow([ep, name, mean_grad, std_grad])

            grad_stats.clear()  # Clear after each epoch

        print(f"Epoch [{ep + 1}/{n_epoch}], Loss: {epoch_loss:.4f}")

        # save model periodically
        if ep % 4 == 0 or ep == int(n_epoch - 1):
            save_checkpoint(nn_model, optim, ep, epoch_loss, save_dir)
            print("saved model at " + save_dir + f"model_{ep}.pth")

    torch.cuda.empty_cache()
    gc.collect()
    
    # Saving all losses in another file
    all_loss_file = os.path.join(save_dir, "all_losses.csv")
    with open(all_loss_file, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["loss"])
        for loss in all_losses:
            writer.writerow([loss])
            
    # Plot losses

    data = pd.read_csv(loss_file_path)
    plt.figure(figsize=(8,6))
    plt.plot(data['epoch'], data['epoch_loss'], marker='o', linestyle='-',color='b', label="Training Loss")
    plt.title("Training Loss Over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()

    return all_losses
        
    
# Test forward pass with real data from the data loader
def test_model(model, data_loader):
    nn_model.eval()
    with torch.autocast(device_type='cuda'):
        for images, seg_masks, text_embeds in data_loader:
            images = images.to(device)
            seg_masks = seg_masks.to(device)
            text_embeds = text_embeds.to(device)
            print('iamges dataloader', images.shape, seg_masks.shape, text_embeds.shape)
            # Use images, segmentation masks, and text embeddings as inputs
            t = torch.randn(
                images.size(0), 1
            )  # Random time steps for testing (replace as needed)
            t = t.to(device)
            # Forward pass
            outputs = model(images, t, text_embeds, seg_masks)

            print("Output shape:", outputs.shape)
            # Output should match the input image shape: (batch_size, in_channels, height, width)
            break  # Test with one batch
    # model.train()  # Set the model back to training mode

# Run the test
def main():
    # test_model(nn_model, dataloader)
    train = train_model(nn_model=nn_model, data_loader=dataloader, start_epoch=0, n_epoch=n_epoch)

if __name__ == '__main__':
    main()