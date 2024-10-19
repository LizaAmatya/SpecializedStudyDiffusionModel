import csv
from matplotlib import pyplot as plt
import torch
from tqdm import tqdm
import os
from data import dataloader 
import torch.nn.functional as F
from torch.amp import GradScaler
from diffusion_utils import load_latest_checkpoint, save_checkpoint
from diff_model import model  as nn_model
import gc
import pandas as pd

from helpers import MonitorParameters

torch.cuda.empty_cache()
gc.collect()

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
print('device', device)
save_dir = "weights/data_context/"

timesteps = 500
n_feat = 64
batch_size = 16
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

optim = torch.optim.Adam(nn_model.parameters(), lr=lrate, weight_decay=0.0001)
# helper function: perturbs an image to a specified noise level
def perturb_input(x, t, noise):
    return (
        ab_t.sqrt()[t, None, None, None] * x + (1 - ab_t[t, None, None, None]) * noise
    )


scaler = GradScaler("cuda")
loss_file_path = os.path.join(save_dir, "loss_val.csv")
all_losses = []

if not os.path.exists(loss_file_path):
    with open(loss_file_path, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "epoch_loss"])

start_epoch = 0
nn_model, optim, start_epoch, loss = load_latest_checkpoint(nn_model, optim, save_dir)


def train_model(model, data_loader, start_epoch, num_epochs):
    for ep in range(start_epoch, n_epoch):
    print(f"!!!epoch {ep}!!!")

    # linearly decay learning rate
    optim.param_groups[0]["lr"] = lrate * (1 - ep / n_epoch)
    epoch_loss = 0.0
    accumulation_steps = 4

    pbar = tqdm(dataloader, mininterval=2)
    for i, x in enumerate(pbar):  # x: images
        optim.zero_grad()
        x = x.to(device)

        # perturb data
        noise = torch.randn_like(x)
        t = torch.randint(1, timesteps + 1, (x.shape[0],)).to(device)
        x_pert = perturb_input(x, t, noise)

        with torch.autocast(device_type='cuda'):        #adding this memory better performance along with scaler as GradScaler
            # use network to recover noise
            pred_noise = nn_model(x_pert, t / timesteps)

            # loss is mean squared error between the predicted and true noise
            loss = F.mse_loss(pred_noise, noise)
            all_losses.append(loss.item())
            print('--- loss',loss)
            
        scaler.scale(loss).backward()
       
        if (i + 1) % accumulation_steps == 0 or i == len(pbar):
            # Clip gradients because gradients exploding that give Nan
            scaler.unscale_(optim)  # Unscale gradients of model parameters
            torch.nn.utils.clip_grad_norm_(nn_model.parameters(), max_norm=1.0)  
            scaler.step(optim)
            scaler.update()
            optim.zero_grad()

        epoch_loss += loss.item()

        del x, loss, x_pert, noise, t
        torch.cuda.empty_cache()
            

    with open(loss_file_path, mode='a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([ep+1, epoch_loss/len(pbar)])
        
        # avg_loss = epoch_loss / len(dataloader)  # Average loss for the epoch
        # loss_values.append(epoch_loss)  # Store loss
        # loss_values_cpu = loss_values.detach().cpu().numpy()

    print(f"Epoch [{ep + 1}/{n_epoch}], Loss: {epoch_loss:.4f}")

    # save model periodically
    if ep % 4 == 0 or ep == int(n_epoch - 1):
        # if not os.path.exists(save_dir):
        #     os.mkdir(save_dir)
        # torch.save(nn_model.state_dict(), save_dir + f"model_{ep}.pth")
        save_checkpoint(nn_model, optim, ep, epoch_loss, save_dir)
        # print("saved model at " + save_dir + f"model_{ep}.pth")

    torch.cuda.empty_cache()
    gc.collect()
        
        
# Test forward pass with real data from the data loader
def test_model(model, data_loader):
    nn_model.eval()
    with torch.no_grad():
        for images, seg_masks, text_embeds in data_loader:
            print('iamges dataloader', images.shape, seg_masks.shape, text_embeds.shape)
            # Use images, segmentation masks, and text embeddings as inputs
            t = torch.randn(
                images.size(0), 1
            )  # Random time steps for testing (replace as needed)

            # Forward pass
            outputs = model(images, t, text_embeds, seg_masks)

            print("Output shape:", outputs.shape)
            # Output should match the input image shape: (batch_size, in_channels, height, width)
            break  # Test with one batch
    # model.train()  # Set the model back to training mode

# Run the test
test_model(nn_model, dataloader)

# Instantiate the monitor
monitor = MonitorParameters()


# Register hooks to each layer of your model
for layer in nn_model.children():
    layer.register_forward_hook(monitor)

# Optionally, if you want to check gradients after a backward pass
# Ensure this is after your loss.backward() call
for name, param in nn_model.named_parameters():
    if param.requires_grad:
        if param.grad is not None:
            mean_grad = param.grad.data.mean().item()
            std_grad = param.grad.data.std().item()
            print(f"{name} grad - mean: {mean_grad:.4f}, std: {std_grad:.4f}")
