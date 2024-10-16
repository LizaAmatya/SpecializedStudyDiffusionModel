import csv
from matplotlib import pyplot as plt
import torch
from tqdm import tqdm
import os
from data_nocontext import dataloader 
import torch.nn.functional as F
from torch.amp import GradScaler
# from helpers import show_image
from diffusion_utils import load_latest_checkpoint, save_checkpoint
from nn_model import nn_model
import gc
import pandas as pd

# For memory optimizes

torch.cuda.empty_cache()
gc.collect()
print('memory mgmt', torch.cuda.memory_allocated())
print('memory reserved', torch.cuda.memory_reserved())

# os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"

# hyperparameters

# diffusion hyperparameters
timesteps = 500
beta1 = 1e-5
beta2 = 0.1

# network hyperparameters
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")

print('----using device---', device)
save_dir = "weights/bird_ds/"

# training hyperparameters
# batch_size = 8    #already set in dataloader
n_epoch = 32
lrate = 1e-4

# construct DDPM noise schedule
b_t = (beta2 - beta1) * torch.linspace(0, 1, timesteps + 1, device=device) + beta1
a_t = 1 - b_t
ab_t = torch.cumsum(a_t.log(), dim=0).exp()
ab_t[0] = 1

os.makedirs(save_dir, exist_ok=True)
# torch.save(nn_model.state_dict(), os.path.join(save_dir, "model_trained.pth"))

# Training
nn_model.train()
optim = torch.optim.Adam(nn_model.parameters(), lr=lrate)


# helper function: perturbs an image to a specified noise level
def perturb_input(x, t, noise):
    return (
        ab_t.sqrt()[t, None, None, None] * x + (1 - ab_t[t, None, None, None]) * noise
    )

scaler = GradScaler('cuda')
loss_file_path = os.path.join(save_dir, 'loss_val.csv')
all_losses = []

if not os.path.exists(loss_file_path):
    with open(loss_file_path, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['epoch', 'epoch_loss'])

start_epoch = 0
nn_model, optim, start_epoch, loss = load_latest_checkpoint(nn_model, optim, save_dir)

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
            # Clip gradients because gradients exploding and give Nan
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

# Saving all losses in another file
all_loss_file = os.path.join(save_dir, 'all_losses.csv')
with open(all_loss_file, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['loss'])
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
