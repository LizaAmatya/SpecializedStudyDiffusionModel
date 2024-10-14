from matplotlib import pyplot as plt
import torch
from tqdm import tqdm
import os
from data_nocontext import dataloader 
import torch.nn.functional as F
from torch.amp import GradScaler
# from helpers import show_image
from nn_model import nn_model
import gc

# For memory optimizes

torch.cuda.empty_cache()
gc.collect()
print('memory mgmt', torch.cuda.memory_allocated())
print('memory reserved', torch.cuda.memory_reserved())

# os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"

# hyperparameters

# diffusion hyperparameters
timesteps = 500
beta1 = 1e-4
beta2 = 0.02

# network hyperparameters
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")

print('----using device---', device)
# n_feat = 64  # 64 hidden dimension feature
# n_cfeat = 5  # context vector is of size 5
# height = 16  # 16x16 image
save_dir = "weights/bird_ds"

# training hyperparameters
# batch_size = 8    #already set in dataloader
n_epoch = 32
lrate = 1e-3

# construct DDPM noise schedule
b_t = (beta2 - beta1) * torch.linspace(0, 1, timesteps + 1, device=device) + beta1
a_t = 1 - b_t
ab_t = torch.cumsum(a_t.log(), dim=0).exp()
ab_t[0] = 1

os.makedirs(save_dir, exist_ok=True)
# torch.save(nn_model.state_dict(), os.path.join(save_dir, "model_trained.pth"))

# print("Model weights saved.")

# Training
nn_model.train()
optim = torch.optim.Adam(nn_model.parameters(), lr=lrate)


# helper function: perturbs an image to a specified noise level
def perturb_input(x, t, noise):
    return (
        ab_t.sqrt()[t, None, None, None] * x + (1 - ab_t[t, None, None, None]) * noise
    )

print('memory allocate before train', torch.cuda.memory_allocated())
print('memory reserved before train', torch.cuda.memory_reserved())

scaler = GradScaler('cuda')
for ep in range(n_epoch):
    print(f"epoch {ep}")

    # linearly decay learning rate
    optim.param_groups[0]["lr"] = lrate * (1 - ep / n_epoch)
    epoch_loss = 0.0
    loss_values = []
    accumulation_Steps = 4

    pbar = tqdm(dataloader, mininterval=2)
    for x in pbar:  # x: images
        optim.zero_grad()
        x = x.to(device)

        print('memory allocate before train', torch.cuda.memory_allocated())
        print('memory reserved before train', torch.cuda.memory_reserved())
        
        # show_image(x[0], title="Original Image")

        # perturb data
        noise = torch.randn_like(x)
        t = torch.randint(1, timesteps + 1, (x.shape[0],)).to(device)
        x_pert = perturb_input(x, t, noise)

        with torch.autocast(device_type='cuda'):        #adding this memory better performance along with scaler as GradScaler
            # use network to recover noise
            pred_noise = nn_model(x_pert, t / timesteps)

            # loss is mean squared error between the predicted and true noise
            loss = F.mse_loss(pred_noise, noise)
            print('--- loss',loss)
            
            scaler.scale(loss).backward()
            scaler.step(optim)    
            scaler.update()

        # # optim.step()
        epoch_loss += loss.item()
        
        # print('len of dataloader', len(dataloader))
        
        avg_loss = epoch_loss / len(dataloader)  # Average loss for the epoch
        loss_values.append(epoch_loss)  # Store loss
        # loss_values_cpu = loss_values.detach().cpu().numpy()

    print(f"Epoch [{ep + 1}/{n_epoch}], Loss: {loss:.4f}")

    print('losss values', loss_values)
    # save model periodically
    if ep % 4 == 0 or ep == int(n_epoch - 1):
    # if ep == int(n_epoch-1):
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        torch.save(nn_model.state_dict(), save_dir + f"/model_{ep}.pth")
        print("saved model at " + save_dir + f"/model_{ep}.pth")

    torch.cuda.empty_cache()
    gc.collect()

print('all lossess', loss_values)
# Plot losses
# plt.figure(figsize=(10, 5))
plt.plot(loss_values, label="Training Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
# plt.title("Training Loss Over Epochs")
# plt.xlim(1, len(loss_values))

# # Set y-axis limits based on your expected loss range (optional, adjust accordingly)
# plt.ylim(0, max(loss_values) + 0.1)
plt.legend()
plt.show()
