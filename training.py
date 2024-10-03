import torch
from tqdm import tqdm
import os
from fashiondata import dataloader
import torch.nn.functional as F

from nn_model import nn_model


# hyperparameters

# diffusion hyperparameters
timesteps = 500
beta1 = 1e-4
beta2 = 0.02

# network hyperparameters
device = torch.device(
    "mps:0" if torch.backends.mps.is_available() else torch.device("cpu")
)
n_feat = 64  # 64 hidden dimension feature
n_cfeat = 5  # context vector is of size 5
height = 16  # 16x16 image
save_dir = "weights/"

# training hyperparameters
batch_size = 100
n_epoch = 32
lrate = 1e-3

# construct DDPM noise schedule
b_t = (beta2 - beta1) * torch.linspace(0, 1, timesteps + 1, device=device) + beta1
a_t = 1 - b_t
ab_t = torch.cumsum(a_t.log(), dim=0).exp()
ab_t[0] = 1

# nn_model = ContextUnet(in_channels=1, n_feat=n_feat, n_cfeat=n_cfeat, height=height).to(
#     device
# )

os.makedirs(save_dir, exist_ok=True)
torch.save(nn_model.state_dict(), os.path.join(save_dir, "model_trained.pth"))

print("Model weights saved.")

# Training
nn_model.train()
optim = torch.optim.Adam(nn_model.parameters(), lr=lrate)


# helper function: perturbs an image to a specified noise level
def perturb_input(x, t, noise):
    return (
        ab_t.sqrt()[t, None, None, None] * x + (1 - ab_t[t, None, None, None]) * noise
    )


for ep in range(n_epoch):
    print(f"epoch {ep}")

    # linearly decay learning rate
    optim.param_groups[0]["lr"] = lrate * (1 - ep / n_epoch)
    epoch_loss = 0.0
    loss_values = []

    pbar = tqdm(dataloader, mininterval=2)
    for x, _ in pbar:  # x: images
        optim.zero_grad()
        x = x.to(device)

        # perturb data
        noise = torch.randn_like(x)
        t = torch.randint(1, timesteps + 1, (x.shape[0],)).to(device)
        x_pert = perturb_input(x, t, noise)

        # use network to recover noise
        pred_noise = nn_model(x_pert, t / timesteps)

        # loss is mean squared error between the predicted and true noise
        loss = F.mse_loss(pred_noise, noise)
        loss.backward()

        optim.step()
        epoch_loss += loss.item()
    
    avg_loss = epoch_loss / len(dataloader)  # Average loss for the epoch
    loss_values.append(avg_loss)  # Store loss

    print(f"Epoch [{ep + 1}/{n_epoch}], Loss: {avg_loss:.4f}")

    print('losss values', loss_values)
    # save model periodically
    if ep % 4 == 0 or ep == int(n_epoch - 1):
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        torch.save(nn_model.state_dict(), save_dir + f"model_{ep}.pth")
        print("saved model at " + save_dir + f"model_{ep}.pth")
