# Function to simulate adding noise
import torch
import tqdm
import os
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from PIL import Image

def add_noise(images, num_steps, device="mps:0"):
    noisy_images = []
    alphas = torch.cos(torch.linspace(0, np.pi / 2, num_steps + 1))  # Cosine schedule

    for t in range(num_steps):
        alpha_t = alphas[t].to(device)
        noise = torch.randn_like(images).to(device)
        noisy_img = images * alpha_t + noise * (1 - alpha_t).sqrt()
        noisy_images.append(noisy_img.cpu().clone())

    return noisy_images


# Function to simulate removing noise with a diffusion model
def remove_noise(noisy_images, model, num_steps, device="mps:0"):
    denoised_images = []

    with torch.no_grad():
        x = noisy_images[-1].to(device)  # Start from the noisiest image

        for t in tqdm.tqdm(reversed(range(num_steps)), desc="Removing noise"):
            t_tensor = torch.full((x.shape[0],), t, dtype=torch.long).to(device)
            predicted_noise = model(x, t_tensor)

            alpha_t = torch.cos(torch.linspace(0, np.pi / 2, num_steps + 1)[t]).to(
                device
            )
            beta_t = 1 - alpha_t

            # Reverse diffusion step to remove noise
            x = (x - beta_t * predicted_noise) / alpha_t.sqrt()
            denoised_images.append(x.cpu().clone())

    return denoised_images


# num_images_per_row = 4

# num_samples = len(samples)
# num_rows = (num_samples + num_images_per_row - 1) // num_images_per_row
# fig, axes = plt.subplots(num_rows, num_images_per_row, figsize=(12, 12))
# for i, ax in enumerate(axes.flat):
#     if i < num_samples:
#         ax.imshow(samples[i, 0].cpu().numpy())
#     ax.axis('off')
# plt.show()


def create_animation(intermediate_samples, save_path="ddpm_animation.gif"):
    fig, ax = plt.subplots(figsize=(5, 5))

    def update(frame):
        ax.clear()
        ax.imshow(intermediate_samples[frame][0][0])
        ax.axis("off")
        ax.set_title(f"Step {frame + 1}")

    anim = FuncAnimation(fig, update, frames=len(intermediate_samples), repeat=False)

    # Save the animation as a GIF
    anim.save(save_path, writer=PillowWriter(fps=5))  # Adjust fps as needed
    plt.close(fig)


class UNet(nn.Module):

    def __init__(self, c_in=3, c_out=3, time_dim=256):

        super().__init__()

        self.time_dim = time_dim

        self.inc = DoubleConv(c_in, 64)

        self.down1 = Down(64, 128)

        self.sa1 = SelfAttention(128)

        self.down2 = Down(128, 256)

        self.sa2 = SelfAttention(256)

        self.down3 = Down(256, 256)

        self.sa3 = SelfAttention(256)

        self.bot1 = DoubleConv(256, 256)

        self.bot2 = DoubleConv(256, 256)

        self.up1 = Up(512, 128)

        self.sa4 = SelfAttention(128)

        self.up2 = Up(256, 64)

        self.sa5 = SelfAttention(64)

        self.up3 = Up(128, 64)

        self.sa6 = SelfAttention(64)

        self.outc = nn.Conv2d(64, c_out, kernel_size=1)

    def unet_forwad(self, x, t):

        "“Classic UNet structure with down and up branches, self attention in between convs”"

        x1 = self.inc(x)

        x2 = self.down1(x1, t)

        x2 = self.sa1(x2)

        x3 = self.down2(x2, t)

        x3 = self.sa2(x3)

        x4 = self.down3(x3, t)

        x4 = self.sa3(x4)

        x4 = self.bot1(x4)

        x4 = self.bot2(x4)

        x = self.up1(x4, x3, t)

        x = self.sa4(x)

        x = self.up2(x, x2, t)

        x = self.sa5(x)

        x = self.up3(x, x1, t)

        x = self.sa6(x)

        output = self.outc(x)

        return output

    def forward(self, x, t):

        "“Positional encoding of the timestep before the blocks”""

        t = t.unsqueeze(-1)

        t = self.pos_encoding(t, self.time_dim)

        return self.unet_forwad(x, t)
    

class EMA:

    def __init__(self, beta):

        super().__init__()

        self.beta = beta

        self.step = 0

    def update_model_average(self, ma_model, current_model):

        for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):

            old_weight, up_weight = ma_params.data, current_params.data

            ma_params.data = self.update_average(old_weight, up_weight)

    def update_average(self, old, new):

        if old is None:

            return new

        return old * self.beta + (1 - self.beta) * new

    def step_ema(self, ema_model, model, step_start_ema=2000):

        if self.step < step_start_ema:

            self.reset_parameters(ema_model, model)

            self.step += 1

            return

        self.update_model_average(ema_model, model)

        self.step += 1

    def reset_parameters(self, ema_model, model):

        ema_model.load_state_dict(model.state_dict())
