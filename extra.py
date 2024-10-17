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


def sample_from_dataset(
    model, dataloader, num_samples=32, num_steps=100, device="mps:0"
):
    model.eval()  # Set the model to evaluation mode

    # Get a batch of real images from the dataset
    data_iter = iter(dataloader)
    batch = next(data_iter)
    images, labels = batch
    images = images.to("mps" if torch.backends.mps.is_available() else "cpu")
    # images = images.to(device)  # Fetch a batch of images

    print("imagesss", images)
    print("dtype", images.dtype, images.shape)
    # Add noise to the images
    noisy_images = add_noise(images, num_steps, device=device)

    # Use the model to remove noise
    # denoised_images = remove_noise(noisy_images, model, num_steps, device=device)
    samples, intermediate_ddpm = sample_ddpm(32)
    return denoised_images


def plot_images_grid(images_list, title="Images", nrow=8):
    # if images_list.numel() == 0:
    #     print("Error: Image tensor is empty!")

    nrow = 8
    for step, images in enumerate(images_list):
        plt.figure(figsize=(10, 10))
        grid = make_grid(images, nrow=nrow, normalize=True, value_range=(0, 1))
        plt.imshow(grid.permute(1, 2, 0).cpu().numpy())
        plt.axis("off")
        plt.title(f"Denoising Step {step}")
        plt.show()


# Visualize the generated samples (replace with your own visualization method)
plot_images_grid(samples, title="Generated Samples")


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
