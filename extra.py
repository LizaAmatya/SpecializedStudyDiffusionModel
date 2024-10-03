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

# create_animation(intermediate_ddpm, save_path="ddpm_animation.gif")

# print("Animation saved as 'ddpm_animation.gif'.")


# animation_ddpm = plot_sample(
#     intermediate_ddpm, 32, 4, save_dir, "ani_run", None, save=False
# )

# # display(HTML(animation_ddpm.to_jshtml()))

# # animation_ddpm.save("animation_ddpm.gif", writer="pillow")
# print('-------here end')


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


# Sample images using the model
samples = sample_from_dataset(
    model=nn_model,
    dataloader=dataloader,
    num_samples=32,
    device="mps" if torch.backends.mps.is_available() else "cpu",
)


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


class CustomDataset(Dataset):
    def __init__(self, sfilename, lfilename, transform, null_context=False):
        self.sprites = np.load(sfilename)
        self.slabels = np.load(lfilename)
        print(f"sprite shape: {self.sprites.shape}")
        print(f"labels shape: {self.slabels.shape}")
        self.transform = transform
        self.null_context = null_context
        self.sprites_shape = self.sprites.shape
        self.slabel_shape = self.slabels.shape

    # Return the number of images in the dataset
    def __len__(self):
        return len(self.sprites)

    # Get the image and label at a given index
    def __getitem__(self, idx):
        # Return the image and label as a tuple
        if self.transform:
            image = self.transform(self.sprites[idx])
            if self.null_context:
                label = torch.tensor(0).to(torch.int64)
            else:
                label = torch.tensor(self.slabels[idx]).to(torch.int64)
        return (image, label)

    def getshapes(self):
        # return shapes of data and labels
        return self.sprites_shape, self.slabel_shape


transform = transforms.Compose(
    [
        transforms.ToTensor(),  # from [0,255] to range [0.0,1.0]
        transforms.Normalize((0.5,), (0.5,)),  # range [-1,1]
    ]
)

def gen_tst_context(n_cfeat):
    """
    Generate test context vectors
    """
    vec = torch.tensor(
        [
            [1, 0, 0, 0, 0],
            [0, 1, 0, 0, 0],
            [0, 0, 1, 0, 0],
            [0, 0, 0, 1, 0],
            [0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0],  # human, non-human, food, spell, side-facing
            [1, 0, 0, 0, 0],
            [0, 1, 0, 0, 0],
            [0, 0, 1, 0, 0],
            [0, 0, 0, 1, 0],
            [0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0],  # human, non-human, food, spell, side-facing
            [1, 0, 0, 0, 0],
            [0, 1, 0, 0, 0],
            [0, 0, 1, 0, 0],
            [0, 0, 0, 1, 0],
            [0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0],  # human, non-human, food, spell, side-facing
            [1, 0, 0, 0, 0],
            [0, 1, 0, 0, 0],
            [0, 0, 1, 0, 0],
            [0, 0, 0, 1, 0],
            [0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0],  # human, non-human, food, spell, side-facing
            [1, 0, 0, 0, 0],
            [0, 1, 0, 0, 0],
            [0, 0, 1, 0, 0],
            [0, 0, 0, 1, 0],
            [0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0],  # human, non-human, food, spell, side-facing
            [1, 0, 0, 0, 0],
            [0, 1, 0, 0, 0],
            [0, 0, 1, 0, 0],
            [0, 0, 0, 1, 0],
            [0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0],
        ]  # human, non-human, food, spell, side-facing
    )
    return len(vec), vec
