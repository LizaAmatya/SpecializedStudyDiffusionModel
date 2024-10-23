import matplotlib.pyplot as plt
import numpy as np
import torch

def show_image(image_tensor, title=None):
    """Helper function to visualize a single image."""
    image = image_tensor.cpu().detach().numpy()
    image = image.transpose(1, 2, 0)  # Convert to HWC format for display
    # Clip the values to [0, 1] range for valid image display
    image = (image - image.min()) / (image.max() - image.min())
    plt.show(image)
    # plt.imshow((image*0.5 +0.5))  # Normalize to [0, 1] for display
    if title:
        plt.title(title)
    plt.axis("off")
    plt.show()


def cosine_beta_schedule(timesteps, beta_start=1e-4, beta_end=0.02, device="cuda"):
    """
    A cosine noise schedule between beta_start and beta_end over `timesteps`.
    """
    # Calculate the values based on cosine interpolation
    steps = torch.linspace(0, 1, timesteps + 1, device=device)
    betas = beta_start + 0.5 * (beta_end - beta_start) * (1 - torch.cos(steps * np.pi))

    # Clamp to avoid too small/large betas
    betas = torch.clamp(betas, min=1e-6, max=0.999)

    return betas


# Example usage
# timesteps = 1000
# beta_schedule = cosine_beta_schedule(timesteps, beta_start=1e-4, beta_end=0.02)

class MonitorParameters:
    def __init__(self):
        self.data = []

    def __call__(self, module, input, output):
        # Get parameters
        for name, param in module.named_parameters():
            if param.requires_grad:
                mean = param.data.mean().item()
                std = param.data.std().item()
                self.data.append({"layer": name, "mean": mean, "std": std})
