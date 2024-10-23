import os
import torch
import torch.nn.functional as F
from diff_model import nn_model
from helpers import show_image

device = (
    torch.device("cuda")
    if torch.cuda.is_available()
    else torch.device("mps")
    if torch.backends.mps.is_available()
    else torch.device("cpu")
)
save_dir = "weights/data_context/"

# load in model weights and set to eval mode
print("curr dir", os.getcwd())
model_path = os.path.join(save_dir + "/model_epoch_31.pth")

print("model path", model_path)
checkpoint = torch.load(f=model_path, map_location=device)

nn_model.load_state_dict(checkpoint["model_state_dict"])
nn_model.eval()
print("Loaded in Model")

def ddim_sample(nn_model, n_samples, timesteps, alphas_cumprod, eta=0.0, device="cuda"):
    """
    DDIM Sampling for a diffusion model.

    Args:
        nn_model: Trained neural network model.
        n_samples: Number of samples to generate.
        timesteps: Total diffusion steps.
        alphas_cumprod: Cumulative product of alphas (beta schedule).
        eta: Controls the amount of noise injected during sampling. Default is 0 for deterministic sampling.
        device: Device to run sampling on, 'cuda' or 'cpu'.

    Returns:
        A batch of generated samples.
    """
    # Start with pure noise
    x = torch.randn(
        (n_samples, 3, 128, 128), device=device
    )  # Adjust image shape as per your dataset (e.g., 3x128x128)

    # Reverse sampling steps
    for t in reversed(range(1, timesteps)):
        t_tensor = torch.full((n_samples,), t, device=device, dtype=torch.long)

        # Predict noise using the neural network
        pred_noise = nn_model(x, t_tensor / timesteps)

        # Calculate x_t_minus_1 (previous step sample)
        alpha_t = alphas_cumprod[t]
        alpha_t_prev = alphas_cumprod[t - 1]
        beta_t = 1 - alpha_t
        sigma_t = eta * torch.sqrt(beta_t)

        # Compute the next step sample
        pred_x0 = (x - torch.sqrt(1 - alpha_t) * pred_noise) / torch.sqrt(alpha_t)
        noise = sigma_t * torch.randn_like(x) if t > 1 else torch.zeros_like(x)

        # Sample the next step
        x = torch.sqrt(alpha_t_prev) * pred_x0 + torch.sqrt(1 - alpha_t_prev) * noise

    return x


# Example usage:
n_samples = 16
timesteps = 1000  # Total timesteps used in the model
alphas_cumprod = torch.cumprod(
    torch.linspace(0.0001, 0.02, timesteps), dim=0
)  # Adjust based on your noise schedule

samples = ddim_sample(nn_model, n_samples, timesteps, alphas_cumprod, eta=0.0)



def pndm_sample(nn_model, n_samples, timesteps, alphas_cumprod, device="cuda"):
    """
    PNDM Sampling for a diffusion model.

    Args:
        nn_model: Trained neural network model.
        n_samples: Number of samples to generate.
        timesteps: Total diffusion steps.
        alphas_cumprod: Cumulative product of alphas (beta schedule).
        device: Device to run sampling on, 'cuda' or 'cpu'.

    Returns:
        A batch of generated samples.
    """
    # Start with pure noise
    x = torch.randn(
        (n_samples, 3, 128, 128), device=device
    )  # Adjust image shape as per your dataset

    # Define step size (can be adjusted for quality vs speed)
    step_size = timesteps // 4  # Skip 4 steps at a time

    for i, t in enumerate(reversed(range(1, timesteps, step_size))):
        t_tensor = torch.full((n_samples,), t, device=device, dtype=torch.long)

        # Predict noise using the neural network
        pred_noise = nn_model(x, t_tensor / timesteps)

        # Compute the cumulative product of alpha_t and the previous one
        alpha_t = alphas_cumprod[t]
        alpha_t_prev = (
            alphas_cumprod[t - step_size] if t - step_size >= 0 else alphas_cumprod[0]
        )

        # Calculate sigma for noise injection
        sigma_t = torch.sqrt(1 - alpha_t)

        # Reconstruct the next x using noise prediction
        pred_x0 = (x - sigma_t * pred_noise) / torch.sqrt(alpha_t)

        # Use the difference between alpha_t and alpha_t_prev to correct x
        x = (
            torch.sqrt(alpha_t_prev) * pred_x0
            + torch.sqrt(1 - alpha_t_prev) * pred_noise
        )
        
        show_image((samples[0]), title=f"After denoising step {i}")

    return x


# Example usage:
samples_pndm = pndm_sample(nn_model, n_samples, timesteps, alphas_cumprod)
