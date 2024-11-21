import os
import numpy as np
import torch
import torch.nn.functional as F
from diff_model import nn_model
from helpers import show_image
from torch.utils.checkpoint import checkpoint as chkpt
from data import test_dataloader

def denoise_add_noise(
    x,
    t,
    pred_noise,
    a_t,
    b_t,
    ab_t,
    z=None,
):
    if z is None:
        z = torch.randn_like(x)
    noise = b_t.sqrt()[t] * z
    mean = (x - pred_noise * ((1 - a_t[t]) / (1 - ab_t[t]).sqrt())) / a_t[t].sqrt()

    return mean + noise

@torch.no_grad()
def sample_ddpm(nn_model, n_sample, device='cuda'):
    # x_T ~ N(0, 1), sample initial noise
    samples = torch.randn(n_sample, 3, 128, 128).to(device)
    show_image(samples[0], title="Initial Noise")

    # array to keep track of generated steps for plotting
    intermediate = []
    for i in range(timesteps, 0, -1):
        print(f"sampling timestep {i:3d}", end="\r")

        # reshape time tensor
        t = torch.tensor([i / timesteps], dtype=torch.int)[:, None, None, None].to(
            device
        )
        seg_mask = torch.randn((n_samples, 1, 128,128), device=device)
        text_emb = torch.randn((n_samples, 512), device=device)

        # sample some random noise to inject back in. For i = 1, don't add back in noise
        z = torch.randn_like(samples) if i > 1 else 0
        # eps: pred noise
        eps = nn_model(samples, t, text_emb, seg_mask)

        if torch.isnan(eps).any() or torch.isinf(eps).any():
            print("invalid eps", eps.min(), eps.max())

        b_t = (beta_end - beta_start) * torch.linspace(0, 1, timesteps + 1, device=device) + beta_start
        a_t = 1 - b_t
        ab_t = torch.cumsum(a_t.log(), dim=0).exp()
        ab_t[0] = 1
        
        samples = denoise_add_noise(samples, i, eps,  a_t=a_t, b_t=b_t, ab_t=ab_t, z=z)
        # if i % save_rate == 0 or i == timesteps or i < 8:
        if i < 8:
            show_image(
                (samples[0]),
                title=f"After denoising step {i} with eps mean {eps.mean().item()}",
            )
            intermediate.append(samples.detach().cpu().numpy())

    intermediate = np.stack(intermediate)
    return samples, intermediate


def ddim_sample(nn_model, n_samples, timesteps, alphas_cumprod, eta=0.5, device="cuda"):
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
    seg_mask = torch.randn((n_samples, 1, 128,128), device=device)
    text_emb = torch.randn((n_samples, 512), device=device)

    # Reverse sampling steps
    with torch.no_grad():
        for t in reversed(range(1, timesteps)):
            t_tensor = torch.full((n_samples,), t, device=device, dtype=torch.long)

            # Predict noise using the neural network
            pred_noise = nn_model(x, t_tensor / timesteps, text_emb, seg_mask)
            # Calculate x_t_minus_1 (previous step sample)
            eps = 1e-8
            alpha_t = alphas_cumprod[t]
            alpha_t = torch.clamp(alpha_t, eps)
            alpha_t_prev = alphas_cumprod[t - 1]
            beta_t = 1 - alpha_t
            print('vals', alpha_t, beta_t)
            sigma_t = eta * torch.sqrt(beta_t)

            print('val of sigma_t', sigma_t)
            # Compute the next step sample
            pred_x0 = (x - torch.sqrt(1 - (alpha_t + eps)) * pred_noise) / torch.sqrt(alpha_t+eps)
            noise = sigma_t * torch.randn_like(x) if t > 1 else torch.zeros_like(x)

            # Sample the next step
            x = torch.sqrt(alpha_t_prev) * pred_x0 + torch.sqrt(1 - alpha_t_prev) * noise
            
            show_image(x[0], title=f"After denoising step {t}")

            del t_tensor, pred_noise
            torch.cuda.empty_cache()
    return x


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
    for batch in test_dataloader:
        image_tensor, seg_masks, text_embeds = batch
        seg_mask = seg_masks.to(device)  # Ensure it's on the right device
        text_emb = text_embeds.to(device)  # Ensure it's on the right device
        
        break
    
    seg_mask = seg_mask[0:1]
    text_emb = text_emb[0:1]
    
    # Start with pure noise
    x = torch.randn(
        (n_samples, 3, 128, 128), device=device
    )  * seg_mask # Adding mask for extra context
    
    seg_mask = torch.randn((n_samples, 1, 128,128), device=device)
    text_emb = torch.randn((n_samples, 512), device=device)
    
    # Define step size (can be adjusted for quality vs speed)
    step_size = timesteps // 20  # Skip 4 steps at a time

    with torch.no_grad():
        for t in reversed(range(1, timesteps, step_size)):
            t_tensor = torch.full((n_samples,), t, device=device, dtype=torch.long)

            # Predict noise using the neural network
            pred_noise = nn_model(x, t_tensor / timesteps, text_emb, seg_mask)
            eps = 1e-8
            # Compute the cumulative product of alpha_t and the previous one
            alpha_t = alphas_cumprod[t]
            alpha_t = torch.clamp(alpha_t, eps)
            alpha_t_prev = (
                alphas_cumprod[t - step_size] if t - step_size >= 0 else alphas_cumprod[0]
            )

            # Calculate sigma for noise injection
            sigma_t = torch.sqrt(1 - alpha_t+eps)

            # Reconstruct the next x using noise prediction
            pred_x0 = (x - sigma_t * pred_noise) / torch.sqrt(alpha_t)

        # Use the difference between alpha_t and alpha_t_prev to correct x
        x = (
            torch.sqrt(alpha_t_prev) * pred_x0
            + torch.sqrt(1 - alpha_t_prev) * pred_noise
        )
        
        show_image((x[0]), title=f"After denoising step {i}")

    return x


# Example usage:
n_samples = 1
timesteps = 500  # Total timesteps used in the model
beta_start = 1e-4
beta_end = 0.02
alphas_cumprod = torch.cumprod(
    torch.linspace(beta_start, beta_end, timesteps), dim=0
)  # Adjust based on your noise schedule


def main():
    # Example usage:
    # samples = ddim_sample(nn_model, n_samples, timesteps, alphas_cumprod, eta=0.5, device=device)
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
    model_path = os.path.join(save_dir + "/model_epoch_20.pth")

    print("model path", model_path)
    checkpoint = torch.load(f=model_path, map_location=device)

    nn_model.load_state_dict(checkpoint["model_state_dict"])
    nn_model.eval()
    print("Loaded in Model")
    
    samples_pndm = pndm_sample(nn_model, n_samples, timesteps, alphas_cumprod, device=device)
    # samples = ddim_sample(nn_model, n_samples, timesteps, alphas_cumprod, device=device)
    # samples = sample_ddpm(nn_model=nn_model, n_sample=n_samples, device=device)


if __name__ == "__main__":
    main()
    