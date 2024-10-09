import torch
import matplotlib.pyplot as plt
from IPython.display import HTML, display
import tqdm
from diffusion_utils import *
import torch.nn.functional as F
from nn_model import nn_model, save_dir, device, b_t, a_t, ab_t, timesteps, height 
import os
from matplotlib.animation import FuncAnimation, PillowWriter
from torchvision.utils import make_grid

def denoise_add_noise(x, t, pred_noise, z=None):
    if z is None:
        z = torch.randn_like(x)
    noise = b_t.sqrt()[t] * z
    mean = (x - pred_noise * ((1 - a_t[t]) / (1 - ab_t[t]).sqrt())) / a_t[t].sqrt()
    return mean + noise

# load in model weights and set to eval mode
print('curr dir', os.getcwd())
model_path = os.path.join(save_dir + "model_31.pth")

print('model path', model_path)
nn_model.load_state_dict(
    torch.load(f=os.path.join(save_dir + "model_31.pth"), map_location=device)
)
nn_model.eval()
print("Loaded in Model")

# sample using standard algorithm
@torch.no_grad()
def sample_ddpm(n_sample, save_rate=20):
    # x_T ~ N(0, 1), sample initial noise
    samples = torch.randn(n_sample, 3, height, height).to(device)

    # array to keep track of generated steps for plotting
    intermediate = []
    for i in range(timesteps, 0, -1):
        print(f"sampling timestep {i:3d}", end="\r")

        # reshape time tensor
        t = torch.tensor([i / timesteps])[:, None, None, None].to(device)

        # sample some random noise to inject back in. For i = 1, don't add back in noise
        z = torch.randn_like(samples) if i > 1 else 0

        eps = nn_model(samples, t)  # predict noise e_(x_t,t)
        samples = denoise_add_noise(samples, i, eps, z)
        if i % save_rate == 0 or i == timesteps or i < 8:
            intermediate.append(samples.detach().cpu().numpy())
        

    intermediate = np.stack(intermediate)
    return samples, intermediate


# visualize samples
samples, intermediate_ddpm = sample_ddpm(5)
# samples_resized = F.interpolate(
#     samples, size=(256, 256), mode="bilinear", align_corners=False
# )

plt.imshow(samples[0].permute(1, 2, 0).cpu().numpy())
plt.show()
# animation_ddpm = plot_sample(
#     intermediate_ddpm, 1, 4, save_dir, "ani_run", None, save=True
# )
# display(HTML(animation_ddpm.to_jshtml()))

# animation_ddpm.save("animation_ddpm.gif", writer=PillowWriter(fps=5))
