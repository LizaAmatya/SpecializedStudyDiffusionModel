import csv
from matplotlib import pyplot as plt
import torch
from tqdm import tqdm
import os
from data import dataloader 
import torch.nn.functional as F
from torch.amp import GradScaler
from diffusion_utils import load_latest_checkpoint, save_checkpoint
from diff_model import model  as nn_model
import gc
import pandas as pd

torch.cuda.empty_cache()
gc.collect()

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
print('device', device)
save_dir = "weights/data_context/"

timesteps = 500
n_feat = 64
batch_size = 16
in_channels = 3
height = 128

beta1 = 1e-4
beta2 = 0.02
n_epoch = 32
lrate = 1e-4


# construct DDPM noise schedule
b_t = (beta2 - beta1) * torch.linspace(0, 1, timesteps + 1, device=device) + beta1
a_t = 1 - b_t
ab_t = torch.cumsum(a_t.log(), dim=0).exp()
ab_t[0] = 1

os.makedirs(save_dir, exist_ok=True)

# Training

optim = torch.optim.Adam(nn_model.parameters(), lr=lrate, weight_decay=0.0001)

# Test forward pass with real data from the data loader
def test_model(model, data_loader):
    nn_model.eval()
    with torch.no_grad():
        for images, seg_masks, text_embeds in data_loader:
            print('iamges dataloader', images.shape, seg_masks.shape, text_embeds.shape)
            # Use images, segmentation masks, and text embeddings as inputs
            t = torch.randn(
                images.size(0), 1
            )  # Random time steps for testing (replace as needed)

            # Forward pass
            outputs = model(images, t, text_embeds, seg_masks)

            print("Output shape:", outputs.shape)
            # Output should match the input image shape: (batch_size, in_channels, height, width)
            break  # Test with one batch
    # model.train()  # Set the model back to training mode

# Run the test
test_model(nn_model, dataloader)
