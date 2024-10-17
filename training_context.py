import csv
from matplotlib import pyplot as plt
import torch
from tqdm import tqdm
import os
from data_nocontext import dataloader 
import torch.nn.functional as F
from torch.amp import GradScaler
from diffusion_utils import load_latest_checkpoint, save_checkpoint
from nn_model import nn_model
import gc
import pandas as pd

# Feed embeddings into your diffusion model or other components
# diffusion_model_output = diffusion_model(images, text_embeddings, image_embeddings)

torch.cuda.empty_cache()
gc.collect()

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")

print('----using device---', device)
save_dir = "weights/data_context/"

timesteps = 500
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
nn_model.train()
optim = torch.optim.Adam(nn_model.parameters(), lr=lrate, weight_decay=0.0001)

