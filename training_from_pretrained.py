import csv
import gc
import os
from matplotlib import pyplot as plt
import pandas as pd
import torch
from diffusers import ControlNetModel, StableDiffusionControlNetPipeline
from transformers import CLIPModel
from tqdm import tqdm
from data import dataloader
from diffusion_utils import load_latest_checkpoint, save_checkpoint
from torch import nn
import torch.nn.functional as F
from torch.amp import GradScaler


device = (
    torch.device("cuda")
    if torch.cuda.is_available()
    else torch.device("mps")
    if torch.backends.mps.is_available()
    else torch.device("cpu")
)

text_encoder = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
model_id = "lllyasviel/control_v11p_sd15_seg"
controlnet = ControlNetModel.from_pretrained(model_id, torch_dtype=torch.float16)
controlnet.to(device)

# print('config',controlnet.config)

# print('model', controlnet)
pipe = StableDiffusionControlNetPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5", controlnet=controlnet, torch_dtype=torch.float16
)
pipe.to(device)
vae = pipe.vae

gc.collect()

save_dir = "weights/controlnet/"
os.makedirs(save_dir, exist_ok=True)

loss_file_path = os.path.join(save_dir, "loss_val.csv")
if not os.path.exists(loss_file_path):
    with open(loss_file_path, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "epoch_loss"])

# Define optimizer and loss
optim = torch.optim.AdamW(controlnet.parameters(), lr=1e-4)
criterion = torch.nn.MSELoss()  # For pixel-wise tasks

nn_model, optim, start_epoch, loss = load_latest_checkpoint(
    controlnet, optim, save_dir, device=device
)

# Training loop
controlnet.train()
num_epochs = 32
timesteps = 500

scaler = GradScaler("cuda")
torch.cuda.empty_cache()
gc.collect()

def train_model(nn_model, data_loader, start_epoch, n_epoch):
    for ep in range(start_epoch, num_epochs):
        epoch_loss = 0.0
        
        pbar = tqdm(data_loader, mininterval=2)
        for batch in data_loader:
            images, masks, text_emb = batch
            print("masks shape---before", masks.shape)

            masks = masks.repeat(1, 3, 1, 1)    # 1 channel to 3 channel conversion
            print('images shaepe', images.shape)
            print('masks shape---after', masks.shape)
            print('text emb shapessss-------', text_emb.shape)
            images, masks, text_emb = (
                images.to(device).to(dtype=torch.float16),
                masks.to(device).to(dtype=torch.float16),
                text_emb.to(device).to(dtype=torch.float16),
            )
            text_emb_resized = nn.Linear(512, 768).to(device).to(dtype=torch.float16)(
                text_emb
            )  # Resize to match 768 features
            t = torch.randint(1, timesteps + 1, (images.shape[0],)).to(device)
            
            latents = vae.encode(images).latent_dist.sample().to(device)
            with torch.autocast(device_type='cuda'):
                # Forward pass
                out_model = nn_model(
                    sample=latents,
                    timestep=t,
                    encoder_hidden_states=text_emb_resized,
                    controlnet_cond=masks,  # Segmentation masks as conditioning
                )
                print("training after", dir(out_model), type(out_model))
                print(out_model.down_block_res_samples[0].shape)  # Check if this contains the image
                print(out_model.mid_block_res_sample.shape) 
                generated_image = out_model.mid_block_res_sample
                generated_image = F.interpolate(generated_image, size=(256, 256), mode='bilinear', align_corners=False)

                # generated images have 1280 channels, so reduction upsampling and conv layer
                upsample_block = (
                    nn.ConvTranspose2d(
                        in_channels=1280,
                        out_channels=1280,
                        kernel_size=4,
                        stride=2,
                        padding=1,
                    )
                    .to(device)
                    .to(dtype=torch.float16)
                ) 
                generated_image = upsample_block(generated_image)

                # Step 2: Reduce the number of channels from 1280 to 3 (for RGB images)
                conv_layer = nn.Conv2d(1280, 3, kernel_size=1).to(device).to(dtype=torch.float16)
                generated_image = conv_layer(generated_image)
                generated_image_resized = F.interpolate(
                    generated_image,
                    size=(256, 256),
                    mode="bilinear",
                    align_corners=False,
                )
                print('gen image------',generated_image.shape)
                print('image orig', images.shape)
                loss = criterion(generated_image_resized, images)   # F.mse_loss
                print(f"Epoch {ep+1}/{num_epochs}, Loss: {loss.item()}")
                
            epoch_loss += loss.item()
            scaler.scale(loss).backward()
            scaler.unscale_(optim)
            # loss.backward()
            optim.step()
            optim.zero_grad()
        
        del images, masks, text_emb, loss, t
        torch.cuda.empty_cache()
        
        # Calculate and log average loss for the epoch
        avg_loss = epoch_loss / len(dataloader)
            
        with open(loss_file_path, mode="a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([ep + 1, avg_loss])

        
        if ep % 4 == 0 or ep == int(n_epoch - 1):
            save_checkpoint(nn_model, optim, ep, epoch_loss, save_dir)
            print("saved model at " + save_dir + f"model_{ep}.pth")

    # Plot losses
    data = pd.read_csv(loss_file_path)
    plt.figure(figsize=(8, 6))
    plt.plot(
        data["epoch"],
        data["epoch_loss"],
        marker="o",
        linestyle="-",
        color="b",
        label="Training Loss",
    )
    plt.title("Training Loss Over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()

        
def main():
    train_model(nn_model=nn_model, data_loader=dataloader, start_epoch=start_epoch, n_epoch=num_epochs)


if __name__ == "__main__":
    main()
