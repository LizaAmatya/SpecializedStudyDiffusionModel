import csv
import gc
import os
import deepspeed
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

text_encoder = CLIPModel.from_pretrained("openai/clip-vit-base-patch32", torch_dtype=torch.float32).to(device)
model_id = "lllyasviel/control_v11p_sd15_seg"
controlnet = ControlNetModel.from_pretrained(model_id, torch_dtype=torch.float32)
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
optim = torch.optim.AdamW(
    controlnet.parameters(), lr=1e-4, weight_decay=1e-2, betas=(0.9, 0.999)
)

# deepspeed_config = {
#     "optimizer": optim,
#     "zero_optimization": {
#         "stage": 2,
#         "offload_optimizer": {
#             "device": "cpu",
#         },
#     },
#     "train_batch_size": 4,
#     "gradient_accumulation_steps": 4,
# }


# criterion = torch.nn.MSELoss()  # For pixel-wise tasks
criterion = nn.SmoothL1Loss()       # For better stability - showing Nan loss for MSE -- HuberLoss (SmoothL1: l1 + MSE loss)

nn_model, optim, start_epoch, loss = load_latest_checkpoint(
    controlnet, optim, save_dir, device=device
)

# model_engine, optimizer, _, _ = deepspeed.initialize(
#     config_params=deepspeed_config, model=nn_model, optimizer=optim
# )

# Training loop
controlnet.train()
num_epochs = 32
timesteps = 500

scaler = GradScaler("cuda")
torch.cuda.empty_cache()
gc.collect()

def initialize_weights(model):
    for m in model.modules():
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
            nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="relu")
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)

def train_model(nn_model, data_loader, start_epoch, n_epoch):
    # upsample_block = (
    #     nn.Sequential(
    #         # Upsample the spatial dimensions from [4, 1280, 4, 4] to [4, 1280, 256, 256]
    #         nn.ConvTranspose2d(
    #             in_channels=1280,
    #             out_channels=640,  # Keep 1280 channels for now (no change)
    #             kernel_size=4,  # Kernel size to upscale
    #             stride=2,  # Stride of 2 to double spatial dimensions
    #             padding=1,  # Ensure the spatial dimensions are doubled
    #         ),
    #         # Reduce the number of channels from 1280 to 3 (for RGB images)
    #         nn.Conv2d(
    #             in_channels=640,
    #             out_channels=3,  # Output channels: 3 (RGB)
    #             kernel_size=1,  # Kernel size of 1 to reduce the channel count
    #             stride=1,  # No change in spatial dimensions from this layer
    #             padding=0,  # No padding necessary
    #         ),
    #     )
    #     .to(device).to(dtype=torch.float32)
    # )
    upsample_block = nn.Sequential(
        # Upsample from [4, 1280, 4, 4] to [4, 1280, 256, 256] using F.interpolate
        nn.Conv2d(
            1280, 640, kernel_size=3, padding=1, stride=1
        ),  # Reduce channels from 1280 to 640
        nn.BatchNorm2d(640),
        nn.ReLU(),
        # Depthwise separable convolution: reduces channels and memory usage
        nn.Conv2d(
            640, 320, kernel_size=3, padding=1, stride=1
        ),  # Reduce channels to 320
        nn.BatchNorm2d(320),
        nn.ReLU(),
        # Final reduction to 3 channels (RGB)
        nn.Conv2d(320, 3, kernel_size=1),
    ).to(device).to(dtype=torch.float32)
    initialize_weights(upsample_block)

    for ep in range(start_epoch, num_epochs):
        epoch_loss = 0.0
        accumulation_steps = 4

        pbar = tqdm(data_loader, mininterval=2)
        for i, batch in enumerate(pbar):
            optim.zero_grad(set_to_none=True)
            images, masks, text_emb = batch
            print("masks shape---before", masks.shape)

            masks = masks.repeat(1, 3, 1, 1)    # 1 channel to 3 channel conversion
            print('images shaepe', images.shape)
            print('masks shape---after', masks.shape)
            print('text emb shapessss-------', text_emb.shape)
            images, masks, text_emb = (
                images.to(device).to(dtype=torch.float32),
                masks.to(device).to(dtype=torch.float32),
                text_emb.to(device).to(dtype=torch.float32),
            )
            with torch.autocast(device_type='cuda'):
                text_emb_resized = nn.Linear(512, 768).to(device)(
                    text_emb
                )  # Resize to match 768 features
                t = torch.randint(1, timesteps + 1, (images.shape[0],)).to(device)
                
                latents = vae.encode(images).latent_dist.sample().to(device)
                
                # Forward pass
                out_model = nn_model(
                    sample=latents,
                    timestep=t,
                    encoder_hidden_states=text_emb_resized,
                    controlnet_cond=masks,  # Segmentation masks as conditioning
                )
                # print("training after", dir(out_model), type(out_model))
                # print(out_model.down_block_res_samples[0].shape)  # Check if this contains the image
                # print(out_model.mid_block_res_sample.shape) 
                generated_image = out_model.mid_block_res_sample
                # generated_image = F.interpolate(input=generated_image.detach(), size=(256, 256), mode='bilinear', align_corners=False)

                generated_image = upsample_block(generated_image)
                generated_image_resized = F.interpolate(
                    generated_image,
                    size=(256, 256),
                    mode="bilinear",
                    align_corners=False,
                )
                print("gen image------", generated_image_resized.shape)
                print('image orig', images.shape)
                loss = criterion(generated_image_resized, images)   # F.mse_loss
                print(f"Epoch {ep+1}/{num_epochs}, Loss: {loss.item()}")
                
            epoch_loss += loss.item()
            # model_engine.backward(loss)
            # model_engine.step()
            scaler.scale(loss).backward()
            # loss.backward()
            
            if (i + 1) % accumulation_steps == 0 or i == len(pbar):
                scaler.unscale_(optim)
                torch.nn.utils.clip_grad_norm_(upsample_block.parameters(), max_norm=1.0)
                # optim.step()
                scaler.step(optim)
                scaler.update()
                optim.zero_grad(set_to_none=True)
        
            del (
                images,
                masks,
                text_emb,
                loss,
                t,
                generated_image,
                generated_image_resized,
            )
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
