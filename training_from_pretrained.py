import csv
import gc
import os
from matplotlib import pyplot as plt
import pandas as pd
import torch
from diffusers import ControlNetModel, StableDiffusionControlNetPipeline
from transformers import CLIPTextModel, CLIPModel
from tqdm import tqdm
from data import dataloader
from diffusion_utils import load_latest_checkpoint, save_checkpoint

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

def train_model(nn_model, data_loader, start_epoch, n_epoch):
    for ep in range(start_epoch, num_epochs):
        epoch_loss = 0.0
        
        pbar = tqdm(data_loader, mininterval=2)
        for batch in data_loader:
            images, masks, text_emb = batch
            images, masks, text_emb = (
                images.to(device).to(dtype=torch.float16),
                masks.to(device).to(dtype=torch.float16),
                text_emb.to(device).to(dtype=torch.float16),
            )
            t = torch.randint(1, timesteps + 1, (images.shape[0],)).to(device)
             
            latents = vae.encode(images).latent_dist.sample().to(device)
            # Forward pass
            generated_images = nn_model(
                sample=latents,
                timestep=t,
                encoder_hidden_states=text_emb,
                controlnet_cond=masks,  # Segmentation masks as conditioning
                guess_mode=True
            )

            # Compute loss
            loss = criterion(generated_images, images)  # Depending on your task

            epoch_loss += loss.item()
            # Backward pass and optimization
            optim.zero_grad()
            loss.backward()
            optim.step()
        
        # Calculate and log average loss for the epoch
        avg_loss = epoch_loss / len(dataloader)
            
        with open(loss_file_path, mode="a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([ep + 1, avg_loss])

        print(f"Epoch {ep+1}/{num_epochs}, Loss: {loss.item()}")
        
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
