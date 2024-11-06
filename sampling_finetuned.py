import os
import torch
from data import test_dataloader
from diffusers import UniPCMultistepScheduler
from training_from_pretrained import controlnet, pipe
from torchvision.utils import save_image
save_dir = "weights/controlnet/"

device = (
    torch.device("cuda")
    if torch.cuda.is_available()
    else torch.device("mps")
    if torch.backends.mps.is_available()
    else torch.device("cpu")
)

prompt = "A myrtle warbler bird flying in a stormy weather" 


def sample_from_controlnet():
    controlnet.eval()
    image, mask, text_emb = next(iter(test_dataloader))
    image, mask = image.to(device), mask.to(device)

    # for image, mask, text_emb in test_dataloader:
    image, mask = image.to(device), mask.to(device)
    
    # Generate images
    with torch.no_grad():
        pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
        pipe.enable_model_cpu_offload()

        generator = torch.manual_seed(0)
        gen_image = pipe(prompt, num_inference_steps=30, generator=generator, image=mask).images[0]

    os.makedirs("generated_samples", exist_ok=True)
    # for i, gen_image in enumerate(generated_images):
    save_image(gen_image, f=os.path.join(save_dir, "gen_images/sample.png"))
        
        
    #     # Get features for FID calculation
    #     real_features.append(get_inception_features(images, inception_model))
    #     generated_features.append(get_inception_features(generated_images, inception_model))

    # # Calculate FID score
    # fid_score = calculate_fid(np.concatenate(real_features), np.concatenate(generated_features))
    # print(f"Epoch {epoch+1}, FID: {fid_score:.4f}")

    # # Log FID score
    # with open("fid_log.txt", "a") as f:
    #     f.write(f"Epoch {epoch+1}, FID: {fid_score:.4f}\n")


def main():
    sample_from_controlnet()

if __name__ == "__main__":
    main()
    