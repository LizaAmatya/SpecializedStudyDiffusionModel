import os
import torch
from data import test_dataloader
from diffusers import UniPCMultistepScheduler
from training_from_pretrained import controlnet, pipe
from PIL import Image
from torch_fidelity import calculate_metrics


save_dir = "weights/controlnet/"
real_images_dir = "weights/controlnet/real_img/"
gen_images = "weights/controlnet/gen_images/"
fid_log = os.path.join(save_dir, "fid.txt")

device = (
    torch.device("cuda")
    if torch.cuda.is_available()
    else torch.device("mps")
    if torch.backends.mps.is_available()
    else torch.device("cpu")
)

prompt = ["A myrtle warbler bird flying in a stormy weather", 
          "Two birds on top of a tree", 
          "A bird flying on a sunny and clear sky",
          "Phoenix rising from ashes"] 


def sample_from_controlnet():
    controlnet.eval()
    image, mask, text_emb = next(iter(test_dataloader))
    image, mask = image.to(device), mask.to(device)

    # for image, mask, text_emb in test_dataloader:
    image, mask = image.to(device), mask.to(device)
    mask_rgb = mask.repeat(1, 3, 1, 1)

    # Save each image to the real_images_dir to calculate FID scores
    for i, img in enumerate(image):
        img = (
            (img * 255).clamp(0, 255).byte()
        )  # Scale and convert to 8-bit format if needed
        img_pil = Image.fromarray(
            img.permute(1, 2, 0).cpu().numpy()
        )  # Convert tensor to PIL image
        img_pil.save(os.path.join(real_images_dir, f"real_image_{i}.png"))

    # Generate images
    with torch.no_grad():
        pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
        pipe.enable_model_cpu_offload()

        generator = torch.manual_seed(0)
        generated_images = pipe(
            prompt, num_inference_steps=30, generator=generator, image=mask_rgb
        ).images

    for i, gen_image in enumerate(generated_images):
        gen_image.save(os.path.join(gen_images, "/sample+{i}.png"))
        
    # Calculate FID
    metrics = calculate_metrics(
        input1=real_images_dir,
        input2=gen_images,
        cuda=True,
        isc=False,  # Inception Score, set to False if not needed
        fid=True,  # FID calculation enabled
    )
    fid_score = metrics["frechet_inception_distance"]
    print(f"FID Score: {fid_score}")
    # # Log FID score
    with open("fid_log.txt", "a") as f:
        f.write(f"FID: {fid_score:.4f}\n")

def main():
    sample_from_controlnet()

if __name__ == "__main__":
    main()
    