import gc
import os
import torch
from data import test_dataloader
from diffusers import UniPCMultistepScheduler
from PIL import Image
from torch_fidelity import calculate_metrics
from diffusers import ControlNetModel, StableDiffusionControlNetPipeline


save_dir = "weights/controlnet/"
real_images_dir = "weights/controlnet/finetuned/real_img/"
gen_images = "weights/controlnet/finetuned/gen_images/"
fid_log = os.path.join(save_dir, "fid.txt")

model_id = "lllyasviel/control_v11p_sd15_seg"
controlnet = ControlNetModel.from_pretrained(model_id, torch_dtype=torch.float16, weights_only=True)
controlnet.eval()

pipe = StableDiffusionControlNetPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5", controlnet=controlnet, torch_dtype=torch.float16
)


device = (
    torch.device("cuda")
    if torch.cuda.is_available()
    else torch.device("mps")
    if torch.backends.mps.is_available()
    else torch.device("cpu")
)
pipe.to(device)

# Sampled using non finetuned -- only controlnet
# prompt = [
#     "A  bird flying in a stormy weather",
#     "Two birds on top of a branch of a tree",
#   "A bird flying on a sunny and clear sky",
#   "Phoenix rising from ashes"]

# Using Finedtuned 
# prompt = ["A  mockingbird flying in a stormy weather", 
#           "A blue bird on top of a branch of a tree", 
#           "A bird flying on a sunny and clear sky",
#           "Phoenix rising from ashes"] 
prompt = [
    "warbler",
    "sparrow",
    "mockingbird",
    "phoenix"
]

torch.cuda.empty_cache()
gc.collect()

model_path = os.path.join(save_dir + "/model_epoch_0.pth")
checkpoint = torch.load(f=model_path, map_location='cpu', weights_only=True)
controlnet.load_state_dict(checkpoint["model_state_dict"], strict=False)
controlnet.to(device)


def sample_from_controlnet():
    image, mask, text_emb = next(iter(test_dataloader))
    image, mask = image.to(device), mask.to(device)

    # for image, mask, text_emb in test_dataloader:
    mask_rgb = mask.repeat(1, 3, 1, 1)

    # Save each image to the real_images_dir to calculate FID scores
    for i, img in enumerate(image):
        img = (
            (img * 255).clamp(0, 255).byte()
        )  # Scale and convert to 8-bit format if needed
        img_pil = Image.fromarray(
            img.permute(1, 2, 0).cpu().numpy()
        )  # Convert tensor to PIL image
        img_pil.save(os.path.join(real_images_dir, f"v3_real_image_{i}.png"))

    # Generate images
    with torch.no_grad():
        pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
        pipe.enable_model_cpu_offload()

        generator = torch.manual_seed(0)
        generated_images = pipe(
            prompt, num_inference_steps=100, generator=generator, image=mask_rgb
        ).images

    for i, gen_image in enumerate(generated_images):
        gen_image.save(os.path.join(gen_images, f"v3_sample_{i}.png"))
        
    def check_image_size(image_path):
        img = Image.open(image_path)
        print('image.size', image.size)
        return img.size  # (width, height)

    real_images_sizes = [check_image_size(os.path.join(real_images_dir, img)) for img in os.listdir(real_images_dir)]
    gen_images_sizes = [check_image_size(os.path.join(gen_images, img)) for img in os.listdir(gen_images)]
    
    if len(set(real_images_sizes)) > 1 or len(set(gen_images_sizes)) > 1:
        raise ValueError("Not all images have the same size.")
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
        f.write(f"v3 - FID: {fid_score:.4f}\n")

def main():
    sample_from_controlnet()

if __name__ == "__main__":
    main()
    