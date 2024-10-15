from datasets import load_dataset

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import numpy as np


# Checking CUDA

# print('Using CUDA-------------',torch.cuda.is_available(), torch.cuda.device_count())


dataset = load_dataset("dpdl-benchmark/caltech_birds2011")

# print('----here', dataset)
# for example in dataset["train"].select(range(5)):
#     image = example["image"]  # Adjust based on your dataset's feature names
    
#     print('image', image)


# print('dataset', dataset['train'][:10])

# Define a transformation to convert images to tensors
transform = transforms.Compose(
    [
        # transforms.Grayscale(num_output_channels=3),
        transforms.Resize((512, 512)),  # Resize images to 512x512
        transforms.ToTensor(),  # Convert image to PyTorch tensor
        transforms.RandomHorizontalFlip(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),  # Normalize images Images already in range [0,1]
    ]
)


transform_mask = transforms.Compose(
    [
        transforms.Resize((512, 512), interpolation=Image.NEAREST),  # Resize the mask
        transforms.ToTensor(),  # Convert the mask to a PyTorch tensor (values between 0 and 1)
    ]
)


# Create a custom dataset class to load the data with transformations
class BirdGenDataset(Dataset):
    def __init__(
        self,
        hf_dataset,
        transform,
        transform_mask=None,
    ):
        self.dataset = hf_dataset
        self.transform = transform
        self.transform_mask = transform_mask
        
        # self.clip_tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        data = self.dataset["train"][idx]
        image = data["image"]
        
        # image_array = np.array(image)
        label = data.get("label_name", None)
        
        print('-----label name', label)

        # Check if the image is a file path (string) or already a PIL image
        if isinstance(image, str):
            # If it's a file path, open it using Image.open()
            image = Image.open(image)
            
        if self.transform_mask is not None:
            seg_mask_img = data['segmentation_mask']
            mask = Image.open(seg_mask_img).convert("L")  # Convert to grayscale
            mask = self.transform_mask(mask)
            mask = mask.long()  # Ensure mask is long tensor
            return image, label, mask

        # If it's already a PIL image, no need to open it, transform it 
        image_tensor = self.transform(image)
        
        # Check the range
        print("Min pixel value:", torch.min(image_tensor))
        print("Max pixel value:", torch.max(image_tensor))
        print("imgae+++++++", image_tensor.shape, type(image_tensor), image_tensor.dtype)
        
        # Process the label into a CLIP embedding
        # Initialize CLIP tokenizer and text model for encoding captions
        
        # text_inputs = self.processor.process([label])  # Tokenize the label
        # inputs = self.processor(text=label, images=images, return_tensors="pt", padding=True)
        # Without text and seg mask
        
        image_inputs = self.processor(images=image, return_tensors="pt", padding=True, do_rescale=False)

        # print('inputs', inputs)
        
        with torch.no_grad():
            # outputs = self.clip_model(**inputs)   For both text and image input
            # text_embeddings = self.clip_model.get_text_features(**text_inputs)
            image_embeddings = self.clip_model.get_image_features(**image_inputs)

        # Feed embeddings into your diffusion model or other components
        # diffusion_model_output = diffusion_model(images, text_embeddings, image_embeddings)

        print('------text and image embeds',  image_embeddings)
        # return image, text_features, seg_mask
        return image_tensor, image_embeddings


# Wrap Hugging Face dataset into a PyTorch Dataset
bird_ds = BirdGenDataset(dataset, transform)    # without seg mask

# Conditional with segmentation mask
# bird_ds = BirdGenDataset(dataset, transform, transform_mask=transform_mask)  # withseg mask

print("dataset----", bird_ds[0])  # transformed image and label tensor data

# Create DataLoader to load batches of data
dataloader = DataLoader(bird_ds, batch_size=16, shuffle=True)

# Iterate over batches of data -- for training and sampling
# Unconditional sampling
for batch in dataloader:

    # Example: load and transform the images and move to gpu
    print('inside loop', batch) 
    # images, labels, seg_mask = batch      # with context 
    
    # without context
    images_tensor, image_embeds = batch
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")

    print('----------device', device)
    image_batch_gpu = images_tensor.to(device)
    image_embeds_gpu = image_embeds.to(device)
    # image_batch_gpu = images.to("cuda" if torch.cuda.is_available() else "cpu")

    print('batch images shape ',image_batch_gpu.shape)
    print('image embed shape', image_embeds.shape)

    break
