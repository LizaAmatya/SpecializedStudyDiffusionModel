from datasets import load_dataset

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
# import numpy as np


dataset = load_dataset("dpdl-benchmark/caltech_birds2011", split='train')

# Define a transformation to convert images to tensors
transform = transforms.Compose(
    [
        transforms.Resize((128, 128)),  # Resize images to 512x512
        transforms.ToTensor(),  # Convert image to PyTorch tensor
        transforms.RandomHorizontalFlip(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),  # Normalize images Images already in range [0,1]
    ]
)


transform_mask = transforms.Compose(
    [
        transforms.Resize((128, 128), interpolation=Image.NEAREST),  # Resize the mask
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
        data = self.dataset[idx]
        image = data["image"]
        
        # image_array = np.array(image)
        label = data.get("label_name", None)
        
        print('-----label name', label)

        # Check if the image is a file path (string) or already a PIL image
        if isinstance(image, str):
            # If it's a file path, open it using Image.open()
            image = Image.open(image)
            
        if self.transform_mask is not None:
            seg_mask_img = data['segmentation_mask']    # seg mask already in grayscale
            mask = self.transform_mask(seg_mask_img)
            mask = mask.long()  # Ensure mask is long tensor

            # if len(mask.shape) == 2:
            #     mask = mask.unsqueeze(0) 
                
        # print('mask', mask.shape)
            
        # If it's already a PIL image, no need to open it, transform it 
        image_tensor = self.transform(image)
        
        # Process the label into a CLIP embedding
        inputs = self.processor(text=label, images=image, return_tensors="pt", padding=True)
        
        text_inputs = inputs['input_ids']
        # image_inputs = inputs['pixel_values']
        with torch.no_grad():
            text_embeddings = self.clip_model.get_text_features(input_ids=text_inputs)
            # image_embeddings = self.clip_model.get_image_features(pixel_values=image_inputs)

        
        # print('------text and image embeds',  image_embeddings.shape, text_embeddings.shape)
       
        return image_tensor, mask, text_embeddings


# Wrap Hugging Face dataset into a PyTorch Dataset
bird_ds = BirdGenDataset(dataset, transform, transform_mask=transform_mask)

batch_size = 16
# Create DataLoader to load batches of data
dataloader = DataLoader(bird_ds, batch_size=batch_size, shuffle=True)

# Iterate over batches of data -- for training and sampling

# for batch in dataloader:

#     image_tensor, image_embeds, text_embeds = batch
#     device = torch.device("cuda") if torch.cuda.is_available() else torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")

#     image_batch_gpu = image_tensor.to(device)
#     image_embeds_gpu = image_embeds.to(device)
#     text_embeds_gpu = text_embeds.to(device)

#     print('batch images shape ',image_tensor.shape)
#     print('image embed shape', image_embeds.shape)
#     print('text embed shape', text_embeds.shape)

#     break

# del image_tensor, text_embeds, image_embeds
# torch.cuda.empty_cache()
