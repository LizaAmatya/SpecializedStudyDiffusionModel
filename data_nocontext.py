from datasets import load_dataset

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image

dataset = load_dataset("dpdl-benchmark/caltech_birds2011", split="train") #without train split it only shows 2 dataset len so add train split 

print('----here', dataset, len(dataset))
# for example in dataset["train"].select(range(5)):
#     image = example["image"]  # Adjust based on your dataset's feature names
    
#     print('image', image)


# print('dataset', dataset['train'][:10])

# Define a transformation to convert images to tensors
transform = transforms.Compose(
    [
        # transforms.Grayscale(num_output_channels=3),
        transforms.Resize((256, 256)),  # Resize images to 512x512
        transforms.ToTensor(),  # Convert image to PyTorch tensor
        transforms.RandomHorizontalFlip(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5,0.5]),  # Normalize images Images already in range [0,1]
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

        # If it's already a PIL image, no need to open it, transform it 
        image_tensor = self.transform(image)
        
        # Check the range
        # print("Min pixel value:", torch.min(image_tensor))
        # print("Max pixel value:", torch.max(image_tensor))
        # print("imgae+++++++", image_tensor.shape, type(image_tensor), image_tensor.dtype)
        
        return image_tensor

batch_size = 16
# Wrap Hugging Face dataset into a PyTorch Dataset
bird_ds = BirdGenDataset(dataset, transform)    # without seg mask

# print("dataset----", bird_ds[0])  # transformed image and label tensor data
# Create DataLoader to load batches of data
dataloader = DataLoader(bird_ds, batch_size=batch_size, shuffle=True,pin_memory=
                        True, num_workers=2, prefetch_factor=2)

torch.cuda.empty_cache()
print('memory allocated after dataload', torch.cuda.memory_allocated())
print('memory reserved', torch.cuda.memory_reserved())

# # Iterate over batches of data -- for training and sampling
# print('len of dataloader',len(dataloader))
# # Unconditional sampling
# for batch in dataloader:

#     device = torch.device("cuda") if torch.cuda.is_available() else torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    
#     print('----------device', device)
#     image_batch_gpu = batch.to(device)

#     # print('image shape ', images_tensor.shape)
#     print("batch images shape ", image_batch_gpu.shape)

#     break
