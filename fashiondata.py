#  https://huggingface.co/zalando-datasets/fashion_mnist

# Label	Description
# 0	T-shirt/top
# 1	Trouser
# 2	Pullover
# 3	Dress
# 4	Coat
# 5	Sandal
# 6	Shirt
# 7	Sneaker
# 8	Bag
# 9	Ankle boot


from datasets import load_dataset
from matplotlib import pyplot as plt
import torch

from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image

dataset = load_dataset("zalando-datasets/fashion_mnist")

# for example in dataset["train"].select(range(5)):
#     image = example["image"]  # Adjust based on your dataset's feature names
#     plt.imshow(image)
#     plt.axis("off")
#     plt.show()
    
    
# print('dataset', dataset['train'][:10])

# Define a transformation to convert images to tensors
transform = transforms.Compose(
    [
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((64, 64)),  # Resize images to 512x512
        transforms.ToTensor(),  # Convert image to PyTorch tensor
        transforms.RandomHorizontalFlip(),
        transforms.Normalize([0.5], [0.7]),  # Normalize images
    ]
)

# Create a custom dataset class to load the data with transformations
class FashionGenDataset(Dataset):
    def __init__(self, hf_dataset, transform):
        self.dataset = hf_dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        data = self.dataset['train'][idx]
        image =data['image']
        
        # Check if the image is a file path (string) or already a PIL image
        if isinstance(image, str):
            # If it's a file path, open it using Image.open()
            image = Image.open(image)
            
        # If it's already a PIL image, no need to open it, transform it
        
        image = self.transform(image)
        print('imgae+++++++', image.shape)
        label = data.get("label", None)
        # transformed image and label tensor data
        return image, label

# image_tensors = [
#     move_to_gpu(sample) for sample in dataset["train"][:10]['image']
# ]  # Example: first 10 images

# print('image tensors', image_tensors)

# Wrap Hugging Face dataset into a PyTorch Dataset
fashion_dataset = FashionGenDataset(dataset, transform)

print('dataset----', fashion_dataset[0])    #transformed image and label tensor data
# Create DataLoader to load batches of data
dataloader = DataLoader(fashion_dataset, batch_size=16, shuffle=True)

# Iterate over batches of data -- for training and sampling
# for batch in dataloader:
    
#     # Example: load and transform the images and move to gpu
#     # images = [transform(img) for img in batch]
#     # print('----', images)
#     # image_batch = torch.stack(images)  
    
#     images, labels = batch
#     image_batch_gpu = images.to(
#         "mps" if torch.backends.mps.is_available() else "cpu"
#     )
    
#     print('batch images shape ',image_batch_gpu.shape)
#     print('images shape', images)

#     break
