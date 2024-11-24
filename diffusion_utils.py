import torch
import torch.nn as nn
import numpy as np
from torchvision.utils import save_image, make_grid
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import os
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from PIL import Image


class ResidualConvBlock(nn.Module):
    def __init__(
        self, in_channels: int, out_channels: int, is_res: bool = True
    ) -> None:
        super().__init__()

        # Check if input and output channels are the same for the residual connection
        self.same_channels = in_channels == out_channels

        # Flag for whether or not to use residual connection
        self.is_res = is_res

        # First convolutional layer
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels, out_channels, 3, 1, 1
            ),  # 3x3 kernel with stride 1 and padding 1
            nn.BatchNorm2d(out_channels),  # Batch normalization
            # nn.GELU(),  # GELU activation function
            nn.SiLU()
        )

        # Initialize weights
        self.apply(self.initialize_weights)

        # Second convolutional layer
        self.conv2 = nn.Sequential(
            nn.Conv2d(
                out_channels, out_channels, 3, 1, 1
            ),  # 3x3 kernel with stride 1 and padding 1
            nn.BatchNorm2d(out_channels),  # Batch normalization
            # nn.GELU(),  # GELU activation function
            nn.SiLU()
        )
        
    def initialize_weights(self, module):
        if isinstance(module, nn.Conv2d):
            nn.init.kaiming_uniform_(module.weight, nonlinearity="relu")
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.BatchNorm2d):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # If using residual connection
        if self.is_res:
            # Apply first convolutional layer
            x1 = self.conv1(x)

            # Apply second convolutional layer
            x2 = self.conv2(x1)

            # If input and output channels are the same, add residual connection directly
            if self.same_channels:
                out = x + x2
            else:
                # If not, apply a 1x1 convolutional layer to match dimensions before adding residual connection
                shortcut = nn.Conv2d(
                    x.shape[1], x2.shape[1], kernel_size=1, stride=1, padding=0
                ).to(x.device)
                out = shortcut(x) + x2
            # print(f"resconv forward: x {x.shape}, x1 {x1.shape}, x2 {x2.shape}, out {out.shape}")

            # Normalize output tensor
            return out / 1.414

        # If not using residual connection, return output of second convolutional layer
        else:
            x1 = self.conv1(x)
            x2 = self.conv2(x1)
            return x2

    # Method to get the number of output channels for this block
    def get_out_channels(self):
        return self.conv2[0].out_channels

    # Method to set the number of output channels for this block
    def set_out_channels(self, out_channels):
        self.conv1[0].out_channels = out_channels
        self.conv2[0].in_channels = out_channels
        self.conv2[0].out_channels = out_channels


class UnetUp(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UnetUp, self).__init__()

        # Create a list of layers for the upsampling block
        # The block consists of a ConvTranspose2d layer for upsampling, followed by two ResidualConvBlock layers
        layers = [
            nn.ConvTranspose2d(in_channels, out_channels, 2, 2),
            ResidualConvBlock(out_channels, out_channels),
            ResidualConvBlock(out_channels, out_channels),
        ]

        # Use the layers to create a sequential model
        self.model = nn.Sequential(*layers)
        

    def forward(self, x, skip):
        # Concatenate the input tensor x with the skip connection tensor along the channel dimension
        x = torch.cat((x, skip), 1)

        # Pass the concatenated tensor through the sequential model and return the output
        x = self.model(x)
        return x


class UnetDown(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UnetDown, self).__init__()

        # Create a list of layers for the downsampling block
        # Each block consists of two ResidualConvBlock layers, followed by a MaxPool2d layer for downsampling
        layers = [
            ResidualConvBlock(in_channels, out_channels),
            ResidualConvBlock(out_channels, out_channels),
            nn.MaxPool2d(kernel_size=2),
        ]

        # Use the layers to create a sequential model
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        # Pass the input through the sequential model and return the output
        return self.model(x)


class EmbedFC(nn.Module):
    def __init__(self, input_dim, emb_dim, activation_fn=nn.GELU(), use_conv=False):
        super(EmbedFC, self).__init__()
        """
        This class defines a generic one layer feed-forward neural network for embedding input data of
        dimensionality input_dim to an embedding space of dimensionality emb_dim.
        """
        self.input_dim = input_dim
        self.use_conv = use_conv

        print('input dim', self.input_dim , emb_dim)
        if use_conv:
            # When using Conv2d, we assume the input is a 4D tensor (batch, channels, height, width)
            layers = [
                nn.Conv2d(input_dim, emb_dim, kernel_size=3, stride=1, padding=1),
                activation_fn,
                nn.Conv2d(emb_dim, emb_dim, kernel_size=3, stride=1, padding=1),
            ]
        else:
            # Use fully connected layers for 1D input data (like text embedding)
            layers = [
                nn.Linear(input_dim, emb_dim),
                activation_fn,
                nn.Linear(emb_dim, emb_dim),
            ]
        
        # create a PyTorch sequential model consisting of the defined layers
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_conv:
            return self.model(x)  # For 2D data, apply convolutions
        else:
            # Flatten the input tensor if it's 2D
            x = x.view(-1, self.input_dim)
            return self.model(x)  # For 1D data, apply FC layers


def unorm(x):
    # unity norm. results in range of [0,1]
    # assume x (h,w,3)
    xmax = x.max((0, 1))
    xmin = x.min((0, 1))
    return (x - xmin) / (xmax - xmin)


def norm_all(store, n_t, n_s):
    # runs unity norm on all timesteps of all samples
    nstore = np.zeros_like(store)
    for t in range(n_t):
        for s in range(n_s):
            nstore[t, s] = unorm(store[t, s])
    return nstore


def norm_torch(x_all):
    # runs unity norm on all timesteps of all samples
    # input is (n_samples, 3,h,w), the torch image format
    x = x_all.cpu().numpy()
    xmax = x.max((2, 3))
    xmin = x.min((2, 3))
    xmax = np.expand_dims(xmax, (2, 3))
    xmin = np.expand_dims(xmin, (2, 3))
    nstore = (x - xmin) / (xmax - xmin)
    return torch.from_numpy(nstore)


def plot_grid(x, n_sample, n_rows, save_dir, w):
    # x:(n_sample, 3, h, w)
    ncols = n_sample // n_rows
    grid = make_grid(
        norm_torch(x), nrow=ncols
    )  # curiously, nrow is number of columns.. or number of items in the row.
    save_image(grid, save_dir + f"run_image_w{w}.png")
    print("saved image at " + save_dir + f"run_image_w{w}.png")
    return grid


def plot_sample(x_gen_store, n_sample, nrows, save_dir, fn, w, save=False):
    ncols = n_sample // nrows
    sx_gen_store = np.moveaxis(
        x_gen_store, 2, 4
    )  # change to Numpy image format (h,w,channels) vs (channels,h,w)
    nsx_gen_store = norm_all(
        sx_gen_store, sx_gen_store.shape[0], n_sample
    )  # unity norm to put in range [0,1] for np.imshow

    # create gif of images evolving over time, based on x_gen_store
    fig, axs = plt.subplots(
        nrows=nrows, ncols=ncols, sharex=True, sharey=True, figsize=(ncols, nrows)
    )

    def animate_diff(i, store):
        print(f"gif animating frame {i} of {store.shape[0]}", end="\r")
        plots = []
        for row in range(nrows):
            for col in range(ncols):
                axs[row, col].clear()
                axs[row, col].set_xticks([])
                axs[row, col].set_yticks([])
                plots.append(axs[row, col].imshow(store[i, (row * ncols) + col]))
        return plots

    ani = FuncAnimation(
        fig,
        animate_diff,
        fargs=[nsx_gen_store],
        interval=200,
        blit=False,
        repeat=True,
        frames=nsx_gen_store.shape[0],
    )
    plt.close()
    if save:
        ani.save(save_dir + f"{fn}_w{w}.gif", dpi=100, writer=PillowWriter(fps=5))
        print("saved gif at " + save_dir + f"{fn}_w{w}.gif")
    return ani


def normalize_image(img):
    # Normalize to [0,1] range
    return (img - img.min()) / (img.max() - img.min())


def save_checkpoint(model, optimizer, epoch, loss, save_dir):
    checkpoint_name = f"model_epoch_{epoch}.pth"
    checkpoint_path = os.path.join(save_dir, checkpoint_name)
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": loss,
    }
    with open(checkpoint_path, 'wb') as f:
        torch.save(checkpoint, f)
    print(f"Checkpoint saved at epoch {epoch}: {checkpoint_path}")
    
    
def load_latest_checkpoint(model, optimizer, save_dir, device='cuda'):
    checkpoint_files = [
        f
        for f in os.listdir(save_dir)
        if f.startswith("model_epoch_") and f.endswith(".pth")
    ]
    if len(checkpoint_files) == 0:
        print("No checkpoints found, starting from scratch.")
        return model, optimizer, 0, None  # No checkpoints found

    # Sort checkpoint files by epoch number (assuming format 'model_epoch_{epoch}.pth')
    checkpoint_files.sort(
        key=lambda f: int(f.split("_")[-1].split(".")[0])
    )  # Sorting by epoch number
    latest_checkpoint = checkpoint_files[-1]  # Get the latest one

    checkpoint_path = os.path.join(save_dir, latest_checkpoint)
    checkpoint = torch.load(checkpoint_path, map_location=device)

    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    epoch = checkpoint["epoch"]
    loss = checkpoint["loss"]

    print(f"Loaded checkpoint from epoch {epoch} at {checkpoint_path}")
    return model, optimizer, epoch, loss


def visualize_feature_maps(feature_map, layer_name, num_channels=6):
    print(f"Visualizing feature maps from: {layer_name}")
    feature_map = feature_map.cpu().detach()
    for i in range(min(num_channels, feature_map.size(0))):
        plt.imshow(feature_map[i], cmap="viridis")
        plt.title(f"Channel {i}")
        plt.axis("off")
        plt.show()
        