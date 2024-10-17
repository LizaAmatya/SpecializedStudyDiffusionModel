import torch
import torch.nn as nn
from diffusion_utils import EmbedFC, ResidualConvBlock, UnetDown, UnetUp
import os
import numpy as np


class ContextUnet(nn.Module):
    def __init__(
        self, in_channels, n_feat=256, n_cfeat=10, height=128, clip_embed_dim=512,
    ):  # cfeat - context features
        super(ContextUnet, self).__init__()

        # number of input channels, number of intermediate feature maps and number of classes
        self.in_channels = in_channels
        self.n_feat = n_feat
        self.n_cfeat = n_cfeat
        self.h = height  # assume h == w. must be divisible by 4, so 28,24,20,16...
        
        # include clip embeds
        self.clip_embedding_layer = nn.Linear(clip_embed_dim, 512)

        # Initialize the initial convolutional layer
        self.init_conv = ResidualConvBlock(in_channels, n_feat, is_res=True)

        # Initialize the down-sampling path of the U-Net with two levels
        self.down1 = UnetDown(n_feat, n_feat)  # down1 #[10, 256, 8, 8]     torch.Size([3, 64, 256, 256]) 
        self.down2 = UnetDown(n_feat, 2 * n_feat)  # down2 #[10, 256, 4,  4]    torch.Size([3, 128, 128, 128])


        # original: self.to_vec = nn.Sequential(nn.AvgPool2d(7), nn.GELU())
        self.to_vec = nn.Sequential(nn.Identity(), nn.GELU())


        # Embed the timestep and context labels with a one-layer fully connected neural network
        self.timeembed1 = EmbedFC(1, 2 * n_feat)
        self.timeembed2 = EmbedFC(1, 1 * n_feat)
        self.contextembed1 = EmbedFC(n_cfeat, 2 * n_feat)
        self.contextembed2 = EmbedFC(n_cfeat, 1 * n_feat)

        # Initialize the up-sampling path of the U-Net with three levels
        self.up0 = nn.Sequential(
            nn.ConvTranspose2d(
                2 * n_feat, 2 * n_feat, kernel_size=3, stride=1, padding=1      # given down2 [3,128,128] gives same output dim for up1
            ),  # up-sample
            nn.GroupNorm(8, 2 * n_feat),  # normalize
            nn.ReLU(),
        )
        self.up1 = UnetUp(4 * n_feat, n_feat)
        self.up2 = UnetUp(2 * n_feat, n_feat)

        # Initialize the final convolutional layers to map to the same number of channels as the input image
        self.out = nn.Sequential(
            nn.Conv2d(
                2 * n_feat, n_feat, 3, 1, 1
            ),  # reduce number of feature maps   #in_channels, out_channels, kernel_size, stride=1, padding=0
            nn.GroupNorm(8, n_feat),  # normalize
            nn.ReLU(),
            nn.Conv2d(
                n_feat, self.in_channels, 3, 1, 1
            ),  # map to same number of channels as input
        )

    def forward(self, x, t, c=None):
        """
        x : (batch, n_feat, h, w) : input image
        t : (batch, n_cfeat)      : time step
        c : (batch, n_classes)    : context label
        """
        # x is the input image, c is the context label, t is the timestep, context_mask says which samples to block the context on

        # pass the input image through the initial convolutional layer
        x = self.init_conv(x)
        # pass the result through the down-sampling path
        down1 = self.down1(x)  # [10, 256, 8, 8]
        down2 = self.down2(down1)  # [10, 256, 4, 4]
        
        # print('down 1 , down 2 dim', down1.shape, down2.shape)
        # convert the feature maps to a vector and apply an activation
        hiddenvec = self.to_vec(down2)      # giving wrong dim [3,128,32,32] should be same as down2 [3,128,128,128]
        # print('hiddenvecc-----', hiddenvec.shape)

        # mask out context if context_mask == 1
        if c is None:
            c = torch.zeros(x.shape[0], self.n_cfeat).to(x)

        # embed context and timestep
        cemb1 = self.contextembed1(c).view(-1, self.n_feat * 2, 1, 1)  # (batch, 2*n_feat, 1,1)
        # print("Shape of t before embedding:", t.shape)
        t = t.unsqueeze(1).float()
        # print("Shape of t before embedding-----:", t.shape)

        temb1 = self.timeembed1(t)
        # print("Shape of temb1 before view():", temb1.shape)

        temb1 = self.timeembed1(t).view(-1, self.n_feat * 2, 1, 1)
        cemb2 = self.contextembed2(c).view(-1, self.n_feat, 1, 1)
        temb2 = self.timeembed2(t).view(-1, self.n_feat, 1, 1)
        # print(f"uunet forward: cemb1 {cemb1.shape}. temb1 {temb1.shape}, cemb2 {cemb2.shape}. temb2 {temb2.shape}")

        up1 = self.up0(hiddenvec)
        # print('up 1', up1.shape)
        up2 = self.up1(cemb1 * up1 + temb1, down2)  # add and multiply embeddings
        up3 = self.up2(cemb2 * up2 + temb2, down1)
        out = self.out(torch.cat((up3, x), 1))
        return out


# hyperparameters

# diffusion hyperparameters
timesteps = 500
beta1 = 1e-4
beta2 = 0.02

# network hyperparameters
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")

n_feat = 64  # 64 hidden dimension feature
n_cfeat = 5  # context vector is of size 5
height = 128  # 64x64 image
save_dir = "weights/bird_ds"

# construct DDPM noise schedule
b_t = (beta2 - beta1) * torch.linspace(0, 1, timesteps + 1, device=device) + beta1
# b_t = (beta2 -beta1 )* torch.tensor([1 + np.cos(np.pi*t/timesteps) for t in range(timesteps+1)], device=device) + beta1
a_t = 1 - b_t
ab_t = torch.cumsum(a_t.log(), dim=0).exp()
# ab_t = torch.cumprod(a_t, dim=0)
ab_t[0] = 1

# Add clip embeds
nn_model = ContextUnet(in_channels=3, n_feat=n_feat, n_cfeat=n_cfeat, height=height).to(device)

# os.makedirs(save_dir, exist_ok=True)
# torch.save(nn_model.state_dict(), os.path.join(save_dir, "model_trained.pth"))

# print("Model weights saved.")
