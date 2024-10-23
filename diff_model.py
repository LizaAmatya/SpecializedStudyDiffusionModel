import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusion_utils import EmbedFC, ResidualConvBlock, UnetDown, UnetUp

# When attn_dim != n_feat (projection required)
class AttentionBlock(nn.Module):
    def __init__(self, n_feat, attn_dim, num_heads=8):
        super(AttentionBlock, self).__init__()

        self.proj_qkv = nn.Linear(
            n_feat, attn_dim
        )  # Project n_feat to attn_dim for attention
        
        # Attention layer for image embeddings and text/mask embeddings
        self.attn = nn.MultiheadAttention(
            embed_dim=attn_dim, num_heads=num_heads, batch_first=True
        )
        self.fc = nn.Linear(attn_dim, n_feat)  # Map back to n_feat after attention
       
        # Layer normalization before and after the attention mechanism
        self.norm1 = nn.LayerNorm(attn_dim)  # Normalizing after projection
        # self.norm2 = nn.LayerNorm(n_feat)  # Normalizing after final output
        self.norm2 = nn.BatchNorm2d(n_feat)     #Normalize after final output for 4D tensors
        
    def forward(self, x):
        batch_size, channels, height, width = x.shape

        # Reshape the feature maps for multi-head attention
        x_reshaped = x.view(batch_size, channels, height * width).permute(
            0, 2, 1
        )  # Shape: [batch, seq_len, n_feat]

        # Project input to attention dimension
        x_proj = self.proj_qkv(x_reshaped)

        # Apply LayerNorm before attention
        x_proj = self.norm1(x_proj)

        # Apply attention mechanism
        attn_output, _ = self.attn(x_proj, x_proj, x_proj)
        print(
            "attn output 111-----", attn_output.shape
        )

        # Map the output back to original feature space (n_feat)
        attn_output = self.fc(attn_output)

        # Reshape back to original shape
        attn_output = attn_output.permute(0, 2, 1).view(
            batch_size, channels, height, width
        )
        
        # print("attn output----- original shape", attn_output.shape)

        # Apply LayerNorm after attention used batch norm for 4D tensor
        attn_output = self.norm2(attn_output)
        print("attn output----- layer norm", attn_output.shape, attn_output.mean(), attn_output.std())

        return attn_output


class ContextUnet(nn.Module):
    def __init__(
        self,
        in_channels,
        n_feat=256,
        seg_mask_dim=128,  # Segmentation mask dimension
        text_embed_dim=512, # text embed dim
        height=128,
        attn_dim=512,
    ):  # cfeat - context features
        super(ContextUnet, self).__init__()

        # number of input channels, number of intermediate feature maps and number of classes
        self.in_channels = in_channels
        self.n_feat = n_feat
        self.text_embed_dim = text_embed_dim
        self.seg_mask_dim = seg_mask_dim
        self.h = height  # assume h == w. must be divisible by 4, so 28,24,20,16...

        # Timestep embeddings
        self.timeembed1 = EmbedFC(1, 2 * n_feat, activation_fn=nn.GELU())
        # self.timeembed2 = EmbedFC(1, 1 * n_feat, activation_fn=nn.GELU())

        # Initialize embedding layers
        # self.clip_embedding_layer = nn.Linear(clip_embed_dim, n_feat*2) #No need used text embed layer
        self.text_embedding_layer = EmbedFC(
            text_embed_dim, n_feat * 2, activation_fn=nn.GELU()
        )
        self.segmentation_embedding_layer = EmbedFC(
            1, seg_mask_dim, activation_fn=nn.SiLU(), use_conv=True 
        ) # For image seg masks 2d images for context use this with Conv2d layer -- and input dim = 1 for grayscale
        
        self.attn_block = AttentionBlock(n_feat=n_feat*2, attn_dim=attn_dim, num_heads=4)
        # Initialize the initial convolutional layer
        self.init_conv = ResidualConvBlock(in_channels, n_feat, is_res=True)

        # Initialize the down-sampling path of the U-Net with two levels
        self.down1 = UnetDown(
            n_feat, n_feat
        ) 
        self.down2 = UnetDown(
            n_feat, 2 * n_feat
        )  

        # original: self.to_vec = nn.Sequential(nn.AvgPool2d(7), nn.GELU())
        self.to_vec = nn.Sequential(nn.Identity(), nn.GELU())

        # Initialize the up-sampling path of the U-Net with three levels
        self.up0 = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=2 * n_feat,
                out_channels=2 * n_feat,
                kernel_size=3,
                stride=1,
                padding=1,  # given down2 [3,128,128] gives same output dim for up1
            ),  # up-sample
            nn.GroupNorm(8, 2 * n_feat),  # normalize
            nn.SiLU(),
        )
        self.up1 = UnetUp(4 * n_feat, n_feat)
        self.up2 = UnetUp(2 * n_feat, n_feat)

        # Initialize the final convolutional layers to map to the same number of channels as the input image
        self.out = nn.Sequential(
            nn.Conv2d(
                2 * n_feat, n_feat, 3, 1, 1
            ),  # reduce number of feature maps   #in_channels, out_channels, kernel_size, stride=1, padding=0
            nn.GroupNorm(8, n_feat),  # normalize
            nn.SiLU(),
            nn.Conv2d(
                n_feat, self.in_channels, 3, 1, 1
            ),  # map to same number of channels as input
        )

    def forward(self, x, t, text_input=None, seg_input=None):
        """
        x : (batch, n_feat, h, w) : input image
        t : (batch, n_cfeat)      : time step
        c : (batch, n_classes)    : context label
        """
        # x is the input image, c is the context label, t is the timestep, context_mask says which samples to block the context on

        # pass the input image through the initial convolutional layer
        x = self.init_conv(x)
        x = self.to_vec(x)      # Apply activation function
        # print('x shape', x.shape)
        
        # pass the result through the down-sampling path
        down1 = self.down1(x)  
        print('down 1 shape', down1.shape)
        down2 = self.down2(down1)  #[16, 128, 32, 32]
        print('down2 shape', down2.shape)
        
        # Embed the text and segmentation mask
        text_embedding = self.text_embedding_layer(text_input).view(
            -1, self.n_feat * 2, 1, 1
        )
        print('seg input', seg_input.shape, self.n_feat)
        seg_input = seg_input.float()   # mismatch in bias type and input type (was long)
        seg_embedding = self.segmentation_embedding_layer(seg_input)
        print('seg embeds', seg_embedding.shape)

        # convert the feature maps to a vector and apply an activation
        down2 = self.to_vec(down2)  
        print(f'down 2 input dtype and device', down2.device, down2.dtype)

        t = t.unsqueeze(1).float()
        # print("Shape of t before embedding-----:", t.shape)

        temb1 = self.timeembed1(t)
        # print("Shape of temb1 before view():", temb1.shape)

        temb1 = self.timeembed1(t).view(-1, self.n_feat * 2, 1, 1)
        # Testing without temb2 see the impact on training and use elsewhere
        # temb2 = self.timeembed2(t).view(-1, self.n_feat, 1, 1)
        
        # Combine embeddings
        # print('all embeds dim', text_embedding.shape, seg_embedding.shape, temb1.shape, temb2.shape)
        combined_embeds = text_embedding + seg_embedding + temb1
        
        # print('combined embeds', combined_embeds.shape, combined_embeds.device, combined_embeds.dtype)
        
        # Downsample the combined embeddings too big for memory consumption when passed to attn block
        downsample_layer = nn.Conv2d(combined_embeds.shape[1], combined_embeds.shape[1], kernel_size=3, stride=2, padding=1).to(device)     # float32 to float16 handle mismatch
        # print(f'layer device -- {next(downsample_layer.parameters()).device}')
        
        combined_embeds_downsampled = downsample_layer(combined_embeds)  # Reduce spatial dimensions

        # print('downsampled layer', combined_embeds_downsampled.shape)
        # Apply attention mechanism after downsampling
        attn_output = self.attn_block(combined_embeds_downsampled)
        
        print('attn output', attn_output.shape)

        # Upsample down2 to solve mismatch in dim during up2
        up_down2 = nn.ConvTranspose2d(128, 128, kernel_size=2, stride=2).to(device)(down2)
        print(f'layer up down2 device -- {up_down2.device}')
        
        up1 = self.up0(attn_output)
        print(f'up1 shape {up1.shape}, down1 {down1.shape}, up_down2 {up_down2.shape}, down2 {down2.shape}, combined embeds {combined_embeds_downsampled.shape}')
        up2 = self.up1(attn_output * up1 + combined_embeds_downsampled, up_down2)
        
        attn_out_adjusted = nn.Conv2d(128, 64, kernel_size=1).to(device)(attn_output)

        combined_embeds_adjusted = nn.Conv2d(128, 64, kernel_size=1).to(device)(combined_embeds_downsampled)
        print('up2 shape', up2.shape)
        print('attn out adjusted', attn_out_adjusted.shape)
        print('combined embeds adjust', combined_embeds_adjusted.shape)
        
        # up2_downsampled = F.interpolate(up2, size=(64, 64), mode='bilinear', align_corners=True) #Use Conv2d layer to preserve while downsampling instead of interpolation
        up2_downsampled = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1).to(device)(up2)
        print('downsampled up2', up2_downsampled.shape)
        # Combine up2 with the adjusted outputs
        # If your architecture incorporates an attention mechanism and you want attn_out_adjusted to influence the features in up2, 
        # then you might use multiplication:
        combined = up2_downsampled * attn_out_adjusted + combined_embeds_adjusted

        print('combined', combined.shape)
        up3 = self.up2(combined, down1)

        print('up3 tensor shape', up3.shape)
        # Final output
        out = self.out(torch.cat((up3, x), 1))
        print('self out', out.shape)
        return out


# Hyperparams

n_feat = 64
batch_size = 4
in_channels = 3
height = 128
device = (
    torch.device("cuda")
    if torch.cuda.is_available()
    else torch.device("mps")
    if torch.backends.mps.is_available()
    else torch.device("cpu")
)
save_dir = "weights/data_context/"


# Instantiate the model
nn_model = ContextUnet(
    in_channels=in_channels,
    n_feat=n_feat,
    height=height,
    seg_mask_dim=128
)

nn_model.to(device)      #mismatch in named params bias and float 16 so converting into float16
