import numpy as np
import torch
import torch.nn.functional as F
import torch
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
#create a VAE with time step as input + noise as input and predict the noise
import torch
import torch.nn as nn
import torch.optim as optim
import math

class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings

#We use basic embedding concatenate time embeddings for training
class Autoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(Autoencoder, self).__init__()
        self.embedding = SinusoidalPositionEmbeddings(input_dim) # Embedding layer for integer input
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),

            nn.Linear(hidden_dim, hidden_dim//2),

            nn.Linear(hidden_dim//2, hidden_dim//4),

        )
        self.decoder = nn.Sequential(

            nn.Linear(hidden_dim//4, hidden_dim//2),

            nn.Linear(hidden_dim//2, hidden_dim),

            nn.Linear(hidden_dim, input_dim),

        )

    def forward(self, x, t):
        t_embed = self.embedding(t)
        #print(x.shape, t.shape, t_embed.shape)
        x = x + t_embed
        x = self.encoder(x)
        x = self.decoder(x)
        return x

class UNet(nn.Module):
    def __init__(self, embedding_dim, hidden_dim):
        super(UNet, self).__init__()
        self.embedding = SinusoidalPositionEmbeddings(embedding_dim) # Embedding layer for integer input
        self.down1 = nn.Linear(embedding_dim , hidden_dim)
        self.down2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.bottom = nn.Linear(hidden_dim // 2, hidden_dim // 2)
        self.up1 = nn.Linear(hidden_dim // 2, hidden_dim)
        self.up2 = nn.Linear(hidden_dim, embedding_dim)

    def forward(self, x, t):
        t_embed = self.embedding(t)
        #print(x.shape, t.shape, t_embed.shape)
        #x = torch.cat([x, t_embed], dim = 1)
        x = x + t_embed
        x1 = self.down1(x)
        x2 = self.down2(x1)
        x3 = self.bottom(x2)
        x4 = self.up1(x3 + x2) # Skip connection
        x5 = self.up2(x4 + x1) # Skip connection
        return x5
    

class FFC(nn.Module):
    def __init__(self, input_dim, output_dim, **kwargs) -> None:
        super().__init__(**kwargs)
        self.ffc = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.BatchNorm1d(output_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.ffc(x)
        return x

    
class UNet_1(nn.Module):
    def __init__(self, embedding_dim, hidden_dim):
        super(UNet_1, self).__init__()
        self.embedding = SinusoidalPositionEmbeddings(embedding_dim) # Embedding layer for integer input
        # self.down1 = nn.Linear(embedding_dim , hidden_dim)
        # self.down2 = nn.Linear(hidden_dim, hidden_dim // 2)
        # self.bottom = nn.Linear(hidden_dim // 2, hidden_dim // 2)
        # self.up1 = nn.Linear(hidden_dim // 2, hidden_dim)
        # self.up2 = nn.Linear(hidden_dim, embedding_dim)

        self.down1 = FFC(embedding_dim, hidden_dim)
        self.down2 = FFC(hidden_dim, hidden_dim // 2)
        self.bottom = FFC(hidden_dim // 2, hidden_dim // 2)
        self.up1 = FFC(hidden_dim // 2, hidden_dim)
        self.up2 = FFC(hidden_dim, embedding_dim)

    def forward(self, x, t):
        t_embed = self.embedding(t)
        #print(x.shape, t.shape, t_embed.shape)
        #x = torch.cat([x, t_embed], dim = 1)
        x = x + t_embed
        x1 = self.down1(x)
        x2 = self.down2(x1)
        x3 = self.bottom(x2)
        x4 = self.up1(x3 + x2) # Skip connection
        x5 = self.up2(x4 + x1) # Skip connection
        return x5

# based on the https://github.com/milesial/Pytorch-UNet/blob/master/unet/unet_parts.py

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv1d(in_channels, mid_channels, kernel_size=3, padding=1, stride=1, bias=False),
            nn.BatchNorm1d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(mid_channels, out_channels, kernel_size=3, padding=1, stride=1, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)
    

class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool1d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose1d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        # diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffY // 2, diffY - diffY // 2])
        # x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
        #                 diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)
    

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class UNet_1_0(nn.Module):
    def __init__(self, embedding_dim, hidden_dim):
        super(UNet_1_0, self).__init__()
        self.UNet_1_0 = SinusoidalPositionEmbeddings(embedding_dim) # Embedding layer for integer input
        # self.down1 = nn.Linear(embedding_dim , hidden_dim)
        # self.down2 = nn.Linear(hidden_dim, hidden_dim // 2)
        # self.bottom = nn.Linear(hidden_dim // 2, hidden_dim // 2)
        # self.up1 = nn.Linear(hidden_dim // 2, hidden_dim)
        # self.up2 = nn.Linear(hidden_dim, embedding_dim)

        self.down1 = DoubleConv(embedding_dim, hidden_dim)
        self.down2 = DoubleConv(hidden_dim, hidden_dim // 2)
        self.bottom = DoubleConv(hidden_dim // 2, hidden_dim // 2)
        self.up1 = DoubleConv(hidden_dim // 2, hidden_dim)
        self.up2 = DoubleConv(hidden_dim, embedding_dim)

    def forward(self, x, t):
        t_embed = self.embedding(t)
        #print(x.shape, t.shape, t_embed.shape)
        #x = torch.cat([x, t_embed], dim = 1)
        x = x + t_embed
        x1 = self.down1(x)
        x2 = self.down2(x1)
        x3 = self.bottom(x2)
        x4 = self.up1(x3 + x2) # Skip connection
        x5 = self.up2(x4 + x1) # Skip connection
        return x5
    
class UNetConv(nn.Module):
    def __init__(self, embedding_dim, n_channels, n_classes, bilinear=False):
        super(UNetConv, self).__init__()

        self.embedding = SinusoidalPositionEmbeddings(embedding_dim) # Embedding layer for integer input

        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = (DoubleConv(n_channels, 64))
        self.down1 = (Down(64, 128))
        self.down2 = (Down(128, 256))
        self.down3 = (Down(256, 512))
        factor = 2 if bilinear else 1
        self.down4 = (Down(512, 1024 // factor))
        self.up1 = (Up(1024, 512 // factor, bilinear))
        self.up2 = (Up(512, 256 // factor, bilinear))
        self.up3 = (Up(256, 128 // factor, bilinear))
        self.up4 = (Up(128, 64, bilinear))
        self.outc = (OutConv(64, n_classes))

    def forward(self, x, t):

        t_embed = self.embedding(t)
        # add time embed for each channel
        t_embed = t_embed.unsqueeze(1)

        x = x + t_embed

        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)

        return logits
