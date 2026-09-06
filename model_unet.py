"""
Simple and lightweight U-Net architecture for SAR + Optical Cloud Removal.
Fuses cloudy Sentinel-2 (13 bands), Sentinel-1 SAR (2 bands: VV, VH), and cloud mask (1 band).
Total input channels: 16
Output channels: 13 (reconstructed cloud-free Sentinel-2 image)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """(Conv2d -> BatchNorm/InstanceNorm -> LeakyReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super(DoubleConv, self).__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super(Down, self).__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv with skip connection"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super(Up, self).__init__()

        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # Handle padding if input dimensions are odd
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class CloudRemovalUNet(nn.Module):
    """
    4-Stage U-Net for Optical + SAR cloud removal.
    
    Args:
        in_channels (int): 16 (13 S2 optical + 2 S1 SAR + 1 Cloud Mask)
        out_channels (int): 13 (Reconstructed S2 optical bands)
        features (list): Base channel sizes for the 4 levels (default: [32, 64, 128, 256, 512])
        bilinear (bool): Whether to use bilinear upsampling or transposed convs
    """

    def __init__(self, in_channels=16, out_channels=13, features=None, bilinear=True):
        super(CloudRemovalUNet, self).__init__()
        if features is None:
            features = [32, 64, 128, 256, 512]
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.bilinear = bilinear

        self.inc = DoubleConv(in_channels, features[0])
        self.down1 = Down(features[0], features[1])
        self.down2 = Down(features[1], features[2])
        self.down3 = Down(features[2], features[3])
        factor = 2 if bilinear else 1
        self.down4 = Down(features[3], features[4] // factor)

        self.up1 = Up(features[4], features[3] // factor, bilinear)
        self.up2 = Up(features[3], features[2] // factor, bilinear)
        self.up3 = Up(features[2], features[1] // factor, bilinear)
        self.up4 = Up(features[1], features[0], bilinear)
        self.outc = OutConv(features[0], out_channels)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """
        Input x: [B, 16, H, W]
        Returns: [B, 13, H, W] in range [0, 1]
        """
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
        return self.sigmoid(logits)


def get_model(in_channels=16, out_channels=13, base_ch=32):
    features = [base_ch, base_ch * 2, base_ch * 4, base_ch * 8, base_ch * 16]
    return CloudRemovalUNet(in_channels=in_channels, out_channels=out_channels, features=features)


if __name__ == '__main__':
    # Sanity check
    model = get_model(in_channels=16, out_channels=13, base_ch=32)
    dummy_input = torch.randn(2, 16, 256, 256)
    out = model(dummy_input)
    print(f"Model instantiated successfully!")
    print(f"Input shape:  {dummy_input.shape}")
    print(f"Output shape: {out.shape}")
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {params:,} (~{params*4/(1024**2):.2f} MB)")
