"""
Advanced Cross-Modal Attention U-Net (X-Attn UNet) for SAR + Optical Fusion.
Features:
- Separate Multi-Modal Encoders for Optical (13 bands) and SAR (2 bands: VV, VH)
- Cross-Modal Spatial & Channel Attention Fusion (CM-Attention)
- Atrous Spatial Pyramid Pooling (ASPP) bottleneck for multi-scale cloud receptive fields
- Residual Skip Connections with Gated Feature Alignment
- High-frequency Detail Recovery Output Layer
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ChannelAttention(nn.Module):
    """Squeeze-and-Excitation Channel Attention"""
    def __init__(self, channels, reduction=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        mid_ch = max(4, channels // reduction)
        self.fc = nn.Sequential(
            nn.Linear(channels, mid_ch, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(mid_ch, channels, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, _, _ = x.size()
        avg_out = self.fc(self.avg_pool(x).view(b, c)).view(b, c, 1, 1)
        max_out = self.fc(self.max_pool(x).view(b, c)).view(b, c, 1, 1)
        return self.sigmoid(avg_out + max_out) * x


class SpatialAttention(nn.Module):
    """Spatial Attention Module"""
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        scale = torch.cat([avg_out, max_out], dim=1)
        scale = self.sigmoid(self.conv(scale))
        return scale * x


class CBAMBlock(nn.Module):
    """Convolutional Block Attention Module"""
    def __init__(self, channels, reduction=16):
        super(CBAMBlock, self).__init__()
        self.ca = ChannelAttention(channels, reduction)
        self.sa = SpatialAttention()

    def forward(self, x):
        x = self.ca(x)
        x = self.sa(x)
        return x


class CrossModalAttentionBridge(nn.Module):
    """
    Efficient Cross-Modal Attention module that aligns radar (SAR) features with optical features
    using Multi-Head Channel Cross-Attention and Spatial Cross-Gating (O(C^2) complexity).
    """
    def __init__(self, channels, num_heads=4):
        super(CrossModalAttentionBridge, self).__init__()
        self.channels = channels
        self.num_heads = num_heads
        
        # Spatial Cross-Gating
        self.spatial_gate = nn.Sequential(
            nn.Conv2d(channels * 2, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.Sigmoid()
        )
        
        # Channel Cross-Attention
        self.q_opt = nn.AdaptiveAvgPool2d(1)
        self.k_sar = nn.AdaptiveAvgPool2d(1)
        self.channel_fc = nn.Sequential(
            nn.Linear(channels * 2, channels // 2, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // 2, channels, bias=False),
            nn.Sigmoid()
        )
        
        # Fusion projection
        self.out_conv = nn.Sequential(
            nn.Conv2d(channels * 2, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.LeakyReLU(0.2, inplace=True)
        )

    def forward(self, feat_opt, feat_sar):
        """
        feat_opt: [B, C, H, W]
        feat_sar: [B, C, H, W]
        """
        B, C, H, W = feat_opt.shape
        
        # 1. Spatial Gating
        combined = torch.cat([feat_opt, feat_sar], dim=1)
        s_gate = self.spatial_gate(combined)
        sar_aligned = feat_sar * s_gate
        
        # 2. Channel Cross-Attention
        opt_vec = self.q_opt(feat_opt).view(B, C)
        sar_vec = self.k_sar(sar_aligned).view(B, C)
        ch_weights = self.channel_fc(torch.cat([opt_vec, sar_vec], dim=1)).view(B, C, 1, 1)
        
        fused = feat_opt * (1.0 - ch_weights) + sar_aligned * ch_weights
        out = self.out_conv(torch.cat([feat_opt, fused], dim=1))
        return out


class ASPPBottleneck(nn.Module):
    """Atrous Spatial Pyramid Pooling for multi-scale context aggregation"""
    def __init__(self, in_channels, out_channels, rates=[1, 6, 12, 18]):
        super(ASPPBottleneck, self).__init__()
        modules = []
        modules.append(nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True)
        ))
        for rate in rates[1:]:
            modules.append(nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, padding=rate, dilation=rate, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.LeakyReLU(0.2, inplace=True)
            ))
        
        # Image level pooling (no BatchNorm after 1x1 pooling to allow batch_size=1)
        self.global_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=True),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.convs = nn.ModuleList(modules)
        self.project = nn.Sequential(
            nn.Conv2d(out_channels * (len(rates) + 1), out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.1)
        )

    def forward(self, x):
        res = []
        for conv in self.convs:
            res.append(conv(x))
        gp = self.global_pool(x)
        gp = F.interpolate(gp, size=x.shape[2:], mode='bilinear', align_corners=True)
        res.append(gp)
        res = torch.cat(res, dim=1)
        return self.project(res)


class ResConvBlock(nn.Module):
    """Residual Convolution Block with CBAM attention"""
    def __init__(self, in_channels, out_channels):
        super(ResConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.cbam = CBAMBlock(out_channels)
        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        residual = self.shortcut(x)
        out = self.conv(x)
        out = self.cbam(out)
        return F.leaky_relu(out + residual, 0.2)


class CrossAttentionUNet(nn.Module):
    """
    State-of-the-Art Cross-Modal Attention U-Net for Optical + SAR cloud removal.
    
    Inputs:
        x: [B, 16, H, W] containing:
           - Cloudy S2 Optical: channels 0..12 (13 channels)
           - Sentinel-1 SAR:    channels 13..14 (2 channels)
           - Cloud Mask:        channel 15 (1 channel)
    Outputs:
        [B, 13, H, W] in [0, 1] (Reconstructed Optical Image)
    """
    def __init__(self, out_channels=13, base_ch=32):
        super(CrossAttentionUNet, self).__init__()

        # Dedicated stream encoders
        # Optical branch (13 optical + 1 mask = 14 ch)
        self.opt_in = ResConvBlock(14, base_ch)
        # SAR branch (2 SAR + 1 mask = 3 ch)
        self.sar_in = ResConvBlock(3, base_ch)

        # Cross-Modal Fusion Bridges at Level 0
        self.fusion0 = CrossModalAttentionBridge(base_ch)

        # Level 1 (Downsampling)
        self.down1 = nn.MaxPool2d(2)
        self.enc1 = ResConvBlock(base_ch, base_ch * 2)

        # Level 2
        self.down2 = nn.MaxPool2d(2)
        self.enc2 = ResConvBlock(base_ch * 2, base_ch * 4)

        # Level 3
        self.down3 = nn.MaxPool2d(2)
        self.enc3 = ResConvBlock(base_ch * 4, base_ch * 8)

        # Bottleneck with ASPP
        self.down4 = nn.MaxPool2d(2)
        self.bottleneck = ASPPBottleneck(base_ch * 8, base_ch * 8)

        # Decoder Stages
        self.up4 = nn.ConvTranspose2d(base_ch * 8, base_ch * 4, kernel_size=2, stride=2)
        self.dec4 = ResConvBlock(base_ch * 8 + base_ch * 4, base_ch * 4)

        self.up3 = nn.ConvTranspose2d(base_ch * 4, base_ch * 2, kernel_size=2, stride=2)
        self.dec3 = ResConvBlock(base_ch * 4 + base_ch * 2, base_ch * 2)

        self.up2 = nn.ConvTranspose2d(base_ch * 2, base_ch, kernel_size=2, stride=2)
        self.dec2 = ResConvBlock(base_ch * 2 + base_ch, base_ch)

        self.up1 = nn.ConvTranspose2d(base_ch, base_ch, kernel_size=2, stride=2)
        self.dec1 = ResConvBlock(base_ch * 2, base_ch)

        # Output head
        self.head = nn.Sequential(
            nn.Conv2d(base_ch, base_ch, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base_ch, out_channels, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        """
        x: [B, 16, H, W]
        """
        opt_input = torch.cat([x[:, 0:13, :, :], x[:, 15:16, :, :]], dim=1) # [B, 14, H, W]
        sar_input = torch.cat([x[:, 13:15, :, :], x[:, 15:16, :, :]], dim=1) # [B, 3, H, W]

        # Stage 0: Separate branch feature extraction & Cross-Attention
        f_opt = self.opt_in(opt_input)
        f_sar = self.sar_in(sar_input)
        e0 = self.fusion0(f_opt, f_sar) # [B, base_ch, H, W]

        # Stage 1
        e1 = self.enc1(self.down1(e0))  # [B, base_ch*2, H/2, W/2]
        # Stage 2
        e2 = self.enc2(self.down2(e1))  # [B, base_ch*4, H/4, W/4]
        # Stage 3
        e3 = self.enc3(self.down3(e2))  # [B, base_ch*8, H/8, W/8]

        # Bottleneck (ASPP)
        b = self.bottleneck(self.down4(e3)) # [B, base_ch*8, H/16, W/16]

        # Decoder with skip connections
        d4 = self.up4(b)
        d4 = self.dec4(torch.cat([d4, e3], dim=1))

        d3 = self.up3(d4)
        d3 = self.dec3(torch.cat([d3, e2], dim=1))

        d2 = self.up2(d3)
        d2 = self.dec2(torch.cat([d2, e1], dim=1))

        d1 = self.up1(d2)
        d1 = self.dec1(torch.cat([d1, e0], dim=1))

        out = self.head(d1)
        return out


def get_advanced_model(model_name="cross_attention_unet", in_channels=16, out_channels=13, base_ch=32):
    if model_name == "cross_attention_unet":
        return CrossAttentionUNet(out_channels=out_channels, base_ch=base_ch)
    else:
        from model_unet import get_model as get_standard_unet
        return get_standard_unet(in_channels=in_channels, out_channels=out_channels, base_ch=base_ch)


if __name__ == '__main__':
    model = get_advanced_model("cross_attention_unet", base_ch=32)
    dummy = torch.randn(2, 16, 256, 256)
    out = model(dummy)
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"CrossAttentionUNet Output Shape: {out.shape}")
    print(f"Total Parameters: {params:,} (~{params*4/(1024**2):.2f} MB)")
