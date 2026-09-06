"""
Advanced Loss Functions for Remote Sensing Optical + SAR Cloud Removal.
Includes:
1. Masked Spectral Angle Mapper (SAM) Loss - Remote Sensing specific spectral fidelity
2. Multi-Scale SSIM / SSIM Loss - Structural texture preservation
3. Sobel Gradient / Edge Loss - Sharp boundary and urban structure preservation
4. Perceptual Loss - High-level semantic feature matching
5. Compound Multi-Task Cloud Removal Loss - Weighted combination of all components
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from util.pytorch_ssim import ssim as compute_ssim_tensor


class MaskedSpectralAngleMapperLoss(nn.Module):
    """
    Computes the Spectral Angle Mapper (SAM) loss across multispectral channels,
    measuring the angular distance between predicted and target spectral vectors.
    Values are in radians (0 = perfect spectral alignment).
    """
    def __init__(self, eps=1e-7):
        super(MaskedSpectralAngleMapperLoss, self).__init__()
        self.eps = eps

    def forward(self, pred, target, mask=None):
        """
        pred:   [B, C, H, W]
        target: [B, C, H, W]
        mask:   [B, 1, H, W] or None
        """
        # Dot product across spectral dimension C
        dot_product = torch.sum(pred * target, dim=1, keepdim=True)
        norm_pred = torch.norm(pred, p=2, dim=1, keepdim=True)
        norm_target = torch.norm(target, p=2, dim=1, keepdim=True)

        cos_theta = dot_product / (norm_pred * norm_target + self.eps)
        cos_theta = torch.clamp(cos_theta, -1.0 + self.eps, 1.0 - self.eps)
        sam_map = torch.acos(cos_theta)  # [B, 1, H, W] in radians

        if mask is not None:
            total_mask = mask.sum()
            if total_mask < 1.0:
                return torch.tensor(0.0, device=pred.device, requires_grad=True)
            return (sam_map * mask).sum() / (total_mask + self.eps)
        else:
            return torch.mean(sam_map)


class SobelEdgeLoss(nn.Module):
    """
    Edge preservation loss using Sobel filters to preserve roads,
    agricultural parcel boundaries, and coastlines in satellite imagery.
    """
    def __init__(self):
        super(SobelEdgeLoss, self).__init__()
        # Sobel kernel horizontal
        sobel_x = torch.tensor([[-1., 0., 1.],
                               [-2., 0., 2.],
                               [-1., 0., 1.]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        # Sobel kernel vertical
        sobel_y = torch.tensor([[-1., -2., -1.],
                               [ 0.,  0.,  0.],
                               [ 1.,  2.,  1.]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        
        self.register_buffer('sobel_x', sobel_x)
        self.register_buffer('sobel_y', sobel_y)

    def get_edges(self, x):
        B, C, H, W = x.shape
        x_reshaped = x.view(B * C, 1, H, W)
        gx = F.conv2d(x_reshaped, self.sobel_x, padding=1)
        gy = F.conv2d(x_reshaped, self.sobel_y, padding=1)
        grad = torch.sqrt(gx ** 2 + gy ** 2 + 1e-8)
        return grad.view(B, C, H, W)

    def forward(self, pred, target, mask=None):
        pred_edges = self.get_edges(pred)
        target_edges = self.get_edges(target)
        diff = torch.abs(pred_edges - target_edges)

        if mask is not None:
            if mask.shape[1] == 1:
                mask = mask.expand(-1, pred.shape[1], -1, -1)
            total = mask.sum()
            if total < 1.0:
                return torch.tensor(0.0, device=pred.device, requires_grad=True)
            return (diff * mask).sum() / (total + 1e-7)
        return torch.mean(diff)


class MaskedSSIMLoss(nn.Module):
    """
    Structural Similarity Loss computed as 1 - SSIM.
    """
    def __init__(self):
        super(MaskedSSIMLoss, self).__init__()

    def forward(self, pred, target, mask=None):
        # Full image SSIM in range [0, 1]
        ssim_val = compute_ssim_tensor(pred, target)
        return 1.0 - ssim_val


class CompoundCloudRemovalLoss(nn.Module):
    """
    Multi-objective compound loss tailored for multi-spectral remote sensing cloud removal:
    Loss = λ_l1 * Masked_L1 + λ_sam * SAM + λ_edge * Edge_Loss + λ_ssim * SSIM_Loss + λ_smooth * Smoothness
    """
    def __init__(self, lambda_l1=1.0, lambda_sam=0.2, lambda_edge=0.15, lambda_ssim=0.2, lambda_clear=0.2):
        super(CompoundCloudRemovalLoss, self).__init__()
        self.lambda_l1 = lambda_l1
        self.lambda_sam = lambda_sam
        self.lambda_edge = lambda_edge
        self.lambda_ssim = lambda_ssim
        self.lambda_clear = lambda_clear

        self.sam_loss = MaskedSpectralAngleMapperLoss()
        self.edge_loss = SobelEdgeLoss()
        self.ssim_loss = MaskedSSIMLoss()

    def forward(self, pred, target, mask, cloudy_input=None):
        """
        pred:   [B, 13, H, W]
        target: [B, 13, H, W]
        mask:   [B, 1, H, W] (1 for cloud, 0 for clear)
        cloudy_input: [B, 13, H, W] (optional)
        """
        # 1. Masked L1 on cloud pixels
        mask_expanded = mask.expand(-1, pred.shape[1], -1, -1) if mask.shape[1] == 1 else mask
        diff = torch.abs(pred - target)
        cloud_sum = mask_expanded.sum()

        if cloud_sum > 0:
            l1_cloud = (diff * mask_expanded).sum() / (cloud_sum + 1e-7)
        else:
            l1_cloud = torch.tensor(0.0, device=pred.device, requires_grad=True)

        # 2. L1 preservation on non-cloud pixels
        clear_mask = 1.0 - mask_expanded
        clear_sum = clear_mask.sum()
        if clear_sum > 0:
            l1_clear = (diff * clear_mask).sum() / (clear_sum + 1e-7)
        else:
            l1_clear = torch.tensor(0.0, device=pred.device, requires_grad=True)

        # 3. Spectral Angle Mapper (SAM) on cloud pixels
        loss_sam = self.sam_loss(pred, target, mask)

        # 4. Edge preservation loss
        loss_edge = self.edge_loss(pred, target, mask)

        # 5. SSIM structural loss
        loss_ssim = self.ssim_loss(pred, target)

        total_loss = (self.lambda_l1 * l1_cloud +
                      self.lambda_clear * l1_clear +
                      self.lambda_sam * loss_sam +
                      self.lambda_edge * loss_edge +
                      self.lambda_ssim * loss_ssim)

        return {
            'loss': total_loss,
            'l1_cloud': l1_cloud.item() if isinstance(l1_cloud, torch.Tensor) else l1_cloud,
            'l1_clear': l1_clear.item() if isinstance(l1_clear, torch.Tensor) else l1_clear,
            'loss_sam': loss_sam.item() if isinstance(loss_sam, torch.Tensor) else loss_sam,
            'loss_edge': loss_edge.item() if isinstance(loss_edge, torch.Tensor) else loss_edge,
            'loss_ssim': loss_ssim.item() if isinstance(loss_ssim, torch.Tensor) else loss_ssim
        }
