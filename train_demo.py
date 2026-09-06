"""
Training script for Scoped-Down SAR + Optical Cloud Removal Demo.
Uses:
- SEN12MS-CR-TS dataset loader (with single small region / subset option)
- 4-Stage lightweight U-Net (16 input channels -> 13 output channels)
- Masked L1 Loss (computed exclusively on cloud-masked pixels)
- Validation tracking with PSNR and SSIM metrics
"""

import os
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR

from model_unet import get_model
from dataset_wrapper import get_dataloaders
from util.pytorch_ssim import ssim as compute_ssim


class MaskedL1Loss(nn.Module):
    """
    L1 reconstruction loss computed ONLY on cloud-masked pixels.
    If no cloud pixels are present in a batch, returns 0 loss.
    """
    def __init__(self, eps=1e-7):
        super(MaskedL1Loss, self).__init__()
        self.eps = eps

    def forward(self, pred, target, mask):
        """
        pred:   [B, 13, H, W]
        target: [B, 13, H, W]
        mask:   [B, 1, H, W] (1 where cloudy, 0 where clear)
        """
        if mask.shape[1] == 1:
            mask = mask.expand(-1, pred.shape[1], -1, -1)
        
        diff = torch.abs(pred - target) * mask
        total_masked_px = mask.sum()
        if total_masked_px < 1.0:
            return torch.tensor(0.0, device=pred.device, requires_grad=True)
        return diff.sum() / (total_masked_px + self.eps)


def compute_psnr(pred, target, max_val=1.0, eps=1e-8):
    """Computes Peak Signal-to-Noise Ratio (PSNR) in dB."""
    mse = torch.mean((pred - target) ** 2)
    if mse < eps:
        return 100.0
    return 10.0 * torch.log10((max_val ** 2) / mse).item()


def validate(model, val_loader, criterion, device):
    model.eval()
    val_loss = 0.0
    total_psnr = 0.0
    total_ssim = 0.0
    num_batches = 0

    with torch.no_grad():
        for batch in val_loader:
            x = batch['input'].to(device)       # [B, 16, H, W]
            target = batch['target'].to(device) # [B, 13, H, W]
            mask = batch['mask'].to(device)     # [B, 1, H, W]

            pred = model(x)
            loss = criterion(pred, target, mask)

            val_loss += loss.item()

            # Measure full-image PSNR and SSIM on RGB channels (bands 3, 2, 1) or all 13 bands
            psnr_val = compute_psnr(pred, target)
            ssim_val = compute_ssim(pred, target).item()

            total_psnr += psnr_val
            total_ssim += ssim_val
            num_batches += 1

    avg_loss = val_loss / max(1, num_batches)
    avg_psnr = total_psnr / max(1, num_batches)
    avg_ssim = total_ssim / max(1, num_batches)
    return avg_loss, avg_psnr, avg_ssim


def main():
    parser = argparse.ArgumentParser(description="Train Scoped Cloud Removal U-Net Demo")
    parser.add_argument("--dataroot", type=str, default="./SEN12MSCRTS", help="Path to SEN12MS-CR-TS dataset directory")
    parser.add_argument("--region", type=str, default="asiaWest", help="Smallest region shard: asiaWest, africa, etc.")
    parser.add_argument("--subset_size", type=int, default=300, help="Train on a scoped subset of N tiles (e.g. 300)")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size (4-8 recommended for consumer GPU)")
    parser.add_argument("--epochs", type=int, default=15, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-5, help="Weight decay")
    parser.add_argument("--save_dir", type=str, default="./checkpoints", help="Directory to save model checkpoints")
    parser.add_argument("--num_workers", type=int, default=0, help="Dataloader workers (0 for Windows compatibility)")
    parser.add_argument("--dummy_data", action="store_true", help="Force synthetic data for zero-setup demo pipeline")
    parser.add_argument("--device", type=str, default="", help="cuda or cpu (auto-detected if empty)")
    args = parser.parse_args()

    # Determine device
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"==================================================")
    print(f"  SAR + Optical Cloud Removal U-Net Demo")
    print(f"==================================================")
    print(f"Device:         {device}")
    print(f"Data Root:      {args.dataroot}")
    print(f"Region:         {args.region}")
    print(f"Subset Size:    {args.subset_size} tiles")
    print(f"Batch Size:     {args.batch_size}")
    print(f"Epochs:         {args.epochs}")
    print(f"Learning Rate:  {args.lr}")
    print(f"Checkpoints:    {args.save_dir}")
    print(f"==================================================")

    os.makedirs(args.save_dir, exist_ok=True)

    # 1. Prepare Data Loaders
    train_loader, val_loader, _ = get_dataloaders(
        dataroot=args.dataroot,
        region=args.region,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        subset_size=args.subset_size,
        use_dummy=args.dummy_data
    )

    # 2. Build U-Net Model
    model = get_model(in_channels=16, out_channels=13, base_ch=32)
    model.to(device)

    # 3. Loss & Optimizer
    criterion = MaskedL1Loss().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    best_val_loss = float("inf")
    start_time = time.time()

    # 4. Training Loop
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        batch_count = 0
        epoch_start = time.time()

        for batch_idx, batch in enumerate(train_loader):
            x = batch['input'].to(device)       # [B, 16, H, W]
            target = batch['target'].to(device) # [B, 13, H, W]
            mask = batch['mask'].to(device)     # [B, 1, H, W]

            optimizer.zero_grad()
            pred = model(x)
            loss = criterion(pred, target, mask)

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += loss.item()
            batch_count += 1

        scheduler.step()

        avg_train_loss = train_loss / max(1, batch_count)
        val_loss, val_psnr, val_ssim = validate(model, val_loader, criterion, device)
        epoch_time = time.time() - epoch_start

        print(f"Epoch [{epoch:02d}/{args.epochs:02d}] ({epoch_time:.1f}s) | "
              f"Train Loss (Masked L1): {avg_train_loss:.5f} | "
              f"Val Loss: {val_loss:.5f} | "
              f"Val PSNR: {val_psnr:.2f} dB | "
              f"Val SSIM: {val_ssim:.4f}")

        # Save latest checkpoint
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_loss': val_loss,
            'val_psnr': val_psnr,
            'val_ssim': val_ssim,
            'args': vars(args)
        }
        torch.save(checkpoint, os.path.join(args.save_dir, "latest_model.pth"))

        # Save best checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(checkpoint, os.path.join(args.save_dir, "best_model.pth"))
            print(f"  --> Saved new best checkpoint (Val Loss: {val_loss:.5f})")

    total_duration = time.time() - start_time
    print(f"\nTraining completed in {total_duration/60:.2f} minutes.")
    print(f"Best model saved to: {os.path.join(args.save_dir, 'best_model.pth')}")


if __name__ == '__main__':
    main()
