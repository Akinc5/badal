"""
Evaluation and visualization script for Scoped-Down SAR + Optical Cloud Removal Demo.
- Loads trained checkpoint
- Evaluates on held-out test split
- Computes comprehensive quantitative metrics: PSNR, SSIM, MAE, RMSE (both global and cloud-masked)
- Generates and saves visual triplet/quad grids (Cloudy Input, SAR VV, Reconstructed, Ground Truth)
"""

import os
import argparse
import json
import numpy as np
import torch
from PIL import Image

from model_unet import get_model
from dataset_wrapper import get_dataloaders
from util.pytorch_ssim import ssim as compute_ssim


def s2_to_rgb(s2_tensor):
    """
    Converts 13-band Sentinel-2 tensor [13, H, W] to RGB numpy array [H, W, 3] in [0, 255] uint8.
    Sentinel-2 standard band indices: B4 (Red) -> index 3, B3 (Green) -> index 2, B2 (Blue) -> index 1.
    """
    if isinstance(s2_tensor, torch.Tensor):
        s2_np = s2_tensor.detach().cpu().float().numpy()
    else:
        s2_np = np.asarray(s2_tensor, dtype=np.float32)

    # RGB bands: Red (idx 3), Green (idx 2), Blue (idx 1)
    rgb = s2_np[[3, 2, 1], :, :]  # [3, H, W]
    rgb = np.transpose(rgb, (1, 2, 0))  # [H, W, 3]

    # Clip to [0, 1] and scale to 255
    rgb = np.clip(rgb, 0.0, 1.0) * 255.0
    return rgb.astype(np.uint8)


def sar_to_gray(sar_tensor):
    """
    Converts 2-band SAR tensor [2, H, W] (VV, VH) to grayscale VV visualization [H, W, 3] uint8.
    """
    if isinstance(sar_tensor, torch.Tensor):
        sar_np = sar_tensor.detach().cpu().float().numpy()
    else:
        sar_np = np.asarray(sar_tensor, dtype=np.float32)

    vv = sar_np[0]  # [H, W]
    vv = np.clip(vv, 0.0, 1.0) * 255.0
    gray_3ch = np.stack([vv, vv, vv], axis=-1).astype(np.uint8)
    return gray_3ch


def mask_to_rgb(mask_tensor):
    """
    Converts 1-band mask tensor [1, H, W] to colored cloud overlay [H, W, 3] uint8.
    """
    if isinstance(mask_tensor, torch.Tensor):
        mask_np = mask_tensor.detach().cpu().float().numpy()
    else:
        mask_np = np.asarray(mask_tensor, dtype=np.float32)

    m = mask_np[0]  # [H, W]
    m = np.clip(m, 0.0, 1.0) * 255.0
    # Show clouds as semi-transparent cyan / white
    vis = np.stack([m * 0.4, m * 0.9, m], axis=-1).astype(np.uint8)
    return vis


def create_comparison_grid(cloudy_rgb, sar_gray, mask_vis, reconstructed_rgb, target_rgb, composite_rgb=None):
    """
    Stitches panels side-by-side into a single high-resolution comparison image.
    Panels: [Cloudy Optical Input | Sentinel-1 SAR VV | Cloud Mask | Reconstructed | Ground Truth]
    """
    panels = [cloudy_rgb, sar_gray, mask_vis, reconstructed_rgb, target_rgb]
    if composite_rgb is not None:
        panels.insert(4, composite_rgb)

    H, W, C = panels[0].shape
    grid = np.concatenate(panels, axis=1) # Horizontal concatenation
    return Image.fromarray(grid)


def compute_metrics_batch(pred, target, mask, eps=1e-8):
    """
    Computes global PSNR, SSIM, MAE, and masked (cloud-only) RMSE/MAE.
    """
    # Global MAE & MSE
    diff = torch.abs(pred - target)
    sq_diff = (pred - target) ** 2

    mae_global = torch.mean(diff).item()
    mse_global = torch.mean(sq_diff).item()
    psnr_global = float(10.0 * np.log10(1.0 / (mse_global + eps)))
    ssim_global = float(compute_ssim(pred, target).item())

    # Masked metrics (only on cloudy pixels)
    mask_expanded = mask.expand(-1, pred.shape[1], -1, -1)
    mask_sum = mask_expanded.sum().item()

    if mask_sum > 0:
        mae_cloudy = (diff * mask_expanded).sum().item() / mask_sum
        rmse_cloudy = np.sqrt((sq_diff * mask_expanded).sum().item() / mask_sum)
    else:
        mae_cloudy = 0.0
        rmse_cloudy = 0.0

    # Non-cloud region metrics (cloud-free pixels)
    clear_mask = 1.0 - mask_expanded
    clear_sum = clear_mask.sum().item()
    if clear_sum > 0:
        mae_clear = (diff * clear_mask).sum().item() / clear_sum
        rmse_clear = np.sqrt((sq_diff * clear_mask).sum().item() / clear_sum)
    else:
        mae_clear = 0.0
        rmse_clear = 0.0

    return {
        'PSNR_global': psnr_global,
        'SSIM_global': ssim_global,
        'MAE_global': mae_global,
        'RMSE_cloudy': float(rmse_cloudy),
        'MAE_cloudy': float(mae_cloudy),
        'RMSE_clear': float(rmse_clear),
        'MAE_clear': float(mae_clear)
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate Scoped Cloud Removal Model")
    parser.add_argument("--checkpoint", type=str, default="./checkpoints/best_model.pth", help="Path to trained checkpoint")
    parser.add_argument("--dataroot", type=str, default="./SEN12MSCRTS", help="Path to SEN12MS-CR-TS dataset")
    parser.add_argument("--region", type=str, default="asiaWest", help="Dataset region shard")
    parser.add_argument("--num_test", type=int, default=20, help="Number of test samples to evaluate")
    parser.add_argument("--num_vis", type=int, default=5, help="Number of visual comparison grids to save")
    parser.add_argument("--output_dir", type=str, default="./results_eval", help="Output directory for results & images")
    parser.add_argument("--dummy_data", action="store_true", help="Use synthetic dataset if dataset files unavailable")
    parser.add_argument("--device", type=str, default="", help="cuda or cpu (auto-detected if empty)")
    args = parser.parse_args()

    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    os.makedirs(args.output_dir, exist_ok=True)
    images_dir = os.path.join(args.output_dir, "visualizations")
    os.makedirs(images_dir, exist_ok=True)

    print("==================================================")
    print("  Cloud Removal Evaluation & Triplet Visualization")
    print("==================================================")
    print(f"Device:         {device}")
    print(f"Checkpoint:     {args.checkpoint}")
    print(f"Output Dir:     {args.output_dir}")
    print("==================================================")

    # 1. Load Data
    _, _, test_loader = get_dataloaders(
        dataroot=args.dataroot,
        region=args.region,
        batch_size=1,
        num_workers=0,
        subset_size=args.num_test,
        use_dummy=args.dummy_data
    )

    # 2. Load Model
    model = get_model(in_channels=16, out_channels=13, base_ch=32)
    if os.path.exists(args.checkpoint):
        print(f"Loading weights from {args.checkpoint}...")
        ckpt = torch.load(args.checkpoint, map_location=device)
        state_dict = ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt
        model.load_state_dict(state_dict)
        print("Checkpoint loaded successfully!")
    else:
        print(f"[Warning] Checkpoint '{args.checkpoint}' not found! Evaluating model with initial weights.")

    model.to(device)
    model.eval()

    # 3. Evaluation Loop
    metrics_list = []
    saved_vis_count = 0

    print(f"\nRunning evaluation on {len(test_loader)} test samples...")
    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            x = batch['input'].to(device)           # [1, 16, H, W]
            target = batch['target'].to(device)     # [1, 13, H, W]
            mask = batch['mask'].to(device)         # [1, 1, H, W]
            cloudy_s2 = batch['cloudy_s2'].to(device) # [1, 13, H, W]
            sar_s1 = batch['sar_s1'].to(device)     # [1, 2, H, W]

            pred = model(x)  # [1, 13, H, W]

            # Composite: replace cloudy pixels with prediction, preserve unclouded optical data
            composite = pred * mask + cloudy_s2 * (1.0 - mask)

            batch_metrics = compute_metrics_batch(pred, target, mask)
            metrics_list.append(batch_metrics)

            # Save visual comparison grids for first N test samples
            if saved_vis_count < args.num_vis:
                cloudy_rgb = s2_to_rgb(cloudy_s2[0])
                sar_gray = sar_to_gray(sar_s1[0])
                mask_vis = mask_to_rgb(mask[0])
                reconstructed_rgb = s2_to_rgb(pred[0])
                composite_rgb = s2_to_rgb(composite[0])
                target_rgb = s2_to_rgb(target[0])

                grid_img = create_comparison_grid(
                    cloudy_rgb=cloudy_rgb,
                    sar_gray=sar_gray,
                    mask_vis=mask_vis,
                    reconstructed_rgb=reconstructed_rgb,
                    composite_rgb=composite_rgb,
                    target_rgb=target_rgb
                )

                img_save_path = os.path.join(images_dir, f"sample_{saved_vis_count + 1:02d}_grid.png")
                grid_img.save(img_save_path)
                print(f"  [Saved Visual Grid] {img_save_path}")
                saved_vis_count += 1

    # 4. Compute Aggregate Metrics
    avg_metrics = {}
    for k in metrics_list[0].keys():
        avg_metrics[k] = float(np.mean([m[k] for m in metrics_list]))

    print("\n=================== EVALUATION RESULTS ===================")
    print(f"Total Test Samples:        {len(metrics_list)}")
    print(f"Global PSNR:               {avg_metrics['PSNR_global']:.2f} dB")
    print(f"Global SSIM:               {avg_metrics['SSIM_global']:.4f}")
    print(f"Global MAE:                {avg_metrics['MAE_global']:.5f}")
    print(f"Cloud-Masked Region RMSE:  {avg_metrics['RMSE_cloudy']:.5f}")
    print(f"Cloud-Masked Region MAE:   {avg_metrics['MAE_cloudy']:.5f}")
    print(f"Clear Region MAE:          {avg_metrics['MAE_clear']:.5f}")
    print("==========================================================")
    print(f"Visual grids layout: [Cloudy S2 | SAR VV | Cloud Mask | Model Output | Composite Output | Ground Truth S2]")
    print(f"Saved {saved_vis_count} comparison images in: {images_dir}")

    # Save summary report to JSON and TXT
    json_path = os.path.join(args.output_dir, "eval_metrics.json")
    with open(json_path, "w") as f:
        json.dump(avg_metrics, f, indent=4)

    txt_path = os.path.join(args.output_dir, "eval_summary.txt")
    with open(txt_path, "w") as f:
        f.write("SEN12MS-CR-TS Scoped Cloud Removal Evaluation Summary\n")
        f.write("=====================================================\n")
        for k, v in avg_metrics.items():
            f.write(f"{k}: {v}\n")

    print(f"Metrics saved to {json_path} and {txt_path}\n")


if __name__ == '__main__':
    main()
