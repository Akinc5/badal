"""
Exhaustive Benchmark Evaluation Suite for SAR + Optical Cloud Removal.
Calculates Remote Sensing standard metrics:
- Peak Signal-to-Noise Ratio (PSNR in dB)
- Structural Similarity Index (SSIM)
- Spectral Angle Mapper (SAM in degrees and radians)
- Erreur Relative Globale Adimensionnelle de Synthèse (ERGAS)
- Universal Image Quality Index (UIQI)
- Masked (Cloud-Only) vs Unmasked (Clear-Only) metric breakdown
- Generates publication-ready HTML + JSON + CSV benchmarking reports
"""

import os
import argparse
import json
import csv
import numpy as np
import torch

from model_unet import get_model as get_standard_unet
from cross_attention_unet import get_advanced_model
from dataset_wrapper import get_dataloaders
from eval_demo import s2_to_rgb, sar_to_gray, mask_to_rgb, create_comparison_grid
from util.pytorch_ssim import ssim as compute_ssim_tensor


def compute_sam(pred, target, mask=None, eps=1e-7):
    """Spectral Angle Mapper in degrees."""
    dot = torch.sum(pred * target, dim=1, keepdim=True)
    norm_p = torch.norm(pred, p=2, dim=1, keepdim=True)
    norm_t = torch.norm(target, p=2, dim=1, keepdim=True)
    cos = torch.clamp(dot / (norm_p * norm_t + eps), -1.0 + eps, 1.0 - eps)
    sam_rad = torch.acos(cos)

    if mask is not None and mask.sum() > 0:
        val = (sam_rad * mask).sum() / (mask.sum() + eps)
    else:
        val = torch.mean(sam_rad)
    return float(val.item() * (180.0 / np.pi))


def compute_ergas(pred, target, scale_ratio=1.0, eps=1e-8):
    """
    Erreur Relative Globale Adimensionnelle de Synthèse (ERGAS).
    Standard remote sensing index for multispectral image synthesis quality (lower is better).
    """
    B, C, H, W = pred.shape
    mean_target = torch.mean(target, dim=[2, 3])  # [B, C]
    rmse_per_band = torch.sqrt(torch.mean((pred - target) ** 2, dim=[2, 3]))  # [B, C]
    ratio_sq = (rmse_per_band / (mean_target + eps)) ** 2
    ergas = 100.0 * scale_ratio * torch.sqrt(torch.mean(ratio_sq, dim=1))
    return float(torch.mean(ergas).item())


def compute_uiqi(pred, target, eps=1e-8):
    """
    Universal Image Quality Index (UIQI / Q-index) in [-1, 1] (1 is perfect quality).
    """
    mu_p = torch.mean(pred, dim=[2, 3], keepdim=True)
    mu_t = torch.mean(target, dim=[2, 3], keepdim=True)
    var_p = torch.var(pred, dim=[2, 3], keepdim=True)
    var_t = torch.var(target, dim=[2, 3], keepdim=True)
    cov_pt = torch.mean((pred - mu_p) * (target - mu_t), dim=[2, 3], keepdim=True)

    numerator = 4.0 * cov_pt * mu_p * mu_t
    denominator = (var_p + var_t + eps) * (mu_p ** 2 + mu_t ** 2 + eps)
    q = numerator / denominator
    return float(torch.mean(q).item())


def main():
    parser = argparse.ArgumentParser(description="Comprehensive Benchmark Evaluation")
    parser.add_argument("--checkpoint", type=str, default="./checkpoints_adv/best_checkpoint.pth")
    parser.add_argument("--model", type=str, default="cross_attention_unet", choices=["unet", "cross_attention_unet"])
    parser.add_argument("--dataroot", type=str, default="./SEN12MSCRTS")
    parser.add_argument("--region", type=str, default="asiaWest")
    parser.add_argument("--num_test", type=int, default=50)
    parser.add_argument("--output_dir", type=str, default="./benchmark_results")
    parser.add_argument("--dummy_data", action="store_true")
    parser.add_argument("--device", type=str, default="")
    args = parser.parse_args()

    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    os.makedirs(args.output_dir, exist_ok=True)
    vis_dir = os.path.join(args.output_dir, "visualizations")
    os.makedirs(vis_dir, exist_ok=True)

    print("==================================================================")
    print("      [+] COMPREHENSIVE CLOUD REMOVAL BENCHMARK EVALUATION")
    print("==================================================================")
    print(f"  Model:        {args.model.upper()}")
    print(f"  Checkpoint:   {args.checkpoint}")
    print(f"  Device:       {device}")
    print(f"  Output Dir:   {args.output_dir}")
    print("==================================================================")

    # 1. DataLoader
    _, _, test_loader = get_dataloaders(
        dataroot=args.dataroot,
        region=args.region,
        batch_size=1,
        num_workers=0,
        subset_size=args.num_test,
        use_dummy=args.dummy_data
    )

    # 2. Model
    if args.model == "cross_attention_unet":
        model = get_advanced_model("cross_attention_unet", base_ch=32)
    else:
        model = get_standard_unet(in_channels=16, out_channels=13, base_ch=32)

    if os.path.exists(args.checkpoint):
        print(f"Loading checkpoint weights from {args.checkpoint}...")
        ckpt = torch.load(args.checkpoint, map_location=device)
        state_dict = ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt
        model.load_state_dict(state_dict, strict=False)
        print("Model checkpoint loaded successfully!")
    else:
        print(f"[Warning] Checkpoint not found at {args.checkpoint}. Running with initial weights.")

    model.to(device)
    model.eval()

    records = []
    saved_grids = 0

    print(f"\nEvaluating on {len(test_loader)} test samples...")

    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            x = batch['input'].to(device)
            target = batch['target'].to(device)
            mask = batch['mask'].to(device)
            cloudy_s2 = batch['cloudy_s2'].to(device)
            sar_s1 = batch['sar_s1'].to(device)

            pred = model(x)
            composite = pred * mask + cloudy_s2 * (1.0 - mask)

            # Global metrics
            diff = torch.abs(pred - target)
            sq_diff = (pred - target) ** 2

            mae_val = float(torch.mean(diff).item())
            mse_val = float(torch.mean(sq_diff).item())
            psnr_val = float(10.0 * np.log10(1.0 / (mse_val + 1e-8)))
            ssim_val = float(compute_ssim_tensor(pred, target).item())
            sam_deg = compute_sam(pred, target, mask=None)
            sam_cloud_deg = compute_sam(pred, target, mask=mask)
            ergas_val = compute_ergas(pred, target)
            uiqi_val = compute_uiqi(pred, target)

            # Cloud-only metrics
            mask_exp = mask.expand(-1, 13, -1, -1)
            cloud_px = mask_exp.sum().item()
            if cloud_px > 0:
                mae_cloud = float((diff * mask_exp).sum().item() / cloud_px)
                rmse_cloud = float(np.sqrt((sq_diff * mask_exp).sum().item() / cloud_px))
            else:
                mae_cloud = 0.0
                rmse_cloud = 0.0

            records.append({
                'Sample_ID': i + 1,
                'PSNR_dB': psnr_val,
                'SSIM': ssim_val,
                'SAM_deg': sam_deg,
                'SAM_Cloud_deg': sam_cloud_deg,
                'ERGAS': ergas_val,
                'UIQI': uiqi_val,
                'MAE_Global': mae_val,
                'MAE_Cloudy': mae_cloud,
                'RMSE_Cloudy': rmse_cloud
            })

            # Save visual comparison grids for first 5 samples
            if saved_grids < 5:
                cloudy_rgb = s2_to_rgb(cloudy_s2[0])
                sar_gray = sar_to_gray(sar_s1[0])
                mask_vis = mask_to_rgb(mask[0])
                rec_rgb = s2_to_rgb(pred[0])
                comp_rgb = s2_to_rgb(composite[0])
                tgt_rgb = s2_to_rgb(target[0])

                grid_img = create_comparison_grid(cloudy_rgb, sar_gray, mask_vis, rec_rgb, tgt_rgb, composite_rgb=comp_rgb)
                grid_save_path = os.path.join(vis_dir, f"benchmark_sample_{saved_grids + 1:02d}.png")
                grid_img.save(grid_save_path)
                saved_grids += 1

    # Aggregate averages
    summary = {}
    for key in records[0].keys():
        if key != 'Sample_ID':
            summary[key] = float(np.mean([r[key] for r in records]))

    # Export CSV
    csv_path = os.path.join(args.output_dir, "benchmark_metrics_per_sample.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=records[0].keys())
        writer.writeheader()
        writer.writerows(records)

    # Export JSON
    json_path = os.path.join(args.output_dir, "benchmark_summary.json")
    with open(json_path, "w") as f:
        json.dump({'summary': summary, 'samples': records}, f, indent=4)

    # Export Markdown & HTML Report
    html_report = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Cloud Removal Benchmark Report</title>
<style>
body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; background: #0f172a; color: #e2e8f0; padding: 2rem; }}
h1 {{ color: #38bdf8; font-size: 1.8rem; }}
.card {{ background: #1e293b; border-radius: 12px; padding: 1.5rem; margin-bottom: 1.5rem; border: 1px solid #334155; }}
table {{ width: 100%; border-collapse: collapse; margin-top: 1rem; }}
th, td {{ padding: 0.75rem 1rem; text-align: left; border-bottom: 1px solid #334155; }}
th {{ background: #0f172a; color: #38bdf8; font-weight: 600; }}
.badge {{ display: inline-block; padding: 0.25rem 0.5rem; border-radius: 4px; font-weight: bold; background: #0369a1; color: #e0f2fe; }}
.metric-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 1rem; margin-top: 1rem; }}
.metric-box {{ background: #0f172a; border-radius: 8px; padding: 1rem; border: 1px solid #334155; text-align: center; }}
.metric-val {{ font-size: 1.6rem; font-weight: bold; color: #38bdf8; }}
.metric-lbl {{ font-size: 0.85rem; color: #94a3b8; margin-top: 0.25rem; }}
</style>
</head>
<body>
<h1>🛰️ SEN12MS-CR-TS Cloud Removal Benchmark Report</h1>
<div class="card">
  <h2>Model Architecture: <span class="badge">{args.model.upper()}</span></h2>
  <div class="metric-grid">
    <div class="metric-box"><div class="metric-val">{summary['PSNR_dB']:.2f} dB</div><div class="metric-lbl">Peak SNR (PSNR)</div></div>
    <div class="metric-box"><div class="metric-val">{summary['SSIM']:.4f}</div><div class="metric-lbl">Structural Similarity (SSIM)</div></div>
    <div class="metric-box"><div class="metric-val">{summary['SAM_deg']:.2f}°</div><div class="metric-lbl">Spectral Angle (SAM)</div></div>
    <div class="metric-box"><div class="metric-val">{summary['ERGAS']:.2f}</div><div class="metric-lbl">ERGAS Synthesis Error</div></div>
    <div class="metric-box"><div class="metric-val">{summary['UIQI']:.4f}</div><div class="metric-lbl">Universal Image Quality (UIQI)</div></div>
    <div class="metric-box"><div class="metric-val">{summary['MAE_Cloudy']:.4f}</div><div class="metric-lbl">Cloud Region Inpainting MAE</div></div>
  </div>
</div>

<div class="card">
  <h2>Visual Evaluation (Comparison Panels)</h2>
  <p style="color: #94a3b8;">Panels: [Cloudy S2 Optical | Sentinel-1 SAR VV | Cloud Mask | Reconstructed Output | Ground Truth S2]</p>
  <p>Saved visual comparison grids in <code>{vis_dir}</code></p>
</div>
</body>
</html>"""
    
    html_path = os.path.join(args.output_dir, "benchmark_report.html")
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html_report)

    print("\n=================== BENCHMARK SUMMARY ===================")
    print(f"Evaluated Test Samples:     {len(records)}")
    print(f"Global PSNR:                {summary['PSNR_dB']:.2f} dB")
    print(f"Global SSIM:                {summary['SSIM']:.4f}")
    print(f"Spectral Angle Mapper (SAM):{summary['SAM_deg']:.2f}°")
    print(f"Cloud-Region SAM:           {summary['SAM_Cloud_deg']:.2f}°")
    print(f"ERGAS Synthesis Index:      {summary['ERGAS']:.2f}")
    print(f"UIQI (Universal Quality):   {summary['UIQI']:.4f}")
    print(f"Cloud Inpainting MAE:       {summary['MAE_Cloudy']:.5f}")
    print(f"Cloud Inpainting RMSE:      {summary['RMSE_Cloudy']:.5f}")
    print("=========================================================")
    print(f"Reports saved to:")
    print(f"  - CSV:  {csv_path}")
    print(f"  - JSON: {json_path}")
    print(f"  - HTML: {html_path}")


if __name__ == '__main__':
    main()
