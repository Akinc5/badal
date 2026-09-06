# Scoped-Down SAR + Optical Cloud Removal Demo Guide

A lightweight, single-day pipeline for satellite cloud removal using Sentinel-1 (SAR) and Sentinel-2 (Optical) data from **SEN12MS-CR-TS**.

---

## 1. Dataset Architecture & Smallest Region Selection

### Smallest Region Archive: `asiaWest`
To avoid downloading hundreds of gigabytes of global data, download exclusively the **`asiaWest`** region shard:

| Sensor & Split | Archive File Name | Direct Download URL | Compressed Size |
| :--- | :--- | :--- | :--- |
| **Sentinel-2 (Optical) Train** | `s2_asiaWest.tar.gz` | `https://dataserv.ub.tum.de/s/m1639953/download?path=/&files=s2_asiaWest.tar.gz` | ~46 GB |
| **Sentinel-1 (SAR) Train** | `s1_asiaWest.tar.gz` | `https://dataserv.ub.tum.de/s/m1639953/download?path=/&files=s1_asiaWest.tar.gz` | ~29 GB |
| **Sentinel-2 (Optical) Test** | `s2_asiaWest_test.tar.gz` | `https://dataserv.ub.tum.de/s/m1659251/download?path=/&files=s2_asiaWest_test.tar.gz` | ~7.2 GB |
| **Sentinel-1 (SAR) Test** | `s1_asiaWest_test.tar.gz` | `https://dataserv.ub.tum.de/s/m1659251/download?path=/&files=s1_asiaWest_test.tar.gz` | ~4.4 GB |

> [!TIP]
> Download credentials for MediaTUM if prompted via FTP: User `m1639953` / Pass `m1639953` (train) and `m1659251` / `m1659251` (test).

### Expected Directory Layout
After extracting the `.tar.gz` archives into a `SEN12MSCRTS` folder, the expected layout is:
```text
SEN12MSCRTS/
├── ROIs1868/
│   ├── 100/
│   │   ├── S1/
│   │   │   └── 0..29/ (*.tif)
│   │   └── S2/
│   │       └── 0..29/ (*.tif)
│   └── 127/
└── ROIs1970/
    ├── 57/
    ├── 83/
    ├── 112/
    ├── 115/
    └── 130/
```

---

## 2. Model & Pipeline Architecture

```
[Cloudy Optical S2 (13 bands)] ──┐
[Sentinel-1 SAR VV/VH (2 bands)] ─┼─► [Concat: 16 ch] ─► [4-Stage U-Net] ─► [Pred Cloud-Free S2 (13 bands)]
[Cloud Mask (1 band)] ───────────┘                                           │
                                                                             ▼
                                         [Masked L1 Loss] ◄── [Ground Truth S2 (13 bands)]
                                         (Computed ONLY on Cloud Pixels M=1)
```

- **U-Net Architecture ([model_unet.py](file:///c:/Users/akshansh%20sharma/Desktop/code%20file/cloudremoval/SEN12MS-CR-TS/model_unet.py))**: 4 downsampling stages (16 $\to$ 32 $\to$ 64 $\to$ 128 $\to$ 256 $\to$ 512) and 4 upsampling stages with skip connections (~4.3M parameters).
- **Masked L1 Loss**: $\mathcal{L} = \frac{\sum |pred - target| \odot M}{\sum M + \epsilon}$. Backpropagation is driven purely by the in-painting of occluded cloud pixels.
- **Composite Output**: $\hat{Y}_{\text{comp}} = pred \odot M + X_{\text{cloudy}} \odot (1 - M)$, preserving 100% original unclouded optical data while synthesizing cloud regions using SAR fusion.

---

## 3. Quickstart Commands

### Step A: Train Model (Single Consumer GPU / CPU / Colab)
```bash
# 1. Train on a subset of 300 tiles from asiaWest region
python train_demo.py --dataroot ./SEN12MSCRTS --region asiaWest --subset_size 300 --batch_size 4 --epochs 15 --lr 2e-4 --save_dir ./checkpoints

# 2. Instant Zero-Setup Dry Run (uses built-in synthetic generator to verify pipeline immediately)
python train_demo.py --dummy_data --subset_size 100 --batch_size 4 --epochs 5 --save_dir ./checkpoints
```

### Step B: Evaluate Model & Generate Visual Grids
```bash
# Evaluate on test split and generate comparison image grids
python eval_demo.py --checkpoint ./checkpoints/best_model.pth --dataroot ./SEN12MSCRTS --region asiaWest --num_test 50 --num_vis 5 --output_dir ./results_eval
```

---

## 4. Compute & Memory Profile

- **VRAM Consumption**: $\approx 1.8 - 2.5\text{ GB}$ VRAM at `batch_size=4` with $256 \times 256$ tiles. Easily runs on any consumer GPU (RTX 3060/4060, T4, Colab free tier).
- **CPU Fallback**: Automatically supported. `batch_size=2` takes $\approx 3-5\text{ s}$ per epoch for 50-100 tile subsets.

---

## 5. Future Work (Out of Scope for Single-Day Demo)

1. **Temporal Stacking & ConvLSTM / Transformer**: Exploiting full 30-timestamp Sentinel-2 acquisition sequences.
2. **Adversarial Loss (GAN Discriminator)**: Adding PatchGAN discriminator to sharpen high-frequency ground textures.
3. **Diffusion Models (e.g. DDPM / Flow Matching)**: Stochastic multi-modal sampling for high-variance terrain synthesis.
4. **Global Multi-Region Training**: Training across all 5 geographical continents (Africa, America, Asia-East, Asia-West, Europa).
