"""
Dataset wrapper and loader utilities for SEN12MS-CR-TS and SEN12MS-CR.
Wraps the repo's original dataLoader.py to output ready-to-train PyTorch tensors:
- input_tensor: [16, H, W] (13 S2 optical + 2 S1 SAR + 1 Cloud Mask)
- target_tensor: [13, H, W] (13 S2 cloud-free optical)
- mask_tensor: [1, H, W] (1 for cloud, 0 for clear)
- cloudy_s2: [13, H, W]
- sar_s1: [2, H, W]
"""

import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader, Subset


class CloudRemovalDatasetWrapper(Dataset):
    """
    Wraps SEN12MSCRTS (or SEN12MSCR) to output PyTorch tensors.
    """
    def __init__(self, raw_dataset):
        self.dataset = raw_dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sample = self.dataset[idx]
        
        # Determine format based on dataloader return structure
        if 'input' in sample and 'target' in sample:
            # S1 SAR
            s1_data = sample['input']['S1']
            if isinstance(s1_data, list):
                s1 = s1_data[0] # take single observation
            else:
                s1 = s1_data
                
            # S2 Cloudy Optical
            s2_data = sample['input']['S2']
            if isinstance(s2_data, list):
                s2_cloudy = s2_data[0]
            else:
                s2_cloudy = s2_data
                
            # Cloud Mask
            mask_data = sample['input']['masks']
            if isinstance(mask_data, list):
                mask = mask_data[0]
            else:
                mask = mask_data
                
            # Target S2 Cloud-free
            target_data = sample['target']['S2']
            if isinstance(target_data, list):
                s2_target = target_data[0]
            else:
                s2_target = target_data
        else:
            raise ValueError(f"Unrecognized sample structure keys: {sample.keys()}")

        # Ensure mask is [1, H, W]
        if mask is None:
            # Fallback: estimate threshold on cloudy image RGB mean if mask is None
            mask = np.zeros((1, s2_cloudy.shape[1], s2_cloudy.shape[2]), dtype=np.float32)
        elif mask.ndim == 2:
            mask = mask[np.newaxis, ...].astype(np.float32)
        elif mask.ndim == 3 and mask.shape[0] != 1:
            mask = mask[:1, ...].astype(np.float32)

        # Ensure float32 numpy arrays
        s2_cloudy = np.asarray(s2_cloudy, dtype=np.float32)
        s1 = np.asarray(s1, dtype=np.float32)
        s2_target = np.asarray(s2_target, dtype=np.float32)
        mask = np.asarray(mask, dtype=np.float32)

        # Convert to torch tensors
        t_cloudy = torch.from_numpy(s2_cloudy)  # [13, H, W]
        t_s1 = torch.from_numpy(s1)            # [2, H, W]
        t_mask = torch.from_numpy(mask)        # [1, H, W]
        t_target = torch.from_numpy(s2_target)  # [13, H, W]

        # Concatenate into single input tensor: [16, H, W]
        input_tensor = torch.cat([t_cloudy, t_s1, t_mask], dim=0)

        return {
            'input': input_tensor,
            'target': t_target,
            'cloudy_s2': t_cloudy,
            'sar_s1': t_s1,
            'mask': t_mask,
            'index': idx
        }


class SyntheticCloudRemovalDataset(Dataset):
    """
    Synthetic dataset generator for testing/demoing when raw satellite archives
    have not yet been downloaded. Generates realistic geometric/textured terrain patches
    with synthetic cloud artifacts and simulated SAR backscatter.
    """
    def __init__(self, num_samples=100, img_size=256):
        self.num_samples = num_samples
        self.img_size = img_size

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        rng = np.random.RandomState(idx + 42)
        H, W = self.img_size, self.img_size

        # Create smooth terrain texture for 13 bands
        x = np.linspace(0, 4 * np.pi, W)
        y = np.linspace(0, 4 * np.pi, H)
        xx, yy = np.meshgrid(x, y)
        base_pattern = 0.5 + 0.3 * np.sin(xx + rng.uniform(0, np.pi)) * np.cos(yy + rng.uniform(0, np.pi))
        
        # 13 optical bands [0, 1]
        target_s2 = np.zeros((13, H, W), dtype=np.float32)
        for b in range(13):
            noise = rng.normal(0, 0.05, (H, W))
            target_s2[b] = np.clip(base_pattern * (0.6 + 0.4 * (b / 13.0)) + noise, 0.05, 0.95)

        # 2 SAR channels (VV, VH) -> correlated with terrain edges
        sar_s1 = np.zeros((2, H, W), dtype=np.float32)
        sar_s1[0] = np.clip(np.abs(np.gradient(base_pattern, axis=0)) * 2.0 + rng.normal(0.3, 0.08, (H, W)), 0, 1)
        sar_s1[1] = np.clip(np.abs(np.gradient(base_pattern, axis=1)) * 2.0 + rng.normal(0.2, 0.08, (H, W)), 0, 1)

        # Synthetic cloud blob & mask
        cx, cy = rng.randint(H // 4, 3 * H // 4), rng.randint(W // 4, 3 * W // 4)
        radius = rng.randint(H // 6, H // 3)
        dist = np.sqrt((xx - (cx / H * 4 * np.pi))**2 + (yy - (cy / W * 4 * np.pi))**2)
        raw_cloud = np.exp(-dist**2 / (2 * (radius / H * 4 * np.pi)**2))
        raw_cloud = np.clip(raw_cloud * 1.5, 0, 1)

        mask = (raw_cloud > 0.35).astype(np.float32)[np.newaxis, ...] # [1, H, W]

        # Cloudy optical image (clouds occlude terrain)
        cloudy_s2 = target_s2.copy()
        for b in range(13):
            cloudy_s2[b] = cloudy_s2[b] * (1 - raw_cloud) + 0.95 * raw_cloud

        t_cloudy = torch.from_numpy(cloudy_s2)
        t_s1 = torch.from_numpy(sar_s1)
        t_mask = torch.from_numpy(mask)
        t_target = torch.from_numpy(target_s2)
        input_tensor = torch.cat([t_cloudy, t_s1, t_mask], dim=0)

        return {
            'input': input_tensor,
            'target': t_target,
            'cloudy_s2': t_cloudy,
            'sar_s1': t_s1,
            'mask': t_mask,
            'index': idx
        }


def get_dataloaders(dataroot=None, region='asiaWest', split_train='train', split_val='val', split_test='test',
                    batch_size=4, num_workers=2, subset_size=None, use_dummy=False):
    """
    Returns train_loader, val_loader, test_loader.
    If dataroot is invalid or empty and use_dummy is True (or dataset fails to find files),
    it falls back cleanly to the synthetic dataset.
    """
    use_synthetic = use_dummy
    train_dataset = None
    val_dataset = None
    test_dataset = None

    if not use_synthetic and dataroot and os.path.exists(dataroot):
        try:
            from data.dataLoader import SEN12MSCRTS
            print(f"[DataLoader] Initializing SEN12MSCRTS from '{dataroot}' (Region: {region})...")
            # Note: Using cloud_cloudshadow_mask to avoid requiring s2cloudless binary install
            raw_train = SEN12MSCRTS(dataroot, split=split_train, region=region,
                                    cloud_masks='cloud_cloudshadow_mask', sample_type='cloudy_cloudfree',
                                    n_input_samples=1, rescale_method='default')
            raw_val = SEN12MSCRTS(dataroot, split=split_val, region=region,
                                  cloud_masks='cloud_cloudshadow_mask', sample_type='cloudy_cloudfree',
                                  n_input_samples=1, rescale_method='default')
            raw_test = SEN12MSCRTS(dataroot, split=split_test, region=region,
                                   cloud_masks='cloud_cloudshadow_mask', sample_type='cloudy_cloudfree',
                                   n_input_samples=1, rescale_method='default')
            
            if len(raw_train) == 0:
                print("[DataLoader] Warning: 0 samples found in dataset directory. Falling back to synthetic dataset.")
                use_synthetic = True
            else:
                train_dataset = CloudRemovalDatasetWrapper(raw_train)
                val_dataset = CloudRemovalDatasetWrapper(raw_val)
                test_dataset = CloudRemovalDatasetWrapper(raw_test)
                print(f"[DataLoader] Successfully loaded SEN12MSCRTS: {len(train_dataset)} train, {len(val_dataset)} val, {len(test_dataset)} test samples.")
        except Exception as e:
            print(f"[DataLoader] Error loading real dataset: {e}. Falling back to synthetic dataset.")
            use_synthetic = True

    if use_synthetic or train_dataset is None:
        print("[DataLoader] Generating synthetic satellite demo dataset (200 train, 40 val, 40 test)...")
        train_dataset = SyntheticCloudRemovalDataset(num_samples=200)
        val_dataset = SyntheticCloudRemovalDataset(num_samples=40)
        test_dataset = SyntheticCloudRemovalDataset(num_samples=40)

    # Apply subsetting if requested (e.g. for quick 1-day demo on few hundred tiles)
    if subset_size is not None and subset_size > 0:
        actual_train_size = min(subset_size, len(train_dataset))
        train_dataset = Subset(train_dataset, range(actual_train_size))
        actual_val_size = min(max(1, subset_size // 5), len(val_dataset))
        val_dataset = Subset(val_dataset, range(actual_val_size))
        actual_test_size = min(subset_size, len(test_dataset))
        test_dataset = Subset(test_dataset, range(actual_test_size))
        print(f"[DataLoader] Subsetting dataset to {len(train_dataset)} train, {len(val_dataset)} val, {len(test_dataset)} test tiles.")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, test_loader
