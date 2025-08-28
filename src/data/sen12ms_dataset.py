import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import rasterio
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List
import albumentations as A
from albumentations.pytorch import ToTensorV2
import json

class SEN12MSDataset(Dataset):
    """SEN12MS dataset for multi-modal geospatial data."""
    
    def __init__(self, 
                 data_root: str,
                 split: str = 'train',
                 modalities: List[str] = ['s1', 's2', 'lc'],
                 transform: Optional[A.Compose] = None,
                 target_size: Tuple[int, int] = (256, 256),
                 normalize: bool = True):
        """
        Args:
            data_root: Root directory of SEN12MS dataset
            split: Dataset split ('train', 'val', 'test')
            modalities: List of modalities to load ['s1', 's2', 'lc']
            transform: Albumentations transform pipeline
            target_size: Target image size (H, W)
            normalize: Whether to normalize the data
        """
        self.data_root = Path(data_root)
        self.split = split
        self.modalities = modalities
        self.transform = transform
        self.target_size = target_size
        self.normalize = normalize
        
        # Load dataset metadata
        self.samples = self._load_samples()
        
        # Normalization statistics for different modalities
        self.norm_stats = {
            's1': {'mean': [-11.76, -18.294], 'std': [4.525, 4.932]},
            's2': {'mean': [1354.40, 1118.24, 1042.47, 947.62, 1199.47, 2004.61, 2376.00, 2303.28, 732.08, 12.11, 1819.12, 1118.92, 2594.14], 
                   'std': [245.71, 333.00, 395.09, 593.75, 566.4, 861.18, 1086.63, 1117.98, 404.89, 4.77, 1002.58, 761.30, 1231.58]},
            'lc': {'num_classes': 17}
        }
        
    def _load_samples(self) -> List[Dict[str, str]]:
        """Load sample paths for the specified split."""
        samples = []
        
        # Define split directories
        split_dirs = {
            'train': ['ROIs1158_spring', 'ROIs1868_summer', 'ROIs1970_fall', 'ROIs2017_winter'],
            'val': ['ROIs1158_spring_val', 'ROIs1868_summer_val'],
            'test': ['ROIs1158_spring_test', 'ROIs1868_summer_test']
        }
        
        for season_dir in split_dirs.get(self.split, []):
            season_path = self.data_root / season_dir
            if not season_path.exists():
                continue
                
            # Find all patches in this season
            for patch_dir in season_path.iterdir():
                if patch_dir.is_dir():
                    sample = {'patch_id': patch_dir.name}
                    
                    # Add paths for each modality
                    if 's1' in self.modalities:
                        s1_path = patch_dir / f"{patch_dir.name}_s1.tif"
                        if s1_path.exists():
                            sample['s1'] = str(s1_path)
                    
                    if 's2' in self.modalities:
                        s2_path = patch_dir / f"{patch_dir.name}_s2.tif"
                        if s2_path.exists():
                            sample['s2'] = str(s2_path)
                    
                    if 'lc' in self.modalities:
                        lc_path = patch_dir / f"{patch_dir.name}_lc.tif"
                        if lc_path.exists():
                            sample['lc'] = str(lc_path)
                    
                    # Only add sample if all required modalities exist
                    if all(mod in sample for mod in self.modalities):
                        samples.append(sample)
        
        return samples
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a sample from the dataset."""
        sample = self.samples[idx]
        data = {'patch_id': sample['patch_id']}
        
        # Load each modality
        for modality in self.modalities:
            if modality in sample:
                data[modality] = self._load_modality(sample[modality], modality)
        
        # Apply transforms
        if self.transform:
            # Prepare data for albumentations
            transform_data = {}
            if 's1' in data:
                transform_data['s1'] = data['s1']
            if 's2' in data:
                transform_data['s2'] = data['s2']
            if 'lc' in data:
                transform_data['mask'] = data['lc']
            
            # Apply transforms
            transformed = self.transform(**transform_data)
            
            # Update data with transformed values
            if 's1' in transformed:
                data['s1'] = transformed['s1']
            if 's2' in transformed:
                data['s2'] = transformed['s2']
            if 'mask' in transformed:
                data['lc'] = transformed['mask']
        
        return data
    
    def _load_modality(self, file_path: str, modality: str) -> np.ndarray:
        """Load and preprocess a specific modality."""
        with rasterio.open(file_path) as src:
            data = src.read()  # Shape: (C, H, W)
            
        # Convert to float32
        data = data.astype(np.float32)
        
        # Transpose to (H, W, C) for albumentations
        if modality != 'lc':
            data = np.transpose(data, (1, 2, 0))
        else:
            # Land cover is single channel, keep as (H, W)
            data = data[0]  # Remove channel dimension
        
        # Resize if needed
        if data.shape[:2] != self.target_size:
            if modality == 'lc':
                # Use nearest neighbor for labels
                data = self._resize_nearest(data, self.target_size)
            else:
                # Use bilinear for continuous data
                data = self._resize_bilinear(data, self.target_size)
        
        # Normalize
        if self.normalize and modality in self.norm_stats:
            if modality != 'lc':
                mean = np.array(self.norm_stats[modality]['mean'])
                std = np.array(self.norm_stats[modality]['std'])
                data = (data - mean) / std
        
        return data
    
    def _resize_nearest(self, data: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
        """Resize using nearest neighbor interpolation."""
        from skimage.transform import resize
        return resize(data, target_size, order=0, preserve_range=True, anti_aliasing=False)
    
    def _resize_bilinear(self, data: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
        """Resize using bilinear interpolation."""
        from skimage.transform import resize
        return resize(data, target_size + (data.shape[-1],), order=1, preserve_range=True, anti_aliasing=True)

def get_sen12ms_transforms(split: str = 'train', target_size: Tuple[int, int] = (256, 256)) -> A.Compose:
    """Get albumentations transforms for SEN12MS dataset."""
    
    if split == 'train':
        transforms = A.Compose([
            A.RandomRotate90(p=0.5),
            A.Flip(p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.3),
            A.GaussNoise(var_limit=(10.0, 50.0), p=0.2),
            A.Resize(target_size[0], target_size[1]),
            ToTensorV2()
        ], additional_targets={'s1': 'image', 's2': 'image'})
    else:
        transforms = A.Compose([
            A.Resize(target_size[0], target_size[1]),
            ToTensorV2()
        ], additional_targets={'s1': 'image', 's2': 'image'})
    
    return transforms

def create_sen12ms_dataloaders(data_root: str,
                              batch_size: int = 16,
                              num_workers: int = 4,
                              target_size: Tuple[int, int] = (256, 256),
                              modalities: List[str] = ['s1', 's2', 'lc']) -> Dict[str, DataLoader]:
    """Create SEN12MS dataloaders for all splits."""
    
    dataloaders = {}
    
    for split in ['train', 'val', 'test']:
        transform = get_sen12ms_transforms(split, target_size)
        
        dataset = SEN12MSDataset(
            data_root=data_root,
            split=split,
            modalities=modalities,
            transform=transform,
            target_size=target_size
        )
        
        shuffle = (split == 'train')
        
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=(split == 'train')
        )
        
        dataloaders[split] = dataloader
    
    return dataloaders