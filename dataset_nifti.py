"""
NIfTI-based Dataset for SCRD4AD Training
Loads NIfTI volumes directly instead of pre-processed PNG slices.
"""

import torch
from torchvision import transforms
import numpy as np
import os
import glob
import nibabel as nib
from models.noise import Simplex_CLASS

torch.multiprocessing.set_sharing_strategy('file_system')


class NiftiSliceDataset(torch.utils.data.Dataset):
    """
    Dataset that loads NIfTI volumes and extracts 2D slices on-the-fly.
    Each sample is a 2D axial slice from a 3D volume.
    
    Args:
        nifti_dirs: List of directories containing .nii.gz files
        min_slice_pct: Minimum slice percentage to use (0-100), default 10%
        max_slice_pct: Maximum slice percentage to use (0-100), default 90%
        mylambda: Noise intensity for simplex noise augmentation
    """
    
    def __init__(self, nifti_dirs, min_slice_pct=10, max_slice_pct=90, mylambda=0.2):
        self.nifti_dirs = nifti_dirs if isinstance(nifti_dirs, list) else [nifti_dirs]
        self.min_slice_pct = min_slice_pct
        self.max_slice_pct = max_slice_pct
        self.mylambda = mylambda
        self.img_size = (256, 256)
        self.simplexNoise = Simplex_CLASS()
        
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            transforms.Resize(self.img_size)
        ])
        
        # Build index of (volume_path, slice_idx) pairs
        self.samples = []
        self._build_sample_index()
        
    def _build_sample_index(self):
        """Build a list of (nifti_path, slice_idx) tuples."""
        for nifti_dir in self.nifti_dirs:
            nifti_files = sorted(glob.glob(os.path.join(nifti_dir, '*.nii.gz')))
            print(f"Found {len(nifti_files)} NIfTI files in {nifti_dir}")
            
            for nifti_path in nifti_files:
                # Load header to get number of slices
                img = nib.load(nifti_path)
                num_slices = img.shape[2]  # Assuming axial is 3rd dimension
                
                # Calculate slice range
                min_slice = int(num_slices * self.min_slice_pct / 100)
                max_slice = int(num_slices * self.max_slice_pct / 100)
                
                for slice_idx in range(min_slice, max_slice):
                    self.samples.append((nifti_path, slice_idx))
        
        print(f"Total training samples (slices): {len(self.samples)}")
    
    def _normalize_slice(self, slice_2d):
        """Normalize a 2D slice to 0-1 range."""
        slice_min = slice_2d.min()
        slice_max = slice_2d.max()
        if slice_max > slice_min:
            slice_2d = (slice_2d - slice_min) / (slice_max - slice_min)
        else:
            slice_2d = np.zeros_like(slice_2d)
        return slice_2d
    
    def _resize_slice(self, slice_2d, target_size=(256, 256)):
        """Resize 2D slice to target size using numpy/scipy."""
        from scipy.ndimage import zoom
        h, w = slice_2d.shape
        zoom_h = target_size[0] / h
        zoom_w = target_size[1] / w
        return zoom(slice_2d, (zoom_h, zoom_w), order=1)
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        nifti_path, slice_idx = self.samples[idx]
        
        # Load volume and extract slice
        img = nib.load(nifti_path)
        volume = img.get_fdata()
        slice_2d = volume[:, :, slice_idx]
        
        # Normalize to 0-1
        slice_2d = self._normalize_slice(slice_2d)
        
        # Resize to 256x256
        slice_2d = self._resize_slice(slice_2d, self.img_size)
        
        # Convert grayscale to 3-channel (for pretrained ResNet)
        img_3ch = np.stack([slice_2d, slice_2d, slice_2d], axis=-1)  # HWC
        
        # Apply transforms (normal image)
        img_normal = self.transform(img_3ch.copy())
        img_normal = img_normal.float()
        
        # Add simplex noise
        size = 256
        h_noise = np.random.randint(10, int(size // 8))
        w_noise = np.random.randint(10, int(size // 8))
        start_h_noise = np.random.randint(1, size - h_noise)
        start_w_noise = np.random.randint(1, size - w_noise)
        noise_size = (h_noise, w_noise)
        simplex_noise = self.simplexNoise.rand_3d_octaves((3, *noise_size), 6, 0.6)
        
        init_zero = np.zeros((256, 256, 3))
        init_zero[start_h_noise:start_h_noise + h_noise, 
                  start_w_noise:start_w_noise + w_noise, :] = self.mylambda * simplex_noise.transpose(1, 2, 0)
        
        img_noise = img_3ch + init_zero
        img_noise = self.transform(img_noise)
        img_noise = img_noise.float()
        
        return img_normal, img_noise


class NiftiSliceDatasetCached(torch.utils.data.Dataset):
    """
    Cached version that loads all volumes into memory for faster training.
    Use this if you have enough RAM.
    
    Args:
        nifti_dirs: List of directories containing .nii.gz files
        min_slice_pct: Minimum slice percentage to use (0-100), default 10%
        max_slice_pct: Maximum slice percentage to use (0-100), default 90%
        mylambda: Noise intensity for simplex noise augmentation
    """
    
    def __init__(self, nifti_dirs, min_slice_pct=10, max_slice_pct=90, mylambda=0.2):
        self.nifti_dirs = nifti_dirs if isinstance(nifti_dirs, list) else [nifti_dirs]
        self.min_slice_pct = min_slice_pct
        self.max_slice_pct = max_slice_pct
        self.mylambda = mylambda
        self.img_size = (256, 256)
        self.simplexNoise = Simplex_CLASS()
        
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            transforms.Resize(self.img_size)
        ])
        
        # Pre-load and cache all slices
        self.slices = []
        self._load_all_slices()
    
    def _normalize_slice(self, slice_2d):
        """Normalize a 2D slice to 0-1 range."""
        slice_min = slice_2d.min()
        slice_max = slice_2d.max()
        if slice_max > slice_min:
            slice_2d = (slice_2d - slice_min) / (slice_max - slice_min)
        else:
            slice_2d = np.zeros_like(slice_2d)
        return slice_2d
    
    def _resize_slice(self, slice_2d, target_size=(256, 256)):
        """Resize 2D slice to target size using scipy."""
        from scipy.ndimage import zoom
        h, w = slice_2d.shape
        zoom_h = target_size[0] / h
        zoom_w = target_size[1] / w
        return zoom(slice_2d, (zoom_h, zoom_w), order=1)
    
    def _load_all_slices(self):
        """Load all slices into memory."""
        print("Loading NIfTI volumes into memory...")
        
        for nifti_dir in self.nifti_dirs:
            nifti_files = sorted(glob.glob(os.path.join(nifti_dir, '*.nii.gz')))
            print(f"Loading {len(nifti_files)} volumes from {nifti_dir}")
            
            for nifti_path in nifti_files:
                img = nib.load(nifti_path)
                volume = img.get_fdata()
                num_slices = volume.shape[2]
                
                min_slice = int(num_slices * self.min_slice_pct / 100)
                max_slice = int(num_slices * self.max_slice_pct / 100)
                
                for slice_idx in range(min_slice, max_slice):
                    slice_2d = volume[:, :, slice_idx]
                    slice_2d = self._normalize_slice(slice_2d)
                    slice_2d = self._resize_slice(slice_2d, self.img_size)
                    
                    # Convert to 3-channel and store as float32
                    img_3ch = np.stack([slice_2d, slice_2d, slice_2d], axis=-1).astype(np.float32)
                    self.slices.append(img_3ch)
        
        print(f"Cached {len(self.slices)} slices in memory")
    
    def __len__(self):
        return len(self.slices)
    
    def __getitem__(self, idx):
        img_3ch = self.slices[idx]
        
        # Apply transforms (normal image)
        img_normal = self.transform(img_3ch.copy())
        img_normal = img_normal.float()
        
        # Add simplex noise
        size = 256
        h_noise = np.random.randint(10, int(size // 8))
        w_noise = np.random.randint(10, int(size // 8))
        start_h_noise = np.random.randint(1, size - h_noise)
        start_w_noise = np.random.randint(1, size - w_noise)
        noise_size = (h_noise, w_noise)
        simplex_noise = self.simplexNoise.rand_3d_octaves((3, *noise_size), 6, 0.6)
        
        init_zero = np.zeros((256, 256, 3), dtype=np.float32)
        init_zero[start_h_noise:start_h_noise + h_noise, 
                  start_w_noise:start_w_noise + w_noise, :] = self.mylambda * simplex_noise.transpose(1, 2, 0)
        
        img_noise = img_3ch + init_zero
        img_noise = self.transform(img_noise)
        img_noise = img_noise.float()
        
        return img_normal, img_noise


if __name__ == '__main__':
    # Test the dataset
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--nifti_dirs', nargs='+', required=True)
    parser.add_argument('--cached', action='store_true')
    args = parser.parse_args()
    
    if args.cached:
        dataset = NiftiSliceDatasetCached(args.nifti_dirs)
    else:
        dataset = NiftiSliceDataset(args.nifti_dirs)
    
    print(f"Dataset size: {len(dataset)}")
    
    # Test loading a sample
    img_normal, img_noise = dataset[0]
    print(f"Normal image shape: {img_normal.shape}")
    print(f"Noise image shape: {img_noise.shape}")
