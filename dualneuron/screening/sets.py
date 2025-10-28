import os
import torch
from glob import glob
    
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import numpy as np
import cv2

from datasets import load_dataset


class MaskTransform:
    """Custom transform for applying a mask to a tensor"""
    def __init__(self, mask, bg_value=0.0):
        self.mask = mask
        self.bg_value = bg_value
        
    def __call__(self, tensor):
        h, w = tensor.shape[1], tensor.shape[2]
        if self.mask.shape != (h, w):
            mask = cv2.resize(
                self.mask, (w, h), 
                interpolation=cv2.INTER_LINEAR
            )
        else:
            mask = self.mask
            
        mask = mask.astype(np.float32)
        mask = (mask - mask.min()) / (mask.max() - mask.min() + 1e-8)
        mask = torch.from_numpy(mask)
        
        if tensor.shape[0] == 1:
            tensor = tensor * mask + self.bg_value * (1 - mask)
        else:
            mask = mask.unsqueeze(0)
            tensor = tensor * mask + self.bg_value * (1 - mask)
            
        return tensor


class NormTransform:
    """Custom transform for normalizing a tensor"""
    def __init__(self, norm):
        self.norm = norm
        
    def __call__(self, tensor):
        current_norm = torch.norm(tensor.view(-1))
        tensor = tensor * self.norm / (current_norm + 1e-8)
        return tensor


class EnsureRGB:
    def __call__(self, img):
        if img.mode != 'RGB':
            return img.convert('RGB')
        return img
    

class ImagenetImages(Dataset):
    def __init__(
        self,
        data_dir=None,
        token=None,
        split='train',
        # Transform options
        use_center_crop=False,
        use_resize_output=False,
        use_grayscale=False,
        use_normalize=False,
        use_mask=False,
        use_norm=False,
        # Transform parameters
        mask=None,
        num_channels=None,
        output_size=(224, 224),
        crop_size=236,
        bg_value=0.0,
        norm=None,
    ):
        """
        ImageNet dataset with flexible transform pipeline.
        
        Base transforms (always applied):
        1. EnsureRGB - Convert to RGB if needed
        2. Resize(256) - Initial resize
        
        Optional transforms (controlled by use_* flags):
        - use_center_crop: Apply CenterCrop(crop_size)
        - use_resize_output: Resize to output_size
        - use_grayscale: Convert to grayscale
        - use_normalize: Apply ImageNet normalization
        - use_mask: Apply mask transform (requires mask parameter)
        - use_norm: Apply norm transform (requires norm parameter)
        
        Args:
            data_dir: Where imagenet is saved locally 
            token: HuggingFace token for dataset download
            split: Dataset split to use ('train', 'validation' or 'test')
            use_center_crop: Whether to apply center cropping
            use_resize_output: Whether to resize to output_size
            use_grayscale: Whether to convert to grayscale
            use_normalize: Whether to apply normalization
            use_mask: Whether to apply mask transform
            use_norm: Whether to apply norm transform
            mask: Mask array for MaskTransform (required if use_mask=True)
            num_channels: Number of output channels (auto-detected if None)
            output_size: Target size for resize (default: (224, 224))
            crop_size: Size for center crop (default: 236)
            bg_value: Background value for mask transform (default: 0.0)
            norm: Norm value for NormTransform (required if use_norm=True)
        """
        
        # Create cache directory if it doesn't exist and was specified
        if data_dir is not None:
            os.makedirs(data_dir, exist_ok=True)
        
        # Load dataset - token only needed for first download
        # After that, it will load from cache
        self.set = load_dataset(
            "ILSVRC/imagenet-1k", 
            token=token,  # Use passed token (can be None if already cached)
            trust_remote_code=False,
            cache_dir=data_dir,
            split=split,
            num_proc=1,
        )
        
        self.mask = mask
        self.output_size = output_size
        self.crop_size = crop_size
        self.use_grayscale = use_grayscale
        
        if num_channels is not None:
            self.num_channels = num_channels
        else:
            self.num_channels = 1 if use_grayscale else 3
        
        # Build transform pipeline
        tlist = []
        
        # Base transforms (always applied)
        tlist.append(EnsureRGB())
        tlist.append(transforms.Resize(256))
        
        # Optional transforms
        if use_center_crop:
            tlist.append(transforms.CenterCrop(crop_size))
            
        if use_resize_output:
            tlist.append(transforms.Resize(output_size))
        
        if use_grayscale:
            tlist.append(transforms.Grayscale())
            
        # Always convert to tensor after PIL transforms
        tlist.append(transforms.ToTensor())
        
        if use_normalize:
            tlist.append(self.get_normalization())
            
        if use_mask:
            if mask is None:
                raise ValueError("mask parameter required when use_mask=True")
            tlist.append(MaskTransform(mask, bg_value))
            
        if use_norm:
            if norm is None:
                raise ValueError("norm parameter required when use_norm=True")
            tlist.append(NormTransform(norm))
        
        self.transform = transforms.Compose(tlist)
    
    def get_normalization(self):
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        
        if self.num_channels == 1:
            mean_gray = sum(mean) / 3
            std_gray = sum(std) / 3
            return transforms.Normalize(
                [mean_gray], [std_gray]
            )
        else:
            return transforms.Normalize(mean, std)

    def __getitem__(self, idx):
        image = self.set[idx]['image']
        label = self.set[idx]['label']
        tensor = self.transform(image)
        
        if self.num_channels == 3 and tensor.shape[0] == 1:
            tensor = tensor.repeat(3, 1, 1)

        return tensor, label
    
    def __len__(self):
        return len(self.set)
    

class RenderedImages(Dataset):
    def __init__(
        self, 
        data_dir,
        # Transform options
        use_center_crop=False,
        use_resize_output=False,
        use_grayscale=False,
        use_normalize=False,
        use_mask=False,
        use_norm=False,
        # Transform parameters
        mask=None,
        num_channels=None,
        output_size=(224, 224),
        crop_size=236,
        bg_value=0.0,
        norm=None,
    ):
        """
        Rendered images dataset with flexible transform pipeline.
        
        No base transforms - all transforms are optional.
        
        Optional transforms (controlled by use_* flags):
        - use_center_crop: Apply CenterCrop(crop_size)
        - use_resize_output: Resize to output_size
        - use_grayscale: Convert to grayscale
        - use_normalize: Apply ImageNet normalization
        - use_mask: Apply mask transform (requires mask parameter)
        - use_norm: Apply norm transform (requires norm parameter)
        
        Args:
            datadir: Where the png data is saved
            use_center_crop: Whether to apply center cropping
            use_resize_output: Whether to resize to output_size
            use_grayscale: Whether to convert to grayscale
            use_normalize: Whether to apply normalization
            use_mask: Whether to apply mask transform
            use_norm: Whether to apply norm transform
            mask: Mask array for MaskTransform (required if use_mask=True)
            num_channels: Number of output channels (auto-detected if None)
            output_size: Target size for resize (default: (224, 224))
            crop_size: Size for center crop (default: 236)
            bg_value: Background value for mask transform (default: 0.0)
            norm: Norm value for NormTransform (required if use_norm=True)
        """
        
        png_files = sorted(glob(os.path.join(data_dir, '*.png')))
        self.png_files = png_files
        self.mask = mask
        self.output_size = output_size
        self.crop_size = crop_size
        self.use_grayscale = use_grayscale
        
        if num_channels is not None:
            self.num_channels = num_channels
        else:
            self.num_channels = 1 if use_grayscale else 3
        
        # Build transform pipeline
        tlist = []
        
        # Optional PIL transforms (before ToTensor)
        if use_center_crop:
            tlist.append(transforms.CenterCrop(crop_size))
            
        if use_resize_output:
            tlist.append(transforms.Resize(output_size))
        
        if use_grayscale:
            tlist.append(transforms.Grayscale())
        
        # Convert to tensor
        tlist.append(transforms.ToTensor())
        
        # Optional tensor transforms
        if use_normalize:
            tlist.append(self.get_normalization())
            
        if use_mask:
            if mask is None:
                raise ValueError("mask parameter required when use_mask=True")
            tlist.append(MaskTransform(mask, bg_value))
            
        if use_norm:
            if norm is None:
                raise ValueError("norm parameter required when use_norm=True")
            tlist.append(NormTransform(norm))
        
        self.transform = transforms.Compose(tlist)
    
    def get_normalization(self):
        """Get appropriate normalization transform"""
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        
        if self.num_channels == 1:  # grayscale
            mean_gray = sum(mean) / 3
            std_gray = sum(std) / 3
            return transforms.Normalize([mean_gray], [std_gray])
        else:
            return transforms.Normalize(mean, std)

    def __len__(self):
        return len(self.png_files)

    def __getitem__(self, idx):
        img_path = self.png_files[idx]
        image = Image.open(img_path)
        tensor = self.transform(image)
        
        if self.num_channels == 3 and tensor.shape[0] == 1:
            tensor = tensor.repeat(3, 1, 1)

        return tensor, img_path