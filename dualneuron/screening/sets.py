import os
import io
import zipfile
import torch
from glob import glob
    
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import numpy as np
import cv2

from dotenv import load_dotenv
load_dotenv()

from datasets import load_dataset

from dualneuron.synthesis.ops import change_norm


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


class CropToMask:
    """
    Crop tensor to the bounding box of a mask and scale up to fill output size.
    Works on tensors (apply after ToTensor and MaskTransform).
    
    Automatically resizes the mask to match tensor dimensions before computing bbox.
    """
    def __init__(self, mask, output_size=(224, 224), padding_frac=0.0):
        """
        Args:
            mask: 2D numpy array, non-zero where content should be kept
            output_size: Target (H, W) after scaling - image will fill this entirely
            padding_frac: Fraction of bbox size to add as padding (default 0.0)
        """
        self.mask = mask
        self.output_size = output_size
        self.padding_frac = padding_frac
        self.bbox_cache = {}  # Cache bbox for different tensor sizes
    
    def _compute_bbox(self, mask, tensor_h, tensor_w):
        """Compute bounding box, resizing mask to match tensor if needed."""
        if mask.shape != (tensor_h, tensor_w):
            mask = cv2.resize(
                mask.astype(np.float32), 
                (tensor_w, tensor_h),
                interpolation=cv2.INTER_LINEAR
            )
        
        # Threshold to find mask region
        binary = (mask > mask.min() + 0.01 * (mask.max() - mask.min()))
        
        rows = np.any(binary, axis=1)
        cols = np.any(binary, axis=0)
        
        if not rows.any() or not cols.any():
            return (0, tensor_h, 0, tensor_w)
        
        y_indices = np.where(rows)[0]
        x_indices = np.where(cols)[0]
        y_min, y_max = y_indices[0], y_indices[-1] + 1
        x_min, x_max = x_indices[0], x_indices[-1] + 1
        
        # Add padding
        bbox_h = y_max - y_min
        bbox_w = x_max - x_min
        pad_y = int(bbox_h * self.padding_frac)
        pad_x = int(bbox_w * self.padding_frac)
        
        y_min = max(0, y_min - pad_y)
        y_max = min(tensor_h, y_max + pad_y)
        x_min = max(0, x_min - pad_x)
        x_max = min(tensor_w, x_max + pad_x)
        
        return (y_min, y_max, x_min, x_max)
    
    def __call__(self, tensor):
        """
        Crop and scale tensor to fill output_size entirely.
        """
        _, h, w = tensor.shape
        
        # Cache bbox per tensor size (computed once per size)
        if (h, w) not in self.bbox_cache:
            self.bbox_cache[(h, w)] = self._compute_bbox(self.mask, h, w)
            bbox = self.bbox_cache[(h, w)]
        
        y_min, y_max, x_min, x_max = self.bbox_cache[(h, w)]
        
        # Crop to bounding box
        cropped = tensor[:, y_min:y_max, x_min:x_max]
        
        # Scale to fill output_size completely
        cropped = cropped.unsqueeze(0)
        scaled = torch.nn.functional.interpolate(
            cropped,
            size=self.output_size,
            mode='bilinear',
            align_corners=False
        )
        return scaled.squeeze(0)
    

class NormTransform:
    """Rescale a tensor to a target L2 norm.

    ``values_range`` (optional) bounds the result as well: a plain rescale is a scalar multiply, so
    it can carry values outside the range the model was trained on whenever it scales up. Passing
    the range defers to :func:`dualneuron.synthesis.ops.change_norm`, which satisfies the norm and
    the bounds together -- the same primitive synthesis uses, so the two cannot drift apart. ``None``
    (the default) keeps the plain rescale, which is what screening has always done.
    """
    def __init__(self, norm, values_range=None):
        self.norm = norm
        self.values_range = values_range

    def __call__(self, tensor):
        if self.values_range is not None:
            return change_norm(tensor, self.norm, self.values_range)
        current_norm = torch.norm(tensor.view(-1))
        tensor = tensor * self.norm / (current_norm + 1e-8)
        return tensor


class ClipTransform:
    """Custom transform for clipping tensor values"""
    def __init__(self, min_val=0.0, max_val=1.0):
        self.min_val = min_val
        self.max_val = max_val
        
    def __call__(self, tensor):
        return torch.clamp(tensor, self.min_val, self.max_val)
    

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
        use_crop_to_mask=False,
        use_norm=False,
        use_clip=False,
        use_experiment_frame=True,
        # Transform parameters
        mask=None,
        num_channels=None,
        output_size=(224, 224),
        crop_size=200,
        bg_value=0.0,
        clip_min=0.0,
        clip_max=1.0,
        crop_padding_frac=0.1,
        norm=None,
        img_mean=None,
        img_std=None,
    ):
        """
        ImageNet dataset with flexible transform pipeline.
        
        Base transforms:
        1. EnsureRGB - Convert to RGB if needed
        2. Frame: if use_experiment_frame, Resize(short->420) + CenterCrop((236,420))
           to build the 236x420 experiment frame; else Resize(256)

        Optional transforms (controlled by use_* flags):
        - use_experiment_frame: Build the 236x420 experiment frame instead of Resize(256)
        - use_center_crop: Apply CenterCrop(crop_size)
        - use_resize_output: Resize to output_size
        - use_grayscale: Convert to grayscale
        - use_normalize: Apply ImageNet normalization
        - use_mask: Apply mask transform (requires mask parameter)
        - use_crop_to_mask: Crop to mask bounding box and scale (requires mask parameter)
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
            use_crop_to_mask: Whether to crop to mask bounding box and scale
            use_norm: Whether to apply norm transform
            use_experiment_frame: If True (default), build the 236x420 experiment
                frame (Resize short->420 + center-band crop) instead of Resize(256)
            mask: Mask array for MaskTransform (required if use_mask=True)
            num_channels: Number of output channels (auto-detected if None)
            output_size: Target size for resize (default: (224, 224))
            crop_size: Size for center crop (default: 236)
            bg_value: Background value for mask transform (default: 0.0)
            norm: Norm value for NormTransform (required if use_norm=True)
            img_mean: Scalar mean (0-255 scale) for the twin's training z-score; if
                set (with img_std), use_normalize applies (x - img_mean/255) /
                (img_std/255) instead of ImageNet stats. Default: None.
            img_std: Scalar std (0-255 scale) paired with img_mean. Default: None.
        """

        # Create cache directory if it doesn't exist and was specified
        if data_dir is not None:
            os.makedirs(data_dir, exist_ok=True)
        
        # Fall back to HF_TOKEN from the environment (.env) if not passed
        if token is None:
            token = os.getenv("HF_TOKEN")

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
        self.img_mean = img_mean
        self.img_std = img_std

        if num_channels is not None:
            self.num_channels = num_channels
        else:
            self.num_channels = 1 if use_grayscale else 3

        # Build transform pipeline
        tlist = []

        # Base transforms
        tlist.append(EnsureRGB())
        if use_experiment_frame:
            # Build the 236x420 experiment frame: resize the short side to 420,
            # then crop the center band.
            tlist.append(transforms.Resize(420))
            tlist.append(transforms.CenterCrop((236, 420)))
        else:
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
        
        if use_crop_to_mask:
            if mask is None:
                raise ValueError("mask parameter required when use_crop_to_mask=True")
            tlist.append(CropToMask(mask, output_size, crop_padding_frac))
            
        if use_norm:
            if norm is None:
                raise ValueError("norm parameter required when use_norm=True")
            tlist.append(NormTransform(norm))
            
        if use_clip:
            tlist.append(ClipTransform(clip_min, clip_max))
        
        self.transform = transforms.Compose(tlist)
    
    def get_normalization(self):
        # Twin models were trained on images z-scored by a single scalar mean/std on a
        # 0-255 scale (nnvision normalize_image). ToTensor yields [0,1], so the stats
        # are divided by 255 and replicated across channels to reproduce training.
        # Falls back to ImageNet stats when no twin stats are provided.
        if self.img_mean is not None and self.img_std is not None:
            mean = self.img_mean / 255.0
            std = self.img_std / 255.0
            return transforms.Normalize([mean] * self.num_channels, [std] * self.num_channels)

        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        if self.num_channels == 1:
            mean_gray = sum(mean) / 3
            std_gray = sum(std) / 3
            return transforms.Normalize([mean_gray], [std_gray])
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
        use_crop_to_mask=False,
        use_norm=False,
        use_clip=False,
        # Transform parameters
        mask=None,
        num_channels=None,
        output_size=(224, 224),
        crop_size=200,
        bg_value=0.0,
        clip_min=0.0,
        clip_max=1.0,
        crop_padding_frac=0.1,
        norm=None,
        img_mean=None,
        img_std=None,
    ):
        """
        Rendered images dataset with flexible transform pipeline.

        Reads rendered scenes from loose .png files in data_dir, or directly from the
        Dryad archives (batch_*.zip) when present (no unzip needed), in global scene order.

        No base transforms - all transforms are optional.
        
        Optional transforms (controlled by use_* flags):
        - use_center_crop: Apply CenterCrop(crop_size)
        - use_resize_output: Resize to output_size
        - use_grayscale: Convert to grayscale
        - use_normalize: Apply ImageNet normalization
        - use_mask: Apply mask transform (requires mask parameter)
        - use_crop_to_mask: Crop to mask bounding box and scale (requires mask parameter)
        - use_norm: Apply norm transform (requires norm parameter)
        
        Args:
            data_dir: Directory of rendered scenes; either loose .png files or the
                Dryad archives (batch_*.zip), which are read directly (no unzip).
            use_center_crop: Whether to apply center cropping
            use_resize_output: Whether to resize to output_size
            use_grayscale: Whether to convert to grayscale
            use_normalize: Whether to apply normalization
            use_mask: Whether to apply mask transform
            use_crop_to_mask: Whether to crop to mask bounding box and scale
            use_norm: Whether to apply norm transform
            mask: Mask array for MaskTransform (required if use_mask=True)
            num_channels: Number of output channels (auto-detected if None)
            output_size: Target size for resize (default: (224, 224))
            crop_size: Size for center crop (default: 236)
            bg_value: Background value for mask transform (default: 0.0)
            norm: Norm value for NormTransform (required if use_norm=True)
            crop_padding_frac: Padding fraction for CropToMask (default: 0.1)
            img_mean: Scalar mean (0-255 scale) for the twin's training z-score; if
                set (with img_std), use_normalize applies (x - img_mean/255) /
                (img_std/255) instead of ImageNet stats. Default: None.
            img_std: Scalar std (0-255 scale) paired with img_mean. Default: None.
        """
        
        # Source can be either a directory of loose PNGs, or the Dryad rendered
        # archives (batch_*.zip), each holding scene_NNNNNN.png files. When zips
        # are present, scenes are read directly from them, preserving the global
        # scene order (batch_001/scene_000000 ... batch_020/scene_199999) so that
        # indices match the ordered npz files in the Dryad release.
        self.zip_paths = sorted(glob(os.path.join(data_dir, 'batch_*.zip')))
        if self.zip_paths:
            self.from_zip = True
            self.index = []  # (zip_idx, member_name) in global scene order
            for zi, zp in enumerate(self.zip_paths):
                with zipfile.ZipFile(zp) as zf:
                    members = sorted(m for m in zf.namelist() if m.lower().endswith('.png'))
                self.index.extend((zi, m) for m in members)
            self._zhandles = {}  # per-process ZipFile cache (DataLoader fork-safe)
        else:
            self.from_zip = False
            self.png_files = sorted(glob(os.path.join(data_dir, '*.png')))
        self.mask = mask
        self.output_size = output_size
        self.crop_size = crop_size
        self.use_grayscale = use_grayscale
        self.img_mean = img_mean
        self.img_std = img_std

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
        
        if use_crop_to_mask:
            if mask is None:
                raise ValueError("mask parameter required when use_crop_to_mask=True")
            tlist.append(CropToMask(mask, output_size, crop_padding_frac))
            
        if use_norm:
            if norm is None:
                raise ValueError("norm parameter required when use_norm=True")
            tlist.append(NormTransform(norm))
            
        if use_clip:
            tlist.append(ClipTransform(clip_min, clip_max))
        
        self.transform = transforms.Compose(tlist)
    
    def get_normalization(self):
        """Get appropriate normalization transform"""
        # Twin models were trained on images z-scored by a single scalar mean/std on a
        # 0-255 scale (nnvision normalize_image). ToTensor yields [0,1], so the stats
        # are divided by 255 and replicated across channels to reproduce training.
        # Falls back to ImageNet stats when no twin stats are provided.
        if self.img_mean is not None and self.img_std is not None:
            mean = self.img_mean / 255.0
            std = self.img_std / 255.0
            return transforms.Normalize([mean] * self.num_channels, [std] * self.num_channels)

        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        if self.num_channels == 1:  # grayscale
            mean_gray = sum(mean) / 3
            std_gray = sum(std) / 3
            return transforms.Normalize([mean_gray], [std_gray])
        else:
            return transforms.Normalize(mean, std)

    def _get_zip(self, zip_idx):
        """Return a per-process ZipFile handle (lazily opened, DataLoader fork-safe)."""
        handles = self._zhandles.setdefault(os.getpid(), {})
        zf = handles.get(zip_idx)
        if zf is None:
            zf = zipfile.ZipFile(self.zip_paths[zip_idx])
            handles[zip_idx] = zf
        return zf

    def __len__(self):
        return len(self.index) if self.from_zip else len(self.png_files)

    def __getitem__(self, idx):
        if self.from_zip:
            zip_idx, member = self.index[idx]
            image = Image.open(io.BytesIO(self._get_zip(zip_idx).read(member)))
            label = member
        else:
            label = self.png_files[idx]
            image = Image.open(label)
        tensor = self.transform(image)

        if self.num_channels == 3 and tensor.shape[0] == 1:
            tensor = tensor.repeat(3, 1, 1)

        return tensor, label