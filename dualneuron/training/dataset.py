"""Datasets and transforms for twin readout training.

The recorded-response matrix and session loading are reused from
:mod:`dualneuron.data.recordings` (the same neuron ordering as the released twin). This module adds
only the training-specific pieces:

* :func:`split_train_val` — deterministic image-level train/val split.
* :func:`make_image_transform` — the eval transform (center-crop -> resize -> single-valued
  z-score) at the backbone's input size.
* :class:`ImageResponseDataset` — raw images + responses, for the one-time feature-extraction pass.
* :class:`CachedFeatureDataset` — pre-extracted frozen-core feature maps + responses, for fast
  readout training.
"""

import os
from typing import Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as T

# Re-exported for convenience so callers can pull everything data-related from one module.
from dualneuron.data.recordings import load_sessions, build_response_matrix  # noqa: F401
from dualneuron.twins import registry


def split_train_val(
    image_ids: np.ndarray,
    val_fraction: float = 0.2,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """Split image indices into train and validation sets.

    Args:
        image_ids: Array of image ids (only its length matters).
        val_fraction: Fraction held out for validation.
        seed: RNG seed for the permutation (reproducible split).

    Returns:
        ``(train_idx, val_idx)`` index arrays into the image / response-matrix rows.
    """
    rng = np.random.RandomState(seed)
    n = len(image_ids)
    perm = rng.permutation(n)
    n_val = int(round(n * val_fraction))
    return perm[n_val:], perm[:n_val]


def make_image_transform(input_size: int = 224, img_mean: float = 113.5,
                         img_std: float = 59.58, crop_size: int = 200,
                         channels: int = 3, upsample_size: int = None) -> T.Compose:
    """Image transform matching the twin's training preprocessing.

    Optionally upsample the frame first, then center-crop, bicubic-resize to the backbone's square
    input, to tensor, then the single-valued (all-channel) z-score ``(pixel - mean) / std``.
    ``ToTensor`` rescales 0-255 -> 0-1, so the stats are divided by 255 to act on the 0-1 tensor.

    Args:
        input_size: Square model input side (224 DINOv3, 100 V4 ResNet, 93 V1 ConvNeXt).
        img_mean: Pixel mean (single value applied to all channels).
        img_std: Pixel std (single value applied to all channels).
        crop_size: Center-crop side (200 for V4's 236x420 frame; 280 for V1 after the 420 upsample).
        channels: 1 (grayscale V1) or 3 (color V4); sets the normalize vector length.
        upsample_size: If set, bicubic-resize the frame to this square side BEFORE the center-crop
            (V1's 233 -> 420 -> crop 280 -> 93 stimulus pipeline). None -> no pre-crop upsample (V4).
    """
    steps = []
    if upsample_size is not None:
        steps.append(T.Resize((upsample_size, upsample_size),
                              interpolation=T.InterpolationMode.BICUBIC, antialias=True))
    steps += [
        T.CenterCrop(crop_size),
        T.Resize((input_size, input_size),
                 interpolation=T.InterpolationMode.BICUBIC, antialias=True),
        T.ToTensor(),
        T.Normalize(mean=[img_mean / 255] * channels, std=[img_std / 255] * channels),
    ]
    return T.Compose(steps)


def training_transform(area: str, backbone: str) -> T.Compose:
    """The twin's training-time image transform, built from the registry geometry: optional stimulus
    upsample -> center-crop -> resize to the twin's input -> z-score. Shared by feature/image caching
    (training) and the accuracy figure (which evaluates the twin on the recorded stimuli), so the two
    cannot diverge. Screening uses its own RF crop (``spec.crop_size``), NOT this transform.
    """
    spec = registry.resolve(area, backbone)
    crop = spec.train_crop or spec.crop_size
    return make_image_transform(spec.input_size, spec.img_mean, spec.img_std, crop,
                                spec.channels, upsample_size=spec.train_upsample)


class ImageResponseDataset(Dataset):
    """Yields ``(image_tensor, response_vector)`` from the ``{id:06d}.npy`` stimuli.

    ``response_vector`` is ``(n_neurons,)`` float32 with NaN where the neuron did not see the image.
    Used for the one-time frozen-core feature-extraction pass and to cache fine-tuned inputs.

    ``channels`` selects the PIL mode: 3 -> ``"RGB"`` (V4 color), 1 -> ``"L"`` (V1 grayscale).
    """

    def __init__(self, image_ids, responses, image_dir, transform=None, channels=3):
        self.image_ids = image_ids
        self.responses = responses          # (n_images, n_neurons) float32
        self.image_dir = image_dir
        self.transform = transform
        self.mode = "RGB" if channels == 3 else "L"

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        img_id = int(self.image_ids[idx])
        img = np.load(os.path.join(self.image_dir, f"{img_id:06d}.npy"))   # (H,W,3) or (H,W) uint8
        img = Image.fromarray(img).convert(self.mode)
        if self.transform is not None:
            img = self.transform(img)
        resp = torch.from_numpy(self.responses[idx])
        return img, resp


class CachedFeatureDataset(Dataset):
    """Yields ``(feature_map, response_vector)`` from in-RAM pre-extracted features.

    Features stay float16 in RAM; the float16 -> float32 cast happens on the GPU after transfer.
    ``feature_map`` is ``(C, H, W)`` float16 and ``response_vector`` is ``(n_neurons,)`` float32.
    """

    def __init__(self, features, responses):
        self.features = features            # (n_images, C, H, W) float16
        self.responses = responses          # (n_images, n_neurons) float32

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        feat = torch.from_numpy(np.array(self.features[idx]))     # (C, H, W) float16
        resp = torch.from_numpy(self.responses[idx].copy())       # (n_neurons,)
        return feat, resp
