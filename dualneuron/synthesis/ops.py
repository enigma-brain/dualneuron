import torch
import torch.nn.functional as F
from torchvision.transforms.functional import gaussian_blur
from torchvision.ops import roi_align
from functools import lru_cache


def standardize(tensor):
    tensor = tensor - torch.mean(tensor)
    tensor = tensor / (torch.std(tensor) + 1e-4)
    return tensor


@lru_cache(maxsize=8)
def color_correlation(device):
    return torch.tensor(
        [[0.56282854, 0.58447580, 0.58447580],
         [0.19482528, 0.00000000, -0.19482528],
         [0.04329450, -0.10823626, 0.06494176]],
        dtype=torch.float32, 
        device=device
    )


def recorrelate_colors(image, device):
    assert len(image.shape) == 3
    correlation = color_correlation(device)
    permuted_image = image.permute(1, 2, 0).contiguous()
    flat_image = permuted_image.view(-1, 3)
    recorrelated = torch.matmul(flat_image, correlation)
    recorrelated = recorrelated.view(permuted_image.shape).permute(2, 0, 1)
    return recorrelated


def create_crops(
    image, nb_crops, box_size, input_size,
    jitter_std=0.03,
    oversample=2,
    reflect_pad_frac=0.05
):
    assert image.ndim == 3
    device = image.device
    C, H, W = image.shape
    s, b = box_size

    pad = int(reflect_pad_frac * min(H, W))
    if pad > 0:
        img_pad = F.pad(
            image.unsqueeze(0), 
            (pad, pad, pad, pad), 
            mode="reflect"
        ).squeeze(0)
        Hpad, Wpad = H + 2*pad, W + 2*pad
        x_offset = pad
        y_offset = pad
    else:
        img_pad = image
        Hpad, Wpad = H, W
        x_offset = 0
        y_offset = 0

    cx = 0.5 + torch.randn(nb_crops, device=device) * jitter_std
    cy = 0.5 + torch.randn(nb_crops, device=device) * jitter_std
    sc = torch.rand(nb_crops, device=device) * (b - s) + s
    bw, bh = sc * W, sc * H

    x1 = (cx * W + x_offset - 0.5 * bw).clamp(0, Wpad)
    y1 = (cy * H + y_offset - 0.5 * bh).clamp(0, Hpad)
    x2 = (cx * W + x_offset + 0.5 * bw).clamp(0, Wpad)
    y2 = (cy * H + y_offset + 0.5 * bh).clamp(0, Hpad)
    batch = torch.zeros_like(x1)

    boxes = torch.stack([batch, x1, y1, x2, y2], dim=1).to(torch.float32)

    hi = input_size * max(1, int(oversample))
    crops_hi = roi_align(
        img_pad.unsqueeze(0),
        boxes,
        output_size=(hi, hi),
        aligned=True
    )

    crops = F.interpolate(
        crops_hi, 
        size=(input_size, input_size),
        mode="bicubic", 
        align_corners=False, 
        antialias=True
    )
    return crops


def add_noise(image, noise_level):
    noisy = image.clone()
    noisy.add_(torch.randn_like(noisy) * noise_level)
    return noisy


def change_norm(image, target_norm):
    if target_norm is None:
        return image
    eps = 1e-8
    if image.ndim == 3:
        current = torch.norm(image.reshape(-1)) + eps
        return image * (target_norm / current)
    else:
        norms = torch.norm(image.reshape(image.shape[0], -1), dim=1, keepdim=True) + eps
        return image * (target_norm / norms.view(-1, 1, 1, 1))
