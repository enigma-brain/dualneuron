import torch
import torch.nn.functional as F
from torchvision.ops import roi_align
from functools import lru_cache


@lru_cache(maxsize=8)
def color_correlation(device):
    """
    Get the color correlation matrix for natural images.
    
    Returns a cached matrix that transforms decorrelated color channels
    into correlated RGB channels matching natural image statistics.
    This is the inverse of a color decorrelation transform.
    
    Args:
        device (str or torch.device): Device to place the matrix on.
    
    Returns:
        torch.Tensor: 3x3 color correlation matrix.
    
    Note:
        Results are cached per device to avoid redundant tensor creation.
        The matrix is derived from ImageNet color statistics.
    """
    return torch.tensor(
        [[0.56282854, 0.58447580, 0.58447580],
         [0.19482528, 0.00000000, -0.19482528],
         [0.04329450, -0.10823626, 0.06494176]],
        dtype=torch.float32, 
        device=device
    )


def recorrelate_colors(image, device):
    """
    Apply color correlation to produce naturalistic RGB values.
    
    Transforms an image from a decorrelated color space back to RGB
    with correlations matching natural images. Used in Fourier-based
    image synthesis to produce more realistic colors.
    
    Args:
        image (torch.Tensor): Input image, shape (3, H, W).
        device (str or torch.device): Device for the correlation matrix.
    
    Returns:
        torch.Tensor: Color-correlated image, shape (3, H, W).
    
    Raises:
        AssertionError: If image is not 3-dimensional.
    """
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
    """
    Create random crops of an image for robust activation maximization.
    
    Generates multiple randomly positioned and scaled crops centered around
    the image center with jitter. Uses reflection padding to handle edge
    cases and oversampling for antialiasing.
    
    Args:
        image (torch.Tensor): Input image, shape (C, H, W).
        nb_crops (int): Number of crops to generate.
        box_size (tuple): (min_scale, max_scale) as fractions of image size.
            For example, (0.2, 0.25) means crops are 20-25% of the image.
        input_size (int): Output size for each crop (square).
        jitter_std (float): Standard deviation for center position jitter,
            as a fraction of image size. Default: 0.03.
        oversample (int): Factor to oversample crops before downscaling,
            improves antialiasing quality. Default: 2.
        reflect_pad_frac (float): Fraction of image size to pad with
            reflection on each side. Default: 0.05.
    
    Returns:
        torch.Tensor: Batch of crops, shape (nb_crops, C, input_size, input_size).
    
    Raises:
        AssertionError: If image is not 3-dimensional.
    
    Note:
        Crop centers are sampled from N(0.5, jitter_std) in normalized
        coordinates, so most crops are near the image center.
    """
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
    """
    Add Gaussian noise to an image.
    
    Used during optimization to improve robustness and prevent
    overfitting to high-frequency artifacts.
    
    Args:
        image (torch.Tensor): Input image, any shape.
        noise_level (float): Standard deviation of Gaussian noise to add.
    
    Returns:
        torch.Tensor: Noisy image with same shape as input.
    
    Note:
        Creates a clone of the input, so the original tensor is not modified.
    """
    noisy = image.clone()
    noisy.add_(torch.randn_like(noisy) * noise_level)
    return noisy


def change_norm(image, target_norm):
    """
    Rescale image(s) to have a specific L2 norm.
    
    Useful for controlling the overall magnitude of synthesized images
    to match the statistics of training data.
    
    Args:
        image (torch.Tensor): Input image, shape (C, H, W) or (N, C, H, W).
        target_norm (float or None): Desired L2 norm. If None, returns
            the image unchanged.
    
    Returns:
        torch.Tensor: Rescaled image with L2 norm ≈ target_norm.
    
    Note:
        For batched inputs (4D), each image in the batch is independently
        rescaled to have the target norm.
    """
    if target_norm is None:
        return image
    eps = 1e-8
    if image.ndim == 3:
        current = torch.norm(image.reshape(-1)) + eps
        return image * (target_norm / current)
    else:
        norms = torch.norm(image.reshape(image.shape[0], -1), dim=1, keepdim=True) + eps
        return image * (target_norm / norms.view(-1, 1, 1, 1))
