import numpy as np
import torch
import torch.nn.functional as F
from torchvision.ops import roi_align


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


def decorrelate_colors(image, device):
    """
    Transform RGB image to decorrelated color space.
    
    Inverse of recorrelate_colors. Transforms an image from RGB to a
    decorrelated color space where channels are statistically independent.
    Useful for parameterizing optimization in a better-conditioned space.
    
    Args:
        image (torch.Tensor): Input RGB image, shape (3, H, W).
        device (str or torch.device): Device for the correlation matrix.
    
    Returns:
        torch.Tensor: Decorrelated image, shape (3, H, W).
    
    Raises:
        AssertionError: If image is not 3-dimensional.
    """
    assert len(image.shape) == 3
    correlation = color_correlation(device)
    decorrelation = torch.linalg.inv(correlation)
    permuted_image = image.permute(1, 2, 0).contiguous()
    flat_image = permuted_image.view(-1, 3)
    decorrelated = torch.matmul(flat_image, decorrelation)
    decorrelated = decorrelated.view(permuted_image.shape).permute(2, 0, 1)
    return decorrelated


def recorrelate_colors(image, device):
    """
    Transform decorrelated image to RGB color space.
    
    Inverse of decorrelate_colors. Transforms an image from a decorrelated
    color space back to RGB with correlations matching natural images.
    Used in image synthesis to produce more realistic colors.
    
    Args:
        image (torch.Tensor): Input decorrelated image, shape (3, H, W).
        device (str or torch.device): Device for the correlation matrix.
    
    Returns:
        torch.Tensor: RGB image, shape (3, H, W).
    
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


def create_neural_crops(
    image, mu, nb_crops, box_size, input_size,
    jitter_std=0.05,
    oversample=1,
    reflect_pad_frac=0.05,
):
    """
    Receptive-field-aware crops for activation maximization.

    Identical to create_crops, except the crops are centered on the neuron's
    receptive-field center (the Gaussian readout center, mu) instead of the image
    center. box_size and jitter_std are absolute fractions of the image, exactly as
    in create_crops; only the anchor point moves. For a centered readout (mu = 0)
    this reduces to create_crops.

    The readout center is the only per-neuron quantity used: the readout sigma is a
    near-point pooling spread on the feature map, not the image-space receptive field
    size (which is set by the fixed convolutional receptive field of the core), so a
    single absolute box_size applies to all neurons.

    Args:
        image (torch.Tensor): Input image, shape (C, H, W).
        mu (sequence or torch.Tensor): RF center in normalized grid coordinates
            [-1, 1], order (x, y) = (width, height), as stored in FullGaussian2d.mu.
        nb_crops (int): Number of crops to generate.
        box_size (float or tuple): Crop size as a fraction of the image. A scalar
            gives a fixed size; a (min, max) tuple samples a per-crop scale.
        input_size (int): Output size for each crop (square).
        jitter_std (float): Std of crop-center jitter, as a fraction of the image.
            Default: 0.05.
        oversample (int): Oversampling factor before downscaling. Default: 1.
        reflect_pad_frac (float): Fraction of image size to reflect-pad. Default: 0.05.

    Returns:
        tuple: (crops, boxes)
            - crops (torch.Tensor): shape (nb_crops, C, input_size, input_size).
            - boxes (torch.Tensor): shape (nb_crops, 4), crop boxes (x1, y1, x2, y2)
              in original-image pixel coordinates, for visualization.
    """
    assert image.ndim == 3
    device = image.device
    C, H, W = image.shape

    # RF center: normalized [-1, 1] (x, y) -> fractional [0, 1].
    mu = torch.as_tensor(mu, dtype=torch.float32, device=device).flatten()
    cx0 = (float(mu[0]) + 1.0) * 0.5
    cy0 = (float(mu[1]) + 1.0) * 0.5

    if isinstance(box_size, (tuple, list)):
        s, b = box_size
    else:
        s = b = box_size

    pad = int(reflect_pad_frac * min(H, W))
    if pad > 0:
        img_pad = F.pad(
            image.unsqueeze(0),
            (pad, pad, pad, pad),
            mode="reflect"
        ).squeeze(0)
        Hpad, Wpad = H + 2 * pad, W + 2 * pad
        x_offset = pad
        y_offset = pad
    else:
        img_pad = image
        Hpad, Wpad = H, W
        x_offset = 0
        y_offset = 0

    # Crop centers: RF center (mu) + jitter; crop scale; all as image fractions.
    cx = cx0 + torch.randn(nb_crops, device=device) * jitter_std
    cy = cy0 + torch.randn(nb_crops, device=device) * jitter_std
    sc = torch.rand(nb_crops, device=device) * (b - s) + s
    bw, bh = sc * W, sc * H

    x1 = (cx * W + x_offset - 0.5 * bw).clamp(0, Wpad)
    y1 = (cy * H + y_offset - 0.5 * bh).clamp(0, Hpad)
    x2 = (cx * W + x_offset + 0.5 * bw).clamp(0, Wpad)
    y2 = (cy * H + y_offset + 0.5 * bh).clamp(0, Hpad)
    batch = torch.zeros_like(x1)
    boxes_roi = torch.stack([batch, x1, y1, x2, y2], dim=1).to(torch.float32)

    hi = input_size * max(1, int(oversample))
    crops_hi = roi_align(
        img_pad.unsqueeze(0),
        boxes_roi,
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

    # Boxes in original-image coordinates (undo padding offset) for visualization.
    boxes = torch.stack([x1, y1, x2, y2], dim=1) - x_offset
    return crops, boxes


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


def _norm_in_range(flat, target_norm, l, h, iters=60):
    """
    Scale-and-clip rows of ``flat`` so each has L2 norm ``target_norm`` AND lies in ``[l, h]``.

    Solves, per row, for the single scalar ``a`` with ``|| clamp(a * x, l, h) ||_2 = target_norm``.
    That ``a`` exists and is found reliably: define ``f(a) = || clamp(a*x, l, h) ||_2``. Each element's
    clipped magnitude is non-decreasing in ``a`` and saturates, so ``f`` is continuous and monotone
    non-decreasing with ``f(0) = 0`` and ``f(inf) = || saturate(x) ||_2``. The intermediate value
    theorem gives existence for any target in that span, and monotonicity makes bisection converge.

    **Rows the box does not bind take a closed-form path** (``a = target_norm / ||x||``, the plain
    rescale) and are returned from it. That is not only cheaper: the closed form is differentiable
    *through the norm*, so those rows keep the exact value **and gradient** the unconstrained rescale
    would have produced. Only binding rows go through the bisection, where ``a`` is a constant and
    gradient reaches the caller through the scale-and-clip (zero on saturated elements, which cannot
    move anyway). So adding a range changes the optimization only where the range actually binds.

    Args:
        flat (torch.Tensor): ``(N, D)`` rows to project.
        target_norm (float): Desired L2 norm per row.
        l, h (float): Inclusive value bounds.
        iters (int): Bisection steps. Default: 60 (far beyond float32 precision).

    Returns:
        torch.Tensor: ``(N, D)`` rows, clipped to ``[l, h]`` with L2 norm ``target_norm``.

    Raises:
        ValueError: If a row cannot reach ``target_norm`` inside the box -- i.e. the target exceeds
            ``|| saturate(x) ||_2``, the most energy that row can hold without leaving the range.
    """
    eps = 1e-8
    # Closed form first: the plain rescale, which is the solution whenever it already lands in the
    # box. Keeping it as an explicit branch matters for more than speed -- it is differentiable
    # *through the norm*, so rows the box does not bind keep exactly the gradient (and the value) the
    # unconstrained rescale would have given. The constraint then changes the optimization only when
    # it genuinely binds, instead of silently altering every step.
    alpha0 = target_norm / (torch.linalg.vector_norm(flat, dim=1, keepdim=True) + eps)
    plain = flat * alpha0
    with torch.no_grad():
        binds = ((plain < l) | (plain > h)).any(dim=1, keepdim=True)
    if not bool(binds.any()):
        return plain

    # Binding rows: solve || clamp(a * x, l, h) ||_2 = target_norm for `a`. The root-find is not part
    # of the model, so it runs under no_grad and `a` is a constant; gradient reaches the caller
    # through the scale-and-clip below (and is zero on saturated elements, which genuinely cannot move).
    with torch.no_grad():
        # Most energy each row can hold in-range: every element pushed to its own bound.
        sat = torch.zeros_like(flat)
        sat[flat > 0] = h
        sat[flat < 0] = l
        reach = torch.linalg.vector_norm(sat, dim=1, keepdim=True)
        if bool((binds & (reach < target_norm)).any()):
            worst = float(reach[binds].min())
            raise ValueError(
                f"target_norm={target_norm} is unreachable inside values_range=({l}, {h}): the most "
                f"a row can hold in-range is {worst:.2f}. Lower the norm or widen the range.")

        def f(a):
            return torch.linalg.vector_norm((flat * a).clamp(l, h), dim=1, keepdim=True)

        lo = torch.zeros_like(alpha0)
        hi = torch.ones_like(alpha0)
        for _ in range(iters):                  # grow hi until it brackets the target
            short = f(hi) < target_norm
            if not bool(short.any()):
                break
            hi = torch.where(short, hi * 2.0, hi)
        for _ in range(iters):                  # bisect
            mid = 0.5 * (lo + hi)
            over = f(mid) > target_norm
            hi = torch.where(over, mid, hi)
            lo = torch.where(over, lo, mid)
        alpha = 0.5 * (lo + hi)

    return torch.where(binds, (flat * alpha).clamp(l, h), plain)


def change_norm(image, target_norm, values_range=None):
    """
    Rescale image(s) to have a specific L2 norm.

    Useful for controlling the overall magnitude of synthesized images
    to match the statistics of training data.

    Args:
        image (torch.Tensor): Input image, shape (C, H, W) or (N, C, H, W).
        target_norm (float or None): Desired L2 norm. If None, returns
            the image unchanged.
        values_range (tuple, optional): ``(min, max)`` the result must also respect. A plain rescale
            is a scalar multiply, so it silently voids any range constraint applied beforehand (e.g.
            by :func:`~dualneuron.synthesis.ascend.precondition`); pass the range here and the norm
            and the bounds are satisfied together instead of one overwriting the other. ``None``
            (default) keeps the original scale-only behaviour.

    Returns:
        torch.Tensor: Rescaled image with L2 norm ≈ target_norm, and within ``values_range``
            when one is given.

    Note:
        For batched inputs (4D), each image in the batch is independently
        rescaled to have the target norm.
    """
    if target_norm is None:
        return image
    if values_range is not None:
        l, h = values_range
        flat = image.reshape(1, -1) if image.ndim == 3 else image.reshape(image.shape[0], -1)
        return _norm_in_range(flat, float(target_norm), float(l), float(h)).reshape(image.shape)
    eps = 1e-8
    if image.ndim == 3:
        current = torch.norm(image.reshape(-1)) + eps
        return image * (target_norm / current)
    else:
        norms = torch.norm(image.reshape(image.shape[0], -1), dim=1, keepdim=True) + eps
        return image * (target_norm / norms.view(-1, 1, 1, 1))
    
    
def get_blur_params(step, total_steps, schedule='cosine', sigma_max=2.5, sigma_min=0.5):
    """
    Compute blur parameters that decrease over optimization.
    
    Early steps use strong smoothing to establish global structure.
    Later steps use weak smoothing to allow fine details.
    
    Args:
        step (int): Current optimization step.
        total_steps (int): Total number of steps.
        schedule (str): How sigma decreases ('linear', 'cosine', 'step').
        sigma_max (float): Initial blur strength.
        sigma_min (float): Final blur strength.
    
    Returns:
        tuple: (kernel_size, sigma) for gaussian_blur.
    """
    progress = step / total_steps
    
    if schedule is None:
        return None, None
    if schedule == 'linear':
        sigma = sigma_max + (sigma_min - sigma_max) * progress
    elif schedule == 'cosine':
        sigma = sigma_min + (sigma_max - sigma_min) * (1 + np.cos(np.pi * progress)) / 2
    elif schedule == 'step':
        if progress < 0.33:
            sigma = sigma_max
        elif progress < 0.66:
            sigma = (sigma_max + sigma_min) / 2
        else:
            sigma = sigma_min
    else:
        raise ValueError(f'Unknown schedule: {schedule}')
    
    kernel_size = int(sigma * 3) // 2 * 2 + 1
    kernel_size = max(3, kernel_size)
    
    return kernel_size, sigma


def create_foveated_crops(
    image, nb_crops, box_size, input_size,
    jitter_std=0.03,
    oversample=2,
    reflect_pad_frac=0.05,
    pyramid_levels=4,
    foveation_strength=0.0,
    foveation_levels=(2, 4),
):
    """
    Create multi-scale crops using a band-limited image pyramid, with optional foveation.

    This is a drop-in alternative to create_crops: it returns a batch
    (N, C, input_size, input_size) suitable for robust activation maximization.

    Compared to create_crops, this transform:
      - Crops from an antialiased pyramid level chosen per crop scale (true scale-space).
      - Optionally applies a simple foveated blur (sharper center, blurrier periphery).

    Args:
        image (torch.Tensor): Input image, shape (C, H, W).
        nb_crops (int): Number of crops to generate.
        box_size (tuple): (min_scale, max_scale) as fractions of image size.
        input_size (int): Output size for each crop (square).
        jitter_std (float): Std for crop center jitter (normalized coords). Default: 0.03.
        oversample (int): Oversample factor before downscaling. Default: 2.
        reflect_pad_frac (float): Reflection pad fraction. Default: 0.05.
        pyramid_levels (int): Number of pyramid levels (>=1). Default: 4.
        foveation_strength (float): 0 disables foveation. Larger => more peripheral blur.
            Default: 0.0.
        foveation_levels (tuple): Downsample factors for blur mixture, e.g. (2,4).
            Default: (2, 4).

    Returns:
        torch.Tensor: Batch of crops, shape (nb_crops, C, input_size, input_size).
    """
    assert image.ndim == 3
    device = image.device
    C, H, W = image.shape
    s, b = box_size

    # --- reflect pad (same as create_crops) ---
    pad = int(reflect_pad_frac * min(H, W))
    if pad > 0:
        img_pad = F.pad(
            image.unsqueeze(0),
            (pad, pad, pad, pad),
            mode="reflect"
        ).squeeze(0)
        Hpad, Wpad = H + 2 * pad, W + 2 * pad
        x_offset = pad
        y_offset = pad
    else:
        img_pad = image
        Hpad, Wpad = H, W
        x_offset = 0
        y_offset = 0

    # --- build antialiased pyramid (band-limited scale-space) ---
    pyramid_levels = max(1, int(pyramid_levels))
    pyr = [img_pad]
    for _ in range(1, pyramid_levels):
        prev = pyr[-1].unsqueeze(0)  # (1, C, H, W)
        down = F.interpolate(
            prev,
            scale_factor=0.5,
            mode="bicubic",
            align_corners=False,
            antialias=True
        ).squeeze(0)
        pyr.append(down)

    # --- sample crop params in original (padded) coordinates ---
    cx = 0.5 + torch.randn(nb_crops, device=device) * jitter_std
    cy = 0.5 + torch.randn(nb_crops, device=device) * jitter_std
    sc = torch.rand(nb_crops, device=device) * (b - s) + s
    bw, bh = sc * W, sc * H

    x1 = (cx * W + x_offset - 0.5 * bw).clamp(0, Wpad)
    y1 = (cy * H + y_offset - 0.5 * bh).clamp(0, Hpad)
    x2 = (cx * W + x_offset + 0.5 * bw).clamp(0, Wpad)
    y2 = (cy * H + y_offset + 0.5 * bh).clamp(0, Hpad)

    # --- choose pyramid level per crop based on scale vs output resolution ---
    hi = input_size * max(1, int(oversample))
    ratio = (bw / max(1.0, float(hi))).clamp(min=1e-6)  # pixels per output pixel
    lvl = torch.floor(torch.log2(ratio)).to(torch.long)
    lvl = lvl.clamp(0, pyramid_levels - 1)

    # --- crop at each level using roi_align (grouped by level) ---
    crops_hi = torch.empty((nb_crops, C, hi, hi), device=device, dtype=img_pad.dtype)

    for L in range(pyramid_levels):
        idx = (lvl == L).nonzero(as_tuple=False).flatten()
        if idx.numel() == 0:
            continue

        scale = float(2 ** L)
        imgL = pyr[L]

        x1L = (x1[idx] / scale).clamp(0, imgL.shape[-1])
        y1L = (y1[idx] / scale).clamp(0, imgL.shape[-2])
        x2L = (x2[idx] / scale).clamp(0, imgL.shape[-1])
        y2L = (y2[idx] / scale).clamp(0, imgL.shape[-2])

        batch = torch.zeros_like(x1L)
        boxes = torch.stack([batch, x1L, y1L, x2L, y2L], dim=1).to(torch.float32)

        cropsL = roi_align(
            imgL.unsqueeze(0),
            boxes,
            output_size=(hi, hi),
            aligned=True
        )
        crops_hi[idx] = cropsL

    if foveation_strength is not None and float(foveation_strength) > 0.0:
        # radial map in [0, 1]
        ys = torch.linspace(-1.0, 1.0, hi, device=device)
        xs = torch.linspace(-1.0, 1.0, hi, device=device)
        yy, xx = torch.meshgrid(ys, xs, indexing="ij")
        r = torch.sqrt(xx ** 2 + yy ** 2) / (2 ** 0.5)
        t = (r * float(foveation_strength)).clamp(0.0, 1.0)

        w0 = (1.0 - t).view(1, 1, hi, hi)
        w1 = (t * (1.0 - t)).view(1, 1, hi, hi)
        w2 = (t ** 2).view(1, 1, hi, hi)

        base = crops_hi
        blur_1 = base
        blur_2 = base

        # first blur level (e.g., /2 then upsample)
        d1 = max(1, int(hi // max(1, int(foveation_levels[0]))))
        if d1 < hi:
            tmp = F.interpolate(
                base,
                size=(d1, d1),
                mode="bicubic",
                align_corners=False,
                antialias=True
            )
            blur_1 = F.interpolate(
                tmp,
                size=(hi, hi),
                mode="bicubic",
                align_corners=False,
                antialias=True
            )

        # second blur level (e.g., /4 then upsample)
        if len(foveation_levels) > 1:
            d2 = max(1, int(hi // max(1, int(foveation_levels[1]))))
            if d2 < hi:
                tmp = F.interpolate(
                    base,
                    size=(d2, d2),
                    mode="bicubic",
                    align_corners=False,
                    antialias=True
                )
                blur_2 = F.interpolate(
                    tmp,
                    size=(hi, hi),
                    mode="bicubic",
                    align_corners=False,
                    antialias=True
                )

        crops_hi = w0 * base + w1 * blur_1 + w2 * blur_2

    crops = F.interpolate(
        crops_hi,
        size=(input_size, input_size),
        mode="bicubic",
        align_corners=False,
        antialias=True
    )
    return crops

