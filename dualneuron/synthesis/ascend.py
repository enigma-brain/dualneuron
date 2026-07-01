import os
import torch
import torch.nn.functional as F
from torchvision.transforms.functional import gaussian_blur
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
import numpy as np

from dualneuron.synthesis.ops import (
    recorrelate_colors,
    decorrelate_colors,
    create_crops,
    add_noise,
    change_norm,
    get_blur_params
)


def buffer(path, init_image=None, target_size=None):
    """
    Load a precomputed magnitude spectrum and initialize phase.
    
    Loads a Fourier magnitude spectrum from a .npy file and optionally resizes it.
    Phase can be initialized from an existing image or randomly.
    
    Args:
        path (str): Filename of the magnitude spectrum in the 'priors' directory.
            Expected files: 'natural_gray.npy' or 'natural_rgb.npy'.
        init_image (np.ndarray or torch.Tensor, optional): Image to extract initial
            phase from. If None, phase is initialized randomly. Shape: (C, H, W) or
            (H, W, C).
        target_size (int or tuple, optional): Target spatial size for the magnitude
            spectrum. If int, used for both height and width.
    
    Returns:
        tuple: (magnitude, phase)
            - magnitude (torch.Tensor): Fourier magnitude spectrum, shape (C, H, W).
            - phase (torch.Tensor): Phase angles in radians [-π, π], shape (C, H, W).
    """
    base_dir = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(base_dir, 'priors', path)
    magnitude = torch.tensor(np.load(path), dtype=torch.float32)
    
    if target_size is not None:
        if isinstance(target_size, int):
            target_size = (target_size, target_size)
        
        current_size = magnitude.shape[-2:]
        
        if current_size != target_size:
            magnitude = magnitude.unsqueeze(0)
            magnitude = F.interpolate(
                magnitude,
                size=target_size,
                mode='bilinear',
                align_corners=False
            )
            magnitude = magnitude.squeeze(0)
            
    if init_image is not None:
        if not torch.is_tensor(init_image):
            init_image = torch.tensor(init_image, dtype=torch.float32)
        
        if init_image.ndim == 3 and init_image.shape[0] not in [1, 3]:
            init_image = init_image.permute(2, 0, 1)  # HWC -> CHW
        
        expected_size = magnitude.shape[-2:]
        if init_image.shape[-2:] != expected_size:
            init_image = F.interpolate(
                init_image.unsqueeze(0),
                size=expected_size,
                mode='bilinear',
                align_corners=False
            ).squeeze(0)
        
        spectrum = torch.fft.fft2(init_image)
        phase = torch.angle(spectrum)
    else:
        phase = torch.rand_like(magnitude) * 2 * np.pi - np.pi
            
    return magnitude, phase


def precondition(magnitude, phase, values_range, range_fn, device):
    """
    Convert Fourier magnitude and phase to a pixel-space image.
    
    Reconstructs an image from its Fourier representation and applies
    color correlation and range normalization.
    
    Args:
        magnitude (torch.Tensor): Fourier magnitude spectrum, shape (C, H, W).
        phase (torch.Tensor): Phase angles in radians, shape (C, H, W).
        values_range (tuple): (min, max) output pixel value range.
        range_fn (str): Function to constrain values to range. One of:
            - 'linear': Hard clipping with torch.clamp
            - 'sigmoid': Smooth mapping via sigmoid
            - 'tanh': Smooth mapping via tanh
        device (str or torch.device): Device for color correlation matrix.
    
    Returns:
        torch.Tensor: Reconstructed image, shape (C, H, W).
    
    Raises:
        ValueError: If range_fn is not 'linear', 'sigmoid', or 'tanh'.
        AssertionError: If values_range[1] < values_range[0].
    """
    l, h = values_range
    assert h >= l
    spectrum = torch.complex(
        torch.cos(phase) * magnitude,
        torch.sin(phase) * magnitude
    )
    # spectrum = magnitude * torch.exp(1j * phase)
    image = torch.fft.ifft2(spectrum).real
    if magnitude.shape[0] == 3: 
        image = recorrelate_colors(image, device)
    if range_fn == 'linear':
        image = torch.clamp(image, l, h)
    elif range_fn == 'sigmoid':
        image = torch.sigmoid(image) * (h - l) + l
    elif range_fn == 'tanh':
        image = torch.tanh(image) * (h - l) / 2 + (h + l) / 2
    else:
        raise ValueError(f'Unknown range function {range_fn}')
    return image


def total_variation_loss(image, weight=1e-4):
    """
    Compute total variation loss for image smoothness regularization.
    
    Penalizes high-frequency noise by computing the sum of absolute differences
    between neighboring pixels.
    
    Args:
        image (torch.Tensor): Image tensor, shape (C, H, W).
        weight (float): Scaling factor for the loss. Default: 1e-4.
    
    Returns:
        torch.Tensor: Scalar TV loss value.
    """
    dx = image[:, :, 1:] - image[:, :, :-1]
    dy = image[:, 1:, :] - image[:, :-1, :]
    tv_loss = torch.sqrt(dx.pow(2).sum() + dy.pow(2).sum() + 1e-8)
    return weight * tv_loss


def _activation(objective_function, population_function, target_index, pole, x):
    """Scalar activation of the target neuron: ``objective_function(x)`` when given, else the
    (pole-signed) target response read from the full population — used for the init/final logging so
    the population path needs no separate objective."""
    if objective_function is not None:
        return objective_function(x)
    return pole * population_function(x)[:, target_index].mean()


def optimization_step(
    objective_function,
    simulation_function,
    simulation_axis,
    simulation_weight,
    image,
    box_size,
    noise_level,
    nb_crops,
    input_size,
    target_norm,
    tv_weight=1e-4,
    jitter_std=0.02,
    oversample=1,
    reflect_pad_frac=0.02,
    population_function=None,
    target_index=None,
    pole=1.0,
    population_axis=None,
    population_mean=None,
    population_std=None,
    population_support=None,
    axis_only=False,
):
    """
    Perform one optimization step for activation maximization.
    
    Applies data augmentation (crops, noise, normalization), computes the
    objective activation, and optionally adds a simulation-based regularization
    term to guide synthesis toward a semantic axis.
    
    Args:
        objective_function (callable): Function that takes (N, C, H, W) tensor
            and returns scalar activation to maximize.
        simulation_function (callable, optional): Feature extractor for semantic
            guidance. Takes (N, C, H, W) and returns embeddings.
        simulation_axis (torch.Tensor, optional): Direction in embedding space
            to project onto for semantic guidance.
        simulation_weight (float): Weight for the simulation loss term.
        image (torch.Tensor): Input image, shape (C, H, W).
        box_size (tuple): (min_frac, max_frac) for random crop sizes.
        noise_level (float): Standard deviation of Gaussian noise to add.
        nb_crops (int): Number of random crops to average over.
        input_size (int): Target size for crops (passed to create_crops).
        target_norm (float, optional): Target L2 norm for images.
        tv_weight (float): Weight for total variation loss. Default: 1e-4.
        jitter_std (float): Standard deviation for color jittering. Default: 0.02.
        oversample (int): Oversampling factor for crops. Default: 1.
        reflect_pad_frac (float): Fraction of image to pad with reflection. Default: 0.02.
    
    Returns:
        tuple: (loss, image)
            - loss (torch.Tensor): Scalar loss to minimize (negative activation + regularization).
            - image (torch.Tensor): Input image with gradients retained.
    
    Note:
        The function calls image.retain_grad() to allow gradient inspection
        for transparency/saliency maps.
    """
    assert image.ndim == 3
    image.retain_grad()

    processed = create_crops(
        image, nb_crops, box_size, input_size,
        jitter_std=jitter_std, 
        oversample=oversample, 
        reflect_pad_frac=reflect_pad_frac
    )
    
    processed = add_noise(processed, noise_level)
    if target_norm is not None:
        processed = change_norm(processed, target_norm)

    # One twin forward: read the target activation from the full population when in population mode
    # (so the axis term below is free), else the plain scalar objective.
    if population_function is not None:
        pop = population_function(processed)                 # (nb_crops, N) full population
        activation = pole * pop[:, target_index].mean()
    else:
        activation = objective_function(processed)
    tv_loss = total_variation_loss(image, weight=tv_weight)
    # axis_only "folds" the drive into the axis: the population-axis cosine below IS the whole
    # objective (use a target-INCLUDED axis), so there is no separate activation term.
    loss = tv_loss if axis_only else (-activation + tv_loss)

    # Population-axis (natural-manifold) term: align the z-scored population response with the neuron's
    # MAI-LAI centroid axis by cosine (+axis for the MEI, -axis for the LEI, via `pole`). The axis lives
    # in the well-predicted subspace, so we select those columns (`population_support`). Whether the
    # target's own component participates is decided by the axis vector itself (kept for the folded
    # a_full, zeroed for a context-only axis). Uses the population forward already done above.
    if population_axis is not None:
        z = (pop[:, population_support].mean(0) - population_mean) / population_std
        z = z / (z.norm() + 1e-8)
        loss = loss - pole * torch.dot(z, population_axis)

    if simulation_function is not None and simulation_axis is not None:
        img = (image - image.min()) / (image.max() - image.min() + 1e-8)
        img = F.interpolate(
            img.unsqueeze(0), 
            size=(224, 224), 
            mode='bilinear', 
            align_corners=False
        )
        
        if img.shape[1] == 1:
            img = img.repeat(1, 3, 1, 1)

        embedding = simulation_function(img).flatten()
        projection = torch.dot(embedding, simulation_axis)
        axis_loss = -simulation_weight * projection
        loss = loss + axis_loss
        
    return loss, image


def fourier_ascending(
    objective_function,
    magnitude_path,
    image_size=None,
    init_image=None,
    total_steps=128,
    learning_rate=1.0,
    lr_schedule=True,
    eta_min=0.0,
    noise=0.01,
    values_range=(-2., 2.),
    range_fn='linear',
    nb_crops=6,
    box_size=(0.20, 0.25),
    target_norm=None,
    tv_weight=1e-4,
    jitter_std=0.02,
    oversample=1, 
    reflect_pad_frac=0.02,
    simulation_function=None,
    simulation_axis=None,
    simulation_weight=0.0,
    population_function=None,
    target_index=None,
    pole=1.0,
    population_axis=None,
    population_mean=None,
    population_std=None,
    population_support=None,
    axis_only=False,
    device='cuda',
    verbose=False,
    save_all_steps=False
):
    """
    Synthesize an image that maximizes a neural objective using Fourier parameterization.
    
    Optimizes phase angles while keeping magnitude fixed to a natural image prior,
    producing images with naturalistic frequency statistics.
    
    Args:
        objective_function (callable): Function that takes (N, C, H, W) tensor
            and returns scalar activation to maximize.
        magnitude_path (str): Filename of magnitude prior in 'priors' directory
            (e.g., 'natural_gray.npy' or 'natural_rgb.npy').
        image_size (int, optional): Target image size. If None, uses size from
            magnitude prior (100x100 for rgb and 93x93 for gray).
        init_image (np.ndarray or torch.Tensor, optional): Image to initialize
            phase from. If None, phase is random.
        total_steps (int): Number of optimization steps. Default: 128.
        learning_rate (float): Learning rate for Adam optimizer. Default: 1.0.
        lr_schedule (bool): Whether to use cosine annealing LR schedule. Default: True.
        eta_min (float): Minimum learning rate for cosine schedule. Default: 0.0.
        noise (float): Noise level for augmentation. Default: 0.01.
        values_range (tuple): (min, max) pixel value range. Default: (-2., 2.).
        range_fn (str): Range constraint function ('linear', 'sigmoid', 'tanh').
            Default: 'linear'.
        nb_crops (int): Number of crops for robustness. Default: 6.
        box_size (tuple): (min_frac, max_frac) crop size range. Default: (0.20, 0.25).
        target_norm (float, optional): Target L2 norm for images.
        tv_weight (float): Total variation regularization weight. Default: 1e-4.
        jitter_std (float): Standard deviation for center position jitter,
            as a fraction of image size. Default: 0.02.
        oversample (int): Factor to oversample crops before downscaling,
            improves antialiasing quality. Default: 1.
        reflect_pad_frac (float): Fraction of image size to pad with
            reflection on each side. Default: 0.02.
        simulation_function (callable, optional): Feature extractor for semantic guidance.
        simulation_axis (np.ndarray, optional): Embedding direction for semantic guidance.
        simulation_weight (float): Weight for simulation loss. Default: 0.0.
        device (str): Device to run on. Default: 'cuda'.
        verbose (bool): Whether to show progress bar. Default: False.
        save_all_steps (bool): Whether to save images at every step. Default: False.
    
    Returns:
        dict: Results dictionary with keys:
            - 'image': Final image tensor (C, H, W) or list of tensors if save_all_steps.
            - 'alpha': Transparency/saliency map (C, H, W) or list if save_all_steps.
            - 'activation': Final activation value or list of all values if save_all_steps.
    """
    assert values_range[1] >= values_range[0]
    magnitude, phase = buffer(
        magnitude_path, 
        init_image=init_image, 
        target_size=image_size
    )
    channels = magnitude.shape[0]
    image_size = magnitude.shape[1]
    
    magnitude = magnitude.to(device)
    phase = phase.to(device)
    phase.requires_grad = True

    optimizer = torch.optim.Adam(
        [phase], lr=learning_rate, 
        betas=(0.9, 0.999), eps=1e-8
    )

    if lr_schedule:
        scheduler = CosineAnnealingLR(
            optimizer, 
            T_max=total_steps,
            eta_min=eta_min
        )
    else:
        scheduler = None
    
    transparency = torch.zeros((3, image_size, image_size)).to(device)
    activations = []

    if save_all_steps:
        images = []
        alphas = []
    else:
        images = None
        alphas = None
    
    if save_all_steps:
        with torch.no_grad():
            if init_image is None:
                init_image = precondition(
                    magnitude, phase, 
                    values_range, range_fn, 
                    device
                )
                
                if target_norm is not None:
                    init_image = change_norm(init_image, target_norm)

            init_image = init_image.to(device)
            init_act = _activation(objective_function, population_function, target_index, pole, init_image.unsqueeze(0)).item()
            images.append(init_image.detach().cpu())
            activations.append(abs(init_act))
            
    if simulation_axis is not None:
        simulation_axis = torch.tensor(
            simulation_axis,
            device=device,
            dtype=torch.float32
        )

    if population_axis is not None:
        population_axis = torch.as_tensor(population_axis, device=device, dtype=torch.float32)
        population_mean = torch.as_tensor(population_mean, device=device, dtype=torch.float32)
        population_std = torch.as_tensor(population_std, device=device, dtype=torch.float32)
        population_support = torch.as_tensor(population_support, device=device, dtype=torch.long)
            
    if verbose:
        pbar = tqdm(range(total_steps))
    else:
        pbar = range(total_steps)

    for _ in pbar:
        optimizer.zero_grad()
        
        img = precondition(
            magnitude, phase, 
            values_range, range_fn, 
            device
        )

        loss, img = optimization_step(
            objective_function, 
            simulation_function,
            simulation_axis,
            simulation_weight,
            img, box_size, noise,
            nb_crops, image_size, target_norm, tv_weight,
            jitter_std, oversample, reflect_pad_frac,
            population_function=population_function, target_index=target_index, pole=pole,
            population_axis=population_axis,
            population_mean=population_mean, population_std=population_std,
            population_support=population_support, axis_only=axis_only,
        )

        loss.backward()
        transparency += torch.abs(img.grad)
        optimizer.step()
        
        if scheduler is not None:
            scheduler.step()

        with torch.no_grad():
            clean_img = precondition(
                magnitude, phase,
                values_range, range_fn, 
                device
            )

            if target_norm is not None:
                clean_img = change_norm(clean_img, target_norm)
            
            act = _activation(objective_function, population_function, target_index, pole, clean_img.unsqueeze(0)).item()
            activations.append(abs(act))

            if save_all_steps:
                images.append(clean_img.detach().cpu())
                alpha = transparency / (transparency.max() + 1e-8)
                if channels == 1:
                    alpha = alpha.mean(dim=0, keepdim=True)
                
                alphas.append(alpha.detach().cpu())
            
        if verbose:
            pbar.set_description(f"Activation: {abs(act):.4f}")

    transparency = transparency / (transparency.max() + 1e-8)
    if channels == 1: 
        transparency = transparency.mean(dim=0, keepdim=True)
    
    if save_all_steps and len(alphas) > 0:
        alphas = [alphas[0]] + alphas
    
    return {
        'image': images if save_all_steps else clean_img, 
        'alpha': alphas if save_all_steps else transparency, 
        'activation': activations if save_all_steps else activations[-1]
    }


def pixel_ascending(
    objective_function,
    image_size=224,
    channels=3,
    init_image=None,
    total_steps=128,
    optimizer='adam',
    learning_rate=0.05,
    lr_schedule=True,
    eta_min=0.0,
    noise=0.01,
    values_range=(-2.5, 2.5),
    nb_crops=6,
    box_size=(0.20, 0.25),
    target_norm=None,
    tv_weight=1e-4,
    init_std=0.01,
    jitter_std=0.1,
    oversample=1, 
    reflect_pad_frac=0.02,
    blur_schedule='linear',
    sigma_max=2.0,
    sigma_min=0.5,
    simulation_function=None,
    simulation_axis=None,
    simulation_weight=0.0,
    population_function=None,
    target_index=None,
    pole=1.0,
    population_axis=None,
    population_mean=None,
    population_std=None,
    population_support=None,
    axis_only=False,
    device='cuda',
    verbose=False,
    save_all_steps=False
):
    """
    Synthesize an image that maximizes a neural objective using direct pixel optimization.
    
    Optimizes pixel values directly with gradient smoothing via Gaussian blur.
    Simpler than Fourier method but may produce less naturalistic textures.
    
    Args:
        objective_function (callable): Function that takes (N, C, H, W) tensor
            and returns scalar activation to maximize.
        image_size (int): Output image size (square). Default: 224.
        channels (int): Number of image channels (1 for grayscale, 3 for RGB).
            Default: 3.
        init_image (np.ndarray or torch.Tensor, optional): Image to initialize
            from. If None, initializes with random noise. Shape: (C, H, W) or
            (H, W, C).
        total_steps (int): Number of optimization steps. Default: 128.
        optimizer (str): Optimizer to use ('adam' or 'sgd'). Default: 'adam'.
        learning_rate (float): Learning rate for SGD optimizer. Default: 0.05.
        lr_schedule (bool): Whether to use cosine annealing LR schedule. Default: True.
        eta_min (float): Minimum learning rate for cosine schedule. Default: 0.0.
        noise (float): Noise level for augmentation. Default: 0.01.
        values_range (tuple): (min, max) pixel value range. Default: (-2.5, 2.5).
        nb_crops (int): Number of crops for robustness. Default: 6.
        box_size (tuple): (min_frac, max_frac) crop size range. Default: (0.20, 0.25).
        target_norm (float, optional): Target L2 norm for images.
        tv_weight (float): Total variation regularization weight. Default: 1e-4.
        init_std (float): Standard deviation for random initialization. Default: 0.01.
        jitter_std (float): Standard deviation for center position jitter,
            as a fraction of image size. Default: 0.1.
        oversample (int): Factor to oversample crops before downscaling,
            improves antialiasing quality. Default: 1.
        reflect_pad_frac (float): Fraction of image size to pad with
            reflection on each side. Default: 0.02.
        blur_schedule (str): Schedule for gradient blur ('linear', 'cosine', 'step', None).
            Default: 'linear'.
        sigma_max (float): Initial blur sigma (strong smoothing). Default: 2.0.
        sigma_min (float): Final blur sigma (weak smoothing). Default: 0.5.
        simulation_function (callable, optional): Feature extractor for semantic guidance.
        simulation_axis (np.ndarray, optional): Embedding direction for semantic guidance.
        simulation_weight (float): Weight for simulation loss. Default: 0.0.
        device (str): Device to run on. Default: 'cuda'.
        verbose (bool): Whether to show progress bar. Default: False.
        save_all_steps (bool): Whether to save images at every step. Default: False.
    
    Returns:
        dict: Results dictionary with keys:
            - 'image': Final image tensor (C, H, W) or list of tensors if save_all_steps.
            - 'alpha': Transparency/saliency map (C, H, W) or list if save_all_steps.
            - 'activation': Final activation value or list of all values if save_all_steps.
    """
    assert channels in [1, 3], "Channels must be 1 (grayscale) or 3 (RGB)"
    assert optimizer in ['adam', 'sgd'], "Optimizer must be 'adam' or 'sgd'"
    assert values_range[1] >= values_range[0]
    l, h = values_range

    # Initialize image parameter
    if init_image is not None:
        if not torch.is_tensor(init_image):
            init_image = torch.tensor(init_image, dtype=torch.float32)
        
        if init_image.ndim == 3 and init_image.shape[0] not in [1, 3]:
            init_image = init_image.permute(2, 0, 1)  # HWC -> CHW
        
        if init_image.shape[-2:] != (image_size, image_size):
            init_image = F.interpolate(
                init_image.unsqueeze(0),
                size=(image_size, image_size),
                mode='bilinear',
                align_corners=False
            ).squeeze(0)
        
        if channels == 3:
            init_image = decorrelate_colors(init_image, device)
        
        # Inverse sigmoid to get parameter values
        init_image_clipped = torch.clamp(init_image, l + 1e-6, h - 1e-6)
        init_image_normalized = (init_image_clipped - l) / (h - l)
        image_param = torch.logit(init_image_normalized).to(device)
        channels = init_image.shape[0]
    else:
        image_param = torch.randn(
            channels, image_size, 
            image_size, device=device
        ) * init_std
    
    image_param.requires_grad = True
    
    if optimizer == 'sgd':
        optimizer = torch.optim.SGD(
            [image_param], 
            lr=learning_rate, 
            momentum=0.9
        )
    else:
        optimizer = torch.optim.Adam(
            [image_param], 
            lr=learning_rate,
            betas=(0.9, 0.999), 
            eps=1e-8
        )   
    
    if lr_schedule:
        scheduler = CosineAnnealingLR(
            optimizer, 
            T_max=total_steps,
            eta_min=eta_min
        )
    else:
        scheduler = None
    
    transparency = torch.zeros((channels, image_size, image_size)).to(device)
    activations = []

    if save_all_steps:
        images = []
        alphas = []
    else:
        images = None
        alphas = None

    # Save initial state if requested
    if save_all_steps:
        with torch.no_grad():
            init_img = torch.sigmoid(image_param) * (h - l) + l
            if channels == 3:
                init_img = recorrelate_colors(init_img, device)
            if target_norm is not None:
                init_img = change_norm(init_img, target_norm)
            init_act = _activation(objective_function, population_function, target_index, pole, init_img.unsqueeze(0)).item()
            images.append(init_img.detach().cpu())
            activations.append(abs(init_act))

    if simulation_axis is not None:
        simulation_axis = torch.tensor(
            simulation_axis,
            device=device,
            dtype=torch.float32
        )

    if population_axis is not None:
        population_axis = torch.as_tensor(population_axis, device=device, dtype=torch.float32)
        population_mean = torch.as_tensor(population_mean, device=device, dtype=torch.float32)
        population_std = torch.as_tensor(population_std, device=device, dtype=torch.float32)
        population_support = torch.as_tensor(population_support, device=device, dtype=torch.long)

    if verbose:
        pbar = tqdm(range(total_steps))
    else:
        pbar = range(total_steps)

    for step in pbar:
        optimizer.zero_grad()
        img = torch.sigmoid(image_param) * (h - l) + l
    
        if channels == 3:
            img = recorrelate_colors(img, device)
        
        loss, img = optimization_step(
            objective_function,
            simulation_function,
            simulation_axis,
            simulation_weight,
            img, box_size, noise,
            nb_crops, image_size, target_norm, tv_weight,
            jitter_std, oversample, reflect_pad_frac,
            population_function=population_function, target_index=target_index, pole=pole,
            population_axis=population_axis,
            population_mean=population_mean, population_std=population_std,
            population_support=population_support, axis_only=axis_only,
        )

        loss.backward()
        with torch.no_grad():
            k, s = get_blur_params(
                step, total_steps, 
                blur_schedule, sigma_max, 
                sigma_min
            )
            if k is not None and s is not None:
                image_param.grad = gaussian_blur(
                    image_param.grad.unsqueeze(0), 
                    kernel_size=k, 
                    sigma=s
                ).squeeze(0)

        transparency += torch.abs(img.grad)
        optimizer.step()
        
        if scheduler is not None:
            scheduler.step()

        with torch.no_grad():
            clean_img = torch.sigmoid(image_param) * (h - l) + l
            if channels == 3:
                clean_img = recorrelate_colors(clean_img, device)
            
            if target_norm is not None:
                clean_img = change_norm(clean_img, target_norm)
            
            act = _activation(objective_function, population_function, target_index, pole, clean_img.unsqueeze(0)).item()
            activations.append(abs(act))

            if save_all_steps:
                images.append(clean_img.detach().cpu())
                alpha = transparency / (transparency.max() + 1e-8)
                if channels == 1:
                    alpha = alpha.mean(dim=0, keepdim=True)
                alphas.append(alpha.detach().cpu())

        if verbose:
            pbar.set_description(f"Activation: {abs(act):.4f}")

    transparency = transparency / (transparency.max() + 1e-8)
    if channels == 1: 
        transparency = transparency.mean(dim=0, keepdim=True)
    
    if save_all_steps and len(alphas) > 0:
        alphas = [alphas[0]] + alphas
    
    return {
        'image': images if save_all_steps else clean_img, 
        'alpha': alphas if save_all_steps else transparency, 
        'activation': activations if save_all_steps else activations[-1]
    }