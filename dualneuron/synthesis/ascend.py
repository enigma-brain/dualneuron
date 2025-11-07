import os
import torch
import torch.nn.functional as F
from torchvision.transforms.functional import gaussian_blur
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
import numpy as np

from dualneuron.synthesis.ops import (
    recorrelate_colors,
    create_crops,
    add_noise,
    change_norm
)


def buffer(path, init_image=None, target_size=None):
    """Load precomputed magnitude spectrum"""
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
    """Compute total variation loss for regularization"""
    dx = image[:, :, 1:] - image[:, :, :-1]
    dy = image[:, 1:, :] - image[:, :-1, :]
    tv_loss = torch.sqrt(dx.pow(2).sum() + dy.pow(2).sum() + 1e-8)
    return weight * tv_loss


def optimization_step(
    objective_function, 
    image, 
    box_size, 
    noise_level, 
    nb_crops, 
    input_size, 
    target_norm,
    tv_weight=1e-4,
    jitter_std=0.03, 
    oversample=2, 
    reflect_pad_frac=0.05
):
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

    score = objective_function(processed)
    tv_loss = total_variation_loss(image, weight=tv_weight)
    loss = -score + tv_loss
    return loss, image


def fourier_ascending(
    objective_function,
    magnitude_path,
    image_size=None,
    init_image=None,
    total_steps=128,
    learning_rate=1.0,
    lr_schedule=True,
    noise=0.01,
    values_range=(-2., 2.),
    range_fn='linear',
    nb_crops=6,
    box_size=(0.20, 0.25),
    target_norm=None,
    tv_weight=1e-4,
    jitter_std=0.1,
    oversample=1, 
    reflect_pad_frac=0.02,
    device='cuda',
    verbose=False,
    save_all_steps=False
):
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
            T_max=total_steps
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
            init_act = objective_function(init_image.unsqueeze(0)).item()
            images.append(init_image.detach().cpu())
            activations.append(abs(init_act))
            
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
            objective_function, img, box_size, noise,
            nb_crops, image_size, target_norm, tv_weight,
            jitter_std, oversample, reflect_pad_frac
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
            
            act = objective_function(clean_img.unsqueeze(0)).item()
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
    total_steps=128,
    learning_rate=0.05,
    lr_schedule=True,
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
    device='cuda',
    verbose=False
):
    assert values_range[1] >= values_range[0]
    l, h = values_range

    image_param = torch.randn(
        channels, image_size, 
        image_size, device=device
    ) * init_std
    
    image_param.requires_grad = True
    
    optimizer = torch.optim.SGD(
        [image_param], 
        lr=learning_rate, 
        momentum=0.9
    )
    
    # optimizer = torch.optim.Adam(
    #     [image_param], 
    #     lr=learning_rate,
    #     betas=(0.9, 0.999), 
    #     eps=1e-8
    # )
    
    if lr_schedule:
        scheduler = CosineAnnealingLR(
            optimizer, 
            T_max=total_steps
        )
    else:
        scheduler = None
    
    transparency = torch.zeros((3, image_size, image_size)).to(device)
    activations = []

    if verbose:
        pbar = tqdm(range(total_steps))
    else:
        pbar = range(total_steps)

    for _ in pbar:
        optimizer.zero_grad()
        img = torch.sigmoid(image_param) * (h - l) + l
        
        loss, img = optimization_step(
            objective_function, img, box_size, noise,
            nb_crops, image_size, target_norm, tv_weight,
            jitter_std, oversample, reflect_pad_frac
        )

        loss.backward()
        with torch.no_grad():
            image_param.grad = gaussian_blur(
                image_param.grad.unsqueeze(0), 
                kernel_size=5, 
                sigma=1.5
            ).squeeze(0)

        transparency += torch.abs(img.grad)
        optimizer.step()
        
        if scheduler is not None:
            scheduler.step()

        with torch.no_grad():
            clean_img = torch.sigmoid(image_param) * (h - l) + l
            
            if target_norm is not None:
                clean_img = change_norm(clean_img, target_norm)
            
            act = objective_function(clean_img.unsqueeze(0)).item()
            activations.append(abs(act))

        if verbose:
            pbar.set_description(f"Activation: {abs(act):.4f}")

    transparency = transparency / (transparency.max() + 1e-8)
    if channels == 1: transparency = transparency.mean(dim=0, keepdim=True)
    return clean_img, transparency, activations