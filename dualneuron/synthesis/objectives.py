import torch
import torch.nn.functional as F


def response_objective(
    model, 
    target, 
    mode='project',
    sign=1,
    loss_type='mse',
    model_kwargs=None
):
    """
    Create an objective function for image synthesis based on population response.
    
    Args:
        model: Neural network that maps images (N, C, H, W) to responses (N, num_neurons)
        target: Target in response space:
            - int: neuron index for single-neuron MEI/LEI
            - Tensor (num_neurons,): response vector or direction
        mode: How to use the target:
            - 'project': Maximize projection onto target direction (for MEI, LEI, PC synthesis)
            - 'reconstruct': Minimize distance to target point (for reconstruction)
        sign: Direction of optimization:
            - For 'project': 1 for positive direction, -1 for negative
            - For 'reconstruct' with 'cosine'/'correlation': 1 for similar, -1 for anti-similar
            - For 'reconstruct' with 'mse': ignored (distance has no sign)
        loss_type: For 'reconstruct' mode only. One of:
            - 'mse': Mean squared error (euclidean distance)
            - 'cosine': Cosine similarity (direction only, ignores magnitude)
            - 'correlation': Pearson correlation (centered cosine)
        model_kwargs: Optional dict of kwargs to pass to model (e.g., {'multiplex': True})
    
    Returns:
        Callable that takes (N, C, H, W) images and returns scalar to maximize
    """
    
    model_kwargs = model_kwargs or {}
    
    # Single neuron case
    if isinstance(target, int):
        def objective(images):
            return sign * model(images, **model_kwargs)[:, target].mean()
        return objective
    
    # Vector target - ensure 1D and detached
    t = target.detach().clone().flatten()
    
    if mode == 'project':
        direction = t / (t.norm() + 1e-8)
        
        def objective(images):
            pred = model(images, **model_kwargs).mean(dim=0).flatten()
            d = direction.to(pred.device)
            return sign * torch.dot(pred, d)
        
    elif mode == 'reconstruct':
        if loss_type == 'mse':
            def objective(images):
                pred = model(images, **model_kwargs).mean(dim=0).flatten()
                tgt = t.to(pred.device)
                return -F.mse_loss(pred, tgt)
            
        elif loss_type == 'cosine':
            t_norm = t.norm() + 1e-8
            
            def objective(images):
                pred = model(images, **model_kwargs).mean(dim=0).flatten()
                tgt = t.to(pred.device)
                return sign * torch.dot(pred, tgt) / (pred.norm() * t_norm + 1e-8)
            
        elif loss_type == 'correlation':
            t_centered = t - t.mean()
            t_centered_norm = t_centered.norm() + 1e-8
            
            def objective(images):
                pred = model(images, **model_kwargs).mean(dim=0).flatten()
                tc = t_centered.to(pred.device)
                pred_centered = pred - pred.mean()
                return sign * torch.dot(pred_centered, tc) / (pred_centered.norm() * t_centered_norm + 1e-8)
        else:
            raise ValueError(f"Unknown loss_type: {loss_type}")
    else:
        raise ValueError(f"Unknown mode: {mode}")
    
    return objective