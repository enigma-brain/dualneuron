import torch
import numpy as np


def semantic_axis(images1, images2, dreamsim_model, device='cuda'):
    """
    Compute semantic axis as the difference between centroids of two image sets.
    
    Args:
        images1: Tensor (N, C, H, W) or list of tensors - first image set (e.g., MAIs)
        images2: Tensor (M, C, H, W) or list of tensors - second image set (e.g., LAIs)
        dreamsim_model: DreamSim model
        device: Device to use
    
    Returns:
        axis: Unit vector pointing from centroid of images2 to centroid of images1
    """
    
    def embed_set(images):
        """Embed a set of images and return mean embedding."""
        if not isinstance(images, (list, tuple)):
            # Single tensor with batch dimension
            images = [images[i] for i in range(len(images))]
        
        embeddings = []
        with torch.no_grad():
            for img in images:
                if img.dim() == 3:
                    img = img.unsqueeze(0)
                img = img.to(device)
                emb = dreamsim_model.embed(img).flatten().cpu()
                embeddings.append(emb)
        
        return torch.stack(embeddings).mean(dim=0)
    
    # Compute centroids
    centroid1 = embed_set(images1)
    centroid2 = embed_set(images2)
    
    # Axis from centroid2 → centroid1
    axis = centroid1 - centroid2
    
    # Normalize to unit vector
    axis = axis / (axis.norm() + 1e-8)
    
    return axis.numpy()