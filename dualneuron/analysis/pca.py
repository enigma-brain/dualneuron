import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader


def extract_population_responses(
    model,
    dataset,
    batch_size=32,
    num_images=5000,
    device='cuda',
    seed=42,
    model_kwargs=None
):
    """
    Extract neural responses for a subset of images from a dataset.
    
    Args:
        model: Neural network that maps images to responses
        dataset: Dataset object with __getitem__ returning (image, label)
        batch_size: Batch size for inference
        num_images: Number of images to sample (None for all)
        device: Device for inference
        seed: Random seed for reproducible sampling
    
    Returns:
        responses: np.ndarray of shape (num_images, num_neurons)
        indices: np.ndarray of dataset indices used
    """
    rng = np.random.RandomState(seed)
    model_kwargs = model_kwargs or {}
    
    if num_images is None or num_images >= len(dataset):
        indices = np.arange(len(dataset))
    else:
        indices = rng.choice(len(dataset), size=num_images, replace=False)
        indices = np.sort(indices)
    
    subset = torch.utils.data.Subset(dataset, indices)
    loader = DataLoader(subset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    responses_list = []
    
    with torch.no_grad():
        for images, _ in tqdm(loader, desc="Extracting responses"):
            images = images.to(device)
            resp = model(images, **model_kwargs).cpu().numpy()
            responses_list.append(resp)
    
    responses = np.concatenate(responses_list, axis=0)
    return responses, indices


def fit_pca(responses, n_components=None, center=True, device='cuda'):
    """
    Fit PCA on population responses using GPU-accelerated SVD.
    
    Args:
        responses: np.ndarray or torch.Tensor of shape (num_images, num_neurons)
        n_components: Number of components to keep (None for all)
        center: Whether to center the data before PCA
        device: Device for computation ('cuda' or 'cpu')
    
    Returns:
        dict with keys:
            - 'components': (n_components, num_neurons) principal components
            - 'explained_variance': variance explained by each component
            - 'explained_variance_ratio': fraction of variance explained
            - 'mean': mean response vector (num_neurons,)
            - 'projections': (num_images, n_components) projected data
        All returned arrays are numpy arrays on CPU.
    """
    # Convert to tensor if needed
    if isinstance(responses, np.ndarray):
        responses = torch.from_numpy(responses).float()
    
    responses = responses.to(device)
    
    if center:
        mean = responses.mean(dim=0)
        centered = responses - mean
    else:
        mean = torch.zeros(responses.shape[1], device=device)
        centered = responses
    
    # SVD on GPU
    U, S, Vt = torch.linalg.svd(centered, full_matrices=False)
    
    # Variance explained
    n_samples = responses.shape[0]
    explained_variance = (S ** 2) / (n_samples - 1)
    total_variance = explained_variance.sum()
    explained_variance_ratio = explained_variance / total_variance
    
    if n_components is not None:
        Vt = Vt[:n_components]
        explained_variance = explained_variance[:n_components]
        explained_variance_ratio = explained_variance_ratio[:n_components]
    
    # Project data onto components
    projections = centered @ Vt.T
    
    # Move back to CPU as numpy
    return {
        'components': Vt.cpu().numpy(),
        'explained_variance': explained_variance.cpu().numpy(),
        'explained_variance_ratio': explained_variance_ratio.cpu().numpy(),
        'mean': mean.cpu().numpy(),
        'projections': projections.cpu().numpy()
    }


def screen_by_pc(
    responses,
    indices,
    pca_result,
    pc_index=0,
    k=10
):
    """
    Find images with highest/lowest projections onto a principal component.
    
    Args:
        responses: np.ndarray of shape (num_images, num_neurons)
        indices: Dataset indices corresponding to responses
        pca_result: Output from fit_pca()
        pc_index: Which PC to screen along (0 = PC1)
        k: Number of images to return from each end
    
    Returns:
        dict with keys:
            - 'high_indices': dataset indices of k highest projecting images
            - 'low_indices': dataset indices of k lowest projecting images
            - 'high_projections': projection values for high images
            - 'low_projections': projection values for low images
    """
    projections = pca_result['projections'][:, pc_index]
    
    sorted_idx = np.argsort(projections)
    
    low_idx = sorted_idx[:k]
    high_idx = sorted_idx[-k:][::-1]  # Reverse for highest first
    
    return {
        'high_indices': indices[high_idx],
        'low_indices': indices[low_idx],
        'high_projections': projections[high_idx],
        'low_projections': projections[low_idx]
    }


def get_pc_vector(pca_result, pc_index=0, as_tensor=True, device='cuda'):
    """
    Get a principal component vector for use with response_objective.
    
    Args:
        pca_result: Output from fit_pca()
        pc_index: Which PC to get (0 = PC1)
        as_tensor: Return as torch.Tensor if True, else np.ndarray
        device: Device for tensor
    
    Returns:
        PC vector of shape (num_neurons,)
    """
    pc = pca_result['components'][pc_index]
    
    if as_tensor:
        return torch.tensor(pc, dtype=torch.float32, device=device)
    return pc