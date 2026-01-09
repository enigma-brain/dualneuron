import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D


def extract_population_responses(
    model,
    dataset,
    batch_size=32,
    num_images=5000,
    indices=None,
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
        num_images: Number of images to sample (None for all). Ignored if indices provided.
        indices: Optional array of specific dataset indices to use. If provided,
            num_images and seed are ignored.
        device: Device for inference
        seed: Random seed for reproducible sampling (ignored if indices provided)
        model_kwargs: Optional dict of kwargs to pass to model
    
    Returns:
        responses: np.ndarray of shape (num_images, num_neurons)
        indices: np.ndarray of dataset indices used
    """
    model_kwargs = model_kwargs or {}
    
    if indices is not None:
        indices = np.asarray(indices)
    else:
        rng = np.random.RandomState(seed)
        
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
            if hasattr(model, 'embed'):
                resp = model.embed(images).cpu().numpy()
            else:
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


def interpolate_pc(
    pca_result,
    pc_index=0,
    n_steps=11,
    range_std=3.0,
    device='cuda'
):
    """
    Generate response vectors interpolating along a single PC direction.
    
    Args:
        pca_result: Output from fit_pca()
        pc_index: Which PC to interpolate along (0 = PC1)
        n_steps: Number of interpolation points
        range_std: Range in standard deviations (e.g., 3.0 means -3*sigma to +3*sigma)
        device: Device for output tensors
    
    Returns:
        dict with keys:
            - 'responses': List of response vectors, each shape (num_neurons,)
            - 'coords': Array of PC coordinates (in std units)
            - 'pc_index': Which PC was used
            - 'std': Standard deviation of projections onto this PC
    """
    projections = pca_result['projections'][:, pc_index]
    std = projections.std()
    mean_proj = projections.mean()  # Should be ~0 for centered PCA
    
    # Coordinates in std units
    coords = np.linspace(-range_std, range_std, n_steps)
    
    # Convert to actual projection values
    proj_values = mean_proj + coords * std
    
    # Build response vectors: mean + projection * PC_direction
    pc_direction = pca_result['components'][pc_index]
    mean_response = pca_result['mean']
    
    responses = []
    for proj in proj_values:
        response = mean_response + proj * pc_direction
        responses.append(torch.tensor(response, dtype=torch.float32, device=device))
    
    return {
        'responses': responses,
        'coords': coords,
        'pc_index': pc_index,
        'std': std
    }


def interpolate_pc_2d(
    pca_result,
    pc_indices=(0, 1),
    n_steps=5,
    range_std=3.0,
    device='cuda'
):
    """
    Generate response vectors on a 2D grid in PC space.
    
    Args:
        pca_result: Output from fit_pca()
        pc_indices: Tuple of two PC indices (e.g., (0, 1) for PC1 vs PC2)
        n_steps: Number of steps along each axis (total grid is n_steps × n_steps)
        range_std: Range in standard deviations along each axis
        device: Device for output tensors
    
    Returns:
        dict with keys:
            - 'responses': 2D list of response vectors, responses[i][j] for (pc1_i, pc2_j)
            - 'coords_1': Array of coordinates along first PC (in std units)
            - 'coords_2': Array of coordinates along second PC (in std units)
            - 'pc_indices': Which PCs were used
            - 'stds': Tuple of standard deviations (std_1, std_2)
    """
    pc_idx_1, pc_idx_2 = pc_indices
    
    proj_1 = pca_result['projections'][:, pc_idx_1]
    proj_2 = pca_result['projections'][:, pc_idx_2]
    
    std_1 = proj_1.std()
    std_2 = proj_2.std()
    mean_1 = proj_1.mean()
    mean_2 = proj_2.mean()
    
    coords_1 = np.linspace(-range_std, range_std, n_steps)
    coords_2 = np.linspace(-range_std, range_std, n_steps)
    
    pc_dir_1 = pca_result['components'][pc_idx_1]
    pc_dir_2 = pca_result['components'][pc_idx_2]
    mean_response = pca_result['mean']
    
    responses = []
    for c1 in coords_1:
        row = []
        proj_val_1 = mean_1 + c1 * std_1
        for c2 in coords_2:
            proj_val_2 = mean_2 + c2 * std_2
            response = mean_response + proj_val_1 * pc_dir_1 + proj_val_2 * pc_dir_2
            row.append(torch.tensor(response, dtype=torch.float32, device=device))
        responses.append(row)
    
    return {
        'responses': responses,
        'coords_1': coords_1,
        'coords_2': coords_2,
        'pc_indices': pc_indices,
        'stds': (std_1, std_2)
    }
    
    
def plot_pca_variance(
    variance_ratio, 
    figsize=(10, 6), 
    accent_color='#00D4FF', 
    title_prefix='PCA Explained Variance'
):
    """
    Create a stylish visualization of PCA explained variance ratio.
    
    Args:
        variance_ratio: Array of explained variance ratios per component
        figsize: Figure size tuple
        accent_color: Color for bars and accents (default: cyan)
        title_prefix: Prefix for the plot title
    """
    n_components = len(variance_ratio)
    cumulative = np.cumsum(variance_ratio)
    x = np.arange(1, n_components + 1)
    
    # Dark style setup
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=figsize, facecolor='black')
    ax.set_facecolor('black')
    
    # Gradient bars using individual colors
    colors = [plt.cm.cool(0.3 + 0.5 * i / n_components) for i in range(n_components)]
    
    # Main bars with glow effect
    for i, (xi, yi, c) in enumerate(zip(x, variance_ratio, colors)):
        # Glow layers
        for w, a in [(0.6, 0.1), (0.5, 0.2), (0.4, 0.4)]:
            ax.bar(xi, yi, width=w, color=accent_color, alpha=a, zorder=2)
        # Solid bar
        ax.bar(xi, yi, width=0.35, color=accent_color, alpha=0.9, zorder=3,
               edgecolor='white', linewidth=0.5)
    
    # Cumulative line with glow
    ax.plot(x, cumulative, color=accent_color, alpha=0.2, linewidth=8, zorder=4)
    ax.plot(x, cumulative, color=accent_color, alpha=0.4, linewidth=4, zorder=4)
    ax.plot(x, cumulative, color='white', linewidth=2, zorder=5, 
            marker='o', markersize=8, markerfacecolor=accent_color,
            markeredgecolor='white', markeredgewidth=1.5)
    
    # Value labels on bars
    for i, (xi, yi) in enumerate(zip(x, variance_ratio)):
        ax.text(xi, yi + 0.012, f'{yi:.1%}', ha='center', va='bottom',
                fontsize=11, fontweight='bold', color='white', zorder=6)
    
    # Cumulative labels
    for i, (xi, yi) in enumerate(zip(x, cumulative)):
        ax.text(xi + 0.15, yi + 0.025, f'{yi:.1%}', ha='left', va='bottom',
                fontsize=9, color=accent_color, alpha=0.9, zorder=6)
    
    # Grid
    ax.yaxis.grid(True, linestyle='--', alpha=0.2, color='white')
    ax.set_axisbelow(True)
    
    # Labels and title
    ax.set_xlabel('Principal Component', fontsize=12, fontweight='bold', 
                  color='white', labelpad=10)
    ax.set_ylabel('Explained Variance Ratio', fontsize=12, fontweight='bold',
                  color='white', labelpad=10)
    ax.set_title(f'{title_prefix}', fontsize=16, fontweight='bold',
                 color='white', pad=20)
    
    # Axes styling
    ax.set_xticks(x)
    ax.set_xticklabels([f'PC{i}' for i in x], fontsize=10, color='white')
    ax.tick_params(colors='white', which='both')
    ax.set_ylim(0, max(cumulative) * 1.15)
    ax.set_xlim(0.4, n_components + 0.6)
    
    # Spine styling
    for spine in ax.spines.values():
        spine.set_color('#333333')
        spine.set_linewidth(0.5)
    
    # Legend
    
    legend_elements = [
        Patch(facecolor=accent_color, alpha=0.9, edgecolor='white', 
              label='Individual Variance'),
        Line2D([0], [0], color='white', linewidth=2, marker='o',
               markerfacecolor=accent_color, markersize=8,
               label='Cumulative Variance')
    ]
    ax.legend(handles=legend_elements, loc='right', fontsize=10,
              facecolor='#111111', edgecolor='#333333', labelcolor='white')
    
    plt.tight_layout()
    plt.show()