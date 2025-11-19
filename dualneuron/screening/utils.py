from scipy import stats
import numpy as np
import os
import pandas as pd
from glob import glob
from tqdm import tqdm
import torch


def gini(x):
    n = len(x)
    if np.sum(x) == 0: return 0.0
    index = np.arange(1, n + 1)
    return (2 * np.sum(index * x)) / (n * np.sum(x)) - (n + 1) / n


def compute_population_statistics(resp_dir, sort_by='gini'):
    """
    Compute statistics for all neurons from their ordered response files.
    
    Args:
        resp_dir: Directory containing ordered response .npy files (named 0.npy, 1.npy, etc.)
        sort_by: Column name to sort by (default: 'gini')
        
    Returns:
        response_stats: DataFrame with statistics for each neuron
        active_neuron_ids: List of active neuron IDs
    """
    
    response_stats = []
    paths = glob(os.path.join(resp_dir, "*.npy"))
    
    for path in tqdm(paths, desc="Computing neuron statistics"):
        neuron_id = int(os.path.splitext(os.path.basename(path))[0])
        unit_responses = np.load(path)
        mean_resp = np.mean(unit_responses)
        std_resp = np.std(unit_responses)
        
        response_stats.append({
            'neuron_id': neuron_id,
            'gini': gini(unit_responses),
            'skewness': stats.skew(unit_responses),
            'mean': mean_resp,
            'std': std_resp,
            'range': np.ptp(unit_responses),
            'cv': std_resp / (mean_resp + 1e-8),
            'max': np.max(unit_responses),
            'q95': np.percentile(unit_responses, 95),
            'q05': np.percentile(unit_responses, 5)
        })
    
    response_stats = pd.DataFrame(response_stats)
    response_stats = response_stats.sort_values(sort_by, ascending=False)
    
    active_neurons = response_stats[
        (response_stats['max'] > 0.5) &           # Has at least some response
        (response_stats['mean'] > 0.01) &         # Non-zero mean activity
        (response_stats['std'] > 0.0)             # Has variance
    ]
    print(f"Active neurons: {len(active_neurons)} / {len(response_stats)} ({100*len(active_neurons)/len(response_stats):.1f}%)")
    return response_stats, active_neurons['neuron_id'].tolist()


def load_poles(neuron_id, dset, idx_dir, k=10, pole='both'):
    """
    Load the k lowest and/or k highest activating images for a neuron.
    
    Args:
        neuron_id: ID of the neuron
        dset: Dataset object (ImagenetImages or RenderedImages) with __getitem__ method
        idx_dir: Directory with ordered indices (.npy files)
        k: Number of images to load from each pole
        pole: 'low', 'high', or 'both'
    
    Returns:
        If pole='both': (low_images, high_images)
        If pole='low': low_images
        If pole='high': high_images
        
        Where images are torch tensors of shape (k, C, H, W)
    """
    # Load ordered responses and indices
    indices = np.load(os.path.join(idx_dir, f"{neuron_id}.npy"))
    
    def load_image_set(idx_list):
        """Load a set of images given their indices."""
        images = []
        for idx in idx_list:
            img, _ = dset[idx]  # Returns (tensor, label) or (tensor, path)
            images.append(img)
        return torch.stack(images)
    
    if pole in ['low', 'both']:
        low_idx = indices[:k]
        low_images = load_image_set(low_idx)
    
    if pole in ['high', 'both']:
        high_idx = indices[-k:]
        high_images = load_image_set(high_idx)
    
    if pole == 'both':
        return low_images, high_images
    elif pole == 'low':
        return low_images
    elif pole == 'high':
        return high_images
    else:
        raise ValueError(f"pole must be 'low', 'high', or 'both', got {pole}")
    
    
def sample_activations_adaptively(responses, num_samples=100):
    """
    Adaptively sample images based on activation derivative approx.
    Samples more densely where activation values change rapidly.
    
    Args:
        responses: Array of activation values for all images (shape: [num_images])
        num_samples: Number of samples to return
    
    Returns:
        sampled_idx: Indices of sampled images
        sorted_responses: Sorted activation values
        sampled_positions: Positions in sorted array (for visualization)
    """
    rng = np.random.default_rng(seed=num_samples)
    sorted_idx = np.argsort(responses)
    sorted_responses = responses[sorted_idx]
    # Get first-order differences
    diffs = np.abs(np.diff(sorted_responses))
    # Normalize to probability distribution
    probs = diffs / np.sum(diffs)
    # Sample positions along the curve
    sampled_transitions = rng.choice(
        len(probs),
        num_samples,
        p=probs,
        replace=False
    )
    # Take the point after each transition
    sampled_positions = sampled_transitions + 1
    sampled_idx = sorted_idx[sampled_positions]
    return sampled_idx, sorted_responses, sampled_positions