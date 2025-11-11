import numpy as np
from tqdm import tqdm
import os
import pandas as pd
import matplotlib.pyplot as plt
from glob import glob


def plot_population_statistics(response_stats, figsize=(12, 8)):
    """
    Plot histograms of key neuron statistics.
    Args: response_stats: List of dicts with neuron statistics
    """
    
    metrics = {
        'gini': 'Gini Coefficient\n(Sparsity)',
        'max': 'Max Response\n(Dynamic Range)',
        'mean': 'Mean Response\n(Overall Activity)',
        'cv': 'Coefficient of Variation\n(Reliability)',
        'skewness': 'Skewness\n(Distribution Shape)',
        'q95': '95th Percentile\n(Strong Responses)'
    }
    
    fig, axs = plt.subplots(2, 3, figsize=figsize, facecolor='black')
    fig.subplots_adjust(hspace=0.4, wspace=0.4)
    axs = axs.flatten()
    
    for idx, (metric, label) in enumerate(metrics.items()):
        ax = axs[idx]
        
        # Plot histogram
        ax.hist(
            response_stats[metric], 
            bins=50, 
            color='#00d4ff', 
            alpha=0.7, 
            edgecolor='#00d4ff',
            linewidth=1.5
        )
        
        # Styling
        ax.set_facecolor('#0a0a0a')
        ax.spines['bottom'].set_color('white')
        ax.spines['left'].set_color('white')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.tick_params(colors='white', labelsize=9)
        ax.grid(True, alpha=0.15, color='white', axis='y')
        
        # Labels
        ax.set_title(label, color='white', fontsize=11, pad=10)
        ax.set_xlabel('Value', color='white', fontsize=9)
        ax.set_ylabel('Count', color='white', fontsize=9)
        
        # Add median line
        median_val = response_stats[metric].median()
        ax.axvline(
            median_val, 
            color='#ff0080', 
            linestyle='--', 
            linewidth=2, 
            alpha=0.8,
            label=f'Median: {median_val:.3f}'
        )
        ax.legend(
            loc='upper right', 
            fontsize=8, 
            facecolor='#0a0a0a', 
            edgecolor='white', 
            labelcolor='white'
        )
    
    plt.tight_layout()
    plt.show()


def plot_neuron_activation(neuron_id, resp_dir, response_stats, figsize=(5, 5)):
    unit_responses = np.load(os.path.join(resp_dir, f"{neuron_id}.npy"))
    nstats = response_stats[response_stats['neuron_id'] == neuron_id].iloc[0]
    fig, ax = plt.subplots(figsize=figsize, facecolor='black')
    ax.plot(unit_responses, color='#00d4ff', linewidth=1.5, alpha=0.8)
    ax.fill_between(
        range(len(unit_responses)), 
        unit_responses, 
        color='#00d4ff', 
        alpha=0.3
    )
    ax.axhline(
        nstats['mean'], 
        color='#ff0080', linestyle='--', 
        linewidth=2, alpha=0.8, 
        label=f"Mean: {nstats['mean']:.3f}"
    )
    ax.set_facecolor('#0a0a0a')
    ax.spines['bottom'].set_color('white')
    ax.spines['left'].set_color('white')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(colors='white', labelsize=9)
    ax.grid(True, alpha=0.15, color='white', axis='y')
    ax.set_title(
        f'Neuron {neuron_id} - Responses\nGini: {nstats["gini"]:.3f} | Max: {nstats["max"]:.2f}', 
        color='white', fontsize=11, pad=10
    )
    ax.set_xlabel('Image Rank', color='white', fontsize=10)
    ax.set_ylabel('Response', color='white', fontsize=10)
    
    ax.legend(
        loc='upper left', fontsize=9, 
        facecolor='#0a0a0a', edgecolor='white', 
        labelcolor='white'
    )
    
    plt.tight_layout()
    plt.show()
    
    
def plot_neuron_poles(
    neuron_id, dset, resp_dir, 
    idx_dir, figsize=(16, 6), 
    vmin=None, vmax=None
):
    """
    Plot the lowest, first-positive, and highest activating images for a neuron.
    
    Args:
        neuron_id: ID of the neuron
        resp_dir: Directory with ordered responses
        idx_dir: Directory with ordered indices
        figsize: Figure size
        vmin, vmax: Color scale limits for images
    """
    # Load ordered responses and indices
    responses = np.load(os.path.join(resp_dir, f"{neuron_id}.npy"))
    indices = np.load(os.path.join(idx_dir, f"{neuron_id}.npy"))
    
    # Find key image sets
    # 1. Lowest 10
    lowest_idx = indices[:10]
    lowest_resp = responses[:10]
    
    # 2. First 10 positive activations
    first_positive_mask = responses > 0
    if np.any(first_positive_mask):
        first_positive_pos = np.where(first_positive_mask)[0][:10]
        first_positive_idx = indices[first_positive_pos]
        first_positive_resp = responses[first_positive_pos]
    else:
        first_positive_idx = []
        first_positive_resp = []
    
    # 3. Highest 10
    highest_idx = indices[-10:]
    highest_resp = responses[-10:]
    
    # Create figure with better spacing
    fig, axs = plt.subplots(3, 10, figsize=figsize, facecolor='black')
    fig.subplots_adjust(
        hspace=0.15, wspace=0.05, top=0.92, 
        bottom=0.02, left=0.02, right=0.98
    )
    
    titles = ['Lowest', 'First Positive', 'Highest']
    image_sets = [
        (lowest_idx, lowest_resp),
        (first_positive_idx, first_positive_resp),
        (highest_idx, highest_resp)
    ]
    
    for row_idx, (title, (img_indices, img_responses)) in enumerate(zip(titles, image_sets)):
        axs[row_idx, 0].text(
            -0.15, 0.5, title,
            transform=axs[row_idx, 0].transAxes,
            fontsize=12, color='#00d4ff',
            ha='center', va='center',
            weight='bold',
            rotation=90,
            bbox=dict(
                boxstyle='round,pad=0.4', 
                facecolor='#0a0a0a', 
                edgecolor='#00d4ff', 
                linewidth=2
            )
        )
        
        for col_idx in range(10):
            ax = axs[row_idx, col_idx]
            
            if col_idx < len(img_indices):
                img_idx = img_indices[col_idx]
                response = img_responses[col_idx]
                img, _ = dset[img_idx]
                img = img.permute(1, 2, 0)
                ax.imshow(img, cmap='gray', vmin=vmin, vmax=vmax)
                
                ax.text(
                    0.5, 1.05, f'{response:.6f}',
                    transform=ax.transAxes,
                    color='#00d4ff' if response > 0 else '#ff0080',
                    fontsize=10, ha='center', va='bottom',
                    weight='bold'
                )
            
            ax.axis('off')
    
    fig.suptitle(
        f'Neuron {neuron_id}', 
        color='white', fontsize=16, y=1.0
    )
    
    plt.show()