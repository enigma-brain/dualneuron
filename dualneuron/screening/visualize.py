import numpy as np
from tqdm import tqdm
import os
import pandas as pd
import matplotlib.pyplot as plt
from glob import glob
from dualneuron.screening.utils import sample_activations_adaptively

import base64
from io import BytesIO
from PIL import Image
import json
import torch


def plot_population_statistics(response_stats, figsize=(12, 8)):
    """
    Plot histograms of key neuron statistics.
    
    Displays six histograms showing the distribution of selectivity and activity
    metrics across the neuron population, with median lines for each metric.
    
    Args:
        response_stats (pd.DataFrame): DataFrame from compute_population_statistics()
            containing columns: 'gini', 'max', 'mean', 'cv', 'skewness', 'q95'.
        figsize (tuple): Figure size as (width, height). Default: (12, 8).
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


def plot_neuron_activation(
    neuron_id, 
    resp_dir, 
    response_stats, 
    figsize=(5, 5)
):
    """
    Plot the sorted activation curve for a single neuron.
    
    Displays the neuron's responses to all images sorted by activation value,
    with a horizontal line indicating the mean response.
    
    Args:
        neuron_id (int): ID of the neuron to plot.
        resp_dir (str): Directory containing ordered response .npy files.
        response_stats (pd.DataFrame): DataFrame from compute_population_statistics()
            containing statistics for each neuron.
        figsize (tuple): Figure size as (width, height). Default: (5, 5).
    """
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
    
    Displays a 3x10 grid showing:
        - Row 1: 10 lowest activating images (most suppressive)
        - Row 2: First 10 images with positive activation
        - Row 3: 10 highest activating images (most excitatory)
    
    Args:
        neuron_id (int): ID of the neuron to visualize.
        dset: Dataset object (ImagenetImages or RenderedImages) with __getitem__
            method that returns (tensor, label) tuples.
        resp_dir (str): Directory containing ordered response .npy files.
        idx_dir (str): Directory containing ordered index .npy files.
        figsize (tuple): Figure size as (width, height). Default: (16, 6).
        vmin (float, optional): Minimum value for image color scaling.
        vmax (float, optional): Maximum value for image color scaling.
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


def plot_synthesized_poles(
    neuron_ids,
    dset,
    resp_dir,
    idx_dir,
    mask,
    lei_path='lei_poles.npy',
    mei_path='mei_poles.npy',
    output_file='poles.png',
    figsize=None,
    vmin=None, 
    vmax=None,
    output_size=(224, 224),
    crop_padding_frac=0.1,
    add_neuron_labels=False
):
    """
    Plot synthesized and screened least/most activating images for multiple neurons.
    
    Creates a grid with 2 rows per neuron:
        - Row 1: 5 LEI seeds (left) and 5 MEI seeds (right) - cropped to mask and scaled
        - Row 2: 5 least activating screened images (left) and 5 most activating screened images (right)
    
    Args:
        neuron_ids (list): List of M neuron IDs to visualize.
        dset: Dataset object with __getitem__ method that returns (tensor, label) tuples.
        resp_dir (str): Directory containing ordered response .npy files.
        idx_dir (str): Directory containing ordered index .npy files.
        mask (np.ndarray): 2D mask array for cropping synthesized images.
        lei_path (str): Path to .npy file with LEI images (shape: [M*5, C, H, W]).
        mei_path (str): Path to .npy file with MEI images (shape: [M*5, C, H, W]).
        figsize (tuple, optional): Figure size. Default: (10, 2*M).
        vmin (float, optional): Minimum value for image color scaling.
        vmax (float, optional): Maximum value for image color scaling.
        output_size (tuple): Target size after cropping and scaling. Default: (224, 224).
        crop_padding_frac (float): Padding fraction for mask bbox. Default: 0.1.
        add_neuron_labels (bool): Whether to add neuron labels on the left side.
    """
    M = len(neuron_ids)
    
    # Load synthesized images
    lei_images = np.load(lei_path)  # Shape: [M*5, C, H, W]
    mei_images = np.load(mei_path)  # Shape: [M*5, C, H, W]
    
    # Create CropToMask transform for synthesized images
    from dualneuron.screening.sets import CropToMask
    import os
    crop_transform = CropToMask(mask, output_size, crop_padding_frac)
    
    # Set default figure size
    if figsize is None:
        figsize = (10, 2 * M)
    
    # Create figure
    fig = plt.figure(figsize=figsize, facecolor='black')
    
    # Create grid spec with extra space between columns 5 and 6
    gs = fig.add_gridspec(
        2 * M, 11,  # 11 columns to accommodate spacing
        hspace=0.15, 
        wspace=0.05,
        top=0.92,  # Reduced from 0.98 to leave more space for headers
        bottom=0.02, 
        left=0.02, 
        right=0.98,
        width_ratios=[1, 1, 1, 1, 1, 0.3, 1, 1, 1, 1, 1]  # Extra space at index 5
    )
    
    for neuron_idx, neuron_id in enumerate(neuron_ids):
        base_row = neuron_idx * 2
        
        # Load ordered responses and indices for this neuron
        responses = np.load(os.path.join(resp_dir, f"{neuron_id}.npy"))
        indices = np.load(os.path.join(idx_dir, f"{neuron_id}.npy"))
        
        # Get lowest 5 and highest 5 screened images
        lowest_idx = indices[:5]
        highest_idx = indices[-10:-5]
        
        # Get synthesized images for this neuron (5 LEIs and 5 MEIs)
        lei_start = neuron_idx * 5
        mei_start = neuron_idx * 5
        neuron_leis = lei_images[lei_start:lei_start + 5]
        neuron_meis = mei_images[mei_start:mei_start + 5]
        
        # Row 1: Synthesized seeds (LEIs and MEIs) - CROPPED AND SCALED
        for col_idx in range(5):
            # LEI seeds (left side)
            ax = fig.add_subplot(gs[base_row, col_idx])
            img = neuron_leis[col_idx]
            
            # Convert to tensor and apply crop transform
            img_tensor = torch.from_numpy(img).float()
            if img_tensor.dim() == 2:  # HW -> CHW
                img_tensor = img_tensor.unsqueeze(0)
            elif img_tensor.shape[0] not in [1, 3]:  # HWC -> CHW
                img_tensor = img_tensor.permute(2, 0, 1)
            
            # Apply crop and scale
            img_cropped = crop_transform(img_tensor)
            
            # Convert back to numpy for display
            img = img_cropped.numpy()
            if img.shape[0] in [1, 3]:  # CHW -> HWC
                img = np.transpose(img, (1, 2, 0))
            if img.shape[-1] == 1:  # Grayscale
                img = img.squeeze(-1)
                ax.imshow(img, cmap='gray', vmin=vmin, vmax=vmax)
            else:
                ax.imshow(img, vmin=vmin, vmax=vmax)
            ax.axis('off')
            
            # MEI seeds (right side, skip column 5 for spacing)
            ax = fig.add_subplot(gs[base_row, col_idx + 6])
            img = neuron_meis[col_idx]
            
            # Convert to tensor and apply crop transform
            img_tensor = torch.from_numpy(img).float()
            if img_tensor.dim() == 2:  # HW -> CHW
                img_tensor = img_tensor.unsqueeze(0)
            elif img_tensor.shape[0] not in [1, 3]:  # HWC -> CHW
                img_tensor = img_tensor.permute(2, 0, 1)
            
            # Apply crop and scale
            img_cropped = crop_transform(img_tensor)
            
            # Convert back to numpy for display
            img = img_cropped.numpy()
            if img.shape[0] in [1, 3]:  # CHW -> HWC
                img = np.transpose(img, (1, 2, 0))
            if img.shape[-1] == 1:  # Grayscale
                img = img.squeeze(-1)
                ax.imshow(img, cmap='gray', vmin=vmin, vmax=vmax)
            else:
                ax.imshow(img, vmin=vmin, vmax=vmax)
            ax.axis('off')
        
        # Row 2: Screened least/most activating images from dataset
        # (These are already cropped and scaled via dset transforms)
        for col_idx in range(5):
            # Least activating screened images (left side)
            ax = fig.add_subplot(gs[base_row + 1, col_idx])
            img_idx = lowest_idx[col_idx]
            img, _ = dset[img_idx]
            img = img.permute(1, 2, 0)  # CHW -> HWC
            if img.shape[-1] == 1:  # Grayscale
                img = img.squeeze(-1)
                ax.imshow(img, cmap='gray', vmin=vmin, vmax=vmax)
            else:
                ax.imshow(img, vmin=vmin, vmax=vmax)
            ax.axis('off')
            
            # Most activating screened images (right side)
            ax = fig.add_subplot(gs[base_row + 1, col_idx + 6])
            img_idx = highest_idx[col_idx]
            img, _ = dset[img_idx]
            img = img.permute(1, 2, 0)  # CHW -> HWC
            if img.shape[-1] == 1:  # Grayscale
                img = img.squeeze(-1)
                ax.imshow(img, cmap='gray', vmin=vmin, vmax=vmax)
            else:
                ax.imshow(img, vmin=vmin, vmax=vmax)
            ax.axis('off')
        
        # Add neuron label on the left
        if add_neuron_labels:
            ax_label = fig.add_subplot(gs[base_row:base_row + 2, 0])
            ax_label.text(
                -0.2, 0.5, f'Neuron {neuron_idx + 1}',
                transform=ax_label.transAxes,
                fontsize=11, color='white',
                ha='right', va='center',
                weight='bold',
                rotation=90
            )
            ax_label.axis('off')
    
    # Add horizontal separator line between neurons (except after last neuron)
    for neuron_idx in range(M - 1):
        base_row = neuron_idx * 2
        # Get position of a subplot in row base_row + 1 (second row of neuron)
        temp_ax = fig.add_subplot(gs[base_row + 1, 0])
        bbox = temp_ax.get_position()
        temp_ax.remove()
        
        # Draw line at bottom of this row
        line_y = bbox.y0
        
        fig.add_artist(plt.Line2D(
            [0.02, 0.98],
            [line_y, line_y],
            transform=fig.transFigure,
            color='white',
            linewidth=1.5,
            alpha=0.5
        ))
    
    # Add column headers
    fig.text(0.25, 0.97, 'Least Activating', 
             ha='center', va='top', color='#00d4ff', 
             fontsize=13, weight='bold')
    fig.text(0.75, 0.97, 'Most Activating', 
             ha='center', va='top', color='#ff0080', 
             fontsize=13, weight='bold')
    
    plt.savefig(
        output_file,
        dpi=200,
        facecolor='black',
        edgecolor='none',
        bbox_inches='tight',
        pad_inches=0.1
    )

    plt.show()
    
    
def visualize_adaptive_sampling(responses, num_samples=100, figsize=(5, 5)):
    """
    Visualize which points were sampled along the activation curve.
    
    Shows the full sorted activation curve with highlighted points indicating
    where adaptive sampling selected images (denser sampling where the curve
    changes rapidly).
    
    Args:
        responses (np.ndarray): 1D array of activation values for all images.
        num_samples (int): Number of points to sample. Default: 100.
        figsize (tuple): Figure size as (width, height). Default: (5, 5).
    """
    sampled_idx, sorted_responses, sampled_positions = sample_activations_adaptively(
        responses, num_samples
    )
    
    fig, ax = plt.subplots(figsize=figsize, facecolor='black')
    ax.plot(sorted_responses, color='#00d4ff', linewidth=1.5, alpha=0.8)
    ax.fill_between(
        range(len(sorted_responses)), 
        sorted_responses, 
        color='#00d4ff', 
        alpha=0.3
    )
    
    ax.scatter(
        sampled_positions, 
        sorted_responses[sampled_positions], 
        c='#ff0080', 
        s=30, 
        zorder=5, 
        alpha=0.8,
        label=f'Sampled (n={num_samples})'
    )
    
    ax.set_facecolor('#0a0a0a')
    ax.spines['bottom'].set_color('white')
    ax.spines['left'].set_color('white')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(colors='white', labelsize=9)
    ax.grid(True, alpha=0.15, color='white', axis='y')
    
    ax.set_title(
        'Adaptive Sampling\nMore samples where curve changes rapidly', 
        color='white', fontsize=11, pad=10
    )
    ax.set_xlabel('Sorted Image Index', color='white', fontsize=10)
    ax.set_ylabel('Response', color='white', fontsize=10)
    
    ax.legend(
        loc='upper left', fontsize=9, 
        facecolor='#0a0a0a', edgecolor='white', 
        labelcolor='white'
    )
    
    plt.tight_layout()
    plt.show()