import numpy as np
import torch 
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.animation import FuncAnimation
from IPython.display import HTML
import cv2

from matplotlib import rc
rc('animation', html='jshtml')


def check_format(arr):
    """
    Convert tensor to numpy and ensure HWC format.
    
    Args:
        arr (torch.Tensor or np.ndarray): Image in any format.
    
    Returns:
        np.ndarray: Image in HWC format (height, width, channels).
    """
    if isinstance(arr, torch.Tensor):
        arr = arr.detach().cpu().numpy()
    if arr.ndim == 3 and arr.shape[0] in [1, 3]:
        return np.moveaxis(arr, 0, -1)
    return arr


def clip_percentile(img, percentile=0.1):
    """
    Clip image values to a percentile range.
    
    Removes extreme outliers by clipping to the specified percentile
    from both ends of the distribution.
    
    Args:
        img (np.ndarray): Input image.
        percentile (float): Percentile to clip from each end. For example,
            0.1 clips to the [0.1, 99.9] percentile range. Default: 0.1.
    
    Returns:
        np.ndarray: Clipped image.
    """
    clipped = np.clip(
        img, np.percentile(img, percentile),
        np.percentile(img, 100 - percentile)
    )
    return clipped


def denormalize(image, mean=0.45, std=0.25, boost=1.0):
    """
    Denormalize an image from standard normalization to [0, 1] range.
    
    Reverses the normalization typically applied before neural network
    input: x_normalized = (x - mean) / std.
    
    Args:
        image (np.ndarray): Normalized image.
        mean (float or None): Mean used in normalization. If None, uses
            min-max scaling instead. Default: 0.45.
        std (float or None): Standard deviation used in normalization.
            If None, uses min-max scaling instead. Default: 0.25.
        boost (float): Multiplier for std to increase contrast. Values > 1
            increase saturation/contrast. Default: 1.0.
    
    Returns:
        np.ndarray: Denormalized image clipped to [0, 1].
    """
    if mean is None or std is None:
        image = (image - image.min()) / (image.max() + 1e-8)
    else:
        image = image * std * boost + mean
    return np.clip(image, 0.0, 1.0)


def blend(
    image, alpha, 
    imagecut=0.0, alphacut=90, 
    mean=0.45, std=0.25, boost=1.0,
    bg_value=0.0, 
):
    """
    Blend an image with a gray background using an alpha mask.
    
    Creates a visualization where salient regions (high alpha) show the
    original image and non-salient regions fade to gray. Useful for
    visualizing which parts of a synthesized image drive neural activation.
    
    Args:
        image (torch.Tensor or np.ndarray): Input image, shape (C, H, W)
            or (H, W, C).
        alpha (torch.Tensor or np.ndarray): Transparency/saliency mask,
            shape (C, H, W) or (H, W, C). Averaged across channels.
        imagecut (float): Percentile for clipping image values. Default: 0.0.
        alphacut (float): Upper percentile for clipping alpha values.
            Prevents very high saliency values from dominating. Default: 90.
        mean (float): Mean for denormalization. Default: 0.45.
        std (float): Std for denormalization. Default: 0.25.
        boost (float): Contrast boost factor. Default: 1.0.
    
    Returns:
        np.ndarray: Blended image in [0, 1] range, shape (H, W, C).
    """
    image, alpha = check_format(image), check_format(alpha)

    image = clip_percentile(image, imagecut)
    image = denormalize(image, mean=mean, std=std, boost=boost)
    
    alpha = np.mean(alpha, -1, keepdims=True)
    alpha = np.clip(alpha, None, np.percentile(alpha, alphacut))
    alpha = alpha / (alpha.max() + 1e-8)

    is_rgb = image.shape[-1] == 3
    if is_rgb:
        alpha = np.repeat(alpha, 3, axis=-1)

    background = np.ones_like(image) * bg_value
    blended = image * alpha + background * (1 - alpha)
    return blended


def plot_poles(images, activations, vmin=0.1, vmax=0.8):
    """
    Plot two synthesized images with their optimization activation curves.
    
    Displays a 2x2 grid with activation curves on top and corresponding
    images below. Intended for comparing low-pole and high-pole MEIs.
    
    Args:
        images (list): Two images to display, each as np.ndarray or tensor.
        activations (list): Two lists of activation values from optimization,
            one per image.
        vmin (float): Minimum value for grayscale colormap. Default: 0.1.
        vmax (float): Maximum value for grayscale colormap. Default: 0.8.
    
    Returns:
        None: Displays the plot using plt.show().
    """
    fig, axs = plt.subplots(2, 2, figsize=(5, 5), facecolor='black')
    fig.subplots_adjust(hspace=0.3, wspace=0.3)

    for i, ax in enumerate(axs[0]):
        ax.plot(activations[i], color='#00d4ff', linewidth=2)
        ax.set_facecolor('#0a0a0a')
        ax.spines['bottom'].set_color('white')
        ax.spines['left'].set_color('white')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.tick_params(colors='white', labelsize=9)
        ax.grid(True, alpha=0.15, color='white')
        ax.set_title(
            ['Least Exciting', 'Most Exciting'][i], 
            color='white', 
            fontsize=11, pad=10
        )

    for i, ax in enumerate(axs[1]):
        ax.imshow(images[i], cmap='gray', vmin=vmin, vmax=vmax)
        ax.axis('off')

    plt.tight_layout()
    plt.show()


def plot_group(images, cols=5, vmin=0.1, vmax=0.8):
    """
    Plot a grid of images.
    
    Args:
        images (list): List of images to display.
        cols (int): Number of columns in the grid. Default: 5.
        vmin (float): Minimum value for grayscale colormap. Default: 0.1.
        vmax (float): Maximum value for grayscale colormap. Default: 0.8.
    
    Returns:
        None: Displays the plot using plt.show().
    """
    nrows = len(images) // cols if len(images) % cols == 0 else len(images) // cols + 1
    ncols = cols if len(images) >= cols else len(images)

    fig, axs = plt.subplots(
        nrows, ncols, 
        figsize=(2*ncols, 2*nrows), 
        facecolor='black',
        squeeze=False
    )
    fig.subplots_adjust(hspace=0.1, wspace=0.1)
    axs = axs.ravel()

    # Hide all axes first
    for ax in axs:
        ax.axis('off')

    # Then show images
    for ax, img in zip(axs, images):
        ax.imshow(
            img, cmap='gray', 
            vmin=vmin, vmax=vmax
        )

    plt.tight_layout()
    plt.show()
    
    
def estimate_mask(mei, zscore_thresh=0.5):
    """
    Estimate a receptive field mask from a synthesized MEI.
    
    Identifies salient regions by z-score thresholding, morphological
    operations, and convex hull extraction. Useful for estimating the
    spatial extent of a neuron's receptive field.
    
    Args:
        mei (np.ndarray): Most exciting image, shape (H, W) or (H, W, C).
        zscore_thresh (float): Z-score threshold for initial segmentation.
            Pixels with |z| > threshold are considered salient. Default: 0.5.
    
    Returns:
        np.ndarray: Soft mask in [0, 1] range, shape (H, W).
    
    Note:
        Requires scipy and skimage packages.
    """
    from scipy import ndimage
    from skimage import morphology
    
    params = {
        'mask_params': 1,
        'zscore_thresh': zscore_thresh,
        'closing_iters': 2,
        'gaussian_sigma': 1
    }

    if mei.shape[-1] == 1:
        mei = np.repeat(mei, 3, axis=-1)
        
    if len(mei.shape) == 3 and mei.shape[2] == 3:
        gray_mei = 0.299 * mei[:,:,0] + 0.587 * mei[:,:,1] + 0.114 * mei[:,:,2]
        mei = gray_mei
    
    norm_mei = (mei - mei.mean()) / (mei.std() + 1e-8)
    thresholded = np.abs(norm_mei) > params['zscore_thresh']
        
    closed = ndimage.binary_closing(
        thresholded, 
        iterations=params['closing_iters']
    )
    
    labeled = morphology.label(closed, connectivity=2)
    most_frequent = np.argmax(np.bincount(labeled.ravel())[1:]) + 1
    oneobject = labeled == most_frequent
    hull = morphology.convex_hull_image(oneobject)
    mask = ndimage.gaussian_filter(
        hull.astype(np.float32), 
        sigma=params['gaussian_sigma']
    )
    return mask


def get_mean_mask(poles, zthreshold=1.0):
    """
    Compute average receptive field mask from multiple MEIs.
    
    Estimates individual masks from each image and averages them to
    get a robust estimate of the neuron's receptive field location.
    
    Args:
        poles (list): List of MEI images, each shape (H, W) or (H, W, C).
        zthreshold (float): Z-score threshold for mask estimation.
            Default: 1.0.
    
    Returns:
        np.ndarray: Averaged and smoothed mask, shape (H, W), values in [0, 1].
    
    Note:
        The final mask is binarized at threshold 0.1 before Gaussian smoothing.
    """
    masks = []

    for i in range(len(poles)):
        mask = estimate_mask(poles[i], zthreshold)
        mask = (mask - mask.min()) / (mask.max() - mask.min())
        masks.append(mask)
    
    meanrf = np.stack(masks).mean(axis=0)
    mask = meanrf.copy()
    mask[mask>0.1] = 1
    
    mask = cv2.GaussianBlur(
        (mask * 255).astype(np.uint8), 
        (5, 5), 
        sigmaX=1.0, 
        sigmaY=1.0
    )
    mask = mask / 255.
    return mask


def sequence_animation(
    imgs, 
    activities,
    savename=None, 
    dpi=100,
    interval=80,
    title="Optimization Progress"
):
    """
    Create animation showing image synthesis optimization progress.
    
    Displays a side-by-side view of the evolving image and activation
    curve, with the current position highlighted. Useful for visualizing
    how activation maximization progresses over optimization steps.
    
    Args:
        imgs (list): List of images at each optimization step.
            Each image can be torch.Tensor or np.ndarray in CHW or HWC format.
        activities (list): Activation values at each step. Must have
            same length as imgs.
        savename (str, optional): Path to save animation as MP4. If None,
            returns an HTML animation for Jupyter display. Default: None.
        dpi (int): Resolution of the output. Default: 100.
        interval (int): Time between frames in milliseconds. Default: 80.
        title (str): Title displayed above the animation.
            Default: "Optimization Progress".
    
    Returns:
        IPython.display.HTML or None: If savename is None, returns HTML
            animation for Jupyter notebook display. Otherwise, saves to
            file and returns None.
    
    Raises:
        AssertionError: If imgs and activities have different lengths.
    
    Note:
        Requires FFmpeg for saving to MP4. Colors indicate activation sign:
        cyan (#00d4ff) for positive, magenta (#ff0080) for negative.
    """
    num_frames = len(imgs)
    assert len(activities) == num_frames, "Images and activities must have same length"
    
    activities = np.array(activities)
    
    # Setup figure with proper proportions
    fig = plt.figure(figsize=(10, 5), dpi=dpi)
    fig.patch.set_facecolor('black')
    
    # Create subplots
    ax_img = fig.add_subplot(1, 2, 1, aspect='equal')
    ax_plot = fig.add_subplot(1, 2, 2)
    
    # Configure image subplot
    ax_img.set_axis_off()
    ax_img.set_facecolor('black')
    ax_img.set_aspect('equal')
    
    # Configure activation plot with cyan aesthetic
    ax_plot.set_facecolor('#0a0a0a')
    ax_plot.set_xlabel('Optimization Step', color='white', fontsize=10, weight='bold')
    ax_plot.set_ylabel('Activation (Hz)', color='white', fontsize=10, weight='bold')
    ax_plot.tick_params(colors='white', labelsize=9)
    
    # Style spines with cyan color
    for spine in ax_plot.spines.values():
        spine.set_color('#00d4ff')
        spine.set_linewidth(2)
    
    # Cyan grid
    ax_plot.grid(True, alpha=0.2, color='#00d4ff', linestyle='--', linewidth=0.8)
    
    # Plot the full trajectory
    steps = np.arange(num_frames)
    
    # Use cyan for positive activations, magenta for negative
    for i in range(len(steps)-1):
        color = '#00d4ff' if activities[i] >= 0 else '#ff0080'
        ax_plot.plot(
            steps[i:i+2], activities[i:i+2], 
            color=color, linewidth=2.5, 
            alpha=0.7, zorder=1
        )
    
    # Add title
    fig.suptitle(title, color='white', fontsize=14, weight='bold', y=0.98)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    # Animation function
    def animate(frame_idx):
        ax_img.clear()
        ax_img.set_axis_off()
        ax_img.set_facecolor('black')
        
        # Show current image
        current_img = imgs[frame_idx]
        
        # Convert to numpy if tensor
        if torch.is_tensor(current_img):
            current_img = current_img.cpu().numpy()
        
        # Handle different input formats
        if current_img.ndim == 3:
            # Check if CHW (channels first) or HWC (channels last)
            if current_img.shape[0] in [1, 3]:  # CHW format
                current_img = np.transpose(current_img, (1, 2, 0))  # Convert to HWC
            
            # If single channel, squeeze it
            if current_img.shape[2] == 1:
                current_img = current_img.squeeze(2)
        
        ax_img.imshow(current_img, cmap='gray')
        
        # Add step counter and activation value
        current_activation = activities[frame_idx] # convert to Hz 
        value_color = '#00d4ff' if current_activation >= 0 else '#ff0080'
        
        ax_img.text(
            0.5, 1.08, f'Step {frame_idx}',
            transform=ax_img.transAxes,
            color='white', fontsize=11, ha='center', va='bottom',
            weight='bold'
        )
        
        ax_img.text(
            0.5, -0.08, f'{current_activation:.4f} Hz',
            transform=ax_img.transAxes,
            color=value_color, fontsize=11, ha='center', va='top',
            weight='bold',
            bbox=dict(
                boxstyle='round,pad=0.5', 
                facecolor='#0a0a0a', 
                edgecolor=value_color, 
                linewidth=2
            )
        )
        
        # Clear old markers from plot
        for line in ax_plot.lines[num_frames-1:]:
            line.remove()
        
        # Add current position marker
        marker_color = '#00d4ff' if current_activation >= 0 else '#ff0080'
        
        # Main marker
        ax_plot.plot(
            frame_idx, current_activation, 'o', 
            color=marker_color, markersize=10, 
            markeredgecolor='white',
            markeredgewidth=2, zorder=5
        )
        
        # Glow effect
        ax_plot.plot(
            frame_idx, current_activation, 'o', 
            color=marker_color, markersize=20, 
            alpha=0.3, zorder=4
        )
        
        # Highlight progress with thicker line
        if frame_idx > 0:
            for i in range(frame_idx):
                color = '#00d4ff' if activities[i] >= 0 else '#ff0080'
                ax_plot.plot(
                    steps[i:i+2], 
                    activities[i:i+2],
                    color=color, 
                    linewidth=4,
                    alpha=0.4, 
                    zorder=2
                )
    
    # Create animation
    ani = FuncAnimation(
        fig, animate,
        frames=num_frames,
        interval=interval,
        repeat=True,
        cache_frame_data=False
    )
    
    # Save or display
    if savename is not None:
        writer = animation.FFMpegWriter(
            fps=max(10, min(30, 1000//interval)),
            metadata=dict(artist='Neural Optimization'),
            bitrate=1200,
            codec='h264',
            extra_args=['-pix_fmt', 'yuv420p', '-preset', 'fast']
        )
        
        ani.save(savename, writer=writer, dpi=dpi)
        plt.close(fig)
    else:
        plt.rcParams['animation.embed_limit'] = 100
        plt.close(fig)
        return HTML(ani.to_jshtml())