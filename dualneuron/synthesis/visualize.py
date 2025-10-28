import numpy as np
import torch 
import matplotlib.pyplot as plt
import cv2


def check_format(arr):
    if isinstance(arr, torch.Tensor):
        arr = arr.detach().cpu().numpy()
    if arr.ndim == 3 and arr.shape[0] in [1, 3]:
        return np.moveaxis(arr, 0, -1)
    return arr


def clip_percentile(img, percentile=0.1):
    clipped = np.clip(
        img, np.percentile(img, percentile),
        np.percentile(img, 100 - percentile)
    )
    return clipped


def denormalize(image, mean=0.45, std=0.25, boost=1.0):
    if mean is None or std is None:
        image = (image - image.min()) / (image.max() + 1e-8)
    else:
        image = image * std * boost + mean
    return np.clip(image, 0.0, 1.0)


def blend(
    image, alpha, 
    imagecut=0.0, alphacut=90, 
    mean=0.45, std=0.25, boost=1.0
):
    image, alpha = check_format(image), check_format(alpha)

    image = clip_percentile(image, imagecut)
    image = denormalize(image, mean=mean, std=std, boost=boost)
    
    alpha = np.mean(alpha, -1, keepdims=True)
    alpha = np.clip(alpha, None, np.percentile(alpha, alphacut))
    alpha = alpha / (alpha.max() + 1e-8)

    is_rgb = image.shape[-1] == 3
    if is_rgb:
        alpha = np.repeat(alpha, 3, axis=-1)

    gray_background = np.ones_like(image) * mean
    blended = image * alpha + gray_background * (1 - alpha)
    return blended


def plot_poles(images, activations, vmin=0.1, vmax=0.8):
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
    nrows = len(images) // cols if len(images) % cols == 0 else len(images) // cols + 1
    ncols = cols if len(images) >= cols else len(images)

    fig, axs = plt.subplots(
        nrows, ncols, 
        figsize=(2*ncols, 2*nrows), 
        facecolor='black'
    )
    fig.subplots_adjust(hspace=0.1, wspace=0.1)
    axs = axs.ravel()

    for ax, img in zip(axs, images):
        ax.imshow(
            img, cmap='gray', 
            vmin=vmin, vmax=vmax
        )
        ax.axis('off')

    plt.tight_layout()
    plt.show()
    
    
def estimate_mask(mei, zscore_thresh=0.5):
    from scipy import ndimage
    from skimage import morphology
    import cv2
    
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