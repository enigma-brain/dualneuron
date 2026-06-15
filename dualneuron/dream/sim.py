import warnings
warnings.filterwarnings("ignore")
import os

from dotenv import load_dotenv
load_dotenv()

from pathlib import Path
from dualneuron.screening.sets import ImagenetImages, RenderedImages
import dualneuron

import numpy as np
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from dreamsim import dreamsim

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
rng = np.random.RandomState(123)

DATA_DIR = os.getenv("DATA_DIR")
MODELS_DIR = os.getenv("MODELS_DIR") or (
    os.path.join(DATA_DIR, "models") if DATA_DIR else "./models"
)


def gray_contrast_normalize(images, gray, norm, eps=1e-8):
    """
    Contrast-normalize a batch around a fixed gray, preserving a gray background.

    Each image's deviation from `gray` is rescaled to L2 norm `norm`, then clipped to
    [0, 1]. Background pixels (set to `gray` by masking) have zero deviation and so
    remain exactly `gray`. This keeps the masked background neutral and the whole
    image within DreamSim's expected [0, 1] input range, while equalizing contrast
    across images. DreamSim normalizes inputs internally per backbone, so a gray near
    the ImageNet/CLIP mean (~0.45) makes the masked background read ~0 to the model.

    Args:
        images (torch.Tensor): Batch of images, shape (N, C, H, W), values in [0, 1].
        gray (float): Neutral background value the deviation is taken around.
        norm (float): Target L2 norm of each image's deviation from gray.
        eps (float): Numerical stabilizer. Default: 1e-8.

    Returns:
        torch.Tensor: Contrast-normalized images (N, C, H, W), clipped to [0, 1].
    """
    dev = images - gray
    scale = norm / (dev.flatten(1).norm(dim=1) + eps)
    out = gray + dev * scale.view(-1, 1, 1, 1)
    return out.clamp(0.0, 1.0)


def embeddings(
    data_dir,
    cache_dir=None,
    output_path=None,
    token=None, 
    split='train', 
    dataset="rendered",
    area='v4',
    use_grayscale=False,
    use_mask=True,
    use_norm=True,
    norm=80.0,
    num_channels=3,
    crop_size=236,
    bg_value=0.45,
    batch_size=32,
    num_workers=0,
    indices=None,
):
    """
    Extract DreamSim embeddings from images with mask applied.
    
    Args:
        data_dir: Path to data directory
        cache_dir: Path to cache directory for the DreamSim weights. Defaults to
            MODELS_DIR (DATA_DIR/models).
        output_path: If provided, path to save an .npz with 'embeddings' and 'indices'
        token: HuggingFace token for ImageNet (if needed)
        split: Dataset split ('train', 'validation', 'test')
        dataset: 'rendered' or 'imagenet'
        area: 'v1' or 'v4' mask to use
        use_grayscale: Whether to convert images to grayscale
        use_mask: Whether to apply mask to images
        use_norm: Whether to contrast-normalize (deviation from gray to L2 'norm')
        norm: Target L2 norm of each image's deviation from the gray background
            (contrast control), applied when use_norm is True. Default: 80.0.
        num_channels: Number of image channels (1 or 3)
        crop_size: Crop size for images
        bg_value: Gray value for the masked background. DreamSim normalizes inputs
            internally per backbone, so a value near the ImageNet/CLIP mean (~0.45)
            makes the background read ~0 to the model. Default: 0.45.
        batch_size: Batch size for dataloader
        num_workers: Number of workers for dataloader
        indices: Optional array of dataset indices to embed (a subset, in the given
            order). If None, embeds the entire dataset. Useful for embedding a subset
            of ImageNet, where embedding all ~1.28M images is impractical.

    Returns:
        If output_path is None:
            embeddings (np.ndarray): (n_images, embedding_dim) DreamSim embeddings,
                in dataset order.
            indices (np.ndarray): (n_images,) dataset index for each embedding row.
        If output_path is provided:
            None: saves an .npz with arrays 'embeddings' and 'indices'.
    """
    assert dataset in ['rendered', 'imagenet']
    assert area in ['v1', 'v4']

    cache_dir = cache_dir or MODELS_DIR

    # Load DreamSim model
    model, _ = dreamsim(
        pretrained=True, 
        device=device, 
        cache_dir=cache_dir,
        dreamsim_type="ensemble",
    )
    model = model.eval()
    
    # Load mask
    package_dir = Path(dualneuron.__file__).parent
    model_name = "V4ColorTaskDriven" if area == 'v4' else "V1GrayTaskDriven"
    mask_path = package_dir / "twins" / model_name / "mask.npy"
    mask = np.load(mask_path)
    
    if dataset == "rendered":    
        dset = RenderedImages(
            data_dir=data_dir,
            use_center_crop=True,
            use_resize_output=True,
            use_grayscale=use_grayscale,
            use_normalize=False,
            use_mask=use_mask,
            use_norm=False,
            use_clip=False, 
            mask=mask,
            num_channels=num_channels,
            output_size=(224, 224),
            crop_size=crop_size,
            bg_value=bg_value,
            clip_min=0.0,
            clip_max=1.0,
            norm=norm,
        )
    else:
        dset = ImagenetImages(
            data_dir=data_dir,
            token=token,
            split=split,
            use_center_crop=True,
            use_resize_output=True,
            use_grayscale=use_grayscale,
            use_normalize=False,
            use_mask=use_mask,
            use_norm=False,
            use_clip=False,
            mask=mask,
            num_channels=num_channels,
            output_size=(224, 224),
            crop_size=crop_size,
            bg_value=bg_value,
            clip_min=0.0,
            clip_max=1.0,
            norm=norm,
        )
        
    # Optionally restrict to a subset of images (e.g. for ImageNet); otherwise embed all.
    if indices is not None:
        indices = np.asarray(indices)
        dset = torch.utils.data.Subset(dset, indices)
    else:
        indices = np.arange(len(dset))

    loader = DataLoader(
        dset,
        batch_size=batch_size,
        shuffle=False, 
        num_workers=num_workers,
        pin_memory=True
    )

    embeddings_list = []
    
    with torch.no_grad():
        for images, _ in tqdm(loader, total=len(loader)):
            images = images.to(device)
            if use_norm:
                images = gray_contrast_normalize(images, gray=bg_value, norm=norm)
            batch_embeddings = model.embed(images)
            embeddings_list.append(batch_embeddings.cpu().numpy())
            
    all_embeddings = np.concatenate(embeddings_list, axis=0)
    
    if output_path is not None:
        np.savez(output_path, embeddings=all_embeddings, indices=indices)
    else:
        return all_embeddings, indices


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Extract DreamSim embeddings")
    parser.add_argument("--data_dir", type=str, help="Where the data is saved")
    parser.add_argument("--cache_dir", type=str, default=MODELS_DIR, help="Where dreamsim models are cached")
    parser.add_argument("--output_path", type=str, default=None, help="Path to save embeddings .npz file")
    parser.add_argument("--token", type=str, default=None, help="Huggingface token for imagenet")
    parser.add_argument("--split", type=str, default="train", help="train, validation, or test for imagenet")
    parser.add_argument("--dataset", type=str, help="rendered or imagenet")
    parser.add_argument("--area", type=str, default="v4", help="v1 or v4 mask to use")
    parser.add_argument("--use_grayscale", type=bool, default=False, help="Use grayscale images")
    parser.add_argument("--use_mask", type=bool, default=True, help="Whether to use mask")
    parser.add_argument("--use_norm", type=bool, default=True, help="Whether to control norm")
    parser.add_argument("--norm", type=float, default=80.0, help="Target L2 norm of deviation from gray (contrast)")
    parser.add_argument("--num_channels", type=int, default=3, help="Number of image channels (1 or 3)")
    parser.add_argument("--crop_size", type=int, default=236, help="Crop size for images")
    parser.add_argument("--bg_value", type=float, default=0.45, help="Gray value for masked background (~0.45 reads ~0 to DreamSim)")
    parser.add_argument("--batch_size", type=int, default=32, help="batch size for dataloader")
    parser.add_argument("--num_workers", type=int, default=0, help="number of workers for dataloader")
    args = parser.parse_args()
    
    embeddings_array = embeddings(
        args.data_dir, args.cache_dir, 
        args.output_path, args.token, 
        args.split, args.dataset,
        args.area, args.use_grayscale, 
        args.use_mask, args.use_norm, 
        args.norm, args.num_channels, 
        args.crop_size, args.bg_value, 
        args.batch_size, args.num_workers
    )
    