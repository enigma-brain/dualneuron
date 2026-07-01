import warnings
warnings.filterwarnings("ignore")
import os
import sys
import time

from dotenv import load_dotenv
load_dotenv()

from pathlib import Path
from dualneuron.screening.sets import ImagenetImages, RenderedImages
from dualneuron.twins import registry
from dualneuron.utils import ensure_dir, env_dir, RewriteLine

import numpy as np
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from dreamsim import dreamsim

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
rng = np.random.RandomState(123)

DATA_DIR = env_dir("DATA_DIR")
MODELS_DIR = env_dir("MODELS_DIR", os.path.join(DATA_DIR, "models") if DATA_DIR else "./models")
ANALYSIS_DIR = env_dir("ANALYSIS_DIR")
RENDERED_DIR = env_dir("RENDERED_DIR")
IMAGENET_CACHE_DIR = env_dir("IMAGENET_CACHE_DIR")


# DreamSim preprocessing constants, independent of the twin backbone: DreamSim's RGB backbones take
# 224px 3-channel input, contrast-normalized to a fixed L2 around a neutral gray. The twin-dependent
# bits come from the registry: the RF mask, the center-crop (area RF, so the mask aligns with what
# the neuron read), and whether to grayscale (V1, repeated to 3 channels for DreamSim).
_DREAMSIM = {"output_size": (224, 224), "num_channels": 3, "norm": 80.0}


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
    area,
    backbone,
    cache_dir=None,
    output_path=None,
    token=None,
    split='train',
    dataset="rendered",
    use_grayscale=None,
    use_mask=True,
    use_norm=True,
    norm=None,
    num_channels=None,
    crop_size=None,
    bg_value=0.45,
    batch_size=32,
    num_workers=0,
    log_path=None,
    log_every=30.0,
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
        area: 'v1' or 'v4'
        backbone: twin backbone ('resnet'/'dino' for v4, 'convnext'/'dino' for v1); with area it
            selects the RF mask, center-crop, and grayscale via the registry
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
        log_path: If provided, path to a progress log file. Its parent folder is
            created on demand, and a one-line header and footer bracket a single
            self-updating progress line. If None, progress prints to stderr.
            Default: None.
        log_every: Minimum seconds between progress-line updates (tqdm mininterval),
            so the log keeps one rewritten line rather than one line per batch.
            Default: 30.0.
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
    spec = registry.resolve(area, backbone)

    # Twin-dependent preprocessing from the registry (crop matched to the screening RF; grayscale for
    # V1, repeated to 3 channels for DreamSim); the rest are DreamSim constants.
    if crop_size is None:
        crop_size = spec.crop_size
    if use_grayscale is None:
        use_grayscale = spec.channels == 1
    if num_channels is None:
        num_channels = _DREAMSIM["num_channels"]
    if norm is None:
        norm = _DREAMSIM["norm"]

    cache_dir = cache_dir or MODELS_DIR

    # Load DreamSim model
    model, _ = dreamsim(
        pretrained=True, 
        device=device, 
        cache_dir=cache_dir,
        dreamsim_type="ensemble",
    )
    model = model.eval()
    
    # Load the twin's RF mask (staged for a shipped twin, regenerated for a trained one).
    mask = np.load(registry.mask_path(area, backbone))

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

    # Optional progress log: a single line rewritten in place (clean in any editor),
    # bracketed by a header and footer. Created on demand. Without log_path, progress
    # goes to stderr as a normal tqdm bar.
    log_file = None
    progress_file = sys.stderr
    if log_path is not None:
        ensure_dir(Path(log_path).parent)
        log_file = open(log_path, "w")
        log_file.write(
            f"embed dataset={dataset} area={area} backbone={backbone} images={len(indices)} "
            f"batch_size={batch_size} "
            f"norm={norm if use_norm else 'off'} bg={bg_value}\n"
        )
        log_file.flush()
        progress_file = RewriteLine(log_file, log_file.tell())

    start = time.time()
    embeddings_list = []
    with torch.no_grad():
        for images, _ in tqdm(
            loader,
            total=len(loader),
            file=progress_file,
            mininterval=log_every,
            ncols=100,
            desc=f"embed {dataset} {area}/{backbone}",
        ):
            images = images.to(device)
            if use_norm:
                images = gray_contrast_normalize(images, gray=bg_value, norm=norm)
            batch_embeddings = model.embed(images)
            embeddings_list.append(batch_embeddings.cpu().numpy())

    all_embeddings = np.concatenate(embeddings_list, axis=0)
    elapsed = time.time() - start

    if output_path is not None:
        ensure_dir(Path(output_path).parent)
        # float16 halves the file; embeddings are unit-norm so cosine sims (computed
        # in fp32 from these) are unaffected at the precision the analyses need.
        np.savez(output_path, embeddings=all_embeddings.astype(np.float16), indices=indices)
    if log_file is not None:
        tail = f" -> {output_path}" if output_path is not None else ""
        log_file.write(f"done: embeddings {all_embeddings.shape} in {elapsed:.0f}s{tail}\n")
        log_file.flush()
        log_file.close()

    if output_path is None:
        return all_embeddings, indices


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Extract DreamSim embeddings")
    parser.add_argument("--data_dir", type=str, default=None, help="Image source (default RENDERED_DIR for rendered, IMAGENET_CACHE_DIR for imagenet)")
    parser.add_argument("--cache_dir", type=str, default=MODELS_DIR, help="Where dreamsim models are cached")
    parser.add_argument("--output_path", type=str, default=None, help="Path to save embeddings .npz file")
    parser.add_argument("--token", type=str, default=None, help="Huggingface token for imagenet")
    parser.add_argument("--split", type=str, default="train", help="train, validation, or test for imagenet")
    parser.add_argument("--dataset", type=str, help="rendered or imagenet")
    parser.add_argument("--area", type=str, required=True, choices=registry.AREAS, help="v1 or v4")
    parser.add_argument("--backbone", type=str, required=True, choices=registry.BACKBONES, help="twin backbone")
    parser.add_argument("--use_grayscale", type=bool, default=None, help="Grayscale (default: per-area; V1 True, V4 False)")
    parser.add_argument("--use_mask", type=bool, default=True, help="Whether to use mask")
    parser.add_argument("--use_norm", type=bool, default=True, help="Whether to control norm")
    parser.add_argument("--norm", type=float, default=None, help="Contrast L2 of deviation from gray (default: per-area, 80)")
    parser.add_argument("--num_channels", type=int, default=None, help="Image channels (default: per-area, 3)")
    parser.add_argument("--crop_size", type=int, default=None, help="Crop matched to screening (default: per-area; V4 200, V1 167)")
    parser.add_argument("--bg_value", type=float, default=0.45, help="Gray value for masked background (~0.45 reads ~0 to DreamSim)")
    parser.add_argument("--batch_size", type=int, default=32, help="batch size for dataloader")
    parser.add_argument("--num_workers", type=int, default=0, help="number of workers for dataloader")
    parser.add_argument("--log_path", type=str, default=None, help="Progress log file (default LOGS_DIR/{area}_dreamsim_{dataset}.log)")
    parser.add_argument("--log_every", type=float, default=30.0, help="Min seconds between progress-line updates")
    parser.add_argument("--indices_path", type=str, default=None, help="Path to a .npy of dataset indices to embed (e.g. the imagenet subset); default embeds all")
    args = parser.parse_args()

    # Subset to embed: an explicit index file (e.g. the imagenet subset), else all images.
    indices = np.load(args.indices_path) if args.indices_path is not None else None

    # Default image source by dataset (RENDERED_DIR / IMAGENET_CACHE_DIR).
    data_dir = args.data_dir
    if data_dir is None:
        data_dir = RENDERED_DIR if args.dataset == "rendered" else IMAGENET_CACHE_DIR
        if data_dir is None:
            raise ValueError(
                "No image source. Set RENDERED_DIR / IMAGENET_CACHE_DIR in .env "
                "or pass --data_dir."
            )

    # Default outputs under ANALYSIS_DIR/{area}/{backbone} and LOGS_DIR, created on demand.
    output_path = args.output_path
    if output_path is None:
        if ANALYSIS_DIR is None:
            raise ValueError(
                "ANALYSIS_DIR is not set. Set it in .env (e.g. "
                "ANALYSIS_DIR=${DATA_DIR}/DUAL-FEATURE-ANALYSIS) or pass --output_path."
            )
        output_path = registry.dreamsim_embeddings_path(args.area, args.backbone, args.dataset)

    log_path = args.log_path
    if log_path is None:
        logs_dir = os.getenv("LOGS_DIR")
        if logs_dir is not None:
            log_path = os.path.join(logs_dir, args.area, args.backbone, f"dreamsim_{args.dataset}.log")

    embeddings(
        data_dir=data_dir,
        cache_dir=args.cache_dir,
        output_path=output_path,
        token=args.token,
        split=args.split,
        dataset=args.dataset,
        area=args.area,
        backbone=args.backbone,
        use_grayscale=args.use_grayscale,
        use_mask=args.use_mask,
        use_norm=args.use_norm,
        norm=args.norm,
        num_channels=args.num_channels,
        crop_size=args.crop_size,
        bg_value=args.bg_value,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        log_path=log_path,
        log_every=args.log_every,
        indices=indices,
    )
    