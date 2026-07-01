import warnings
warnings.filterwarnings("ignore")
import os
import sys
import time

from dotenv import load_dotenv
load_dotenv()
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

from pathlib import Path
from dualneuron.screening.sets import ImagenetImages, RenderedImages
from dualneuron.twins.nets import load_model
from dualneuron.twins import registry
from dualneuron.twins.activations import get_layer_info, get_spatial_activation
from dualneuron.utils import ensure_dir, env_dir, RewriteLine

import numpy as np
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader

TOKEN = os.getenv("HF_TOKEN")
DATA_DIR = env_dir("DATA_DIR")
ANALYSIS_DIR = env_dir("ANALYSIS_DIR")
RENDERED_DIR = env_dir("RENDERED_DIR")
IMAGENET_CACHE_DIR = env_dir("IMAGENET_CACHE_DIR")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
rng = np.random.RandomState(123)


def _run_tag(ensemble, member):
    # Identifies a run in output filenames: a single ensemble member, the full
    # ensemble, or a single (non-ensemble) model.
    if member is not None:
        return f"member{member}"
    return "ensemble" if ensemble else "single"


def screen_activations(
    area,
    backbone,
    data_dir,
    output_dir=None,
    token=None,
    split='train',
    dataset="rendered",
    use_experiment_frame=True,
    field="masked",
    n_sample=None,
    subset_seed=0,
    layer=None,
    location='center',
    ensemble=True,
    member=None,
    weights_dir=None,
    batch_size=32,
    num_workers=0,
    save_format="npz",
    device="cuda",
    log_path=None,
    log_every=30.0,
):
    """
    Screen a dataset to compute neuron activations and sort images by response strength.
    
    This function loads a neural network model, processes images through it, and extracts
    activations for all neurons at a specified spatial location. Results are sorted by
    activation value for each neuron, enabling identification of most/least activating images.
    
    Args:
        area (str): Recorded area — 'v4' or 'v1'.
        backbone (str): Twin backbone — 'resnet'/'dino' (v4) or 'convnext'/'dino' (v1). The
            (area, backbone) pair resolves the twin, geometry, normalization, and RF mask via
            :mod:`dualneuron.twins.registry`.
        data_dir (str): Path to the directory containing the image dataset.
        output_dir (str, optional): Directory to save ordered responses and indices
            (default ANALYSIS_DIR/{area}/{backbone}). If None, returns results instead of saving.
        token (str, optional): HuggingFace token for accessing ImageNet dataset.
            Required when dataset="imagenet".
        split (str): Dataset split to use. One of 'train', 'validation', or 'test'.
            Default: 'train'.
        dataset (str): Type of dataset to load. Must be 'rendered' or 'imagenet'.
            Default: 'rendered'.
        use_experiment_frame (bool): Imagenet only. If True, build the 236x420
            experiment frame (resize short side to 420 + center-band crop) instead
            of Resize(256), matching how the rendered stimulus frames were made.
            Ignored for the rendered dataset. Default: True.
        field (str): 'masked' (default) applies the RF mask + L2 contrast norm (the MAI/LAI
            screening regime). 'full' is full-field natural — no mask, no L2 — keeping only the
            twin's z-score + crop/resize; it needs no mask, so it can run before any MEI/mask exists
            (the population-axis regime). Output files carry a 'fullfield' tag in this case.
        n_sample (int, optional): Screen a uniform random subset of this many images (e.g. 200000 for
            the full-field ImageNet axis) instead of the whole dataset. Ordered indices are the global
            image ids of the sampled subset. Default: None (screen all).
        subset_seed (int): RNG seed for --n_sample; fixed so every twin screens the same images.
        layer (str, optional): Layer name for activation extraction. If None, uses
            the model's final output layer. Default: None.
        location (str): Spatial location for activation extraction. Typically 'center'
            to extract activations from the center of the spatial map. Default: 'center'.
        ensemble (bool): Whether to use ensemble averaging across multiple model
            instances. Default: True.
        member (int, optional): Screen a single ensemble member by index instead of
            the ensemble average; output files are tagged member{member}. Default:
            None (use the ensemble).
        weights_dir (str, optional): Trained-ensemble directory. If None (default), the
            staged weights for resnet/convnext, or TRAINED_MODELS_DIR/{area}/{backbone} for dino.
        batch_size (int): Number of images per batch for processing. Default: 32.
        num_workers (int): Number of worker processes for data loading. Default: 0.
        save_format (str): Output format when output_dir is provided. 'npz' writes the
            files {tag}_{dataset}_ordered_responses.npz and ..._indices.npz,
            each keyed by unit_{neuron_id}; 'npy' writes per-neuron .npy files in two
            folders. Default: 'npz'.
        device (str): Device to run computations on ('cuda' or 'cpu'). Default: 'cuda'.
        log_path (str, optional): If provided, path to a progress log file. Its parent
            folder is created on demand, and a one-line header and footer bracket a
            single self-updating progress line. If None, progress prints to stderr.
            Default: None.
        log_every (float): Minimum seconds between progress-line updates (tqdm
            mininterval), so the log keeps one rewritten line rather than one per
            batch. Default: 30.0.

    Returns:
        If output_dir is None:
            tuple: (sorted_responses, sorted_indices)
                - sorted_responses (np.ndarray): Shape (num_images, num_neurons).
                  Activation values sorted in ascending order for each neuron.
                - sorted_indices (np.ndarray): Shape (num_images, num_neurons).
                  Dataset indices corresponding to sorted responses.
        
        If output_dir is provided:
            None: Results are saved to disk. With save_format='npz' (default):
                - {output_dir}/{tag}_{dataset}_ordered_responses.npz
                - {output_dir}/{tag}_{dataset}_ordered_indices.npz
              both keyed by unit_{neuron_id}, each a per-neuron array sorted ascending.
              With save_format='npy', per-neuron .npy files are written instead:
                - {output_dir}/{tag}_{layer}_{dataset}_ordered_responses/{neuron_id}.npy
                - {output_dir}/{tag}_{layer}_{dataset}_ordered_indices/{neuron_id}.npy

    Notes:
        - Images are preprocessed with center cropping, resizing, normalization,
          and masking according to the twin's registry geometry.
        - Each twin uses its own receptive-field mask (dualneuron.twins.registry.mask_path):
          the staged mask for a shipped twin, the regenerated one for a trained twin.
        - Results are sorted in ascending order, so the last indices correspond
          to the most strongly activating images (highest pole), and the first
          indices correspond to the least activating images (lowest pole).
    """

    assert dataset in ['rendered', 'imagenet']

    spec = registry.resolve(area, backbone)
    tag = _run_tag(ensemble, member)

    function = load_model(
        architecture=spec.arch,
        layer=layer,
        ensemble=ensemble or member is not None,
        centered=True,
        weights_dir=weights_dir,
        device=device
    )
    if member is not None:
        function = function.members[member]

    # Per-twin screening geometry + normalization, all from the central registry (single source of
    # truth). img_mean/img_std are the twin's training z-score stats (0-255 scale); norm is the L2
    # contrast target; grayscale/crop/output_size/num_channels match the twin's input. These are the
    # established per-run constants (e.g. v4/resnet: 100px, crop 200, L2 40, 113.5/59.58; v1/convnext:
    # 93px, crop 167, L2 12, 124.54/70.28) — screening a twin means exactly what it did before.
    output_size = (spec.input_size, spec.input_size)
    norm = spec.screen_norm
    num_channels = spec.channels
    grayscale = spec.channels == 1
    crop_size = spec.crop_size
    img_mean, img_std = spec.img_mean, spec.img_std

    # "masked" screening applies the RF mask + L2 contrast norm (the MAI/LAI regime). "full" screening
    # is full-field natural (no mask, no L2 — only the twin's z-score + crop/resize), so it needs no
    # mask and can run first, before any MEI/mask exists.
    masked = field == "masked"
    mask = np.load(registry.mask_path(area, backbone)) if masked else None

    if dataset == "rendered":
        dset = RenderedImages(
            data_dir=data_dir,
            use_center_crop=True,
            use_resize_output=True,
            use_grayscale=grayscale,
            use_normalize=True,
            use_mask=masked,
            use_crop_to_mask=False,
            use_norm=masked,
            use_clip=False,
            mask=mask,
            num_channels=num_channels,
            output_size=output_size,
            crop_size=crop_size,
            bg_value=0.0,
            clip_min=0.0,
            clip_max=1.0,
            crop_padding_frac=0.05,
            norm=norm,
            img_mean=img_mean,
            img_std=img_std,
        )
    else:
        dset = ImagenetImages(
            data_dir=data_dir,
            token=token,
            split=split,
            use_center_crop=True,
            use_resize_output=True,
            use_grayscale=grayscale,
            use_normalize=True,
            use_mask=masked,
            use_crop_to_mask=False,
            use_norm=masked,
            use_clip=False,
            use_experiment_frame=use_experiment_frame,
            mask=mask,
            num_channels=num_channels,
            output_size=output_size,
            crop_size=crop_size,
            bg_value=0.0,
            crop_padding_frac=0.05,
            clip_min=0.0,
            clip_max=1.0,
            norm=norm,
            img_mean=img_mean,
            img_std=img_std,
        )

    # Optional uniform subset (e.g. 200k ImageNet for the full-field population axis); a fixed seed
    # means every twin screens the same images. Sorted indices are remapped to global ids after sorting.
    subset_indices = None
    if n_sample is not None and n_sample < len(dset):
        rng_sub = np.random.RandomState(subset_seed)
        subset_indices = np.sort(rng_sub.choice(len(dset), size=n_sample, replace=False))
        dset = torch.utils.data.Subset(dset, subset_indices)

    loader = DataLoader(
        dset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    
    _, neurons = get_layer_info(
        function, 
        (1, num_channels, *output_size), 
        device=device
    )
    neurons = list(range(neurons))
    bsize = loader.batch_size
    outs = torch.zeros(len(loader.dataset), len(neurons))
    
    # Optional progress log: a single line rewritten in place (clean in any editor),
    # bracketed by a header and footer. Created on demand. Without log_path, progress
    # goes to stderr as a normal tqdm bar.
    log_file = None
    progress_file = sys.stderr
    if log_path is not None:
        ensure_dir(Path(log_path).parent)
        log_file = open(log_path, "w")
        log_file.write(
            f"screen {area}/{backbone} dataset={dataset} images={len(loader.dataset)} "
            f"neurons={len(neurons)} batch_size={batch_size}\n"
        )
        log_file.flush()
        progress_file = RewriteLine(log_file, log_file.tell())

    start = time.time()
    with torch.no_grad():
        for i, (scenes, _) in enumerate(tqdm(
            loader,
            total=len(loader),
            file=progress_file,
            mininterval=log_every,
            ncols=100,
            desc=f"screen {area}/{backbone} {dataset}",
        )):
            output = function(scenes.to(device))
            output = get_spatial_activation(
                output,
                neurons=neurons,
                location=location
            )
            outs[i*bsize:i*bsize+len(scenes)] = output.cpu().detach()
    elapsed = time.time() - start

    resps, idx = torch.sort(outs, dim=0)
    sresps = resps.numpy()
    sidx = idx.numpy()
    if subset_indices is not None:                 # subset positions -> global image ids
        sidx = subset_indices[sidx]

    def _finish(msg):
        if log_file is not None:
            log_file.write(
                f"done: screened {sresps.shape[0]} images x {len(neurons)} "
                f"neurons in {elapsed:.0f}s{msg}\n"
            )
            log_file.flush()
            log_file.close()

    if output_dir is None:
        _finish(" (returned, not saved)")
        return sresps, sidx

    ensure_dir(output_dir)
    regime = "" if masked else "fullfield_"   # full-field files carry the regime, masked keep the name

    if save_format == "npz":
        # One npz per (responses | indices), keyed by unit_{neuron_id}, each value a per-neuron array
        # sorted ascending. Bare, folder-namespaced under ANALYSIS_DIR/{area}/{backbone}/
        # (e.g. ensemble_rendered_ordered_responses.npz, ensemble_imagenet_fullfield_ordered_*.npz).
        resp_path = os.path.join(output_dir, f"{tag}_{dataset}_{regime}ordered_responses.npz")
        idx_path = os.path.join(output_dir, f"{tag}_{dataset}_{regime}ordered_indices.npz")
        np.savez(resp_path, **{f"unit_{unit}": sresps[:, i] for i, unit in enumerate(neurons)})
        np.savez(idx_path, **{f"unit_{unit}": sidx[:, i] for i, unit in enumerate(neurons)})
        print(f"saved {resp_path}")
        print(f"saved {idx_path}")
        _finish(f" -> {resp_path} | {idx_path}")
        return

    elif save_format == "npy":
        # Per-neuron .npy files in two folders ({tag}_{layer}_{dataset}_ordered_*).
        layer_name = layer if layer is not None else "output"
        endrespdir = os.path.join(output_dir, f"{tag}_{layer_name}_{dataset}_{regime}ordered_responses")
        idxdir = os.path.join(output_dir, f"{tag}_{layer_name}_{dataset}_{regime}ordered_indices")
        ensure_dir(endrespdir)
        ensure_dir(idxdir)
        for i, unit in tqdm(enumerate(neurons), total=len(neurons)):
            np.save(os.path.join(endrespdir, f"{str(unit)}.npy"), sresps[:, i])
            np.save(os.path.join(idxdir, f"{str(unit)}.npy"), sidx[:, i])
        _finish(f" -> {endrespdir}/ | {idxdir}/")
        return

    else:
        raise ValueError(f"Unknown save_format: {save_format}. Use 'npz' or 'npy'.")

        
if __name__ == "__main__":
    import argparse
    if DATA_DIR is None:
        raise ValueError(
            "DATA_DIR is not set. Copy .env.example to .env and set "
            "DATA_DIR=/path/to/your/data (ImageNet caches to DATA_DIR/datasets)."
        )
    parser = argparse.ArgumentParser(description="Get DERS")
    parser.add_argument("--area", type=str, required=True, choices=registry.AREAS, help="recorded area")
    parser.add_argument("--backbone", type=str, required=True, choices=registry.BACKBONES, help="twin backbone")
    parser.add_argument("--weights_dir", type=str, default=None, help="trained-ensemble dir (default: staged for resnet/convnext; TRAINED_MODELS_DIR/{area}/{backbone} for dino)")
    parser.add_argument("--data_dir", type=str, default=None, help="Image source (default RENDERED_DIR for rendered, IMAGENET_CACHE_DIR for imagenet)")
    parser.add_argument("--output_dir", type=str, default=None, help="Where ordered responses/indices are saved (default ANALYSIS_DIR/{area}/{backbone})")
    parser.add_argument("--token", type=str, default=TOKEN, help="Huggingface token for imagenet")
    parser.add_argument("--split", type=str, default="train", help="train, validation, or test")
    parser.add_argument("--dataset", type=str, default="imagenet", help="rendered or imagenet")
    parser.add_argument("--field", type=str, default="masked", choices=["masked", "full"],
                        help="'masked' (RF mask + L2 norm, MAI/LAI regime) or 'full' (full-field natural: no mask, no L2 -> needs no mask, runs first)")
    parser.add_argument("--n_sample", type=int, default=None,
                        help="uniform image subset to screen (e.g. 200000 for the full-field ImageNet axis); default: all")
    parser.add_argument("--subset_seed", type=int, default=0, help="seed for --n_sample (fixed -> same images across twins)")
    parser.add_argument("--use_experiment_frame", action=argparse.BooleanOptionalAction, default=True, help="imagenet only: build the 236x420 experiment frame instead of Resize(256)")
    parser.add_argument("--layer", type=str, default=None, help="layer name, if none use final layer")
    parser.add_argument("--location", default="center", help="spatial location for activation extraction")
    parser.add_argument("--ensemble", type=bool, default=True, help="use ensemble model")
    parser.add_argument("--member", type=int, default=None, help="screen a single ensemble member by index (e.g. 0); default uses the ensemble")
    parser.add_argument("--batch_size", type=int, default=64, help="batch size for dataloader")
    parser.add_argument("--num_workers", type=int, default=4, help="dataloader workers (>=4: the JPEG decode is the bottleneck, so 0 starves the GPU ~4x slower)")
    parser.add_argument("--save_format", type=str, default="npz", help="npz (Dryad format) or npy (per-neuron folders)")
    parser.add_argument("--device", type=str, default="cuda", help="device to run on")
    parser.add_argument("--log_path", type=str, default=None, help="Progress log file (default LOGS_DIR/{area}/{backbone}/screening_{tag}_{dataset}.log)")
    parser.add_argument("--log_every", type=float, default=30.0, help="Min seconds between progress-line updates")
    args = parser.parse_args()

    # Default output location: ANALYSIS_DIR/{area}/{backbone}, created on demand.
    output_dir = args.output_dir
    if output_dir is None:
        if ANALYSIS_DIR is None:
            raise ValueError(
                "ANALYSIS_DIR is not set. Set it in .env (e.g. "
                "ANALYSIS_DIR=${DATA_DIR}/DUAL-FEATURE-ANALYSIS) or pass --output_dir."
            )
        output_dir = registry.analysis_dir(args.area, args.backbone)

    # Default image source by dataset (RENDERED_DIR / IMAGENET_CACHE_DIR).
    data_dir = args.data_dir
    if data_dir is None:
        data_dir = RENDERED_DIR if args.dataset == "rendered" else IMAGENET_CACHE_DIR
        if data_dir is None:
            raise ValueError(
                "No image source. Set RENDERED_DIR / IMAGENET_CACHE_DIR in .env "
                "or pass --data_dir."
            )

    # Default progress log under LOGS_DIR, created on demand.
    log_path = args.log_path
    if log_path is None:
        logs_dir = os.getenv("LOGS_DIR")
        if logs_dir is not None:
            tag = _run_tag(args.ensemble, args.member)
            log_path = os.path.join(logs_dir, args.area, args.backbone,
                                    f"screening_{args.field}_{tag}_{args.dataset}.log")

    screen_activations(
        area=args.area,
        backbone=args.backbone,
        data_dir=data_dir,
        output_dir=output_dir,
        token=args.token,
        split=args.split,
        dataset=args.dataset,
        use_experiment_frame=args.use_experiment_frame,
        field=args.field,
        n_sample=args.n_sample,
        subset_seed=args.subset_seed,
        layer=args.layer,
        location=args.location,
        ensemble=args.ensemble,
        member=args.member,
        weights_dir=args.weights_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        save_format=args.save_format,
        device=args.device,
        log_path=log_path,
        log_every=args.log_every,
    )