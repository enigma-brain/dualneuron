"""Receptive-field mask estimated from the twin's input gradients -- no synthesis required.

:mod:`dualneuron.synthesis.mask` builds a twin's RF mask from its synthesized MEIs/LEIs. That makes
the mask a *product* of synthesis, which is circular for anything synthesis itself needs: the L2
constraint should be measured over the region the stimulus occupies, but that region is only known
once the stimuli exist.

The receptive field, though, is a property of the trained model. It can be read off directly as the
sensitivity of each neuron's response to each input pixel -- one backward pass per neuron, averaged
over natural stimuli because the model is nonlinear and a single input gives a noisy estimate:

    map = mean_n mean_images | d r_n / d x |

thresholded and softened by exactly the recipe :func:`dualneuron.synthesis.mask.build_rf_mask` uses,
so the two are the same kind of object built two ways and cannot drift apart.

Measured on ``v4/staged`` (32 stimuli, 205 well-predicted neurons): correlation **0.9945** against
the shipped mask, in ~6 s versus the 12-17 h of synthesis the alpha-derived mask costs. The estimate
is also insensitive to input scale -- rescaling the same stimuli to the synthesis energy budget
moves the correlation only to 0.9902 -- which is what lets it run before the norm is known.

The model is loaded CENTERED, matching how MEIs are synthesized and how the shipped mask was built.

This mask defines the support for :mod:`dualneuron.screening.norms`. It does NOT feed
:func:`dualneuron.twins.registry.mask_path`, so screening and DreamSim keep reading the
synthesis-derived mask exactly as before.

    python -m dualneuron.synthesis.rf --area v4 --backbone staged
"""
import warnings
warnings.filterwarnings("ignore")

import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

import numpy as np
import torch
from tqdm import tqdm

from dualneuron.data.recordings import load_sessions, build_response_matrix
from dualneuron.training.dataset import ImageResponseDataset, training_transform
from dualneuron.twins import registry
from dualneuron.twins.nets import load_model
from dualneuron.utils import ensure_dir, env_dir, RewriteLine, should_compute

LOGS_DIR = env_dir("LOGS_DIR")

#: Defaults shared with :mod:`dualneuron.synthesis.mask`, so both masks are thresholded alike.
PERCENTILE, SIGMA = 77.5, 1.3


def sensitivity_map(area, backbone, n_images=32, seed=0, weights_dir=None, device="cuda",
                    progress=None, mininterval=10.0):
    """Mean ``|d r_n / d x|`` over a twin's well-predicted neurons and a sample of its stimuli.

    Args:
        area, backbone: The twin. Geometry and neuron selection resolve from the registry.
        n_images: Recorded training stimuli to average the gradient over. Default: 32.
        seed: RNG seed selecting those stimuli, so the estimate is reproducible.
        weights_dir: Ensemble weights; default the registry's choice for this twin.
        device: Torch device.
        progress: File-like sink for the tqdm bar (a :class:`RewriteLine`); None -> stderr.
        mininterval: Minimum seconds between progress-line updates.

    Returns:
        ``(H, W)`` float64 sensitivity map, averaged over neurons and images.
    """
    spec = registry.resolve(area, backbone)
    neurons = registry.well_predicted_neurons(area, backbone, weights_dir=weights_dir)
    model = load_model(architecture=spec.arch, ensemble=True, centered=True,
                       weights_dir=weights_dir or registry.weights_dir(area, backbone),
                       device=device).eval()

    image_ids, responses, _ = build_response_matrix(load_sessions(area), "train")
    pick = np.sort(np.random.RandomState(seed).choice(len(image_ids), size=min(n_images,
                                                       len(image_ids)), replace=False))
    dset = ImageResponseDataset(image_ids[pick], responses[pick],
                                os.path.join(env_dir("EXPERIMENT_DIR"), area, "images"),
                                transform=training_transform(area, backbone),
                                channels=spec.channels)
    x = torch.stack([dset[i][0] for i in range(len(dset))]).to(device).requires_grad_(True)

    out = model(x)
    acc = torch.zeros(x.shape[-2], x.shape[-1], device=device)
    for nid in tqdm(neurons, desc=f"rf {area}/{backbone}", file=progress or sys.stderr,
                    mininterval=mininterval):
        g, = torch.autograd.grad(out[:, int(nid)].sum(), x, retain_graph=True)
        acc += g.abs().mean(0).sum(0)          # mean over images, sum over channels
    return (acc / len(neurons)).detach().cpu().numpy().astype(np.float64), len(neurons)


def build_rf_mask(area, backbone, percentile=PERCENTILE, sigma=SIGMA, n_images=32, seed=0,
                  weights_dir=None, device="cuda", progress=None, mininterval=10.0):
    """Gradient RF mask: the sensitivity map thresholded at ``percentile`` and Gaussian-softened.

    Identical post-processing to :func:`dualneuron.synthesis.mask.build_rf_mask`, so the gradient
    and alpha masks are directly comparable.

    Returns:
        ``(mask, n_neurons, sensitivity_map)``.
    """
    from scipy.ndimage import gaussian_filter
    smap, n = sensitivity_map(area, backbone, n_images=n_images, seed=seed,
                              weights_dir=weights_dir, device=device, progress=progress,
                              mininterval=mininterval)
    binary = (smap > np.percentile(smap, percentile)).astype(np.float64)
    return gaussian_filter(binary, sigma), n, smap


def compare_to_synthesis_mask(mask, area, backbone):
    """Compare against whatever mask :func:`registry.mask_path` currently resolves to, if any.

    A self-check where ground truth exists (a shipped or already-regenerated mask); ``None`` when
    the twin has neither, which is exactly the case this module is built to serve.
    """
    p = registry.mask_path(area, backbone)
    if p is None or not os.path.exists(p):
        return None
    other = np.load(p)
    if other.shape != mask.shape:
        return None
    a, b = mask.ravel() - mask.mean(), other.ravel() - other.mean()
    return {
        "path": p,
        "corr": float((a * b).sum() / (np.sqrt((a * a).sum() * (b * b).sum()) + 1e-12)),
        "mse": float(((mask - other) ** 2).mean()),
        "max_abs": float(np.abs(mask - other).max()),
    }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="Estimate a twin's RF mask from input gradients (no synthesis needed)")
    parser.add_argument("--area", type=str, required=True, choices=registry.AREAS)
    parser.add_argument("--backbone", type=str, required=True, choices=registry.BACKBONES)
    parser.add_argument("--n_images", type=int, default=32,
                        help="recorded stimuli to average the gradient over (default: 32)")
    parser.add_argument("--seed", type=int, default=0, help="seed selecting those stimuli")
    parser.add_argument("--percentile", type=float, default=PERCENTILE,
                        help=f"threshold percentile of the sensitivity map (default: {PERCENTILE}, "
                             f"matching synthesis.mask)")
    parser.add_argument("--sigma", type=float, default=SIGMA,
                        help=f"Gaussian edge-smoothing width in pixels (default: {SIGMA})")
    parser.add_argument("--weights_dir", type=str, default=None,
                        help="trained-ensemble dir (default: the registry's choice for this twin)")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--rewrite", action="store_true",
                        help="rebuild + overwrite even if the mask exists")
    parser.add_argument("--log_path", type=str, default=None,
                        help="progress log (default LOGS_DIR/{area}/{backbone}/rf/rf.log)")
    parser.add_argument("--log_every", type=float, default=10.0,
                        help="min seconds between progress-line updates")
    args = parser.parse_args()
    registry.check_pair(args.area, args.backbone, parser)

    output = registry.rf_mask_path(args.area, args.backbone)
    if output is None:
        raise ValueError("ANALYSIS_DIR is not set. Set it in .env (e.g. "
                         "ANALYSIS_DIR=${DATA_DIR}/DUAL-FEATURE-ANALYSIS).")
    if not should_compute(output, args.rewrite):
        print(f"cached (use --rewrite to rebuild): {output}")
        raise SystemExit(0)

    log_path = args.log_path
    if log_path is None and LOGS_DIR is not None:
        log_path = registry.log_path(args.area, args.backbone, *registry.rel_rf(), "rf.log")

    header = (f"rf {args.area}/{args.backbone} n_images={args.n_images} seed={args.seed} "
              f"percentile={args.percentile} sigma={args.sigma} centered=True")
    log_file = None
    progress = None
    if log_path is not None:
        ensure_dir(Path(log_path).parent)
        log_file = open(log_path, "w")
        log_file.write(header + "\n")
        log_file.flush()
        progress = RewriteLine(log_file, log_file.tell())

    start = time.time()
    mask, n, smap = build_rf_mask(
        args.area, args.backbone, percentile=args.percentile, sigma=args.sigma,
        n_images=args.n_images, seed=args.seed, weights_dir=args.weights_dir, device=args.device,
        progress=progress, mininterval=args.log_every)
    elapsed = time.time() - start

    cmp = compare_to_synthesis_mask(mask, args.area, args.backbone)
    summary = (f"rf mask {mask.shape} from {n} neurons x {args.n_images} stimuli in {elapsed:.0f}s"
               + (f" | vs {os.path.basename(os.path.dirname(cmp['path']))} mask: "
                  f"corr={cmp['corr']:.4f} mse={cmp['mse']:.5f} max|diff|={cmp['max_abs']:.3f}"
                  if cmp else " | no existing mask to compare against"))
    print(summary)

    ensure_dir(Path(output).parent)
    np.save(output, mask)
    np.save(os.path.join(os.path.dirname(output), "sensitivity.npy"), smap)
    print(f"saved {output}")
    if log_file is not None:
        log_file.write(f"\n{summary} -> {output}\n")
        log_file.close()
