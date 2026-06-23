"""
Build an area's receptive-field (RF) mask from its synthesized MEIs/LEIs.

The mask is the mean alpha (the per-pixel envelope each synthesis settles on) over every
MEI and LEI of the area's neurons, thresholded to the RF core and given a Gaussian-softened
boundary:

    average alpha over all MEIs/LEIs  ->  keep the top (100 - percentile)% pixels (the RF
    core)  ->  Gaussian-smooth the binary edge.

This reproduces the shipped twins/{model}/mask.npy (correlation > 0.99 for both areas) and
is the shared RF input to screening and DreamSim, so each evaluates exactly the retinotopic
region its neuron drives.

    python -m dualneuron.synthesis.mask --area v4
"""
import warnings
warnings.filterwarnings("ignore")
import os
import sys
import glob
import time
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()
import numpy as np
from scipy.ndimage import gaussian_filter
from tqdm import tqdm

import dualneuron
from dualneuron.utils import env_dir, ensure_dir, RewriteLine

ANALYSIS_DIR = env_dir("ANALYSIS_DIR")
TWINS_DIR = os.path.join(os.path.dirname(dualneuron.__file__), "twins")

# Analysis area -> twin model folder name.
AREAS = {"v1": "V1GrayTaskDriven", "v4": "V4ColorTaskDriven"}


def average_alpha(area, synthesis_dir=None, progress=None, mininterval=10.0):
    """
    Mean synthesis alpha over every MEI and LEI of an area (channels averaged).

    Args:
        area (str): "v1" or "v4".
        synthesis_dir (str, optional): Folder of per-neuron synthesis npz. Default:
            ANALYSIS_DIR/{area}/synthesis.
        progress (file-like, optional): If given, a tqdm bar is written to it.
        mininterval (float): Min seconds between progress updates. Default: 10.0.

    Returns:
        (np.ndarray, int): the (H, W) mean alpha and the number of neurons averaged.

    Raises:
        FileNotFoundError: If no synthesis npz are present.
    """
    if synthesis_dir is None:
        synthesis_dir = os.path.join(ANALYSIS_DIR, area, "synthesis")
    files = sorted(glob.glob(os.path.join(synthesis_dir, f"{area}_neuron*.npz")))
    if not files:
        raise FileNotFoundError(
            f"No synthesis npz in {synthesis_dir}. Run the synthesis first: "
            f"python -m dualneuron.synthesis.generate --area {area}"
        )
    iterator = files if progress is None else tqdm(
        files, file=progress, mininterval=mininterval, ncols=100, desc=f"avg alpha {area}")
    acc = None
    n = 0
    for f in iterator:
        z = np.load(f)
        for key in ("mei_alpha", "lei_alpha"):
            a = z[key].mean(axis=1)            # (seeds, H, W); average over channels
            acc = a.sum(0) if acc is None else acc + a.sum(0)
            n += a.shape[0]
    return acc / n, len(files)


def build_rf_mask(area, percentile=77.5, sigma=1.3, synthesis_dir=None,
                  progress=None, mininterval=10.0):
    """
    Receptive-field mask: Gaussian-smoothed binary of the thresholded average alpha.

    The average alpha over all MEIs/LEIs is thresholded at its `percentile` (keeping the
    most-attended ~RF-coverage fraction of pixels as the core), then the binary core's edge
    is softened with a Gaussian of width `sigma`. Defaults reproduce the shipped masks.

    Args:
        area (str): "v1" or "v4".
        percentile (float): Threshold percentile of the average alpha. Default: 77.5.
        sigma (float): Gaussian edge-smoothing width, in pixels. Default: 1.3.
        synthesis_dir (str, optional): Folder of per-neuron synthesis npz.
        progress (file-like, optional): tqdm sink for progress.
        mininterval (float): Min seconds between progress updates. Default: 10.0.

    Returns:
        (np.ndarray, int): the (H, W) float64 mask in [0, 1] and the number of neurons used.
    """
    alpha, n = average_alpha(area, synthesis_dir, progress, mininterval)
    binary = (alpha > np.percentile(alpha, percentile)).astype(np.float64)
    return gaussian_filter(binary, sigma), n


def compare_to_shipped(mask, area):
    """
    Compare a computed mask to the shipped twins/{model}/mask.npy.

    Args:
        mask (np.ndarray): Computed (H, W) mask.
        area (str): "v1" or "v4".

    Returns:
        dict or None: {"corr", "mse", "max_abs"} versus the shipped mask, or None if no
            shipped mask of matching shape exists.
    """
    ref = os.path.join(TWINS_DIR, AREAS[area], "mask.npy")
    if not os.path.exists(ref):
        return None
    M = np.load(ref)
    if M.shape != mask.shape:
        return None
    return {
        "corr": float(np.corrcoef(mask.ravel(), M.ravel())[0, 1]),
        "mse": float(np.mean((mask - M) ** 2)),
        "max_abs": float(np.max(np.abs(mask - M))),
    }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="Build an area's RF mask from its synthesized MEIs/LEIs")
    parser.add_argument("--area", type=str, required=True, choices=["v1", "v4"])
    parser.add_argument("--percentile", type=float, default=77.5,
                        help="threshold percentile of the average alpha (default: 77.5)")
    parser.add_argument("--sigma", type=float, default=1.3,
                        help="Gaussian edge-smoothing width in pixels (default: 1.3)")
    parser.add_argument("--synthesis_dir", type=str, default=None,
                        help="synthesis npz folder (default ANALYSIS_DIR/{area}/synthesis)")
    parser.add_argument("--output", type=str, default=None,
                        help="output .npy (default dualneuron/twins/{model}/mask.npy)")
    parser.add_argument("--log_path", type=str, default=None,
                        help="progress log file (default LOGS_DIR/{area}_mask.log)")
    parser.add_argument("--log_every", type=float, default=10.0,
                        help="min seconds between progress-line updates")
    args = parser.parse_args()

    output = args.output or os.path.join(TWINS_DIR, AREAS[args.area], "mask.npy")
    LOGS_DIR = env_dir("LOGS_DIR")
    log_path = args.log_path
    if log_path is None and LOGS_DIR is not None:
        log_path = os.path.join(LOGS_DIR, f"{args.area}_mask.log")

    log_file = None
    progress = None
    if log_path is not None:
        ensure_dir(Path(log_path).parent)
        log_file = open(log_path, "w")
        log_file.write(
            f"build mask area={args.area} percentile={args.percentile} sigma={args.sigma}\n")
        log_file.flush()
        progress = RewriteLine(log_file, log_file.tell())

    start = time.time()
    mask, n = build_rf_mask(
        args.area, percentile=args.percentile, sigma=args.sigma,
        synthesis_dir=args.synthesis_dir, progress=progress, mininterval=args.log_every,
    )
    elapsed = time.time() - start

    cmp = compare_to_shipped(mask, args.area)
    summary = (f"mask {mask.shape} from {n} neurons in {elapsed:.0f}s"
               + (f" | vs shipped: corr={cmp['corr']:.4f} mse={cmp['mse']:.5f} "
                  f"max|diff|={cmp['max_abs']:.3f}" if cmp else ""))
    print(summary)

    ensure_dir(Path(output).parent)
    np.save(output, mask)
    print(f"saved {output}")
    if log_file is not None:
        log_file.write(summary + f"\nsaved {output}\n")
        log_file.close()
