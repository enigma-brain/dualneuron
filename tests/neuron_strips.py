"""
One figure per area (and dataset) showing four neurons spanning the sparsity range.

Neurons: 4, 5, the most non-sparse (lowest skewness) and the most sparse (highest
skewness) well-predicted neuron of the area, ordered top->bottom by ascending skewness.
Each neuron is a 2-row block: the 10 least-activating images (LAI) and the 10 most-
activating (MAI), ordered low->high activation, receptive-field masked at bg=0.5
(screening geometry, no z-score / no L2 so the natural image shows). The response
range of each row is annotated at its right edge.

    python tests/neuron_strips.py                 # all four (v4/v1 x rendered/imagenet)
    python tests/neuron_strips.py --area v4 --dataset rendered
"""
import os
import argparse
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from dotenv import load_dotenv

import dualneuron
from dualneuron.screening.sets import ImagenetImages, RenderedImages
from dualneuron.utils import ensure_dir, env_dir, sparse_split

REPO_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(REPO_ROOT / ".env")

ANALYSIS_DIR = env_dir("ANALYSIS_DIR")
RENDERED_DIR = env_dir("RENDERED_DIR")
IMAGENET_CACHE_DIR = env_dir("IMAGENET_CACHE_DIR")
FIGS = env_dir("PAPER_FIG_DIR", str(REPO_ROOT / "figs"))

# Per-area display geometry (crop matches the screening so the RF mask aligns) + accent.
AREAS = {
    "v4": dict(model_name="V4ColorTaskDriven", channels=3, output_size=(100, 100),
               crop_size=200, grayscale=False, cmap=None, accent="#2c6fbb"),
    "v1": dict(model_name="V1GrayTaskDriven", channels=1, output_size=(93, 93),
               crop_size=167, grayscale=True, cmap="gray", accent="#e08a1e"),
}
POLE_COLOR = {"LAI": "#2f6db0", "MAI": "#c0392b"}


def _build_dataset(area, dataset):
    cfg = AREAS[area]
    mask = np.load(Path(dualneuron.__file__).parent / "twins" / cfg["model_name"] / "mask.npy")
    common = dict(
        use_center_crop=True, use_resize_output=True, use_grayscale=cfg["grayscale"],
        use_normalize=False, use_mask=True, use_crop_to_mask=False, use_norm=False,
        use_clip=False, mask=mask, num_channels=cfg["channels"],
        output_size=cfg["output_size"], crop_size=cfg["crop_size"], bg_value=0.5,
    )
    if dataset == "imagenet":
        return ImagenetImages(data_dir=IMAGENET_CACHE_DIR, split="train",
                              use_experiment_frame=True, **common)
    return RenderedImages(data_dir=RENDERED_DIR, **common)


def _show(ds, i, channels):
    tensor, _ = ds[int(i)]                          # (C, H, W) in [0,1], masked at bg=0.5
    a = np.clip(tensor.detach().cpu().numpy(), 0, 1)
    return a[0] if channels == 1 else np.transpose(a, (1, 2, 0))


def select_neurons(area):
    """[(neuron, skewness)] for {4, 5, most non-sparse, most sparse}, ascending skewness."""
    sp = sparse_split(area)
    skew = {int(n): float(s) for n, s in zip(sp["neurons"], sp["skewness"])}
    ids = [4, 5, int(sp["neurons"][np.argmin(sp["skewness"])]),
           int(sp["neurons"][np.argmax(sp["skewness"])])]
    seen, kept = set(), []
    for n in ids:
        if n in skew and n not in seen:
            seen.add(n)
            kept.append((n, skew[n]))
    return sorted(kept, key=lambda t: t[1])


def figure(area, dataset, neurons):
    """One figure for `neurons` (list of (id, skewness), ascending), saved as PDF."""
    cfg = AREAS[area]
    ds = _build_dataset(area, dataset)
    base = os.path.join(ANALYSIS_DIR, area)
    idx = np.load(os.path.join(base, f"{area}_ensemble_{dataset}_ordered_indices.npz"))
    resp = np.load(os.path.join(base, f"{area}_ensemble_{dataset}_ordered_responses.npz"))
    imshow_kw = {} if cfg["cmap"] is None else dict(vmin=0, vmax=1)

    nb = len(neurons)
    fig = plt.figure(figsize=(13, 1.9 * nb + 0.7))
    gs = fig.add_gridspec(nb, 1, hspace=0.5, left=0.11, right=0.92, top=0.99, bottom=0.08)

    for bi, (nid, sk) in enumerate(neurons):
        oi, orr = idx[f"unit_{nid}"], resp[f"unit_{nid}"]
        rows = [("LAI", oi[:10], orr[:10]), ("MAI", oi[-10:], orr[-10:])]
        inner = gs[bi].subgridspec(2, 10, hspace=0.05, wspace=0.05)
        for r, (label, idxs, resps) in enumerate(rows):
            ax = None
            for c in range(10):
                ax = fig.add_subplot(inner[r, c])
                ax.imshow(_show(ds, idxs[c], cfg["channels"]), cmap=cfg["cmap"], **imshow_kw)
                ax.set_xticks([])
                ax.set_yticks([])
                for spine in ax.spines.values():
                    spine.set_visible(False)
                if c == 0:
                    ax.set_ylabel(label, rotation=0, ha="right", va="center", labelpad=10,
                                  color=POLE_COLOR[label], fontsize=11, fontweight="bold")
            lo, hi = float(resps.min()), float(resps.max())
            ax.text(1.05, 0.5, f"{lo:.2f}–{hi:.2f}", transform=ax.transAxes,
                    ha="left", va="center", fontsize=9, color="0.35")
        pos = gs[bi].get_position(fig)
        fig.text(0.05, 0.5 * (pos.y0 + pos.y1), f"n{nid}\nskew {sk:.2f}",
                 ha="center", va="center", fontsize=10, color=cfg["accent"], fontweight="bold")

    fig.text(0.5 * (0.11 + 0.92), 0.025, "weaker  ←  activation  →  stronger",
             ha="center", va="center", fontsize=10, color="0.3")

    out_dir = ensure_dir(FIGS)
    path = os.path.join(out_dir, f"{area}_{dataset}_neuron_strips.pdf")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"{area} {dataset}: neurons {[n for n, _ in neurons]} -> {path}", flush=True)
    return path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Per-area neuron-strip figure across the sparsity range")
    parser.add_argument("--area", choices=["v4", "v1"], default=None, help="default: both")
    parser.add_argument("--dataset", choices=["rendered", "imagenet"], default=None, help="default: both")
    parser.add_argument("--neurons", type=int, nargs="+", default=None,
                        help="override the neuron set (still ordered by skewness)")
    args = parser.parse_args()

    areas = [args.area] if args.area else ["v4", "v1"]
    datasets = [args.dataset] if args.dataset else ["rendered", "imagenet"]
    for area in areas:
        if args.neurons:
            sp = sparse_split(area)
            skew = {int(n): float(s) for n, s in zip(sp["neurons"], sp["skewness"])}
            neurons = sorted(((n, skew[n]) for n in args.neurons if n in skew), key=lambda t: t[1])
        else:
            neurons = select_neurons(area)
        for dataset in datasets:
            figure(area, dataset, neurons)
