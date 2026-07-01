"""
One figure per area showing the synthesized LEIs and MEIs of four neurons.

For the same neurons as neuron_strips (4, 5, the most non-sparse and the most sparse
well-predicted neuron of the area, ordered top->bottom by ascending skewness), each block
shows the 10 seeds of the least-exciting input (LEI) over the 10 seeds of the most-exciting
input (MEI). Each seed is its synthesized image blended with its alpha (the RF envelope the
optimization settled on) over a gray background via synthesis.visualize.blend -- the same
recipe used in the Deis notebook.

    python -m dualneuron.figures.make_fig_mei_lei --area v4 --backbone resnet
    python -m dualneuron.figures.make_fig_mei_lei --area v1 --backbone convnext
"""
import os
import argparse
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from dotenv import load_dotenv

from dualneuron.utils import ensure_dir, env_dir
from dualneuron.twins import registry
from dualneuron.synthesis.visualize import blend
from dualneuron.figures.neuron_strips import select_neurons, ACCENT

REPO_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(REPO_ROOT / ".env")
FIGS = env_dir("PAPER_FIG_DIR", str(REPO_ROOT / "figs"))

POLE = {"LEI": "#2f6db0", "MEI": "#c0392b"}   # least- / most-exciting accents
N_SEEDS = 10
BLEND = dict(alphacut=90.0, boost=1.0, bg_value=0.4)   # as in the Deis notebook


def _seed_image(image, alpha, channels):
    """Blend one seed's synthesized image with its alpha; return (H,W) or (H,W,3)."""
    b = blend(image, alpha, **BLEND)              # (H, W, C) in [0,1]
    return b[..., 0] if channels == 1 else b


def figure(area, backbone, neurons, variant="free"):
    """One figure for `neurons` (list of (id, skewness), ascending), saved as PDF.

    `variant` selects the synthesis method to display ("free" or "axis")."""
    spec = registry.resolve(area, backbone)
    cmap = "gray" if spec.channels == 1 else None
    accent = ACCENT[area]
    imshow_kw = {} if cmap is None else dict(vmin=0, vmax=1)

    nb = len(neurons)
    fig = plt.figure(figsize=(13, 1.9 * nb + 0.7))
    gs = fig.add_gridspec(nb, 1, hspace=0.5, left=0.11, right=0.95, top=0.99, bottom=0.06)

    for bi, (nid, sk) in enumerate(neurons):
        z = np.load(registry.synthesis_neuron_path(area, backbone, nid, variant=variant))
        rows = [("LEI", z["lei_image"], z["lei_alpha"]),
                ("MEI", z["mei_image"], z["mei_alpha"])]
        inner = gs[bi].subgridspec(2, N_SEEDS, hspace=0.05, wspace=0.05)
        for r, (label, imgs, alphas) in enumerate(rows):
            for c in range(N_SEEDS):
                ax = fig.add_subplot(inner[r, c])
                ax.imshow(_seed_image(imgs[c], alphas[c], spec.channels), cmap=cmap, **imshow_kw)
                ax.set_xticks([])
                ax.set_yticks([])
                for spine in ax.spines.values():
                    spine.set_visible(False)
                if c == 0:
                    ax.set_ylabel(label, rotation=0, ha="right", va="center", labelpad=10,
                                  color=POLE[label], fontsize=11, fontweight="bold")
        pos = gs[bi].get_position(fig)
        fig.text(0.05, 0.5 * (pos.y0 + pos.y1), f"n{nid}\nskew {sk:.2f}",
                 ha="center", va="center", fontsize=10, color=accent, fontweight="bold")

    fig.text(0.5 * (0.11 + 0.95), 0.02, "synthesis seeds  1 → 10",
             ha="center", va="center", fontsize=10, color="0.3")

    stem = "mei_lei_seeds" if variant == "free" else f"mei_lei_seeds_{variant}"
    out = os.path.join(ensure_dir(os.path.join(FIGS, area, backbone)), f"{stem}.pdf")
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"{area}/{backbone} ({variant}): neurons {[n for n, _ in neurons]} -> {out}", flush=True)
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Synthesized LEI/MEI seed strips per neuron")
    parser.add_argument("--area", required=True, choices=registry.AREAS)
    parser.add_argument("--backbone", required=True, choices=registry.BACKBONES)
    parser.add_argument("--neurons", type=int, nargs="+", default=None,
                        help="override the neuron set (still ordered by skewness)")
    parser.add_argument("--variant", default="free", choices=registry.SYNTHESIS_VARIANTS,
                        help="synthesis method to display: 'free' (default) or 'axis'")
    args = parser.parse_args()

    if args.neurons:
        sp = registry.sparse_split(args.area, args.backbone)
        skew = {int(n): float(s) for n, s in zip(sp["neurons"], sp["skewness"])}
        neurons = sorted(((n, skew[n]) for n in args.neurons if n in skew), key=lambda t: t[1])
    else:
        neurons = select_neurons(args.area, args.backbone)
    figure(args.area, args.backbone, neurons, variant=args.variant)
