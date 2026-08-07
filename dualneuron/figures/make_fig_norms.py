"""A twin's computed L2 constraint against the training-stimulus norms that define it.

Reads what :mod:`dualneuron.screening.norms` measured and shows the constraint as what it is -- a
low-tail threshold on the energy a training stimulus carries *inside the twin's receptive field*:

* **a** -- the RF-masked distribution, with the computed constraint marked, and the unmasked
  full-frame distribution behind it for scale. The gap between the two is the whole reason the
  support matters: a synthesized stimulus concentrates its energy in the RF, so the full-frame
  distribution sits several times higher and would give a badly inflated constraint.
* **b** -- the masked distribution as an empirical CDF. **This is the panel to choose the percentile
  from**: read a percentile off the y-axis, get the norm it implies. The default 2.56 is a choice,
  not a law -- it is where V4's established 40 falls on its own masked distribution.

    python -m dualneuron.figures.make_fig_norms --area v4 --backbone staged
"""
import warnings
warnings.filterwarnings("ignore")

import os
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from dotenv import load_dotenv

from dualneuron.utils import ensure_dir
from dualneuron.twins import registry
from dualneuron.figures.neuron_strips import ACCENT

REPO_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(REPO_ROOT / ".env")

FULL_COLOR = "0.65"          # unmasked full-frame distribution, shown for scale
MARK_COLOR = "0.3"


def _despine(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def main(area, backbone, split="train"):
    path = registry.norms_path(area, backbone, split)
    if path is None or not os.path.exists(path):
        raise FileNotFoundError(
            f"Measured norms not found: {path}. Run them first: "
            f"python -m dualneuron.screening.norms --area {area} --backbone {backbone} "
            f"--split {split}")
    d = np.load(path)

    accent = ACCENT[area]
    masked, full = d["masked"], d["full"]
    norm, pct = float(d["norm"]), float(d["percentile"])
    spec = registry.resolve(area, backbone)

    fig, ax = plt.subplots(1, 2, figsize=(7.2, 3.0))

    # (a) the distribution the constraint thresholds, with the full-frame one behind it for scale
    hi = float(np.percentile(full, 99.5))
    hkw = dict(bins=60, range=(0.0, hi), histtype="step", density=True)
    ax[0].hist(full, color=FULL_COLOR, label=r"full frame $\|x\|_2$", **hkw)
    ax[0].hist(masked, color=accent, label=r"RF-masked $\|x\cdot m\|_2$", **hkw)
    ax[0].axvline(norm, color=MARK_COLOR, lw=1, ls="--")
    ax[0].annotate(f"p{pct:g} = {norm:.1f}", xy=(norm, 0.92), xycoords=("data", "axes fraction"),
                   xytext=(-5, 0), textcoords="offset points",
                   ha="right", va="top", fontsize=8, color=MARK_COLOR)
    ax[0].set_xlabel(r"$\|x\|_2$ of training stimulus")
    ax[0].set_ylabel("density")
    ax[0].set_title("a  norm distribution", loc="left", fontsize=10)
    ax[0].legend(frameon=False, fontsize=8)
    _despine(ax[0])

    # (b) cumulative -- read a percentile off the axis, get the norm it implies
    v = np.sort(masked)
    ax[1].plot(v, np.linspace(0, 100, len(v)), color=accent, linewidth=1.4)
    ax[1].axhline(pct, color="0.6", lw=0.8, ls=":")
    ax[1].axvline(norm, color=MARK_COLOR, lw=1, ls="--")
    ax[1].plot([norm], [pct], "o", color=MARK_COLOR, ms=4)
    ax[1].annotate(f"p{pct:g} = {norm:.1f}", xy=(norm, pct), xytext=(6, 8),
                   textcoords="offset points", fontsize=8, color=MARK_COLOR)
    ax[1].set_xlim(0, float(np.percentile(masked, 99.5)))
    ax[1].set_ylim(0, 100)
    ax[1].set_xlabel(r"RF-masked $\|x\cdot m\|_2$")
    ax[1].set_ylabel("percentile of training stimuli")
    ax[1].set_title("b  cumulative — pick the percentile here", loc="left", fontsize=10)
    _despine(ax[1])

    fig.suptitle(f"{area}/{backbone} — L2 constraint from {len(masked)} recorded {split} stimuli "
                 f"(literal: {spec.synth_target_norm:g})", fontsize=10)
    fig.tight_layout()
    out = registry.fig_path(area, backbone, *registry.rel_norms(split), "norms.pdf")
    ensure_dir(os.path.dirname(out))
    fig.savefig(out, dpi=300)
    plt.close(fig)
    print(f"saved {out}")
    return out


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(
        description="Figure: a twin's computed L2 constraint against its training-stimulus norms")
    p.add_argument("--area", type=str, required=True, choices=registry.AREAS)
    p.add_argument("--backbone", type=str, required=True, choices=registry.BACKBONES)
    p.add_argument("--split", type=str, default="train", choices=["train", "test"])
    args = p.parse_args()
    registry.check_pair(args.area, args.backbone, p)
    main(args.area, args.backbone, args.split)
