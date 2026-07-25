"""Recorded-response verification figure (paper Fig. 7 / Fig_3_Verify_Data): for each neuron we
take the test image the model predicts to be most- and least-activating, and show where those
images fall within the neuron's *recorded* response distribution over the test set -- separately for
non-sparse and sparse neurons.

For non-sparse neurons the model captures both extremes (predicted-most -> high recorded percentile,
predicted-least -> low recorded percentile). For sparse neurons only the most-activating end is
predicted well; the least-activating percentiles are ~uniform (the lower tail has little dynamic
range). A non-selective ordering would yield a uniform distribution.

Predictions use the same eval pipeline as the accuracy figure -- learned readout positions
(``centered=False``) and the training transform (no RF mask / L2 norm) -- because we are predicting
recorded responses to the actual test stimuli. Recorded responses come from dualneuron.data.recordings;
the sparse/non-sparse split is the screening-skewness split (threshold 2.0).

Requires the area's recordings + its canonical SESSION_ORDER (V4 set; V1 pending its order).

    python -m dualneuron.figures.make_fig_verify_data --area v4 --backbone resnet
"""
import os
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from dotenv import load_dotenv

from dualneuron.utils import ensure_dir
from dualneuron.twins import registry
from dualneuron.data.recordings import load_sessions, build_response_matrix
from dualneuron.figures.make_fig_accuracy import _predict  # shared eval pipeline (centered=False)

REPO_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(REPO_ROOT / ".env")
POLE = {"most": "#c0392b", "least": "#2f6db0"}   # predicted most- / least-activating


def _despine(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def _percentile(values, x):
    """Percentile rank (0-100) of value x within `values` (fraction <= x)."""
    return 100.0 * np.mean(values <= x)


def main(area, backbone, weights_dir=None):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    sessions = load_sessions(area)
    image_ids, recorded, _ = build_response_matrix(sessions, split="test")   # (n_img, N) NaN-sparse
    preds = _predict(area, backbone, image_ids, device, weights_dir=weights_dir)   # (n_img, N)

    split = registry.sparse_split(area, backbone)       # well-predicted neurons split by skewness 2.0
    groups = {"non-sparse": np.asarray(split["non_sparse"]), "sparse": np.asarray(split["sparse"])}

    # per-neuron recorded percentile of the predicted most- and least-activating test image
    pct = {g: {"most": [], "least": []} for g in groups}
    for g, neurons in groups.items():
        for j in neurons:
            m = ~np.isnan(recorded[:, j])
            if m.sum() < 5 or preds[m, j].std() == 0:
                continue
            p = preds[m, j]
            r = recorded[m, j]
            pct[g]["most"].append(_percentile(r, r[int(np.argmax(p))]))
            pct[g]["least"].append(_percentile(r, r[int(np.argmin(p))]))

    # figure: non-sparse (top) and sparse (bottom); predicted-most (red) vs predicted-least (blue)
    fig, ax = plt.subplots(2, 1, figsize=(4.0, 5.2), sharex=True)
    for row, g in enumerate(("non-sparse", "sparse")):
        for end, color in (("least", POLE["least"]), ("most", POLE["most"])):
            ax[row].hist(pct[g][end], range=(0, 100), bins=20, histtype="step",
                         color=color, linewidth=1.5, label=f"predicted {end}")
        ax[row].set_title(f"{area.upper()} {g} (n={len(pct[g]['most'])})", fontsize=10)
        ax[row].set_ylabel("# neurons")
        _despine(ax[row])
        med = {e: np.median(pct[g][e]) for e in ("most", "least")}
        print(f"[stats] {g}: median percentile  most={med['most']:.1f}  least={med['least']:.1f}", flush=True)
    ax[0].legend(frameon=False, fontsize=8)
    ax[1].set_xlabel("recorded response percentile")
    fig.tight_layout()
    out = registry.fig_path(area, backbone, "verify_data.pdf")
    ensure_dir(os.path.dirname(out))
    fig.savefig(out, dpi=300)
    plt.close(fig)
    print(f"saved {out}", flush=True)


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Recorded-response verification figure (Fig 7) for one twin")
    p.add_argument("--area", required=True, choices=registry.AREAS)
    p.add_argument("--backbone", required=True, choices=registry.BACKBONES)
    p.add_argument("--weights_dir", default=None)
    args = p.parse_args()
    registry.check_pair(args.area, args.backbone, p)
    main(args.area, args.backbone, args.weights_dir)
