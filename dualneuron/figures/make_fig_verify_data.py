"""Recorded-response verification figure (paper Fig. 7 / Fig_3_Verify_Data, V4): for each neuron we
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

V4 only -- there are no V1 recordings in this dataset.

    python -m dualneuron.figures.make_fig_verify_data
"""
import os
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from dotenv import load_dotenv

from dualneuron.utils import env_dir, ensure_dir, sparse_split
from dualneuron.data.recordings import load_sessions, build_response_matrix
from dualneuron.figures.make_fig_accuracy import _predict  # shared eval pipeline (centered=False)

REPO_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(REPO_ROOT / ".env")
FIGS = env_dir("PAPER_FIG_DIR", str(REPO_ROOT / "figs"))
POLE = {"most": "#c0392b", "least": "#2f6db0"}   # predicted most- / least-activating


def _despine(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def _percentile(values, x):
    """Percentile rank (0-100) of value x within `values` (fraction <= x)."""
    return 100.0 * np.mean(values <= x)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    sessions = load_sessions()
    image_ids, recorded, _ = build_response_matrix(sessions, split="test")   # (n_img, 394) NaN-sparse
    preds = _predict(image_ids, device)                                      # (n_img, 394)

    split = sparse_split("v4")                          # well-predicted neurons split by skewness 2.0
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
        ax[row].set_title(f"V4 {g} (n={len(pct[g]['most'])})", fontsize=10)
        ax[row].set_ylabel("# neurons")
        _despine(ax[row])
        med = {e: np.median(pct[g][e]) for e in ("most", "least")}
        print(f"[stats] {g}: median percentile  most={med['most']:.1f}  least={med['least']:.1f}", flush=True)
    ax[0].legend(frameon=False, fontsize=8)
    ax[1].set_xlabel("recorded response percentile")
    fig.tight_layout()
    out = os.path.join(ensure_dir(FIGS), "fig_verify_data_v4.pdf")
    fig.savefig(out, dpi=300)
    plt.close(fig)
    print(f"saved {out}", flush=True)


if __name__ == "__main__":
    main()
