"""Sparsity-continuum figure (paper Fig. 2): lifetime-sparsity skewness across a twin's population.

Panels (baseline-firing panels e,f are deferred until the gray-screen baseline is available):
* b: sorted response profiles for the most non-sparse and most sparse well-predicted neuron --
     model predictions over the screening set (gray) and recorded test responses (black), with the
     skewness annotated.
* c: model (screening) skewness vs recorded (test) skewness, per neuron, with Pearson r.
* d: population distribution of model skewness, with the 2.0 non-sparse/sparse threshold.

Two skewness measures, two regimes:
* MODEL skewness comes from the screening responses (``*_ensemble_imagenet_ordered_responses`` --
  the centered, RF-masked, L2-normed screening regime), via ``scipy.stats.skew`` per neuron. This is
  exactly what ``utils.sparse_split`` computes, so we reuse it.
* RECORDED skewness comes from the recordings loader: ``scipy.stats.skew`` over each neuron's mean
  (over-repeats) recorded test responses (observed images only).

Requires the area's recordings + its canonical SESSION_ORDER (V4 set; V1 pending its order).

    python -m dualneuron.figures.make_fig_sparsity --area v4 --backbone resnet
"""
import os
from pathlib import Path

import numpy as np
from scipy.stats import skew
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from dotenv import load_dotenv

from dualneuron.utils import ensure_dir
from dualneuron.twins import registry
from dualneuron.data.recordings import load_sessions, build_response_matrix
from dualneuron.figures.neuron_strips import ACCENT

REPO_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(REPO_ROOT / ".env")
SKEW_THRESHOLD = 2.0


def _despine(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def _profile(ax, values, color, label):
    """Sorted activation curve over a normalized rank axis [0, 1]."""
    v = np.sort(np.asarray(values, dtype=np.float64))
    ax.plot(np.linspace(0, 1, len(v)), v, color=color, linewidth=1.4, label=label)


def main(area, backbone):
    color = ACCENT[area]
    sp = registry.sparse_split(area, backbone)
    neurons = np.asarray(sp["neurons"])               # well-predicted global indices
    model_skew = np.asarray(sp["skewness"])           # screening skewness, aligned to `neurons`

    sessions = load_sessions(area)
    image_ids, recorded, _ = build_response_matrix(sessions, split="test")   # (n_img, N) NaN-sparse

    # recorded skewness per well-predicted neuron (observed test images only)
    rec_skew = np.full(len(neurons), np.nan)
    for k, n in enumerate(neurons):
        obs = recorded[~np.isnan(recorded[:, n]), n]
        if obs.size >= 5 and obs.std() > 0:
            rec_skew[k] = skew(obs)

    # screening ordered responses for the example profiles (sorted ascending, per neuron)
    ordered = np.load(registry.screening_path(area, backbone, "imagenet", "responses"))

    ns = int(neurons[np.argmin(model_skew)])          # most non-sparse
    sparse_n = int(neurons[np.argmax(model_skew)])    # most sparse
    examples = [("non-sparse", ns), ("sparse", sparse_n)]

    fig, ax = plt.subplots(2, 2, figsize=(8.4, 6.4))

    # panel b: example sorted profiles (model = screening, recorded = test)
    for col, (label, n) in enumerate(examples):
        a = ax[0, col]
        _profile(a, ordered[f"unit_{n}"], "0.55", "model (screening)")
        k = int(np.where(neurons == n)[0][0])
        _profile(a, recorded[~np.isnan(recorded[:, n]), n], "black", "recorded (test)")
        a.set_title(f"{area.upper()} {label} neuron {n}\nskew model={model_skew[k]:.2f}  rec={rec_skew[k]:.2f}", fontsize=9)
        a.set_xlabel("sorted image rank")
        a.set_ylabel("response")
        _despine(a)
        if col == 0:
            a.legend(frameon=False, fontsize=8)

    # panel c: model vs recorded skewness
    valid = ~np.isnan(rec_skew)
    r = np.corrcoef(model_skew[valid], rec_skew[valid])[0, 1]
    ax[1, 0].scatter(model_skew[valid], rec_skew[valid], s=12, color=color, alpha=0.6, edgecolors="none")
    ax[1, 0].set_xlabel("model skewness (screening)")
    ax[1, 0].set_ylabel("recorded skewness (test)")
    ax[1, 0].set_title(f"r = {r:.2f}  (n = {int(valid.sum())})", fontsize=9)
    _despine(ax[1, 0])

    # panel d: population distribution of model skewness + threshold
    ax[1, 1].hist(model_skew, bins=30, color=color, alpha=0.8)
    ax[1, 1].axvline(SKEW_THRESHOLD, ls=":", color="gray", linewidth=1.2)
    ax[1, 1].set_xlabel("model skewness")
    ax[1, 1].set_ylabel("# neurons")
    ax[1, 1].set_title(f"non-sparse < {SKEW_THRESHOLD} < sparse", fontsize=9)
    _despine(ax[1, 1])

    print(f"[stats] model-vs-recorded skewness r={r:.3f} (n={int(valid.sum())}); "
          f"non-sparse={int((model_skew < SKEW_THRESHOLD).sum())} sparse={int((model_skew >= SKEW_THRESHOLD).sum())}", flush=True)
    fig.tight_layout()
    out = registry.fig_path(area, backbone, "sparsity.pdf")
    ensure_dir(os.path.dirname(out))
    fig.savefig(out, dpi=300)
    plt.close(fig)
    print(f"saved {out}", flush=True)


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Sparsity-continuum figure (Fig 2) for one twin")
    p.add_argument("--area", required=True, choices=registry.AREAS)
    p.add_argument("--backbone", required=True, choices=registry.BACKBONES)
    args = p.parse_args()
    registry.check_pair(args.area, args.backbone, p)
    main(args.area, args.backbone)
