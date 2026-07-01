"""Population figure (paper Fig. 10 / Fig_Population): most- and least-activating images of one
neuron tend to strongly or weakly activate other neurons, revealing shared feature selectivity.

For each (non-sparse) source neuron we take its top-N most-activating (MAI) and bottom-N
least-activating (LAI) screening images, plus N random images, and compute their response percentile
within every *other* non-sparse neuron's response distribution over the 1.2M ImageNet screening set
(rank in that neuron's sorted responses). Per source we histogram those percentiles (10 bins); the
panels show the population mean +/- 99% CI (2.58*SEM), following single_neuron_selectivity's
analyze_neuron_relationships / plot_ci.

* within-population: MAIs are right-skewed (drive other neurons strongly), LAIs bimodal, random uniform.
* cross-animal: the same, but MAIs/LAIs from one monkey scored on the other monkey's neurons.

Everything is from the SCREENING responses (centered=True, RF-masked, L2-normed regime) -- the MAI/LAI
regime -- so no model forwarding here; the sparse/non-sparse split is the screening-skewness split.
N=10 matches the reference code (the paper text states 15; the distributions are robust to this).

Requires the area's recordings (subject_id per neuron) + its canonical SESSION_ORDER + ImageNet
screening (V4 set; V1 pending its order).

    python -m dualneuron.figures.make_fig_population --area v4 --backbone resnet
"""
import os
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from dotenv import load_dotenv

from dualneuron.utils import env_dir, ensure_dir
from dualneuron.twins import registry
from dualneuron.data.recordings import load_sessions, build_response_matrix

REPO_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(REPO_ROOT / ".env")
FIGS = env_dir("PAPER_FIG_DIR", str(REPO_ROOT / "figs"))
N_EXTREME = 10           # reference analyze_neuron_relationships uses 10 (paper text says 15)
N_BINS = 10
COND = {"MAI": "#c0392b", "LAI": "#2f6db0", "random": "0.6"}


def _despine(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def _percentile_distributions(idx_by_neuron, sources, targets, n_images, n=N_EXTREME, seed=0):
    """Per-source INDIVIDUAL image percentiles across targets, for top-n / bottom-n / random-n.

    idx_by_neuron[g] = image indices sorted ascending (worst->best) for neuron g. For each source we
    collect the percentile of every one of its top-n / bottom-n / random-n images within every other
    target neuron's distribution (not averaged -- matching the reference's neuron_*_percentiles, so
    the random control is uniform). Returns three lists (MAI/LAI/random), one percentile array/source.
    """
    rng = np.random.default_rng(seed)
    top = {s: idx_by_neuron[s][-n:] for s in sources}
    bot = {s: idx_by_neuron[s][:n] for s in sources}
    rnd = {s: rng.choice(n_images, n, replace=False) for s in sources}
    out = {"MAI": {s: [] for s in sources}, "LAI": {s: [] for s in sources}, "random": {s: [] for s in sources}}
    for t in targets:
        rank = np.empty(n_images, dtype=np.float32)
        rank[idx_by_neuron[t]] = np.arange(n_images, dtype=np.float32) / n_images * 100.0
        for s in sources:
            if s == t:
                continue
            out["MAI"][s].extend(rank[top[s]].tolist())
            out["LAI"][s].extend(rank[bot[s]].tolist())
            out["random"][s].extend(rank[rnd[s]].tolist())
    return {c: [np.asarray(out[c][s]) for s in sources if out[c][s]] for c in out}


def _ci_hist(per_source):
    """Per-source normalized histograms (N_BINS over 0-100) -> (mean, 99% CI half-width) per bin."""
    H = np.zeros((len(per_source), N_BINS))
    for i, d in enumerate(per_source):
        w = np.ones_like(d) / len(d)
        H[i] = np.histogram(d, bins=N_BINS, range=(0, 100), weights=w)[0]
    mean = H.mean(0)
    ci = 2.58 * H.std(0, ddof=1) / np.sqrt(H.shape[0])
    return mean, ci


def _panel(ax, dists, title):
    x = np.arange(N_BINS)
    for c in ("MAI", "LAI", "random"):
        mean, ci = _ci_hist(dists[c])
        ax.bar(x + (0.27 * (-1 if c == "LAI" else 1 if c == "MAI" else 0)), mean, width=0.26,
               yerr=ci, color=COND[c], alpha=0.8, capsize=2,
               error_kw={"linewidth": 0.8}, label=c)
    ax.set_xticks([0, N_BINS / 2 - 0.5, N_BINS - 1])
    ax.set_xticklabels([0, 50, 100])
    ax.set_xlabel("response percentile in other neurons")
    ax.set_ylabel("probability")
    ax.set_title(title, fontsize=10)
    _despine(ax)


def main(area, backbone):
    sp = registry.sparse_split(area, backbone)
    nonsparse = [int(n) for n in np.asarray(sp["non_sparse"])]

    # subject id per global neuron (for the cross-animal split)
    _, _, meta = build_response_matrix(load_sessions(area), split="test")
    subject = {m["global_idx"]: m["subject_id"] for m in meta}

    # screening (imagenet) sorted indices for the non-sparse neurons
    oi = np.load(registry.screening_path(area, backbone, "ensemble", "imagenet", "indices"))
    idx_by_neuron = {g: oi[f"unit_{g}"] for g in nonsparse}
    n_images = len(next(iter(idx_by_neuron.values())))
    print(f"[info] non-sparse neurons={len(nonsparse)}  images={n_images}  subjects={sorted(set(subject.values()))}", flush=True)

    within = _percentile_distributions(idx_by_neuron, nonsparse, nonsparse, n_images)

    subj_vals = sorted(set(subject[g] for g in nonsparse))
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.6))
    _panel(ax[0], within, f"{area.upper()} within-population (n={len(nonsparse)} non-sparse)")
    ax[0].legend(frameon=False, fontsize=8)

    if len(subj_vals) == 2:
        a, b = subj_vals
        src = [g for g in nonsparse if subject[g] == a]
        tgt = [g for g in nonsparse if subject[g] == b]
        cross = _percentile_distributions(idx_by_neuron, src, tgt, n_images)
        _panel(ax[1], cross, f"cross-animal: monkey {a} -> monkey {b}\n(src={len(src)}, tgt={len(tgt)})")
    for c in ("MAI", "LAI", "random"):
        m, _ = _ci_hist(within[c])
        print(f"[stats] within {c}: bin0(low%)={m[0]:.3f} bin9(high%)={m[-1]:.3f}", flush=True)

    fig.tight_layout()
    out = os.path.join(ensure_dir(os.path.join(FIGS, area, backbone)), "fig_population.pdf")
    fig.savefig(out, dpi=300)
    plt.close(fig)
    print(f"saved {out}", flush=True)


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Population shared-selectivity figure (Fig 10) for one twin")
    p.add_argument("--area", required=True, choices=registry.AREAS)
    p.add_argument("--backbone", required=True, choices=registry.BACKBONES)
    args = p.parse_args()
    main(args.area, args.backbone)
