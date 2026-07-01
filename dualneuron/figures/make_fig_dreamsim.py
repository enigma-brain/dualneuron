"""
DreamSim similarity figures: Fig 6 (MAI/LAI coherence d') and Fig 9 (2D similarity
space R^2 + controls), for V4 and V1 on the rendered and imagenet image sets.

Aggregate panels are read from the saved similarity_{dataset}.npz under
ANALYSIS_DIR/{area}/{backbone}/ (produced by dualneuron.dream.similarity); the per-neuron example
panels are computed on the fly from the DreamSim embeddings via similarity_space_neuron /
coherence_pooled. One twin per run (per-twin figures); PDFs are written to PAPER_FIG_DIR.

    python -m dualneuron.figures.make_fig_dreamsim --area v4 --backbone resnet   # both datasets
    python -m dualneuron.figures.make_fig_dreamsim --area v1 --backbone convnext --dataset rendered
"""
import os
import argparse
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from dotenv import load_dotenv

from dualneuron.utils import env_dir, ensure_dir
from dualneuron.twins import registry
from dualneuron.dream.similarity import similarity_space_neuron, coherence_pooled

REPO_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(REPO_ROOT / ".env")
FIGS = env_dir("PAPER_FIG_DIR", str(REPO_ROOT / "figs"))

AREA = {"v4": dict(color="#2c6fbb", label="V4"), "v1": dict(color="#e08a1e", label="V1")}
POLE = {"mai": "#c0392b", "lai": "#2f6db0"}   # most- / least-activating accents


def _despine(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def _panel(ax, letter):
    ax.text(-0.2, 1.06, letter, transform=ax.transAxes, fontsize=13,
            fontweight="bold", va="top", ha="right")


def _example_neurons(area, backbone):
    """The most non-sparse (min skewness) and most sparse (max skewness) well-predicted
    neurons of a twin -- the same pair used in the neuron_strips figures. Skewness is
    from the imagenet screening (sparse_split), so the examples are identical across the
    rendered and imagenet figures."""
    sp = registry.sparse_split(area, backbone)
    return (int(sp["neurons"][np.argmin(sp["skewness"])]),
            int(sp["neurons"][np.argmax(sp["skewness"])]))


def _results(area, backbone, dataset):
    return np.load(registry.similarity_path(area, backbone, dataset))


def _embeddings(area, backbone, dataset):
    z = np.load(registry.dreamsim_embeddings_path(area, backbone, dataset))
    return z["embeddings"], z["indices"]


def _ordered(area, backbone, dataset):
    resp = np.load(registry.screening_path(area, backbone, "ensemble", dataset, "responses"))
    idx = np.load(registry.screening_path(area, backbone, "ensemble", dataset, "indices"))
    return resp, idx


def _heatmap(ax, x, y, activity, bins=60):
    """Binned mean-activity map over (cos-to-LAI, cos-to-MAI), shown at full extent.

    Only empty bins are masked, so the entire occupied cloud is displayed (the MAI/LAI
    poles at its corners included -- no truncation). The color range is the binned means'
    robust 2-98th percentile: that is the lever that makes the low->high transition span
    the colormap. (Using the raw per-image activity range instead leaves the populated
    bulk in mid-tones, because per-image tails are far wider than the bin means.)
    """
    xb = np.linspace(x.min(), x.max(), bins + 1)
    yb = np.linspace(y.min(), y.max(), bins + 1)
    summed, _, _ = np.histogram2d(x, y, bins=[xb, yb], weights=activity)
    counts, _, _ = np.histogram2d(x, y, bins=[xb, yb])
    mean = np.ma.masked_array(summed, mask=counts == 0) / np.ma.masked_array(counts, mask=counts == 0)
    cmap = plt.cm.inferno.copy()
    cmap.set_bad("white", 1.0)
    vmin, vmax = np.percentile(mean.compressed(), [2, 98])
    im = ax.imshow(mean.T, origin="lower", aspect="auto", extent=[x.min(), x.max(), y.min(), y.max()],
                   cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest")
    ax.set_xlabel("similarity to LAI")
    ax.set_ylabel("similarity to MAI")
    return im


def _gradient_dir(x, y, activity):
    """OLS gradient (b_x, b_y) of the fitted plane activity ~ 1 + x + y.

    Population planarity check (degree-2 adds a median ~1% R^2) justifies the linear model,
    so this gradient is a faithful summary; its direction is the variance-maximizing axis.
    """
    A = np.column_stack([np.ones_like(x), x, y])
    beta = np.linalg.lstsq(A.T @ A, A.T @ np.asarray(activity, np.float64), rcond=None)[0]
    return float(beta[1]), float(beta[2])


def _arrow(ax, x, y, bx, by, frac=0.30):
    """Draw the activity-gradient direction (toward higher activity) from the centroid."""
    g = np.hypot(bx, by)
    if g == 0:
        return
    L = frac * min(x.max() - x.min(), y.max() - y.min())
    cx, cy = x.mean(), y.mean()
    ax.annotate("", xy=(cx + L * bx / g, cy + L * by / g), xytext=(cx - L * bx / g, cy - L * by / g),
                arrowprops=dict(arrowstyle="-|>", color="white", lw=2.2, mutation_scale=15))


def _projection_panel(ax, x, y, activity, bx, by, color, nb=30):
    """Mean activity vs position along the activity-gradient axis -- the 1-D transition.

    Projecting onto the OLS gradient direction is the variance-maximizing linear axis, so
    the 1-D R^2 equals the 2-D linear R^2 -- this collapse loses no explained variance and
    shows the low->high rise as directly as possible.
    """
    t = bx * np.asarray(x) + by * np.asarray(y)
    a = np.asarray(activity)
    edges = np.linspace(np.percentile(t, 0.5), np.percentile(t, 99.5), nb + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    k = np.clip(np.digitize(t, edges) - 1, 0, nb - 1)
    means = np.array([a[k == i].mean() if np.any(k == i) else np.nan for i in range(nb)])
    sems = np.array([a[k == i].std() / max(np.sqrt(np.sum(k == i)), 1.0) if np.any(k == i) else np.nan
                     for i in range(nb)])
    ax.fill_between(centers, means - sems, means + sems, color=color, alpha=0.25, linewidth=0)
    ax.plot(centers, means, "-", color=color, lw=2)
    ax.set_xlabel("projection onto activity gradient")
    ax.set_ylabel("mean activity")


def fig_dprime(area, backbone, dataset):
    """Fig 6: for one twin, example within/random cosine distributions (b) and d' scatter (c)."""
    fig, (axb, axc) = plt.subplots(1, 2, figsize=(9.5, 4.3))
    color, label = AREA[area]["color"], AREA[area]["label"]

    # Panel b: within-MAI / within-LAI vs pole-to-random cosine distributions, pooled over all
    # non-sparse neurons of this twin (population view, not one cherry-picked cell).
    res = _results(area, backbone, dataset)
    neurons_ns = res["coh_neurons"][res["coh_non_sparse"]]
    emb, idx = _embeddings(area, backbone, dataset)
    _, oi = _ordered(area, backbone, dataset)
    cp = coherence_pooled(emb, idx, oi, neurons_ns)
    rand = np.concatenate([cp["mai_random"], cp["lai_random"]])
    allc = np.concatenate([rand, cp["within_mai"], cp["within_lai"]])
    rng_lo, rng_hi = np.percentile(allc, [0.1, 99.9])
    hkw = dict(bins=60, range=(rng_lo, rng_hi), histtype="step", density=True)
    axb.hist(rand, color="0.6", label="MAI/LAI – random", **hkw)
    axb.hist(cp["within_mai"], color=POLE["mai"], label="within MAI", **hkw)
    axb.hist(cp["within_lai"], color=POLE["lai"], label="within LAI", **hkw)
    axb.set_xlabel("cosine similarity")
    axb.set_ylabel("density")
    axb.set_title(f"{label}/{backbone} non-sparse (n={len(neurons_ns)})", fontsize=9)
    axb.legend(frameon=False, fontsize=8)
    _despine(axb)
    _panel(axb, "b")

    # Panel c: d'(MAI vs random) vs d'(LAI vs random), this twin's non-sparse neurons, over the
    # gray rand-vs-rand control cloud near the origin.
    rng = np.random.default_rng(0)
    m = res["coh_non_sparse"]
    dc = res["coh_dprime_control"][m]
    dm, dl = res["coh_dprime_mai"][m], res["coh_dprime_lai"][m]
    axc.scatter(dc, rng.permutation(dc), facecolors="none", edgecolors="0.78",
                s=16, linewidths=0.7, label="random")
    axc.scatter(dm, dl, facecolors="none", edgecolors=color, s=26, linewidths=1.2, label=label)
    # Data-driven square limits showing the entire scatter (our preprocessing differs from
    # Franke's, so the d' range differs; never clip to a fixed window).
    allv = np.concatenate([dc, dm, dl])
    lo, hi = float(np.nanmin(allv)), float(np.nanmax(allv))
    margin = 0.05 * (hi - lo)
    lim = [lo - margin, hi + margin]
    axc.plot(lim, lim, color="0.3", lw=1)
    axc.set_xlim(lim)
    axc.set_ylim(lim)
    axc.set_aspect("equal", "box")
    axc.set_xlabel("d′ (MAI vs random)")
    axc.set_ylabel("d′ (LAI vs random)")
    axc.legend(frameon=False, fontsize=9)
    _despine(axc)
    _panel(axc, "c")

    fig.tight_layout()
    out = os.path.join(ensure_dir(os.path.join(FIGS, area, backbone)), f"dreamsim_dprime_{dataset}.pdf")
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"dprime {area}/{backbone} {dataset}: non-sparse n={len(neurons_ns)} -> {out}", flush=True)


def fig_similarity(area, backbone, dataset):
    """Fig 9: example non-sparse (b,b') and sparse (c,c') 2D similarity spaces with the
    activity-gradient arrow and its 1D projection, the R^2 histogram (d), and the
    R^2-vs-control scatter (f). Linear model (CV-validated planar); R^2 is the linear fit."""
    res = _results(area, backbone, dataset)
    ns, r2, neurons = res["sp_non_sparse"], res["sp_r2"], res["sp_neurons"]
    emb, idx = _embeddings(area, backbone, dataset)
    resp, oi = _ordered(area, backbone, dataset)

    ex_ns, ex_sp = _example_neurons(area, backbone)   # most non-sparse / most sparse (as neuron_strips)
    Sns = similarity_space_neuron(emb, idx, resp, oi, ex_ns)
    Ssp = similarity_space_neuron(emb, idx, resp, oi, ex_sp)

    fig = plt.figure(figsize=(13, 7.6))
    gs = fig.add_gridspec(2, 3, hspace=0.45, wspace=0.42)

    # Rows: non-sparse (b, b') and sparse (c, c'). Col 0 = 2D space + gradient arrow,
    # col 1 = the 1D projection of activity onto that gradient axis.
    for row, (S, nid, tag, pl) in enumerate(
            [(Sns, ex_ns, "non-sparse", "b"), (Ssp, ex_sp, "sparse", "c")]):
        bx, by = _gradient_dir(S["x"], S["y"], S["activity"])
        ax = fig.add_subplot(gs[row, 0])
        im = _heatmap(ax, S["x"], S["y"], S["activity"])
        _arrow(ax, S["x"], S["y"], bx, by)
        ax.set_title(f"{tag} n{nid} (skew {S['skewness']:.1f})   R²={S['r2']:.2f}", fontsize=9)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        _panel(ax, pl)

        ax = fig.add_subplot(gs[row, 1])
        _projection_panel(ax, S["x"], S["y"], S["activity"], bx, by, AREA[area]["color"])
        ax.set_title(f"activity along gradient   R²={S['r2']:.2f}", fontsize=9)
        _despine(ax)
        _panel(ax, pl + "'")

    # (d) R^2 histogram, sparse vs non-sparse  (top-right)
    ax = fig.add_subplot(gs[0, 2])
    ax.hist(r2[ns], bins=15, range=(0, 0.7), histtype="step", density=True,
            color=AREA[area]["color"], label=f"non-sparse (n={int(ns.sum())})")
    if (~ns).any():
        ax.hist(r2[~ns], bins=15, range=(0, 0.7), histtype="step", density=True,
                color="0.4", label=f"sparse (n={int((~ns).sum())})")
    ax.axvline(np.nanmean(r2[ns]), color=AREA[area]["color"], lw=1, ls="--")
    ax.set_xlabel("variance explained R²")
    ax.set_ylabel("density")
    ax.set_title(f"non-sparse mean R²={np.nanmean(r2[ns]):.2f}", fontsize=9)
    ax.legend(frameon=False, fontsize=8)
    _despine(ax)
    _panel(ax, "d")

    # (f) R^2 (true) vs control R^2, all non-sparse neurons  (bottom-right)
    ax = fig.add_subplot(gs[1, 2])
    rr = r2[ns]
    cb = res["sp_r2_control_both"][ns]
    ckm = res["sp_r2_control_keep_mai"][ns]
    ckl = res["sp_r2_control_keep_lai"][ns]
    ax.scatter(rr, cb, facecolors="none", edgecolors="0.55", s=20, linewidths=0.9, label="both random")
    ax.scatter(rr, ckm, facecolors="none", edgecolors=POLE["mai"], s=20, linewidths=0.9, label="keep MAI")
    ax.scatter(rr, ckl, facecolors="none", edgecolors=POLE["lai"], s=20, linewidths=0.9, label="keep LAI")
    # Limits span every point (real and all controls), so no part of the cloud is cut.
    allp = np.concatenate([rr, cb, ckm, ckl]) if len(rr) else np.array([0.0, 0.7])
    lo = min(0.0, float(np.nanmin(allp)))
    hi = float(np.nanmax(allp))
    margin = 0.05 * (hi - lo)
    lim = [lo - margin, hi + margin]
    ax.plot(lim, lim, color="0.3", lw=1)
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_aspect("equal", "box")
    ax.set_xlabel("R² (true MAI & LAI)")
    ax.set_ylabel("R² (control)")
    ax.legend(frameon=False, fontsize=8)
    _despine(ax)
    _panel(ax, "f")

    out = os.path.join(ensure_dir(os.path.join(FIGS, area, backbone)), f"dreamsim_similarity_{dataset}.pdf")
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"similarity {area}/{backbone} {dataset}: non-sparse n{ex_ns}, sparse n{ex_sp} -> {out}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DreamSim coherence (Fig 6) and similarity-space (Fig 9) figures")
    parser.add_argument("--area", required=True, choices=registry.AREAS)
    parser.add_argument("--backbone", required=True, choices=registry.BACKBONES)
    parser.add_argument("--dataset", choices=["rendered", "imagenet"], default=None, help="default: both")
    args = parser.parse_args()

    datasets = [args.dataset] if args.dataset else ["rendered", "imagenet"]
    for dataset in datasets:
        fig_dprime(args.area, args.backbone, dataset)
        fig_similarity(args.area, args.backbone, dataset)
