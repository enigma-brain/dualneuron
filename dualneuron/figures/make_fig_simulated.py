"""Simulated simple/complex-cell control (paper Suppl. Fig. 4): idealized Gabor cells do not
reproduce the structured, coherent least-activating images seen in real neurons.

A library of Gabor simple cells (5 spatial frequencies x 6 orientations x 4 phases) is dotted with
grayscale center patches of the rendered scenes. Simple-cell response = image . Gabor (linear), which
is bimodal -- its most- and least-activating images are both coherent (the LAIs are phase-shifted
versions of the MAIs). Complex-cell response = sqrt(r^2 + (-r)^2) = sqrt(2)|r| (the reference's
full-wave-rectified energy), which is non-negative and sparse -- its least-activating images sit near
zero and are incoherent. This is a pure simulation: no digital twin, hence no transform/centering.

Ports single_neuron_selectivity's create_gabor_patch / generate_simple_cells_library /
compute_simple_cell_responses / compute_complex_cell_responses.

    python -m dualneuron.figures.make_fig_simulated
"""
import os
from glob import glob
from pathlib import Path

import numpy as np
from scipy.stats import skew
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from dotenv import load_dotenv

from dualneuron.utils import env_dir, ensure_dir

REPO_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(REPO_ROOT / ".env")
IMAGES = os.path.join(env_dir("EXPERIMENT_DIR"), "images_all_types")
FIGS = env_dir("PAPER_FIG_DIR", str(REPO_ROOT / "figs"))
RENDERED_LO, RENDERED_HI = 49076, 75000     # rendered-scene id block in images_all_types
SIZE, SIGMA = 100, 10
POLE = {"most": "#c0392b", "least": "#2f6db0"}
LUM = np.array([0.299, 0.587, 0.114], np.float32)


def create_gabor_patch(size=SIZE, sigma=SIGMA, frequency=0.1, theta=0.0, phase=0.0):
    """Gabor patch (Gaussian envelope x cosine carrier), normalized to [-1, 1]."""
    tr = np.deg2rad(theta)
    x, y = np.meshgrid(np.arange(size) - size / 2, np.arange(size) - size / 2)
    xr = x * np.cos(tr) + y * np.sin(tr)
    yr = -x * np.sin(tr) + y * np.cos(tr)
    gabor = np.exp(-(xr ** 2 + yr ** 2) / (2 * sigma ** 2)) * np.cos(2 * np.pi * frequency * xr + phase)
    return gabor / np.max(np.abs(gabor))


def simple_cell_library(n_cells=100, size=SIZE, sigma=SIGMA, seed=0):
    """5 frequencies x 6 orientations x 4 phases (120) -> a random n_cells subset (reference params)."""
    combos = [(f, o, p)
              for f in np.linspace(0.01, 0.2, 5)
              for o in np.linspace(0, 150, 6)
              for p in np.linspace(0, 3 * np.pi / 4, 4)]
    rng = np.random.default_rng(seed)
    combos = [combos[i] for i in rng.choice(len(combos), n_cells, replace=False)]
    return [create_gabor_patch(size, sigma, f, o, p) for (f, o, p) in combos]


def _load_rendered_patches(n=8000, seed=0):
    """Grayscale SIZE x SIZE center patches of n rendered scenes from images_all_types."""
    ids = sorted(int(os.path.basename(f)[:-4]) for f in glob(os.path.join(IMAGES, "*.npy")))
    ids = [i for i in ids if RENDERED_LO <= i <= RENDERED_HI]
    rng = np.random.default_rng(seed)
    ids = rng.choice(ids, min(n, len(ids)), replace=False)
    patches = np.zeros((len(ids), SIZE, SIZE), np.float32)
    for k, i in enumerate(ids):
        a = np.load(os.path.join(IMAGES, f"{int(i):06d}.npy")).astype(np.float32)
        g = a @ LUM if a.ndim == 3 else a
        h, w = g.shape
        t, l = (h - SIZE) // 2, (w - SIZE) // 2
        patches[k] = g[t:t + SIZE, l:l + SIZE]
    return patches


def _strip(fig, gs_cell, patches, title, color):
    """Row of small image patches (MAI or LAI examples) inside a gridspec cell."""
    inner = gs_cell.subgridspec(1, len(patches), wspace=0.05)
    for j, p in enumerate(patches):
        ax = fig.add_subplot(inner[0, j])
        ax.imshow(p, cmap="gray"); ax.set_xticks([]); ax.set_yticks([])
        if j == 0:
            ax.set_ylabel(title, fontsize=8, color=color)


def main():
    cells = simple_cell_library()
    patches = _load_rendered_patches()
    flat = patches.reshape(len(patches), -1)
    gab = np.stack(cells).reshape(len(cells), -1)
    simple = flat @ gab.T                                   # (n_images, n_cells), linear -> bimodal
    complex_ = np.sqrt(2.0) * np.abs(simple)                # full-wave rectified -> sparse

    idx = 0                                                 # example cell
    sk_s, sk_c = skew(simple[:, idx]), skew(complex_[:, idx])
    print(f"[stats] example cell: simple skew={sk_s:.2f} (bimodal~0)  complex skew={sk_c:.2f} (sparse>0)", flush=True)

    fig = plt.figure(figsize=(9, 5.5))
    gs = fig.add_gridspec(3, 3, height_ratios=[1.4, 1, 1], hspace=0.55, wspace=0.35)

    # example Gabor RF
    axg = fig.add_subplot(gs[0, 0]); axg.imshow(cells[idx], cmap="gray"); axg.set_xticks([]); axg.set_yticks([])
    axg.set_title("Gabor RF (simple cell)", fontsize=9)

    # sorted response curves
    for col, (resp, name, sk) in enumerate(((simple[:, idx], "simple (linear)", sk_s),
                                            (complex_[:, idx], "complex (energy)", sk_c)), start=1):
        ax = fig.add_subplot(gs[0, col])
        ax.plot(np.sort(resp), color="0.3", linewidth=1.4)
        ax.axhline(0, ls=":", color="gray", linewidth=0.8)
        ax.set_title(f"{name}\nskew={sk:.2f}", fontsize=9)
        ax.set_xlabel("sorted image rank"); ax.set_ylabel("response")
        for s in ("top", "right"): ax.spines[s].set_visible(False)

    # MAI / LAI example patches for simple (row 1) and complex (row 2)
    for row, resp in ((1, simple[:, idx]), (2, complex_[:, idx])):
        order = np.argsort(resp)
        _strip(fig, gs[row, 0:2], patches[order[-3:][::-1]], "MAI", POLE["most"])
        _strip(fig, gs[row, 2:3], patches[order[:3]], "LAI", POLE["least"])
    fig.text(0.5, 0.66, "simple: coherent MAIs (left) and LAIs (right, phase-shifted)", fontsize=8, ha="center")
    fig.text(0.5, 0.34, "complex: coherent MAIs but incoherent near-zero LAIs", fontsize=8, ha="center")

    out = os.path.join(ensure_dir(FIGS), "fig_simulated_cells.pdf")
    fig.savefig(out, dpi=300, bbox_inches="tight"); plt.close(fig)
    print(f"saved {out}", flush=True)


if __name__ == "__main__":
    main()
