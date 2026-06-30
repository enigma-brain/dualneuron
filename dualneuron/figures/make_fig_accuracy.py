"""Prediction-accuracy figure (paper Fig. 1c): distributions of macaque V4 digital-twin test
correlations -- single-trial (left) and correlation-to-average (right) -- with the 0.4 inclusion
threshold.

Predictions are the V4 ensemble's responses (learned readout positions, ``centered=False``) to the
recorded test images, evaluated with the model's training transform (``CenterCrop(200) ->
Resize(100, bicubic) -> z-score 113.5/59.58``); the RF mask and L2 norm are screening-only and are
NOT applied here. Recorded responses come from
``dualneuron.data.recordings`` (spike counts summed over time-bins 2:, averaged over repeats for the
correlation-to-average; single trials for the single-trial correlation). The two metrics match
nnvision's ``get_avg_correlations`` / ``get_correlations``. The recomputed correlation-to-average is
cross-checked against the shipped ``correlations.npy`` to confirm the neuron alignment.

V4 only -- there are no V1 recordings in this dataset.

    python -m dualneuron.figures.make_fig_accuracy
"""
import os
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from dotenv import load_dotenv

import dualneuron
from dualneuron.utils import env_dir, ensure_dir
from dualneuron.twins.nets import load_model
from dualneuron.data.recordings import load_sessions, build_response_matrix, SKIP_BINS

REPO_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(REPO_ROOT / ".env")
FIGS = env_dir("PAPER_FIG_DIR", str(REPO_ROOT / "figs"))
IMAGES = os.path.join(env_dir("EXPERIMENT_DIR"), "images_all_types")
CORR_NPY = os.path.join(os.path.dirname(dualneuron.__file__), "twins", "V4ColorTaskDriven", "correlations.npy")
V4 = dict(color="#2c6fbb", label="V4")
THRESHOLD = 0.4

# Evaluation transform (matches the dev repo's make_image_transform): center-crop the RF region,
# bicubic-downsample to the model's 100x100 input, then the single-valued 113.5/59.58 z-score on all
# three channels (the model has no internal normalization, so we normalize here). No RF mask / L2
# norm -- those control contrast for the MAI/LAI screening only.
EVAL_TF = T.Compose([
    T.CenterCrop(200),
    T.Resize((100, 100), interpolation=T.InterpolationMode.BICUBIC, antialias=True),
    T.ToTensor(),
    T.Normalize([113.5 / 255] * 3, [59.58 / 255] * 3),
])


def _corr(a, b, eps=1e-8):
    """Centered Pearson correlation over the sample axis (nnvision ``corr`` convention)."""
    a = a - a.mean()
    b = b - b.mean()
    return float((a * b).mean() / (a.std() * b.std() + eps))


def _despine(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def _predict(image_ids, device, batch_size=64):
    """V4 ensemble predictions for the given image ids -> (n_images, n_neurons).

    Uses the LEARNED readout positions (``centered=False``): accuracy predicts recorded responses
    to the actual stimuli, so each neuron reads from its own receptive field, not the image center
    (centering is only for the screening / MAI-LAI analysis, where stimuli are centered on the RF).
    """
    model = load_model(architecture="v4", ensemble=True, centered=False, device=device).eval()
    preds = []
    with torch.no_grad():
        for s in range(0, len(image_ids), batch_size):
            batch = image_ids[s:s + batch_size]
            x = torch.stack([
                EVAL_TF(Image.fromarray(np.load(os.path.join(IMAGES, f"{int(i):06d}.npy"))).convert("RGB"))
                for i in batch
            ]).to(device)
            preds.append(model(x).detach().float().cpu().numpy())
    return np.concatenate(preds, 0)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    sessions = load_sessions()
    image_ids, recorded_avg, meta = build_response_matrix(sessions, split="test")  # (n_img, 394) NaN-sparse
    n_neurons = recorded_avg.shape[1]
    preds = _predict(image_ids, device)                                            # (n_img, 394)
    id_to_row = {int(i): r for r, i in enumerate(image_ids)}

    # --- correlation to average: per neuron over its observed images ---
    avg_corr = np.full(n_neurons, np.nan, dtype=np.float64)
    for j in range(n_neurons):
        m = ~np.isnan(recorded_avg[:, j])
        if m.sum() >= 3 and preds[m, j].std() > 0 and recorded_avg[m, j].std() > 0:
            avg_corr[j] = _corr(preds[m, j], recorded_avg[m, j])

    # --- single-trial correlation: per neuron over all of its session's test trials ---
    single_corr = np.full(n_neurons, np.nan, dtype=np.float64)
    g = 0
    for sess in sessions:
        sc = sess["testing_responses"][:, SKIP_BINS:, :].sum(axis=1).astype(np.float32)  # (units, trials)
        rows = np.array([id_to_row[int(t)] for t in sess["testing_image_ids"]])          # (trials,)
        for ui in range(sc.shape[0]):
            p = preds[rows, g + ui]
            r = sc[ui]
            if p.std() > 0 and r.std() > 0:
                single_corr[g + ui] = _corr(p, r)
        g += sc.shape[0]

    # --- cross-check correlation-to-average against the shipped correlations.npy ---
    shipped = np.load(CORR_NPY)
    v = ~np.isnan(avg_corr)
    print(f"[check] recomputed corr-to-avg vs shipped correlations.npy: "
          f"r={_corr(avg_corr[v], shipped[v]):.4f}  mean|delta|={np.nanmean(np.abs(avg_corr[v]-shipped[v])):.4f}", flush=True)
    print(f"[check] n>{THRESHOLD}: recomputed={int(np.nansum(avg_corr > THRESHOLD))}  "
          f"shipped={int((shipped > THRESHOLD).sum())}", flush=True)
    print(f"[stats] single-trial median={np.nanmedian(single_corr):.3f}  "
          f"corr-to-avg median={np.nanmedian(avg_corr):.3f}", flush=True)

    # --- figure (paper Fig. 1c): two step-histograms, 0.4 line on corr-to-average ---
    fig, ax = plt.subplots(1, 2, figsize=(7.2, 3.0))
    ax[0].hist(single_corr[~np.isnan(single_corr)], range=(0, 1), bins=20, histtype="step",
               color=V4["color"], linewidth=1.5)
    ax[0].set_title("single-trial correlation", fontsize=10)
    ax[1].hist(avg_corr[~np.isnan(avg_corr)], range=(0, 1), bins=20, histtype="step",
               color=V4["color"], linewidth=1.5)
    ax[1].axvline(THRESHOLD, ls=":", color="gray", linewidth=1.2)
    ax[1].set_title("correlation to average", fontsize=10)
    for a in ax:
        _despine(a)
        a.set_xlabel("correlation")
        a.set_ylabel("# neurons")
        a.set_xlim(0, 1)
    fig.tight_layout()
    out = os.path.join(ensure_dir(FIGS), "fig_accuracy_v4.pdf")
    fig.savefig(out, dpi=300)
    plt.close(fig)
    print(f"saved {out}", flush=True)


if __name__ == "__main__":
    main()
