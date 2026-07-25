"""Prediction-accuracy figure (paper Fig. 1c): distributions of a digital twin's test correlations
-- single-trial (left) and correlation-to-average (right) -- with the 0.4 inclusion threshold, for
one ``(area, backbone)`` twin.

Predictions are the twin ensemble's responses (learned readout positions, ``centered=False``) to the
recorded test images, evaluated with the twin's exact training transform (``training_transform``:
optional stimulus upsample -> crop -> resize -> z-score); the RF mask and L2 norm are screening-only and are NOT
applied here. Recorded responses come from ``dualneuron.data.recordings`` (spike counts summed over
time-bins 2:, averaged over repeats for the correlation-to-average; single trials for the single-trial
correlation). The two metrics match nnvision's ``get_avg_correlations`` / ``get_correlations``. The
recomputed correlation-to-average is cross-checked against the twin's ``correlations.npy`` to confirm
the neuron alignment.

Requires the area's recordings + the area's canonical SESSION_ORDER (in data/recordings.py) for the
neuron alignment; V4 is set, V1 pending its order.

    python -m dualneuron.figures.make_fig_accuracy --area v4 --backbone resnet
"""
import os
from pathlib import Path

import numpy as np
import torch
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from dotenv import load_dotenv

from dualneuron.utils import env_dir, ensure_dir
from dualneuron.twins import registry
from dualneuron.twins.nets import load_model
from dualneuron.training.dataset import training_transform
from dualneuron.data.recordings import load_sessions, build_response_matrix, SKIP_BINS
from dualneuron.figures.neuron_strips import ACCENT

REPO_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(REPO_ROOT / ".env")
THRESHOLD = 0.4


def _corr(a, b, eps=1e-8):
    """Centered Pearson correlation over the sample axis (nnvision ``corr`` convention)."""
    a = a - a.mean()
    b = b - b.mean()
    return float((a * b).mean() / (a.std() * b.std() + eps))


def _despine(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def _predict(area, backbone, image_ids, device, weights_dir=None, batch_size=64):
    """Twin-ensemble predictions for the given image ids -> (n_images, n_neurons).

    Uses the LEARNED readout positions (``centered=False``): accuracy predicts recorded responses
    to the actual stimuli, so each neuron reads from its own receptive field, not the image center
    (centering is only for the screening / MAI-LAI analysis, where stimuli are centered on the RF).
    The eval transform is the twin's training transform (crop -> resize -> z-score) from the registry.
    """
    spec = registry.resolve(area, backbone)
    weights_dir = weights_dir or registry.weights_dir(area, backbone)   # staged read-only vs trained dir
    images_dir = os.path.join(env_dir("EXPERIMENT_DIR"), area, "images")
    tf = training_transform(area, backbone)          # exact training transform (never diverges from training)
    mode = "RGB" if spec.channels == 3 else "L"
    model = load_model(architecture=spec.arch, ensemble=True, centered=False,
                       weights_dir=weights_dir, device=device).eval()
    preds = []
    with torch.no_grad():
        for s in range(0, len(image_ids), batch_size):
            batch = image_ids[s:s + batch_size]
            x = torch.stack([
                tf(Image.fromarray(np.load(os.path.join(images_dir, f"{int(i):06d}.npy"))).convert(mode))
                for i in batch
            ]).to(device)
            preds.append(model(x).detach().float().cpu().numpy())
    return np.concatenate(preds, 0)


def main(area, backbone, weights_dir=None):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    sessions = load_sessions(area)
    image_ids, recorded_avg, meta = build_response_matrix(sessions, split="test")  # (n_img, N) NaN-sparse
    n_neurons = recorded_avg.shape[1]
    preds = _predict(area, backbone, image_ids, device, weights_dir=weights_dir)   # (n_img, N)
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
        sc = sess["testing_responses"][:, SKIP_BINS[area]:, :].sum(axis=1).astype(np.float32)  # (units, trials)
        rows = np.array([id_to_row[int(t)] for t in sess["testing_image_ids"]])          # (trials,)
        for ui in range(sc.shape[0]):
            p = preds[rows, g + ui]
            r = sc[ui]
            if p.std() > 0 and r.std() > 0:
                single_corr[g + ui] = _corr(p, r)
        g += sc.shape[0]

    # --- cross-check correlation-to-average against this twin's correlations.npy ---
    shipped = np.load(registry.correlations_path(area, backbone, weights_dir))
    v = ~np.isnan(avg_corr)
    print(f"[check] recomputed corr-to-avg vs shipped correlations.npy: "
          f"r={_corr(avg_corr[v], shipped[v]):.4f}  mean|delta|={np.nanmean(np.abs(avg_corr[v]-shipped[v])):.4f}", flush=True)
    print(f"[check] n>{THRESHOLD}: recomputed={int(np.nansum(avg_corr > THRESHOLD))}  "
          f"shipped={int((shipped > THRESHOLD).sum())}", flush=True)
    print(f"[stats] single-trial median={np.nanmedian(single_corr):.3f}  "
          f"corr-to-avg median={np.nanmedian(avg_corr):.3f}", flush=True)

    # --- figure (paper Fig. 1c): two step-histograms, 0.4 line on corr-to-average ---
    color = ACCENT[area]
    fig, ax = plt.subplots(1, 2, figsize=(7.2, 3.0))
    ax[0].hist(single_corr[~np.isnan(single_corr)], range=(0, 1), bins=20, histtype="step",
               color=color, linewidth=1.5)
    ax[0].set_title("single-trial correlation", fontsize=10)
    ax[1].hist(avg_corr[~np.isnan(avg_corr)], range=(0, 1), bins=20, histtype="step",
               color=color, linewidth=1.5)
    ax[1].axvline(THRESHOLD, ls=":", color="gray", linewidth=1.2)
    ax[1].set_title("correlation to average", fontsize=10)
    for a in ax:
        _despine(a)
        a.set_xlabel("correlation")
        a.set_ylabel("# neurons")
        a.set_xlim(0, 1)
    fig.tight_layout()
    out = registry.fig_path(area, backbone, "accuracy.pdf")
    ensure_dir(os.path.dirname(out))
    fig.savefig(out, dpi=300)
    plt.close(fig)
    print(f"saved {out}", flush=True)


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Prediction-accuracy figure (Fig 1c) for one twin")
    p.add_argument("--area", required=True, choices=registry.AREAS)
    p.add_argument("--backbone", required=True, choices=registry.BACKBONES)
    p.add_argument("--weights_dir", default=None,
                   help="trained-ensemble dir (default: staged for resnet/convnext; "
                        "TRAINED_MODELS_DIR/{area}/{backbone} for dino)")
    args = p.parse_args()
    registry.check_pair(args.area, args.backbone, p)
    main(args.area, args.backbone, args.weights_dir)
