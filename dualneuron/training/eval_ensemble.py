"""Evaluate a trained twin ensemble on the recorded V4 test set.

Reports per-neuron **single-trial** correlation and **correlation-to-average**, using exactly the
verified procedure of ``figures/make_fig_accuracy.py`` (Fig 1c): the ensemble predicts recorded
responses to the actual stimuli with learned readout positions (``centered=False``) and the training
transform, then both metrics follow nnvision's centered-Pearson convention
(``get_correlations`` / ``get_avg_correlations``). Results are compared to the staged ResNet
``correlations.npy`` (and, when evaluating the staged release itself, reproduce it at r ~ 0.9997).

    # evaluate a trained ensemble
    python -m dualneuron.training.eval_ensemble --area v4 --backbone dino \
        --weights_dir $TRAINED_MODELS_DIR/v4/dino
    # sanity-check the evaluator against the staged ResNet release (weights_dir omitted)
    python -m dualneuron.training.eval_ensemble --area v4 --backbone resnet
"""

import argparse
import os
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
import torch
from PIL import Image
from dotenv import load_dotenv

import dualneuron
from dualneuron.utils import ensure_dir
from dualneuron.twins.nets import load_model
from dualneuron.data.recordings import load_sessions, build_response_matrix, SKIP_BINS
from dualneuron.training.config import TrainConfig
from dualneuron.training.dataset import make_image_transform

load_dotenv()

# (area, backbone) -> load_model architecture string.
_ARCH = {("v4", "resnet"): "v4", ("v4", "dino"): "v4_dino",
         ("v1", "convnext"): "v1", ("v1", "dino"): "v1_dino"}
# Staged twin folder per area, whose correlations.npy is the reference for the sanity comparison.
_STAGED_FOLDER = {"v4": "V4ColorTaskDriven", "v1": "V1GrayTaskDriven"}
THRESHOLD = 0.4


def _staged_corr_path(area):
    return os.path.join(os.path.dirname(dualneuron.__file__),
                        "twins", _STAGED_FOLDER[area], "correlations.npy")


def _corr(a, b, eps=1e-8):
    """Centered Pearson correlation over the sample axis (nnvision ``corr`` convention)."""
    a = a - a.mean()
    b = b - b.mean()
    return float((a * b).mean() / (a.std() * b.std() + eps))


def correlation_to_average(preds, recorded_avg):
    """Per-neuron correlation between predictions and the repeat-averaged recorded responses.

    Args:
        preds: ``(n_test_images, n_neurons)`` ensemble predictions (test_ids row order).
        recorded_avg: ``(n_test_images, n_neurons)`` recorded means, NaN where unobserved.

    Returns:
        ``(n_neurons,)`` correlations (NaN for neurons with <3 observed images / no variance).
    """
    n = recorded_avg.shape[1]
    out = np.full(n, np.nan)
    for j in range(n):
        m = ~np.isnan(recorded_avg[:, j])
        if m.sum() >= 3 and preds[m, j].std() > 0 and recorded_avg[m, j].std() > 0:
            out[j] = _corr(preds[m, j], recorded_avg[m, j])
    return out


def single_trial_correlation(preds, sessions, test_ids, area):
    """Per-neuron correlation between predictions and every individual test trial.

    For each neuron, pairs the prediction for an image with each single-trial response to that image
    (spike count summed over time-bins ``SKIP_BINS[area]:``), and correlates across all pairs.
    """
    id_to_row = {int(i): r for r, i in enumerate(test_ids)}
    n_neurons = preds.shape[1]
    out = np.full(n_neurons, np.nan)
    g = 0
    for sess in sessions:
        sc = sess["testing_responses"][:, SKIP_BINS[area]:, :].sum(axis=1).astype(np.float32)  # (units, trials)
        rows = np.array([id_to_row[int(t)] for t in sess["testing_image_ids"]])
        for ui in range(sc.shape[0]):
            p = preds[rows, g + ui]
            r = sc[ui]
            if p.std() > 0 and r.std() > 0:
                out[g + ui] = _corr(p, r)
        g += sc.shape[0]
    return out


@torch.no_grad()
def predict(config, weights_dir, test_ids, device, batch_size=64):
    """Ensemble predictions for the test images (full forward, ``centered=False``, eval transform)."""
    arch = _ARCH[(config.area, config.backbone)]
    tf = make_image_transform(config.input_size, config.img_mean, config.img_std,
                              config.crop_size, config.channels)
    mode = "RGB" if config.channels == 3 else "L"
    model = load_model(architecture=arch, ensemble=True, centered=False,
                       weights_dir=weights_dir, device=device).eval()
    preds = []
    for s in range(0, len(test_ids), batch_size):
        batch = test_ids[s:s + batch_size]
        x = torch.stack([
            tf(Image.fromarray(np.load(os.path.join(config.image_dir, f"{int(i):06d}.npy"))).convert(mode))
            for i in batch
        ]).to(device)
        preds.append(model(x).detach().float().cpu().numpy())
    return np.concatenate(preds, 0)


def main():
    p = argparse.ArgumentParser(description="Evaluate a trained twin ensemble on the V4 test set")
    p.add_argument("--area", default="v4", choices=["v4", "v1"])
    p.add_argument("--backbone", default="resnet", choices=["resnet", "dino", "convnext"])
    p.add_argument("--weights_dir", default=None,
                   help="Trained ensemble dir; omit to evaluate the GitHub-staged release.")
    p.add_argument("--device", default=None)
    args = p.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    config = TrainConfig(area=args.area, backbone=args.backbone)

    sessions = load_sessions(config.area)
    test_ids, recorded_avg, _ = build_response_matrix(sessions, "test")
    preds = predict(config, args.weights_dir, test_ids, device)

    avg = correlation_to_average(preds, recorded_avg)
    single = single_trial_correlation(preds, sessions, test_ids, config.area)
    staged = np.load(_staged_corr_path(args.area))

    va = ~np.isnan(avg)
    vs = ~np.isnan(single)

    log_path = (os.path.join(config.logs_dir, args.area, args.backbone, "eval.log")
                if config.logs_dir else None)
    lf = None
    if log_path:
        ensure_dir(Path(log_path).parent)
        lf = open(log_path, "w")

    def emit(msg):
        print(msg, flush=True)
        if lf:
            lf.write(msg + "\n")

    emit(f"[eval] {args.area}/{args.backbone}  weights={'staged' if args.weights_dir is None else args.weights_dir}")
    emit(f"  single-trial : mean={np.nanmean(single):.4f}  median={np.nanmedian(single):.4f}  (n={vs.sum()})")
    emit(f"  corr-to-avg  : mean={np.nanmean(avg):.4f}  median={np.nanmedian(avg):.4f}  n>{THRESHOLD}={int(np.nansum(avg > THRESHOLD))}")
    emit(f"  staged {args.area}   : mean={staged.mean():.4f}  median={np.median(staged):.4f}  n>{THRESHOLD}={int((staged > THRESHOLD).sum())}")
    emit(f"  corr(recomputed corr-to-avg, staged) r={_corr(avg[va], staged[va]):.4f}  "
         f"mean delta={np.nanmean(avg[va] - staged[va]):+.4f}")
    if lf:
        lf.close()


if __name__ == "__main__":
    main()
