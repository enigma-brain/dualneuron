"""
2x10 LAI/MAI strips for given neurons, from the ensemble screening.

Row 0 = the 10 least-activating images (LAIs, low->high activity), row 1 = the 10
most-activating images (MAIs, low->high). Images use the screening geometry
(area crop -> resize) and the RF mask at bg=0.5, but no z-score / no L2, so the
natural image shows through. Saves one PNG per neuron to PAPER_FIG_DIR.

Run from anywhere:
    python tests/neuron_strips.py --area v4 --dataset rendered --neurons 4 5 6
"""
import os
import argparse
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from dotenv import load_dotenv

import dualneuron
from dualneuron.screening.sets import ImagenetImages, RenderedImages
from dualneuron.utils import ensure_dir, env_dir

REPO_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(REPO_ROOT / ".env")

ANALYSIS_DIR = env_dir("ANALYSIS_DIR")
RENDERED_DIR = env_dir("RENDERED_DIR")
IMAGENET_CACHE_DIR = env_dir("IMAGENET_CACHE_DIR")
FIGS = env_dir("PAPER_FIG_DIR", str(REPO_ROOT / "figs"))

# Per-area display geometry (crop matches the screening so the RF mask aligns).
AREAS = {
    "v4": dict(model_name="V4ColorTaskDriven", channels=3, output_size=(100, 100),
               crop_size=200, grayscale=False, cmap=None),
    "v1": dict(model_name="V1GrayTaskDriven", channels=1, output_size=(93, 93),
               crop_size=167, grayscale=True, cmap="gray"),
}


def _build_dataset(area, dataset):
    cfg = AREAS[area]
    mask = np.load(Path(dualneuron.__file__).parent / "twins" / cfg["model_name"] / "mask.npy")
    common = dict(
        use_center_crop=True, use_resize_output=True, use_grayscale=cfg["grayscale"],
        use_normalize=False, use_mask=True, use_crop_to_mask=False, use_norm=False,
        use_clip=False, mask=mask, num_channels=cfg["channels"],
        output_size=cfg["output_size"], crop_size=cfg["crop_size"], bg_value=0.5,
    )
    if dataset == "imagenet":
        return ImagenetImages(data_dir=IMAGENET_CACHE_DIR, split="train",
                              use_experiment_frame=True, **common)
    return RenderedImages(data_dir=RENDERED_DIR, **common)


def _show(ds, i, channels):
    tensor, _ = ds[int(i)]                          # (C, H, W) in [0,1], masked at bg=0.5
    a = np.clip(tensor.detach().cpu().numpy(), 0, 1)
    return a[0] if channels == 1 else np.transpose(a, (1, 2, 0))


def strips(area, dataset, neurons):
    cfg = AREAS[area]
    ds = _build_dataset(area, dataset)
    idx = np.load(os.path.join(ANALYSIS_DIR, area, f"{area}_ensemble_{dataset}_ordered_indices.npz"))
    resp = np.load(os.path.join(ANALYSIS_DIR, area, f"{area}_ensemble_{dataset}_ordered_responses.npz"))
    out_dir = ensure_dir(FIGS)
    imshow_kw = {} if cfg["cmap"] is None else dict(vmin=0, vmax=1)

    for n in neurons:
        key = f"unit_{n}"
        order_idx, order_resp = idx[key], resp[key]
        rows = [("LAI", order_idx[:10], order_resp[:10]),       # 10 lowest, low->high
                ("MAI", order_idx[-10:], order_resp[-10:])]      # 10 highest, low->high
        fig, axes = plt.subplots(2, 10, figsize=(20, 4.4))
        for r, (label, idxs, resps) in enumerate(rows):
            for c in range(10):
                ax = axes[r, c]
                ax.imshow(_show(ds, idxs[c], cfg["channels"]), cmap=cfg["cmap"], **imshow_kw)
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_title(f"{resps[c]:.2f}", fontsize=8)
            axes[r, 0].set_ylabel(label, fontsize=12)
        fig.suptitle(f"{area} {dataset} ensemble — {key}: LAIs and MAIs (low->high activity)", fontsize=12)
        fig.tight_layout()
        out = os.path.join(out_dir, f"neuron{n}_{area}_{dataset}_lai_mai_2x10.png")
        fig.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"{area} {dataset} {key} -> {out}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="2x10 LAI/MAI strips for given neurons")
    parser.add_argument("--area", required=True, choices=["v4", "v1"])
    parser.add_argument("--dataset", required=True, choices=["rendered", "imagenet"])
    parser.add_argument("--neurons", type=int, nargs="+", required=True, help="neuron indices")
    args = parser.parse_args()
    strips(args.area, args.dataset, args.neurons)
