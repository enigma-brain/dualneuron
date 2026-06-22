"""
Synthesize most- and least-exciting inputs (MEIs / LEIs) for the well-predicted
neurons of a twin via gradient ascent, saving one npz per neuron.

Each neuron is written as soon as it finishes, so a run is crash-safe and resumable:
neurons whose npz already exists are skipped. Run one area per process so each can be
pinned to its own GPU, e.g.:
    CUDA_VISIBLE_DEVICES=0 python -m dualneuron.synthesis.generate --area v4
    CUDA_VISIBLE_DEVICES=1 python -m dualneuron.synthesis.generate --area v1
"""
import warnings
warnings.filterwarnings("ignore")
import os
import sys
import time

from dotenv import load_dotenv
load_dotenv()

from pathlib import Path
import numpy as np
from tqdm import tqdm
import torch

from dualneuron.twins.nets import V1GrayTaskDriven, V4ColorTaskDriven
from dualneuron.synthesis.ascend import pixel_ascending, fourier_ascending
from dualneuron.utils import ensure_dir, env_dir, RewriteLine, well_predicted_neurons

ANALYSIS_DIR = env_dir("ANALYSIS_DIR")
LOGS_DIR = env_dir("LOGS_DIR")


# Per-area gradient-ascent hyperparameters.
v1_params = {
    'image_size': 93,
    'channels': 1,
    'init_image': None,
    'total_steps': 128,
    'optimizer': 'adam',
    'learning_rate': 0.05,
    'lr_schedule': True,
    'eta_min': 0.0,
    'noise': 0.0,
    'values_range': (-1.77, 1.86),
    'nb_crops': 4,
    'box_size': (1.0, 1.0),
    'target_norm': 12.0,
    'tv_weight': 0.0,
    'init_std': 0.01,
    'jitter_std': 0.1,
    'oversample': 1,
    'reflect_pad_frac': 0.0,
    'blur_schedule': 'cosine',
    'sigma_max': 1.0,
    'sigma_min': 0.1,
    'device': 'cuda',
    'verbose': False,
    'save_all_steps': False
}

v4_params = {
    'magnitude_path': 'natural_rgb.npy',
    'image_size': 100,
    'init_image': None,
    'total_steps': 128,
    'learning_rate': 0.5,
    'lr_schedule': True,
    'eta_min': 0.0,
    'noise': 0.0,
    'values_range': (-1.9, 2.3),
    'range_fn': 'sigmoid',
    'nb_crops': 4,
    'box_size': (1.0, 1.0),
    'target_norm': 40.0,
    'tv_weight': 0.0,
    'jitter_std': 0.1,
    'oversample': 1,
    'reflect_pad_frac': 0.0,
    'device': 'cuda',
    'verbose': False,
    'save_all_steps': False
}

# Area -> twin class, ascent function, hyperparameters, and twin folder name.
AREAS = {
    "v1": dict(model_class=V1GrayTaskDriven, ascend=pixel_ascending,
               params=v1_params, model_name="V1GrayTaskDriven"),
    "v4": dict(model_class=V4ColorTaskDriven, ascend=fourier_ascending,
               params=v4_params, model_name="V4ColorTaskDriven"),
}


def _objective(model, neuron_id, weight):
    # weight=+1 maximizes the neuron's response (MEI); weight=-1 minimizes it (LEI).
    return lambda images: weight * torch.mean(model(images)[:, neuron_id])


def generate(area, output_dir=None, num_seeds=10, neurons=None,
             device="cuda", log_path=None, log_every=30.0):
    """
    Synthesize MEIs and LEIs for an area's neurons, one npz per neuron.

    For each neuron and each of `num_seeds` reproducible seeds, runs gradient ascent
    to produce a most-exciting (MEI) and a least-exciting (LEI) image. Each neuron is
    saved to its own npz as soon as it finishes, so the run is crash-safe and
    resumable: neurons whose file already exists are skipped.

    Args:
        area (str): "v1" or "v4".
        output_dir (str, optional): Folder for the per-neuron npz files.
            Default: ANALYSIS_DIR/{area}/synthesis.
        num_seeds (int): Random seeds per neuron. Default: 10.
        neurons (sequence, optional): Explicit neuron indices. Default: the
            well-predicted set (correlation-to-average > 0.4) for the area's twin.
        device (str): Torch device. Default: "cuda".
        log_path (str, optional): Progress-log file; a single line is rewritten in
            place, bracketed by a header and footer. Without it, progress goes to
            stderr. Default: None.
        log_every (float): Minimum seconds between progress-line updates. Default: 30.0.

    Saves:
        {output_dir}/{area}_neuron{id:04d}.npz per neuron, each with arrays:
            - neuron_id (scalar), seeds (S,)
            - mei_image, lei_image (S, C, H, W)
            - mei_alpha, lei_alpha (S, C, H, W)
            - mei_activation, lei_activation (S,)
    """
    cfg = AREAS[area]

    if neurons is None:
        neurons = well_predicted_neurons(cfg["model_name"])
    neurons = [int(n) for n in neurons]
    seeds = list(range(num_seeds))               # reproducible, shared across neurons
    params = dict(cfg["params"])
    params["device"] = device

    if output_dir is None:
        if ANALYSIS_DIR is None:
            raise ValueError(
                "ANALYSIS_DIR is not set. Set it in .env "
                "(e.g. ANALYSIS_DIR=${DATA_DIR}/DUAL-FEATURE-ANALYSIS) or pass output_dir."
            )
        output_dir = os.path.join(ANALYSIS_DIR, area, "synthesis")
    output_dir = ensure_dir(output_dir)

    def out_file(neuron_id):
        return os.path.join(output_dir, f"{area}_neuron{neuron_id:04d}.npz")

    # Resume: skip neurons already written.
    todo = [n for n in neurons if not os.path.exists(out_file(n))]
    done = len(neurons) - len(todo)

    # Optional progress log: one line rewritten in place, bracketed by a header and
    # footer (clean in any editor). Without log_path, progress goes to stderr.
    log_file = None
    progress_file = sys.stderr
    if log_path is not None:
        ensure_dir(Path(log_path).parent)
        log_file = open(log_path, "w")
        log_file.write(
            f"synthesize area={area} neurons={len(neurons)} "
            f"(done {done}, todo {len(todo)}) seeds={num_seeds}\n"
        )
        log_file.flush()
        progress_file = RewriteLine(log_file, log_file.tell())

    def _finish(msg):
        if log_file is not None:
            log_file.write(msg + "\n")
            log_file.flush()
            log_file.close()

    if not todo:
        _finish(f"done: nothing to do, all {len(neurons)} neurons present -> {output_dir}")
        return

    model = cfg["model_class"](centered=True, ensemble=True).eval().to(device)
    poles = (("mei", 1), ("lei", -1))
    fields = ("image", "alpha", "activation")

    start = time.time()
    for neuron_id in tqdm(
        todo,
        file=progress_file,
        mininterval=log_every,
        ncols=100,
        desc=f"synthesize {area}",
    ):
        results = {f"{pole}_{field}": [] for pole, _ in poles for field in fields}
        for seed in seeds:
            torch.manual_seed(seed)
            np.random.seed(seed)
            for pole, weight in poles:
                res = cfg["ascend"](_objective(model, neuron_id, weight), **params)
                results[f"{pole}_image"].append(res["image"].detach().cpu().numpy())
                results[f"{pole}_alpha"].append(res["alpha"].detach().cpu().numpy())
                results[f"{pole}_activation"].append(float(res["activation"]))
        np.savez_compressed(
            out_file(neuron_id),
            neuron_id=neuron_id,
            seeds=np.array(seeds),
            **{key: np.stack(values) for key, values in results.items()},
        )
    elapsed = time.time() - start

    _finish(
        f"done: synthesized {len(todo)} neurons x {num_seeds} seeds in {elapsed:.0f}s "
        f"-> {output_dir}"
    )


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(
        description="Synthesize MEIs/LEIs for an area's well-predicted neurons"
    )
    parser.add_argument("--area", type=str, required=True, choices=["v1", "v4"],
                        help="visual area to synthesize")
    parser.add_argument("--num_seeds", type=int, default=10,
                        help="random seeds per neuron (default: 10)")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="output directory (default ANALYSIS_DIR/{area}/synthesis)")
    parser.add_argument("--neurons", type=int, nargs="+", default=None,
                        help="explicit neuron indices (default: well-predicted set)")
    parser.add_argument("--device", type=str, default="cuda", help="device to run on")
    parser.add_argument("--log_path", type=str, default=None,
                        help="progress log file (default LOGS_DIR/{area}_synthesis.log)")
    parser.add_argument("--log_every", type=float, default=30.0,
                        help="min seconds between progress-line updates")
    args = parser.parse_args()

    log_path = args.log_path
    if log_path is None and LOGS_DIR is not None:
        log_path = os.path.join(LOGS_DIR, f"{args.area}_synthesis.log")

    generate(
        area=args.area,
        output_dir=args.output_dir,
        num_seeds=args.num_seeds,
        neurons=args.neurons,
        device=args.device,
        log_path=log_path,
        log_every=args.log_every,
    )
