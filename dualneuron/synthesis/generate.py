"""
Synthesize most- and least-exciting inputs (MEIs / LEIs) for the well-predicted
neurons of a twin via gradient ascent, saving one npz per neuron.

Each neuron is written as soon as it finishes, so a run is crash-safe and resumable:
neurons whose npz already exists are skipped. Run one twin per process so each can be
pinned to its own GPU, e.g.:
    CUDA_VISIBLE_DEVICES=0 python -m dualneuron.synthesis.generate --area v4 --backbone resnet
    CUDA_VISIBLE_DEVICES=1 python -m dualneuron.synthesis.generate --area v1 --backbone convnext
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

from dualneuron.twins.nets import load_model
from dualneuron.twins import registry
from dualneuron.synthesis.ascend import pixel_ascending, fourier_ascending
from dualneuron.utils import ensure_dir, env_dir, RewriteLine

ANALYSIS_DIR = env_dir("ANALYSIS_DIR")
LOGS_DIR = env_dir("LOGS_DIR")


# Gradient-ascent hyperparameters per synthesis method (the ascent *algorithm* settings only). The
# twin geometry — image_size, channels, values_range, target_norm — is injected from the central
# registry in generate(), so these dicts are not duplicated per twin.
_PIXEL_PARAMS = {
    'init_image': None,
    'total_steps': 128,
    'optimizer': 'adam',
    'learning_rate': 0.05,
    'lr_schedule': True,
    'eta_min': 0.0,
    'noise': 0.0,
    'nb_crops': 4,
    'box_size': (1.0, 1.0),
    'tv_weight': 0.0,
    'init_std': 0.01,
    'jitter_std': 0.1,
    'oversample': 1,
    'reflect_pad_frac': 0.0,
    'blur_schedule': 'cosine',
    'sigma_max': 1.0,
    'sigma_min': 0.1,
    'verbose': False,
    'save_all_steps': False,
}

_FOURIER_PARAMS = {
    'magnitude_path': 'natural_rgb.npy',
    'init_image': None,
    'total_steps': 128,
    'learning_rate': 0.5,
    'lr_schedule': True,
    'eta_min': 0.0,
    'noise': 0.0,
    'range_fn': 'sigmoid',
    'nb_crops': 4,
    'box_size': (1.0, 1.0),
    'tv_weight': 0.0,
    'jitter_std': 0.1,
    'oversample': 1,
    'reflect_pad_frac': 0.0,
    'verbose': False,
    'save_all_steps': False,
}

_ASCEND = {"pixel": pixel_ascending, "fourier": fourier_ascending}
_ASCEND_PARAMS = {"pixel": _PIXEL_PARAMS, "fourier": _FOURIER_PARAMS}


def _ascend_params(spec, device):
    """Assemble the full ascent kwargs for a twin: the method's algorithm params + the twin geometry
    (image_size, values_range, target_norm, and channels for the pixel method) from the registry."""
    params = dict(_ASCEND_PARAMS[spec.synth_method])
    params.update(image_size=spec.input_size, values_range=spec.synth_values_range,
                  target_norm=spec.synth_target_norm, device=device)
    if spec.synth_method == "pixel":
        params["channels"] = spec.channels
    return params


def _objective(model, neuron_id, weight):
    # weight=+1 maximizes the neuron's response (MEI); weight=-1 minimizes it (LEI).
    return lambda images: weight * torch.mean(model(images)[:, neuron_id])


def generate(area, backbone, output_dir=None, num_seeds=10, neurons=None,
             weights_dir=None, device="cuda", log_path=None, log_every=30.0):
    """
    Synthesize MEIs and LEIs for a twin's neurons, one npz per neuron.

    For each neuron and each of `num_seeds` reproducible seeds, runs gradient ascent
    to produce a most-exciting (MEI) and a least-exciting (LEI) image. Each neuron is
    saved to its own npz as soon as it finishes, so the run is crash-safe and
    resumable: neurons whose file already exists are skipped.

    Args:
        area (str): "v1" or "v4".
        backbone (str): Twin backbone ("resnet"/"dino" for v4, "convnext"/"dino" for v1). The
            (area, backbone) pair selects the twin, ascent method, and geometry via the registry.
        output_dir (str, optional): Folder for the per-neuron npz files.
            Default: ANALYSIS_DIR/{area}/{backbone}/synthesis.
        num_seeds (int): Random seeds per neuron. Default: 10.
        neurons (sequence, optional): Explicit neuron indices. Default: the
            well-predicted set (correlation-to-average > 0.4) for the twin.
        weights_dir (str, optional): Trained-ensemble dir (default: staged for resnet/convnext,
            TRAINED_MODELS_DIR/{area}/{backbone} for dino).
        device (str): Torch device. Default: "cuda".
        log_path (str, optional): Progress-log file; a single line is rewritten in
            place, bracketed by a header and footer. Without it, progress goes to
            stderr. Default: None.
        log_every (float): Minimum seconds between progress-line updates. Default: 30.0.

    Saves:
        {output_dir}/neuron{id:04d}.npz per neuron, each with arrays:
            - neuron_id (scalar), seeds (S,)
            - mei_image, lei_image (S, C, H, W)
            - mei_alpha, lei_alpha (S, C, H, W)
            - mei_activation, lei_activation (S,)
    """
    spec = registry.resolve(area, backbone)

    if neurons is None:
        neurons = registry.well_predicted_neurons(area, backbone, weights_dir=weights_dir)
    neurons = [int(n) for n in neurons]
    seeds = list(range(num_seeds))               # reproducible, shared across neurons
    params = _ascend_params(spec, device)
    ascend = _ASCEND[spec.synth_method]

    if output_dir is None:
        if ANALYSIS_DIR is None:
            raise ValueError(
                "ANALYSIS_DIR is not set. Set it in .env "
                "(e.g. ANALYSIS_DIR=${DATA_DIR}/DUAL-FEATURE-ANALYSIS) or pass output_dir."
            )
        output_dir = registry.synthesis_dir(area, backbone)
    output_dir = ensure_dir(output_dir)

    def out_file(neuron_id):
        return os.path.join(output_dir, f"neuron{neuron_id:04d}.npz")

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
            f"synthesize area={area} backbone={backbone} neurons={len(neurons)} "
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

    model = load_model(spec.arch, ensemble=True, centered=True,
                       weights_dir=weights_dir, device=device).eval()
    poles = (("mei", 1), ("lei", -1))
    fields = ("image", "alpha", "activation")

    start = time.time()
    for neuron_id in tqdm(
        todo,
        file=progress_file,
        mininterval=log_every,
        ncols=100,
        desc=f"synthesize {area}/{backbone}",
    ):
        results = {f"{pole}_{field}": [] for pole, _ in poles for field in fields}
        for seed in seeds:
            torch.manual_seed(seed)
            np.random.seed(seed)
            for pole, weight in poles:
                res = ascend(_objective(model, neuron_id, weight), **params)
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
    parser.add_argument("--area", type=str, required=True, choices=registry.AREAS,
                        help="visual area to synthesize")
    parser.add_argument("--backbone", type=str, required=True, choices=registry.BACKBONES,
                        help="twin backbone")
    parser.add_argument("--num_seeds", type=int, default=10,
                        help="random seeds per neuron (default: 10)")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="output directory (default ANALYSIS_DIR/{area}/{backbone}/synthesis)")
    parser.add_argument("--neurons", type=int, nargs="+", default=None,
                        help="explicit neuron indices (default: well-predicted set)")
    parser.add_argument("--weights_dir", type=str, default=None,
                        help="trained-ensemble dir (default: staged for resnet/convnext; "
                             "TRAINED_MODELS_DIR/{area}/{backbone} for dino)")
    parser.add_argument("--device", type=str, default="cuda", help="device to run on")
    parser.add_argument("--log_path", type=str, default=None,
                        help="progress log file (default LOGS_DIR/{area}_{backbone}_synthesis.log)")
    parser.add_argument("--log_every", type=float, default=30.0,
                        help="min seconds between progress-line updates")
    args = parser.parse_args()

    log_path = args.log_path
    if log_path is None and LOGS_DIR is not None:
        log_path = os.path.join(LOGS_DIR, args.area, args.backbone, "synthesis.log")

    generate(
        area=args.area,
        backbone=args.backbone,
        output_dir=args.output_dir,
        num_seeds=args.num_seeds,
        neurons=args.neurons,
        weights_dir=args.weights_dir,
        device=args.device,
        log_path=log_path,
        log_every=args.log_every,
    )
