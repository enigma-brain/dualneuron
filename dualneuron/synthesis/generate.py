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
from dualneuron.dream.axis import population_context, sampled_axis
from dualneuron.synthesis.ascend import pixel_ascending, fourier_ascending
from dualneuron.utils import ensure_dir, env_dir, RewriteLine, should_compute

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
    (image_size, values_range, target_norm, and channels for the pixel method) from the registry.

    ``target_norm`` comes from :func:`~dualneuron.twins.registry.resolve_synth_norm` -- the value
    measured from this twin's own training stimuli if it has been computed, else the ``TwinSpec``
    literal. ``values_range`` is passed through to the ascent so the norm constraint and the value
    bounds are satisfied together rather than the rescale silently voiding the range.
    """
    params = dict(_ASCEND_PARAMS[spec.synth_method])
    params.update(image_size=spec.input_size, values_range=spec.synth_values_range,
                  target_norm=registry.resolve_synth_norm(spec.area, spec.backbone), device=device)
    if spec.synth_method == "pixel":
        params["channels"] = spec.channels
    return params


def _objective(model, neuron_id, weight):
    # weight=+1 maximizes the neuron's response (MEI); weight=-1 minimizes it (LEI).
    return lambda images: weight * torch.mean(model(images)[:, neuron_id])


def generate(area, backbone, output_dir=None, num_seeds=10, neurons=None,
             weights_dir=None, mode="free", axis_pool=100, axis_sample=15,
             axis_field="full", rewrite=False, device="cuda", log_path=None, log_every=30.0):
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
            Default: the mode's synthesis dir (ANALYSIS_DIR/{area}/{backbone}/synthesis[_axis]).
        num_seeds (int): Random seeds per neuron. Default: 10.
        neurons (sequence, optional): Explicit neuron indices. Default: the
            well-predicted set (correlation-to-average > 0.4) for the twin.
        weights_dir (str, optional): Trained-ensemble dir (default: staged for resnet/convnext,
            TRAINED_MODELS_DIR/{area}/{backbone} for dino).
        mode (str): Synthesis method. "free" (default) is the original free ascent that maximizes the
            neuron's activation. "axis" folds the drive into the natural population axis: it maximizes
            cos(z_pop, a_full) over the well-predicted subspace (target component kept) -- a single
            bounded objective that reproduces the neuron's natural MAI/LAI endpoint state. "axis" needs
            the twin's full-field screening.
        axis_pool (int): Size of the extreme pool at each pole the axis centroids are drawn from
            (mode="axis"). Default: 100.
        axis_sample (int): Images drawn from each pool per seed to form the centroids; None uses the
            whole pool. Drawing a fresh subsample per seed makes the axis vary across seeds, so the
            seeds sample the pole's invariances. Default: 15.
        axis_field (str): Screening regime the axis is built from (mode="axis"). Default: "full".
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
    if mode not in registry.SYNTHESIS_VARIANTS:
        raise ValueError(f"unknown mode {mode!r}; expected one of {registry.SYNTHESIS_VARIANTS}")
    spec = registry.resolve(area, backbone)
    # Resolve where this twin's ensemble lives (staged read-only folder vs trained dir), keyed off the
    # registry so weights, correlations and mask all come from the same place. Explicit arg overrides.
    weights_dir = weights_dir or registry.weights_dir(area, backbone)

    if neurons is None:
        neurons = registry.well_predicted_neurons(area, backbone, weights_dir=weights_dir)
    neurons = [int(n) for n in neurons]
    seeds = list(range(num_seeds))               # reproducible, shared across neurons
    params = _ascend_params(spec, device)
    ascend = _ASCEND[spec.synth_method]

    # "axis" mode folds the drive into the natural population axis a_full (target component KEPT), from
    # the full-field screening; the cosine to it is the whole objective (axis_only). "free" needs none.
    # The context (screening -> z-scored population matrix) is loaded once here; the axis itself is
    # drawn per (neuron, seed) inside the loop below, so seeds sample the pole rather than sharing one
    # fixed direction. Setting axis_sample >= axis_pool disables the subsampling.
    axis_ctx = pop_mean = pop_std = pop_support = None
    if mode == "axis":
        axis_ctx = population_context(area, backbone, neurons=neurons, field=axis_field,
                                      weights_dir=weights_dir)
        pop_mean, pop_std, pop_support = axis_ctx["mean"], axis_ctx["std"], axis_ctx["support"]

    if output_dir is None:
        if ANALYSIS_DIR is None:
            raise ValueError(
                "ANALYSIS_DIR is not set. Set it in .env "
                "(e.g. ANALYSIS_DIR=${DATA_DIR}/DUAL-FEATURE-ANALYSIS) or pass output_dir."
            )
        output_dir = registry.synthesis_output_dir(area, backbone, variant=mode)
    output_dir = ensure_dir(output_dir)

    def out_file(neuron_id):
        return os.path.join(output_dir, f"neuron{neuron_id:04d}.npz")

    # Resume: skip neurons already written.
    todo = [n for n in neurons if should_compute(out_file(n), rewrite)]
    done = len(neurons) - len(todo)

    # Optional progress log: one line rewritten in place, bracketed by a header and
    # footer (clean in any editor). Without log_path, progress goes to stderr.
    log_file = None
    progress_file = sys.stderr
    if log_path is not None:
        ensure_dir(Path(log_path).parent)
        log_file = open(log_path, "w")
        log_file.write(
            f"synthesize area={area} backbone={backbone} mode={mode} neurons={len(neurons)} "
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
        axis_ids = {"axis_mai_ids": [], "axis_lai_ids": []}
        for seed in seeds:
            torch.manual_seed(seed)
            np.random.seed(seed)
            # A fresh subsample of this neuron's extreme pools per seed, so the axis differs between
            # seeds and the MEI/LEI pair samples the pole's invariances. The MEI and LEI of one seed
            # share the axis -- the pole sign picks which end of it is ascended.
            axis_vec = None
            if mode == "axis":
                axis_vec, mai_ids, lai_ids = sampled_axis(
                    axis_ctx, neuron_id, pool=axis_pool, n_sample=axis_sample,
                    rng=np.random.RandomState((neuron_id * 100003 + seed) % (2 ** 32)),
                    exclude_target=False, return_ids=True)
                axis_ids["axis_mai_ids"].append(mai_ids)
                axis_ids["axis_lai_ids"].append(lai_ids)
            for pole, weight in poles:
                if mode == "axis":
                    res = ascend(None, population_function=model, target_index=neuron_id,
                                 pole=float(weight), population_axis=axis_vec,
                                 population_mean=pop_mean, population_std=pop_std,
                                 population_support=pop_support, axis_only=True, **params)
                else:
                    res = ascend(_objective(model, neuron_id, weight), **params)
                results[f"{pole}_image"].append(res["image"].detach().cpu().numpy())
                results[f"{pole}_alpha"].append(res["alpha"].detach().cpu().numpy())
                results[f"{pole}_activation"].append(float(res["activation"]))
        # axis mode also records the images each seed's axis was drawn from: with a per-seed rng the
        # axis is not recoverable from the twin alone, so without these the run is not reproducible.
        meta = {} if mode == "free" else {                 # keep the free npz byte-identical
            "mode": mode,
            **{k: np.stack(v) for k, v in axis_ids.items() if v},
        }
        np.savez_compressed(
            out_file(neuron_id),
            neuron_id=neuron_id,
            seeds=np.array(seeds),
            **meta,
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
                        help="output directory (default: the mode's synthesis[_axis] dir)")
    parser.add_argument("--neurons", type=int, nargs="+", default=None,
                        help="explicit neuron indices (default: well-predicted set)")
    parser.add_argument("--weights_dir", type=str, default=None,
                        help="trained-ensemble dir (default: staged for resnet/convnext; "
                             "TRAINED_MODELS_DIR/{area}/{backbone} for dino)")
    parser.add_argument("--mode", type=str, default="free", choices=registry.SYNTHESIS_VARIANTS,
                        help="'free' (original activation ascent) or 'axis' (fold the drive into the "
                             "natural population axis; needs the twin's full-field screening)")
    parser.add_argument("--axis_pool", type=int, default=100,
                        help="extreme pool per pole the axis centroids are drawn from (mode=axis)")
    parser.add_argument("--axis_sample", type=int, default=15,
                        help="images drawn from each pool per SEED to form the centroids; the axis "
                             "then varies across seeds (mode=axis)")
    parser.add_argument("--axis_field", type=str, default="full", help="screening regime for the axis")
    parser.add_argument("--rewrite", action="store_true", help="re-synthesize neurons even if their npz exists")
    parser.add_argument("--device", type=str, default="cuda", help="device to run on")
    parser.add_argument("--log_path", type=str, default=None,
                        help="progress log file (default LOGS_DIR/{area}/{backbone}/synthesis/{mode}/generate.log)")
    parser.add_argument("--log_every", type=float, default=30.0,
                        help="min seconds between progress-line updates")
    args = parser.parse_args()
    registry.check_pair(args.area, args.backbone, parser)

    log_path = args.log_path
    if log_path is None and LOGS_DIR is not None:
        log_path = registry.log_path(args.area, args.backbone,
                                     *registry.rel_synthesis(args.mode), "generate.log")

    generate(
        area=args.area,
        backbone=args.backbone,
        output_dir=args.output_dir,
        num_seeds=args.num_seeds,
        neurons=args.neurons,
        weights_dir=args.weights_dir,
        mode=args.mode,
        axis_pool=args.axis_pool,
        axis_sample=args.axis_sample,
        axis_field=args.axis_field,
        rewrite=args.rewrite,
        device=args.device,
        log_path=log_path,
        log_every=args.log_every,
    )
