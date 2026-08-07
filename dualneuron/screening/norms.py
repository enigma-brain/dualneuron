"""Compute a twin's L2 constraint from the data it was trained on.

MEI/LEI synthesis constrains its solution to a fixed L2 norm, and masked screening rescales every
image to one. Those numbers should not be inherited as literals -- they are thresholds on the norms
the twin's own training stimuli already have, so they are computed per twin.

**What is measured.** The energy a recorded training stimulus carries *inside the twin's receptive
field*: ``|| x * m ||_2``, with ``x`` the stimulus under
:func:`~dualneuron.training.dataset.training_transform` (the twin's own optional upsample ->
center-crop -> resize -> z-score) and ``m`` the gradient RF mask from
:mod:`dualneuron.synthesis.rf`. Not ImageNet -- that is the screening corpus, a distribution the
twin never learned from -- and not the full frame either: a synthesized stimulus concentrates its
energy in the receptive field, so full-frame energy is the wrong support and gives a number several
times too large.

**Why the gradient mask and not the shipped one.** Every twin is then measured the same way, whether
or not it happens to ship a mask, and the estimate exists straight after training rather than after
a synthesis run. That is what makes this runnable at the *start* of a twin's pipeline.

**The percentile is a choice.** :data:`~dualneuron.twins.registry.NORM_PERCENTILE` (2.56) was
calibrated against the *shipped* masks, where V4's established 40 falls at p2.56 and V1's 12 at
p1.65 -- two independent areas agreeing on a low single-digit percentile. Measured here against the
*gradient* mask it yields **38.92** for ``v4/staged``, 2.7% under the published 40, because the two
masks correlate 0.9945 rather than being identical. Inspect ``figures.make_fig_norms`` and pass
``--percentile`` to choose differently; the value used is recorded in the npz and the log.

The result is read back by :func:`~dualneuron.twins.registry.resolve_synth_norm` and
:func:`~dualneuron.twins.registry.resolve_screen_norm`, which fall back to the ``TwinSpec`` literals
until it exists -- so a twin behaves exactly as before until its norm has been measured.

No GPU is touched here. ``EXPERIMENT_DIR`` is typically a network share, so the cost is
latency-bound small-file reads and ``--num_workers`` dominates the runtime; it defaults to a value
sized to the machine (:func:`~dualneuron.utils.default_workers`) rather than a fixed number.

    python -m dualneuron.synthesis.rf     --area v4 --backbone staged   # the RF mask first
    python -m dualneuron.screening.norms  --area v4 --backbone staged
"""
import warnings
warnings.filterwarnings("ignore")

import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from dualneuron.data.recordings import load_sessions, build_response_matrix
from dualneuron.training.dataset import ImageResponseDataset, training_transform
from dualneuron.twins import registry
from dualneuron.utils import default_workers, ensure_dir, env_dir, RewriteLine, should_compute

LOGS_DIR = env_dir("LOGS_DIR")

#: Percentiles stored in the npz and printed in the summary.
PERCENTILES = (1, 2.56, 5, 10, 25, 50, 75, 90, 95, 99)


def rf_mask(area, backbone):
    """This twin's gradient RF mask, or a pointer to the command that builds it.

    Raises:
        FileNotFoundError: If it has not been built yet -- the norm's support is undefined without
            it, so this reports rather than guessing at one.
    """
    path = registry.rf_mask_path(area, backbone)
    if path is None:
        raise ValueError("ANALYSIS_DIR is not set. Set it in .env (e.g. "
                         "ANALYSIS_DIR=${DATA_DIR}/DUAL-FEATURE-ANALYSIS).")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"No gradient RF mask for {area}/{backbone}: {path}. Build it first: "
            f"python -m dualneuron.synthesis.rf --area {area} --backbone {backbone}")
    return np.load(path), path


def measure_norms(area, backbone, split="train", n_sample=None, subset_seed=0, batch_size=64,
                  num_workers=None, progress=None, mininterval=10.0):
    """RF-masked L2 norms of the recorded stimuli under this twin's training transform.

    Uses the project's own machinery end to end -- :func:`load_sessions` /
    :func:`build_response_matrix` for the split's image ids, :class:`ImageResponseDataset` for the
    frames, :func:`training_transform` for the geometry -- so the tensors measured are the ones the
    twin was fed during training, not a reimplementation of them.

    Args:
        area, backbone: The twin; geometry and z-score resolve from the registry.
        split: ``"train"`` (the fitting distribution; default) or ``"test"``.
        n_sample: Measure a uniform random subset of this many frames; None measures all.
        subset_seed: Seed for that subset, so twins are compared on the same frames.
        batch_size: Loader batch size.
        num_workers: Loader workers; None (default) -> :func:`~dualneuron.utils.default_workers`,
            which sizes itself to this machine's usable cores.
        progress: File-like sink for the tqdm bar (a :class:`RewriteLine`); None -> stderr.
        mininterval: Minimum seconds between progress-line updates.

    Returns:
        dict with ``masked`` ((n,) float64 RF-masked norms), ``full`` (the unmasked norms, for
        reference), ``image_ids``, ``n``, ``mask_path`` and ``mask_area``.
    """
    spec = registry.resolve(area, backbone)
    mask, mask_path = rf_mask(area, backbone)
    m = torch.from_numpy(mask).float()

    image_ids, responses, _ = build_response_matrix(load_sessions(area), split)
    dset = ImageResponseDataset(image_ids, responses,
                                os.path.join(env_dir("EXPERIMENT_DIR"), area, "images"),
                                transform=training_transform(area, backbone),
                                channels=spec.channels)
    if n_sample is not None and n_sample < len(dset):
        rng = np.random.RandomState(subset_seed)
        keep = np.sort(rng.choice(len(dset), size=n_sample, replace=False))
        dset, image_ids = Subset(dset, keep), image_ids[keep]

    if num_workers is None:
        num_workers = default_workers()
    loader = DataLoader(dset, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers, pin_memory=False)

    masked, full = [], []
    for images, _ in tqdm(loader, desc=f"norms {area}/{backbone} {split}",
                          file=progress or sys.stderr, mininterval=mininterval):
        images = images.float()
        masked.append(torch.linalg.vector_norm((images * m).flatten(1), dim=1).numpy())
        full.append(torch.linalg.vector_norm(images.flatten(1), dim=1).numpy())

    masked = np.concatenate(masked).astype(np.float64)
    return {
        "masked": masked,
        "full": np.concatenate(full).astype(np.float64),
        "image_ids": np.asarray(image_ids),
        "n": int(masked.size),
        "mask_path": mask_path,
        "mask_area": float(mask.mean()),
    }


def summarize(norms, percentile=registry.NORM_PERCENTILE):
    """Percentile summary of a norm array and the constraint it defines.

    Returns a dict with ``mean``/``std``/``min``/``max``, the :data:`PERCENTILES` table, the
    ``percentile`` used and ``norm`` -- the value at it, which is the twin's L2 constraint.
    """
    return {
        "mean": float(norms.mean()), "std": float(norms.std()),
        "min": float(norms.min()), "max": float(norms.max()),
        "percentiles": {p: float(np.percentile(norms, p)) for p in PERCENTILES},
        "percentile": float(percentile),
        "norm": float(np.percentile(norms, percentile)),
    }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="Compute a twin's L2 constraint from its RF-masked TRAINING stimuli")
    parser.add_argument("--area", type=str, required=True, choices=registry.AREAS)
    parser.add_argument("--backbone", type=str, required=True, choices=registry.BACKBONES)
    parser.add_argument("--split", type=str, default="train", choices=["train", "test"],
                        help="'train' is the fitting distribution (default)")
    parser.add_argument("--percentile", type=float, default=registry.NORM_PERCENTILE,
                        help=f"threshold percentile defining the constraint (default: "
                             f"{registry.NORM_PERCENTILE}; a choice -- see the norms figure)")
    parser.add_argument("--n_sample", type=int, default=None,
                        help="uniform subset of frames (default: every frame)")
    parser.add_argument("--subset_seed", type=int, default=0, help="seed for --n_sample")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=None,
                        help="loader workers (default: sized to this machine's usable cores, "
                             "capped -- see dualneuron.utils.default_workers)")
    parser.add_argument("--rewrite", action="store_true",
                        help="re-measure and overwrite even if norms.npz exists")
    parser.add_argument("--log_path", type=str, default=None,
                        help="progress log (default LOGS_DIR/{area}/{backbone}/norms/{split}/norms.log)")
    parser.add_argument("--log_every", type=float, default=10.0,
                        help="min seconds between progress-line updates")
    args = parser.parse_args()
    registry.check_pair(args.area, args.backbone, parser)

    output = registry.norms_path(args.area, args.backbone, args.split)
    if output is None:
        raise ValueError("ANALYSIS_DIR is not set. Set it in .env (e.g. "
                         "ANALYSIS_DIR=${DATA_DIR}/DUAL-FEATURE-ANALYSIS).")
    if not should_compute(output, args.rewrite):
        print(f"cached (use --rewrite to re-measure): {output}")
        raise SystemExit(0)

    log_path = args.log_path
    if log_path is None and LOGS_DIR is not None:
        log_path = registry.log_path(args.area, args.backbone,
                                     *registry.rel_norms(args.split), "norms.log")

    spec = registry.resolve(args.area, args.backbone)
    crop = spec.train_crop or spec.crop_size
    workers = args.num_workers if args.num_workers is not None else default_workers()
    header = (f"norms {args.area}/{args.backbone} split={args.split} "
              f"crop={crop} -> {spec.input_size}px {spec.channels}ch "
              f"z-score={spec.img_mean}/{spec.img_std} "
              f"n_sample={args.n_sample or 'all'} workers={workers} p={args.percentile}")

    log_file = None
    progress = None
    if log_path is not None:
        ensure_dir(Path(log_path).parent)
        log_file = open(log_path, "w")
        log_file.write(header + "\n")
        log_file.flush()
        progress = RewriteLine(log_file, log_file.tell())

    start = time.time()
    res = measure_norms(args.area, args.backbone, split=args.split, n_sample=args.n_sample,
                        subset_seed=args.subset_seed, batch_size=args.batch_size,
                        num_workers=workers, progress=progress, mininterval=args.log_every)
    elapsed = time.time() - start

    s = summarize(res["masked"], args.percentile)
    print(f"\n[norms] {args.area}/{args.backbone}  recorded {args.split} stimuli  n={res['n']}")
    print(f"  training transform: crop {crop} -> {spec.input_size}px, {spec.channels}ch, "
          f"z-score {spec.img_mean}/{spec.img_std}")
    print(f"  RF mask: {res['mask_path']}  (mean weight {res['mask_area']:.4f})")
    print(f"\n  RF-masked ||x*m||_2  (what the L2 constraint is a threshold on)")
    print(f"    mean={s['mean']:.2f}  std={s['std']:.2f}  min={s['min']:.2f}  max={s['max']:.2f}")
    print("    " + "  ".join(f"p{p:g}={s['percentiles'][p]:.1f}" for p in PERCENTILES))
    print(f"    (unmasked full-frame median, for reference: {np.median(res['full']):.1f})")
    print(f"\n  computed L2 constraint = p{s['percentile']:g} = {s['norm']:.2f}")
    print(f"  (registry literals: synth_target_norm={spec.synth_target_norm}  "
          f"screen_norm={spec.screen_norm})")

    ensure_dir(Path(output).parent)
    np.savez(
        output,
        masked=res["masked"],
        full=res["full"],
        image_ids=res["image_ids"],
        percentiles=np.array(PERCENTILES, dtype=np.float64),
        percentile_values=np.array([s["percentiles"][p] for p in PERCENTILES]),
        percentile=np.float64(s["percentile"]),
        norm=np.float64(s["norm"]),
        mask_area=np.float64(res["mask_area"]),
    )
    summary = (f"measured {res['n']} frames in {elapsed:.0f}s | "
               f"L2 constraint = p{s['percentile']:g} = {s['norm']:.2f}")
    print(f"\n{summary}\nsaved {output}\n")
    if log_file is not None:
        log_file.write(f"\n{summary} -> {output}\n")
        log_file.close()
