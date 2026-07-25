"""Cache the fixed input to the trainable part of a twin (extract once, reuse across members).

The one principle, two regimes (selected by ``config.cache_kind``):

* **frozen core** (``"features"``) — the core's feature map never changes during readout training, so
  :func:`extract_features` computes it once, *before* the trainable head:
    - ``dino``   : raw intermediate block map ``get_intermediate_layers(..., norm=False)``; the
      trainable head is ``DINOv3Core.norm`` (BatchNorm2d) + nonlinearity + Gaussian readout.
    - ``resnet`` : the nnvision ``core.features.TaskDriven`` output (frozen ResNet trunk, before the
      trainable ``OutBatchNorm``/ReLU); the trainable head is ``OutBatchNorm`` + ReLU + readout.

* **fine-tuned core** (``"images"``) — the backbone changes every step, so a feature cache is invalid;
  :func:`cache_images` caches the fixed *transformed input images* instead (the trainable part's
  input), and the whole (truncated) backbone + head train end-to-end on them.

Both are written as a float16 memmap under ``FEATURES_DIR/{area}/{backbone}/`` and reused across every
ensemble member and sweep.
"""

import os

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from dualneuron.utils import ensure_dir, RewriteLine, should_compute
from dualneuron.training.dataset import ImageResponseDataset, training_transform


def _single_thread_worker(worker_id):
    """DataLoader worker init: cap each extraction worker to one CPU thread, so N parallel workers
    (reading + transforming stimuli off CIFS) don't oversubscribe the host's cores."""
    torch.set_num_threads(1)


def input_cache_path(config, split: str) -> str:
    """Cache path for a split's trainable-part input: ``FEATURES_DIR/{area}/{backbone}/{split}_inputs.npy``.

    One constant name for both regimes -- frozen-core feature maps (``cache_kind="features"``) or
    transformed input images (``cache_kind="images"``). The ``{area}/{backbone}/`` folder is the
    disambiguator, so the filename carries no backbone/block/input-size tag.
    """
    return os.path.join(config.features_dir, f"{split}_inputs.npy")


def _build_extractor(config, device):
    """Return a callable ``images -> frozen feature map`` for the configured backbone.

    The returned closure runs the frozen core only (no trainable head), so its output is the exact
    tensor that :class:`dualneuron.twins.dino.DINONeuralPredictor.forward_from_features` (dino) or
    the nnvision head (resnet) consumes during training.
    """
    if config.backbone == "dino":
        from dualneuron.twins.dino import DINOv3Core
        core = DINOv3Core(
            model_name=config.model_name, feature_dim=config.feature_dim,
            block=config.block, model_dir=config.dino_model_dir).to(device).eval()
        backbone, block = core.backbone, core.block

        def extract(x):
            return backbone.get_intermediate_layers(x, n=[block], reshape=True, norm=False)[0]
        return extract

    if config.backbone == "resnet":
        from dualneuron.twins.nets import load_model
        # The staged twin carries the frozen, pretrained robust-ResNet trunk we cache from; the
        # readout/OutBatchNorm it also carries are unused here (we only run TaskDriven).
        model = load_model(architecture=config.area, ensemble=False, untrained=False,
                           centered=False, device=device)
        task_driven = model.core.features.TaskDriven.eval()

        def extract(x):
            return task_driven(x)
        return extract

    raise ValueError(f"Unknown backbone {config.backbone!r}.")


@torch.no_grad()
def extract_features(config, image_ids, split, batch_size=None, device=None, log_path=None, rewrite=False) -> str:
    """Extract and cache the frozen-core feature maps for ``image_ids``.

    Idempotent: returns immediately if the cache file already exists. The feature-map shape
    ``(C, H, W)`` is discovered from the first batch (not hardcoded), so this works for any backbone.

    Args:
        config: :class:`dualneuron.training.config.TrainConfig`.
        image_ids: Image ids to extract (row order of the cache).
        split: ``"train"`` or ``"test"`` (names the cache file).
        batch_size: Override the config batch size for extraction.
        device: Torch device; defaults to ``config.device`` if CUDA is available, else CPU.
        log_path: Progress-log file; defaults to ``LOGS_DIR/{area}_{backbone}_extract_{split}.log``
            (a single self-rewriting tqdm line + a summary). Pass "" to disable.

    Returns:
        str: Path to the ``.npy`` cache file.
    """
    out_path = input_cache_path(config, split)
    if not should_compute(out_path, rewrite):
        print(f"  cached: {out_path}", flush=True)
        return out_path

    device = device or (config.device if torch.cuda.is_available() else "cpu")
    bs = batch_size or config.batch_size
    ensure_dir(config.features_dir)

    if log_path is None and config.logs_dir:
        log_path = os.path.join(config.logs_dir, config.area, config.backbone, f"extract_{split}.log")
    log_file, progress_file = None, None
    if log_path:
        ensure_dir(Path(log_path).parent)
        log_file = open(log_path, "w")
        log_file.write(f"extract {config.area}/{config.backbone} {split}: "
                       f"{len(image_ids)} images -> {out_path}\n")
        progress_file = RewriteLine(log_file, log_file.tell())

    extract = _build_extractor(config, device)
    transform = training_transform(config.area, config.backbone)
    dummy = np.zeros((len(image_ids), 1), dtype=np.float32)
    loader = DataLoader(
        ImageResponseDataset(image_ids, dummy, config.image_dir, transform, channels=config.channels),
        batch_size=bs, shuffle=False, num_workers=config.extract_num_workers, pin_memory=True,
        worker_init_fn=_single_thread_worker,
    )

    tmp_path = out_path + ".tmp"
    fp = None
    idx = 0
    try:
        for imgs, _ in tqdm(loader, desc=f"extract {config.area}/{config.backbone} {split}",
                            unit="batch", file=progress_file):
            feat = extract(imgs.to(device, non_blocking=True)).cpu().to(torch.float16).numpy()
            if fp is None:   # allocate the memmap once the (C, H, W) shape is known
                C, H, W = feat.shape[1:]
                fp = np.lib.format.open_memmap(
                    tmp_path, mode="w+", dtype=np.float16, shape=(len(image_ids), C, H, W))
            fp[idx:idx + feat.shape[0]] = feat
            idx += feat.shape[0]
        fp.flush()
        del fp
        os.rename(tmp_path, out_path)   # atomic: a partial file never gets the final name
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    finally:
        if log_file:
            log_file.write(f"\nwrote {out_path}  ({idx} images)\n")
            log_file.close()
    print(f"  wrote {out_path}  ({idx} images)", flush=True)
    return out_path


@torch.no_grad()
def cache_images(config, image_ids, split, batch_size=None, log_path=None, rewrite=False) -> str:
    """Transform and cache the input images for the fine-tuned (end-to-end) training path.

    For a fine-tuned backbone the frozen-feature cache is invalid (the backbone changes every step),
    so we cache the fixed *transformed input images* once — the input to the trainable part — then
    train the whole (truncated) backbone + head on them. Written as a float16 memmap to
    ``FEATURES_DIR/{area}/{backbone}/{split}_inputs.npy`` and reused across ensemble members.
    Idempotent (returns immediately if the cache already exists).

    Returns:
        str: Path to the ``.npy`` cache file.
    """
    out_path = input_cache_path(config, split)
    if not should_compute(out_path, rewrite):
        print(f"  cached: {out_path}", flush=True)
        return out_path

    bs = batch_size or config.batch_size
    ensure_dir(config.features_dir)

    if log_path is None and config.logs_dir:
        log_path = os.path.join(config.logs_dir, config.area, config.backbone, f"cacheimg_{split}.log")
    log_file, progress_file = None, None
    if log_path:
        ensure_dir(Path(log_path).parent)
        log_file = open(log_path, "w")
        log_file.write(f"cache images {config.area}/{config.backbone} {split}: "
                       f"{len(image_ids)} images -> {out_path}\n")
        progress_file = RewriteLine(log_file, log_file.tell())

    transform = training_transform(config.area, config.backbone)
    dummy = np.zeros((len(image_ids), 1), dtype=np.float32)
    loader = DataLoader(
        ImageResponseDataset(image_ids, dummy, config.image_dir, transform, channels=config.channels),
        batch_size=bs, shuffle=False, num_workers=config.extract_num_workers, pin_memory=False,
        worker_init_fn=_single_thread_worker,
    )

    tmp_path = out_path + ".tmp"
    fp, idx = None, 0
    try:
        for imgs, _ in tqdm(loader, desc=f"cache-img {config.area}/{config.backbone} {split}",
                            unit="batch", file=progress_file):
            arr = imgs.to(torch.float16).numpy()
            if fp is None:   # allocate the memmap once the (C, H, W) shape is known
                C, H, W = arr.shape[1:]
                fp = np.lib.format.open_memmap(
                    tmp_path, mode="w+", dtype=np.float16, shape=(len(image_ids), C, H, W))
            fp[idx:idx + arr.shape[0]] = arr
            idx += arr.shape[0]
        fp.flush()
        del fp
        os.rename(tmp_path, out_path)   # atomic: a partial file never gets the final name
    finally:
        if log_file:
            log_file.write(f"\nwrote {out_path}  ({idx} images)\n")
            log_file.close()
    print(f"  wrote {out_path}  ({idx} images)", flush=True)
    return out_path


def _load_progress_bar(log, desc, total):
    """One-line tqdm bar over ``total`` rows, rewritten into the seed log (a :class:`_Log`, via its
    ``fh``) when given, else stderr -- so a multi-GB cache load reports progress instead of a silent gap."""
    fh = getattr(log, "fh", None)
    if fh is not None:
        return tqdm(total=total, desc=desc, unit="img", file=RewriteLine(fh, fh.tell()),
                    mininterval=1.0, ncols=100)
    return tqdm(total=total, desc=desc, unit="img", mininterval=1.0, ncols=100)


def load_features(path: str, log=None, desc: str = "load features") -> np.ndarray:
    """Load a cached feature/image .npy fully into RAM (float16), reporting load progress.

    Equivalent to ``np.load(path)`` but reads the (multi-GB, CIFS-backed) array in sequential chunks
    with a tqdm bar written to ``log`` (the member's seed log), so materializing it isn't a silent gap
    before training. Plain chunked reads (no mmap) keep it safe on the network filesystem.
    """
    from numpy.lib import format as npfmt
    with open(path, "rb") as f:
        version = npfmt.read_magic(f)
        npfmt._check_version(version)
        shape, fortran_order, dtype = npfmt._read_array_header(f, version)
        if fortran_order or not shape:
            return np.load(path)                       # uncommon layout -> defer to numpy
        out = np.empty(shape, dtype=dtype)
        n, row_bytes = shape[0], out[0].nbytes
        chunk = max(1, (256 * 1024 * 1024) // max(row_bytes, 1))       # ~256 MB reads
        bar = _load_progress_bar(log, f"{desc} ({out.nbytes / 1e9:.1f} GB)", n)
        try:
            for start in range(0, n, chunk):
                end = min(start + chunk, n)
                f.readinto(memoryview(out[start:end]))
                bar.update(end - start)
        finally:
            bar.close()
    return out
