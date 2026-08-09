import os
import time
from pathlib import Path

import torch
import numpy as np
from tqdm import tqdm

from dualneuron.twins import registry
from dualneuron.utils import ensure_dir, env_dir, RewriteLine, should_compute


def build_context(area, backbone, dataset="imagenet", field="full", weights_dir=None, eps=1e-8,
                  progress=None, mininterval=10.0):
    """Reconstruct the z-scored (images x support-neuron) matrix from a screening.

    This is the expensive step: every support neuron's key must be read out of both
    ``responses.npz`` and ``indices.npz``, because each neuron's responses are stored sorted by that
    neuron's own rank. Recovering one image's population vector therefore costs a full pass, which is
    why :func:`population_context` caches the result.

    Returns:
        ``(zmat, support, row_ids, mean, std)``.
    """
    resp = np.load(registry.screening_path(area, backbone, dataset, "responses", field=field))
    idx = np.load(registry.screening_path(area, backbone, dataset, "indices", field=field))

    # The axis space = the well-predicted neurons (training correlation > 0.4), per twin.
    support = np.sort(registry.well_predicted_neurons(area, backbone, weights_dir=weights_dir))

    # Scatter each neuron's sorted responses back to their images (row = screened image id).
    row_ids = np.unique(idx[f"unit_{int(support[0])}"])
    mat = np.empty((len(row_ids), len(support)), dtype=np.float32)
    it = support if progress is None else tqdm(support, file=progress, mininterval=mininterval,
                                               ncols=100, desc=f"context {area}/{backbone}")
    for c, u in enumerate(it):
        mat[np.searchsorted(row_ids, idx[f"unit_{int(u)}"]), c] = resp[f"unit_{int(u)}"]

    mean = mat.mean(0)
    std = mat.std(0) + eps
    mat -= mean
    mat /= std                                             # in place: avoid a second 164 MB copy
    return mat, support, row_ids, mean, std


def population_context(area, backbone, neurons=None, dataset="imagenet", field="full",
                       weights_dir=None, eps=1e-8, cache=True, rewrite=False):
    """Everything the axes are built from, loaded ONCE.

    Reading the screening npz and rebuilding the (image x support-neuron) matrix is the expensive
    part; the axis itself is a couple of means over rows of it. Separating them is what makes a
    *per-seed* axis affordable -- :func:`sampled_axis` can then be called once per (neuron, seed)
    without touching disk again.

    With ``cache`` the matrix is persisted next to its screening
    (:func:`~dualneuron.twins.registry.context_path`) and reopened with ``mmap_mode="r"``, so a
    later call reads nothing up front and an axis draw faults in only the rows it sampled. Build it
    ahead of time with ``python -m dualneuron.dream.axis``.

    Args:
        area, backbone: The twin (its ``field`` screening must exist).
        neurons: Neuron ids whose ranked image order to read; default the whole support. Only these
            keys are read from ``indices.npz``, so asking for a few neurons is cheap.
        dataset, field: Screening the axis is built from ("full" = the natural, unmasked regime).
        weights_dir: Override for the twin's ``correlations.npy`` (which defines the support).
        eps: Numerical stabilizer for the z-score.
        cache: Use (and populate) the on-disk context. False always rebuilds in memory.
        rewrite: Rebuild and overwrite an existing cache.

    Returns:
        dict with ``zmat`` ((images, P) z-scored responses; a memmap when cached), ``support``
        ((P,) global ids), ``pos`` (global id -> column), ``row_ids`` ((images,) screened image ids,
        sorted), ``orders`` ({neuron: image ids ascending by that neuron's response}), ``neurons``,
        and the ``mean``/``std`` used for the z-score.
    """
    mat_p = registry.context_path(area, backbone, dataset, "matrix", field) if cache else None
    meta_p = registry.context_path(area, backbone, dataset, "meta", field) if cache else None
    cached = (mat_p is not None
              and not should_compute(mat_p, rewrite) and not should_compute(meta_p, rewrite))

    if cached:
        meta = np.load(meta_p)
        support, row_ids = meta["support"], meta["row_ids"]
        mean, std = meta["mean"], meta["std"]
        zmat = np.load(mat_p, mmap_mode="r")               # nothing read until rows are touched
    else:
        zmat, support, row_ids, mean, std = build_context(
            area, backbone, dataset=dataset, field=field, weights_dir=weights_dir, eps=eps)
        if mat_p is not None:
            ensure_dir(Path(mat_p).parent)
            np.save(mat_p, zmat)
            np.savez(meta_p, support=support, row_ids=row_ids, mean=mean, std=std)
            zmat = np.load(mat_p, mmap_mode="r")           # drop the resident copy

    pos = {int(u): c for c, u in enumerate(support)}       # global id -> column in the support space
    neurons = np.array([int(n) for n in (support if neurons is None else neurons)])
    # Only the requested neurons' orders are read; each is one key out of indices.npz.
    idx = np.load(registry.screening_path(area, backbone, dataset, "indices", field=field))
    orders = {int(n): idx[f"unit_{int(n)}"] for n in neurons}
    return {"zmat": zmat, "support": support, "pos": pos, "row_ids": row_ids,
            "orders": orders, "neurons": neurons, "mean": mean, "std": std}


def sampled_axis(ctx, neuron, pool=100, n_sample=15, rng=None, exclude_target=False, eps=1e-8,
                 return_ids=False):
    """One neuron's population axis, from a random subsample of its extreme images.

    The centroids are taken over ``n_sample`` images drawn without replacement from the neuron's
    top-``pool`` and bottom-``pool`` full-field responses, rather than over a fixed extreme set. With
    a different ``rng`` per synthesis seed this makes the axis itself vary across seeds, so a neuron's
    MEIs/LEIs sample the invariances of its high/low poles instead of re-deriving one fixed direction.

    Args:
        ctx: A :func:`population_context` dict.
        neuron: Target neuron id (must be in ``ctx["orders"]``).
        pool: Size of the extreme pool at each pole to draw from. Default: 100.
        n_sample: Images drawn from each pool for the centroid; ``None`` (or >= ``pool``) uses the
            whole pool, which reproduces the fixed-extreme behaviour. Default: 15.
        rng: ``np.random.RandomState`` for the draw; ``None`` uses the global RNG.
        exclude_target: Zero the target's own component, leaving only the surrounding population's
            context (the regularizer form). False keeps it -- the full MAI-LAI direction used to fold
            the drive into a single bounded cosine.
        eps: Numerical stabilizer.
        return_ids: Also return the drawn image ids. With a per-seed ``rng`` the axis is not
            recoverable from the twin alone, so a caller that persists its results should record
            them alongside.

    Returns:
        ``(P,)`` float32 unit axis over ``ctx["support"]``; with ``return_ids``, the tuple
        ``(axis, mai_ids, lai_ids)`` where the id arrays are the drawn high/low image ids.
    """
    n = int(neuron)
    order = ctx["orders"][n]                                # ascending by this neuron's response
    hi_pool, lo_pool = order[-pool:], order[:pool]
    if n_sample is not None and n_sample < len(hi_pool):
        draw = (rng or np.random).choice
        hi = draw(hi_pool, size=n_sample, replace=False)
        lo = draw(lo_pool, size=n_sample, replace=False)
    else:
        hi, lo = hi_pool, lo_pool

    zmat, row_ids = ctx["zmat"], ctx["row_ids"]
    a = zmat[np.searchsorted(row_ids, hi)].mean(0) - zmat[np.searchsorted(row_ids, lo)].mean(0)
    if exclude_target and n in ctx["pos"]:
        a[ctx["pos"][n]] = 0.0                              # exclude the target from its own axis
    axis = (a / (np.linalg.norm(a) + eps)).astype(np.float32)
    return (axis, np.asarray(hi), np.asarray(lo)) if return_ids else axis


def semantic_axis(
    images1, 
    images2, 
    dreamsim_model, 
    device='cuda'
):
    """
    Compute semantic axis as the difference between centroids of two image sets.
    
    Args:
        images1: Tensor (N, C, H, W) or list of tensors - first image set (e.g., MAIs)
        images2: Tensor (M, C, H, W) or list of tensors - second image set (e.g., LAIs)
        dreamsim_model: DreamSim model
        device: Device to use
    
    Returns:
        axis: Unit vector pointing from centroid of images1 to centroid of images2
    """
    
    def embed_set(images):
        """Embed a set of images and return mean embedding."""
        if not isinstance(images, (list, tuple)):
            images = [images[i] for i in range(len(images))]
        
        embeddings = []
        with torch.no_grad():
            for img in images:
                if img.dim() == 3:
                    img = img.unsqueeze(0)
                img = img.to(device)
                emb = dreamsim_model.embed(img).flatten().cpu()
                embeddings.append(emb)
        
        return torch.stack(embeddings).mean(dim=0)
    
    # Compute centroids
    centroid1 = embed_set(images1)
    centroid2 = embed_set(images2)
    # Axis from centroid1 → centroid2
    axis = centroid2 - centroid1
    # Normalize to unit vector
    axis = axis / (axis.norm() + 1e-8)
    return axis.numpy()

if __name__ == "__main__":
    import argparse
    from dotenv import load_dotenv
    load_dotenv()

    parser = argparse.ArgumentParser(
        description="Build and cache the population context a twin's axes are drawn from")
    parser.add_argument("--area", type=str, required=True, choices=registry.AREAS)
    parser.add_argument("--backbone", type=str, required=True, choices=registry.BACKBONES)
    parser.add_argument("--dataset", type=str, default="imagenet", choices=["imagenet", "rendered"])
    parser.add_argument("--field", type=str, default="full", choices=["masked", "full"],
                        help="screening regime the context is built from (default: full)")
    parser.add_argument("--rewrite", action="store_true", help="rebuild even if the cache exists")
    parser.add_argument("--log_path", type=str, default=None,
                        help="progress log (default LOGS_DIR/.../{dataset}/screening/{field}/context.log)")
    parser.add_argument("--log_every", type=float, default=10.0)
    args = parser.parse_args()
    registry.check_pair(args.area, args.backbone, parser)

    out = registry.context_path(args.area, args.backbone, args.dataset, "matrix", args.field)
    if out is None:
        raise ValueError("ANALYSIS_DIR is not set. Set it in .env (e.g. "
                         "ANALYSIS_DIR=${DATA_DIR}/DUAL-FEATURE-ANALYSIS).")
    src = registry.screening_path(args.area, args.backbone, args.dataset, "responses", field=args.field)
    if not os.path.exists(src):
        raise FileNotFoundError(
            f"No {args.field} {args.dataset} screening for {args.area}/{args.backbone}: {src}. "
            f"Run it first: python -m dualneuron.screening.run --area {args.area} "
            f"--backbone {args.backbone} --dataset {args.dataset} --field {args.field}")
    if not should_compute(out, args.rewrite):
        print(f"cached (use --rewrite to rebuild): {out}")
        raise SystemExit(0)

    log_path = args.log_path
    if log_path is None and env_dir("LOGS_DIR") is not None:
        log_path = registry.log_path(args.area, args.backbone,
                                     *registry.rel_screening(args.dataset, args.field), "context.log")

    log_file = None
    progress = None
    if log_path is not None:
        ensure_dir(Path(log_path).parent)
        log_file = open(log_path, "w")
        log_file.write(f"context {args.area}/{args.backbone} dataset={args.dataset} "
                       f"field={args.field}\n")
        log_file.flush()
        progress = RewriteLine(log_file, log_file.tell())

    start = time.time()
    ctx = population_context(args.area, args.backbone, neurons=[], dataset=args.dataset,
                             field=args.field, rewrite=args.rewrite)
    elapsed = time.time() - start
    z = ctx["zmat"]
    summary = (f"context {z.shape} ({z.dtype}, {z.nbytes / 1e6:.0f} MB) from "
               f"{len(ctx['support'])} support neurons in {elapsed:.0f}s")
    print(summary)
    print(f"saved {out}")
    if log_file is not None:
        log_file.write(f"\n{summary} -> {out}\n")
        log_file.close()
