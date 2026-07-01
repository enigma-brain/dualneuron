"""Train a digital-twin readout ensemble for an ``(area, backbone)``.

Extracts the frozen-core features once (to ``FEATURES_DIR``), trains 5 readout members on them, and
saves the ensemble + ``correlations.npy`` + ``mask.npy`` to ``TRAINED_MODELS_DIR/{area}/{backbone}/``
(loadable via ``twins.nets.load_model(architecture, weights_dir=...)``). Logs go to ``LOGS_DIR``.

    # sequential (one device)
    python -m dualneuron.training.run --area v4 --backbone dino
    python -m dualneuron.training.run --area v4 --backbone resnet

    # multi-GPU: one member per GPU (parent extracts features once, then a queue-based pool, then
    # aggregates the ensemble). Each worker holds the feature set in RAM, so peak RAM ~ n_gpus x cache.
    python -m dualneuron.training.run --area v4 --backbone dino --gpus 0,1,2,3,4

    # train a single member (used internally by the pool; also runnable directly)
    python -m dualneuron.training.run --area v4 --backbone dino --member 3
"""

import argparse
import os
import subprocess
import sys
import time
import warnings

warnings.filterwarnings("ignore")

import torch
from dotenv import load_dotenv

from dualneuron.training.config import TrainConfig, BACKBONES, AREAS

load_dotenv()


def _build_config(args):
    kw = dict(area=args.area, backbone=args.backbone)
    if args.block is not None:
        kw["block"] = args.block
    for k in ("batch_size", "max_epochs", "lr", "gamma_readout", "num_workers"):
        v = getattr(args, k)
        if v is not None:
            kw[k] = v
    if args.device is not None:
        kw["device"] = args.device
    return TrainConfig(**kw)


def _member_cmd(args, seed):
    """The subprocess command to train a single member (overrides passed through)."""
    cmd = [sys.executable, "-u", "-m", "dualneuron.training.run",
           "--area", args.area, "--backbone", args.backbone,
           "--member", str(seed), "--device", "cuda"]
    if args.block is not None:
        cmd += ["--block", str(args.block)]
    for k in ("batch_size", "max_epochs", "lr", "gamma_readout", "num_workers"):
        v = getattr(args, k)
        if v is not None:
            cmd += [f"--{k}", str(v)]
    return cmd


def _memory_cap(cache_path, n_gpus, override=None, safety=0.9):
    """Max members to run concurrently so total RAM stays under the cgroup memory limit.

    Each member loads its own copy of the feature cache into RAM (the CIFS-backed cache can't be
    safely memory-mapped, and /dev/shm is too small to share it), so concurrency is bounded by
    ``floor(safety * cgroup_limit / per_member_RAM)``. Backbone-agnostic — the same rule yields all
    GPUs for ResNet's small cache and fewer for DINO's 21 GB cache.
    """
    if override:
        return max(1, min(n_gpus, override))
    try:
        limit = int(open("/sys/fs/cgroup/memory.max").read().strip())   # bytes ("max" if unlimited)
    except (OSError, ValueError):
        return n_gpus
    cache = os.path.getsize(cache_path)
    overhead = 5 * 1024 ** 3            # measured per-member process + DataLoader-worker footprint
    # The OS also keeps ~1 shared page-cache copy of the (CIFS) cache file, counted by the cgroup;
    # subtract it from the budget, then each member adds its own anon array + overhead.
    cap = int((safety * limit - cache) / (cache + overhead))
    return max(1, min(n_gpus, cap))


def _run_pool(args, seeds, gpus):
    """Queue-based pool: cache the trainable-part input once, train members across GPUs (capped by
    RAM), aggregate. The cache is frozen-core features (frozen twins) or transformed images
    (fine-tuned twins), per ``config.cache_kind``."""
    from dualneuron.data.recordings import load_sessions, build_response_matrix
    from dualneuron.training.features import extract_features, cache_images
    from dualneuron.training.trainer import aggregate_ensemble
    from dualneuron.utils import ensure_dir

    config = _build_config(args)
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    lf = None
    if config.logs_dir:
        pool_log = os.path.join(config.logs_dir, config.area, config.backbone, "pool.log")
        ensure_dir(os.path.dirname(pool_log))
        lf = open(pool_log, "w")

    def plog(msg, **kw):
        print(msg, flush=True)
        if lf:
            lf.write(msg + "\n")
            lf.flush()

    # 1) Cache the trainable-part input ONCE (writes the shared cache; workers reuse it): frozen-core
    #    features for a frozen twin, or transformed images for a fine-tuned twin.
    sessions = load_sessions(config.area)
    train_ids, _, _ = build_response_matrix(sessions, "train")
    test_ids, _, _ = build_response_matrix(sessions, "test")
    if config.cache_kind == "images":
        cache_train = cache_images(config, train_ids, "train")
        cache_images(config, test_ids, "test")
    else:
        cache_train = extract_features(config, train_ids, "train", device=dev)
        extract_features(config, test_ids, "test", device=dev)

    # 2) Queue-based pool, capped so concurrent members fit the cgroup memory limit.
    gpus = list(dict.fromkeys(gpus))
    max_concurrent = _memory_cap(cache_train, len(gpus), args.max_parallel)
    pending, active, failed = list(seeds), {}, []
    plog(f"[pool] {len(pending)} members, GPUs {gpus}, max {max_concurrent} concurrent "
         f"(cgroup-aware: each member loads the cache into its own RAM)", flush=True)
    while pending or active:
        for gpu in gpus:
            if gpu in active or not pending or len(active) >= max_concurrent:
                continue
            seed = pending.pop(0)
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = str(gpu)
            active[gpu] = (seed, subprocess.Popen(_member_cmd(args, seed), env=env))
            plog(f"[pool] seed {seed} -> GPU {gpu}", flush=True)
        time.sleep(5)
        for gpu in list(active):
            seed, proc = active[gpu]
            rc = proc.poll()
            if rc is None:
                continue
            if rc == 0:
                plog(f"[pool] seed {seed} done (GPU {gpu})", flush=True)
            else:
                failed.append((seed, rc))
                plog(f"[pool] seed {seed} FAILED rc={rc} (GPU {gpu})", flush=True)
            del active[gpu]

    if failed:
        plog(f"[pool] failed members: {[s for s, _ in failed]}", flush=True)
        sys.exit(1)

    # 3) Aggregate the saved members into correlations.npy (+ mask).
    aggregate_ensemble(config, device=dev)


def main():
    p = argparse.ArgumentParser(
        description="Train a digital-twin ensemble (frozen-core readout, or fine-tuned backbone+readout)")
    p.add_argument("--area", default="v4", choices=AREAS)
    p.add_argument("--backbone", default="dino", choices=BACKBONES)
    p.add_argument("--seeds", default="1,2,3,4,5", help="Comma-separated member seeds (-> file index).")
    p.add_argument("--member", type=int, default=None,
                   help="Train only this one member (no aggregation); used by the --gpus pool.")
    p.add_argument("--gpus", default=None, help="Comma-separated GPU ids for one-member-per-GPU training.")
    p.add_argument("--max-parallel", dest="max_parallel", type=int, default=None,
                   help="Cap concurrent members (default: auto from cgroup memory limit vs cache size).")
    p.add_argument("--block", type=int, default=None, help="DINOv3 block to read out (default: config).")
    p.add_argument("--batch_size", type=int, default=None)
    p.add_argument("--max_epochs", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--gamma_readout", type=float, default=None)
    p.add_argument("--num_workers", type=int, default=None)
    p.add_argument("--device", default=None, help="Torch device (default: cuda if available, else cpu).")
    p.add_argument("--log_path", default=None, help="Log file (default LOGS_DIR/{area}_{backbone}_*.log).")
    args = p.parse_args()

    # Single member (pool worker, or standalone).
    if args.member is not None:
        from dualneuron.training.trainer import train_member
        train_member(_build_config(args), args.member, device=args.device, log_path=args.log_path)
        return

    # Multi-GPU pool.
    if args.gpus is not None:
        seeds = [int(s) for s in args.seeds.split(",")]
        gpus = [int(g) for g in args.gpus.split(",")]
        _run_pool(args, seeds, gpus)
        return

    # Sequential ensemble.
    from dualneuron.training.trainer import train_ensemble
    config = _build_config(args)
    seeds = tuple(int(s) for s in args.seeds.split(","))
    train_ensemble(config, seeds=seeds, device=args.device, log_path=args.log_path)


if __name__ == "__main__":
    main()
