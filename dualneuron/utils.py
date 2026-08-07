"""
Small shared utilities for the dualneuron package.

Twin-aware helpers (well_predicted_neurons, sparse_split, the area/backbone catalog and its analysis
paths) live in :mod:`dualneuron.twins.registry` -- the single source of truth for the pipeline.
"""
import os
from pathlib import Path


def env_dir(name, default=None):
    """
    Read a directory path from an environment variable, normalized.

    Returns os.path.normpath of the variable's value, collapsing any '//' or
    trailing slash so paths are clean regardless of how the variable was written
    in .env (e.g. a trailing slash on DATA_DIR no longer leaks into ${DATA_DIR}/x
    interpolations). Falls back to `default` (also normalized when it is a string)
    if the variable is unset.

    Args:
        name: Environment variable name to read.
        default: Value returned when the variable is unset. Default: None.

    Returns:
        str or None: The normalized directory path, or `default` when unset and
            `default` is not a string.
    """
    value = os.getenv(name)
    if value is None:
        value = default
    return os.path.normpath(value) if isinstance(value, str) else value


def default_workers(cap=16):
    """
    DataLoader workers to default to on *this* machine.

    Loading the recorded stimuli is latency-bound small-file I/O -- EXPERIMENT_DIR is often a
    network share -- so concurrency is the lever that matters. But the right number is a property of
    the machine, not of this repository: a hard-coded default either starves a big host or
    oversubscribes a small one. This reads the process's actual CPU affinity, so a cgroup/cpuset
    limit is respected (unlike os.cpu_count(), which reports the host's cores regardless), and caps
    it so a very large host does not open an unreasonable number of concurrent readers.

    Callers should keep exposing an explicit override (e.g. ``--num_workers``); this is only the
    value used when the caller does not choose one.

    Args:
        cap: Upper bound on the returned count. Default: 16.

    Returns:
        int: Worker count in [1, cap].
    """
    try:
        available = len(os.sched_getaffinity(0))      # respects cgroup/cpuset limits
    except AttributeError:                            # not available off Linux
        available = os.cpu_count() or 1
    return max(1, min(cap, available))


def ensure_dir(path):
    """
    Create a directory (and any missing parents), returning it as a Path.

    Idempotent: an existing directory is left untouched. Call this at the point a
    script is about to write into an output location (figures, logs, analysis
    files) so the folder is created on demand rather than being committed to the
    repository.

    Args:
        path: Directory to create. Parent directories are created as needed.

    Returns:
        pathlib.Path: The directory, as a Path object.
    """
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def should_compute(path, rewrite=False):
    """Central rewrite gate: True if an artifact must be (re)computed rather than reused.

    Returns True when ``rewrite`` is set, when ``path`` is None, or when the file is absent — i.e.
    exactly when a producer should compute + write. This is the single primitive behind every
    ``--rewrite`` flag, replacing the ad-hoc ``if os.path.exists(...): skip`` scattered across
    features/screening/synthesis. Producers with a bespoke save (multi-array npz, per-neuron loops,
    figures) call this directly; simple one-file producers can use :func:`load_or_compute`.
    """
    return rewrite or path is None or not os.path.exists(path)


def load_or_compute(path, compute, rewrite=False, load=None, save=None):
    """Load ``path`` if present (and not ``rewrite``), else ``compute()`` + save to ``path``.

    Convenience wrapper over :func:`should_compute` for the common one-file case: default reuses the
    saved form; ``rewrite=True`` recomputes and overwrites. Parent dirs are created on write.

    Args:
        path: Cache file path.
        compute: Zero-arg callable producing the result when (re)computing.
        rewrite: If True, always recompute + overwrite.
        load: Loader ``load(path)`` (default ``numpy.load``).
        save: Saver ``save(path, result)`` (default ``numpy.save``).
    """
    import numpy as np
    load = load or np.load
    save = save or (lambda p, r: np.save(p, r))
    if not should_compute(path, rewrite):
        return load(path)
    result = compute()
    if path is not None:
        ensure_dir(os.path.dirname(path))
        save(path, result)
    return result


class RewriteLine:
    """
    File-like sink that collapses a tqdm progress bar to one rewritten line.

    tqdm emits each update as a carriage-return-prefixed string; written verbatim
    to a file this leaves many '\\r'-separated segments that a text editor renders
    as separate lines. This sink instead truncates back to a fixed anchor (just
    past any header already written) and rewrites the latest bar, so the log file
    always holds exactly one progress line, readable in any editor.

    Args:
        fileobj: Open writable text file the progress line is rewritten into.
        anchor: Byte offset (from fileobj.tell()) to truncate back to on each
            update; everything before it (the header) is preserved.
    """
    def __init__(self, fileobj, anchor):
        self._f = fileobj
        self._anchor = anchor

    def write(self, text):
        text = text.strip("\r\n")
        if not text:
            return
        self._f.seek(self._anchor)
        self._f.truncate()
        self._f.write(text + "\n")
        self._f.flush()

    def flush(self):
        self._f.flush()
