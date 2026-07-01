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
