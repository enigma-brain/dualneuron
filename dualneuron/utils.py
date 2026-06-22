"""
Small shared utilities for the dualneuron package.
"""
import os
from pathlib import Path

import numpy as np


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


def well_predicted_neurons(model_name, threshold=0.4):
    """
    Global indices of the well-predicted neurons of a twin model.

    Reads the per-model correlations.npy shipped under
    dualneuron/twins/{model_name}/ (each neuron's correlation-to-average on
    held-out test images) and returns the indices that clear the inclusion
    threshold used throughout the analyses.

    Args:
        model_name: Twin model folder name, e.g. "V4ColorTaskDriven" or
            "V1GrayTaskDriven".
        threshold: Minimum correlation-to-average for inclusion. Default: 0.4.

    Returns:
        np.ndarray: Sorted 1-D array of global neuron indices with
            correlation > threshold.
    """
    path = Path(__file__).resolve().parent / "twins" / model_name / "correlations.npy"
    corr = np.load(path)
    return np.where(corr > threshold)[0]


# Analysis area -> twin model folder name.
_AREA_MODELS = {"v1": "V1GrayTaskDriven", "v4": "V4ColorTaskDriven"}


def sparse_split(area, threshold=2.0, responses_path=None):
    """
    Split an area's well-predicted neurons into sparse and non-sparse sets.

    Following Franke et al., a neuron's lifetime sparsity is the skewness of its
    predicted responses to the ImageNet screening set; neurons with skewness below
    `threshold` (2.0 in the paper) are non-sparse, the rest sparse. The split is
    restricted to the well-predicted neurons (correlation-to-average > 0.4) of the
    area, obtained via well_predicted_neurons.

    Args:
        area: "v1" or "v4".
        threshold: Skewness cutoff; skewness < threshold is non-sparse. Default: 2.0.
        responses_path: Path to the imagenet screening responses npz (keyed
            unit_{neuron}). Default:
            ANALYSIS_DIR/{area}/{area}_ensemble_imagenet_ordered_responses.npz.

    Returns:
        dict: {
            "neurons": well-predicted neuron indices (np.ndarray),
            "skewness": per-neuron skewness aligned to "neurons" (np.ndarray),
            "non_sparse": neuron indices with skewness < threshold (np.ndarray),
            "sparse": neuron indices with skewness >= threshold (np.ndarray),
        }

    Raises:
        FileNotFoundError: If the screening responses npz is missing, with a message
            pointing to the screening command that produces it.
    """
    neurons = well_predicted_neurons(_AREA_MODELS[area])

    if responses_path is None:
        responses_path = os.path.join(
            env_dir("ANALYSIS_DIR"), area, f"{area}_ensemble_imagenet_ordered_responses.npz"
        )
    if not os.path.exists(responses_path):
        raise FileNotFoundError(
            f"Screening responses not found: {responses_path}. "
            f"Run the imagenet screening first: "
            f"python -m dualneuron.screening.run --model {area} --dataset imagenet"
        )
    responses = np.load(responses_path)

    # Fisher-Pearson skewness (scipy default); invariant to the ascending sort of the
    # screening responses. Imported lazily to keep the rest of utils dependency-light.
    from scipy.stats import skew
    skewness = np.array([skew(responses[f"unit_{int(n)}"]) for n in neurons])

    return {
        "neurons": neurons,
        "skewness": skewness,
        "non_sparse": neurons[skewness < threshold],
        "sparse": neurons[skewness >= threshold],
    }
