"""Central ``(area, backbone)`` twin registry — the single source of truth for the whole pipeline.

Every stage (training, screening, synthesis, DreamSim, similarity, figures) resolves a twin from an
``(area, backbone)`` pair here: which :func:`dualneuron.twins.nets.load_model` architecture to build,
its input geometry and normalization, its screening / synthesis constants, and where its analysis
artifacts, ``correlations.npy`` and RF ``mask.npy`` live.

**Storage convention** mirrors training: analysis outputs go under ``ANALYSIS_DIR/{area}/{backbone}/``
(folder-namespaced, bare filenames), exactly like ``FEATURES_DIR/{area}/{backbone}/`` and
``TRAINED_MODELS_DIR/{area}/{backbone}/``.

**Staged twins are read-only.** The shipped twins under ``dualneuron/twins/<folder>/``
(``V4ColorTaskDriven``, ``V1GrayTaskDriven``, ``V4GrayTaskDriven``) — their weights,
``correlations.npy`` and ``mask.npy`` — are never modified. A newly trained twin's weights +
``correlations.npy`` live in ``TRAINED_MODELS_DIR/{area}/{backbone}/``, and its regenerated RF mask
in ``ANALYSIS_DIR/{area}/{backbone}/mask.npy``.
"""

import os
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from dualneuron.utils import env_dir

# Directory holding the GitHub-staged twins (dualneuron/twins/<folder>/).
_TWINS_DIR = os.path.dirname(os.path.abspath(__file__))


@dataclass(frozen=True)
class TwinSpec:
    """Everything the pipeline needs to resolve one ``(area, backbone)`` twin.

    ``arch`` is the :func:`load_model` string. Geometry (``input_size``/``crop_size``/``channels``/
    ``img_mean``/``img_std``) drives every stage's transform. ``screen_norm`` is the L2 norm applied
    to masked images in screening; ``synth_*`` parameterize MEI/LEI synthesis. ``staged_folder`` names
    the read-only shipped-twin folder, or is ``None`` for a twin produced only by training.
    """

    area: str
    backbone: str
    arch: str
    input_size: int
    crop_size: int
    channels: int
    img_mean: float
    img_std: float
    n_neurons: int
    screen_norm: float
    synth_method: str                 # "fourier" (RGB freq-domain) or "pixel" (grayscale)
    synth_target_norm: float
    synth_values_range: Tuple[float, float]
    staged_folder: Optional[str]
    # Training-transform geometry (how the RECORDED stimuli are fed to the twin): optional pre-crop
    # upsample, then a center-crop, then resize to input_size. Distinct from crop_size, which is the
    # SCREENING RF crop -- screening never reads these. Defaults reproduce the plain V4 crop.
    train_crop: Optional[int] = None       # center-crop side for training (None -> use crop_size)
    train_upsample: Optional[int] = None   # pre-crop upsample side (V1 stimulus: 420); None -> off


# The twin catalog: the four (area, backbone) twins of the dual-model paper.
# Geometry (input_size/crop_size/channels/img_mean/img_std/n_neurons) and arch are verified — from
# the staged twins (v4/resnet, v1/convnext) and from training (v4/dino, v1/dino). The screening /
# synthesis L2 constants (screen_norm, synth_target_norm, synth_values_range) are the established,
# validated values for the staged twins; for the DINO twins they are inherited from the area's staged
# twin as provisional starting points — NOT yet re-derived for the 224px DINO input, so revisit them
# when the DINO synthesis/screening is actually run.
TWINS = {
    ("v4", "resnet"):   TwinSpec("v4", "resnet", "v4", 100, 200, 3, 113.5, 59.58, 394,
                                 40.0, "fourier", 40.0, (-1.9, 2.3), "V4ColorTaskDriven"),
    ("v4", "dino"):     TwinSpec("v4", "dino", "v4_dino", 224, 200, 3, 113.5, 59.58, 394,
                                 40.0, "fourier", 40.0, (-1.9, 2.3), None),
    ("v1", "convnext"): TwinSpec("v1", "convnext", "v1", 93, 167, 1, 124.54466, 70.28, 458,
                                 12.0, "pixel", 12.0, (-1.77, 1.86), None,
                                 train_crop=280, train_upsample=420),
    ("v1", "dino"):     TwinSpec("v1", "dino", "v1_dino", 224, 167, 1, 124.54466, 70.28, 458,
                                 12.0, "pixel", 12.0, (-1.77, 1.86), None,
                                 train_crop=280, train_upsample=420),
}

AREAS = list(dict.fromkeys(a for a, _ in TWINS))
BACKBONES = list(dict.fromkeys(b for _, b in TWINS))
# Read-only shipped-twin folder names (guards against writing into them).
STAGED_FOLDERS = {s.staged_folder for s in TWINS.values() if s.staged_folder}


def resolve(area: str, backbone: str) -> TwinSpec:
    """Return the :class:`TwinSpec` for ``(area, backbone)`` (raises on an unknown pair)."""
    key = (area, backbone)
    if key not in TWINS:
        raise ValueError(f"Unknown (area, backbone)={key}; expected one of {list(TWINS)}.")
    return TWINS[key]


def backbones_for(area: str):
    """The backbones available for an area (e.g. ``['resnet', 'dino']`` for v4)."""
    return [b for (a, b) in TWINS if a == area]


# ---------------------------------------------------------------------------
#  Analysis-output paths — ANALYSIS_DIR/{area}/{backbone}/... (bare, folder-namespaced)
# ---------------------------------------------------------------------------

def analysis_dir(area: str, backbone: str) -> Optional[str]:
    """``ANALYSIS_DIR/{area}/{backbone}`` (None if ANALYSIS_DIR is unset)."""
    base = env_dir("ANALYSIS_DIR")
    return os.path.join(base, area, backbone) if base else None


def screening_path(area: str, backbone: str, run: str, dataset: str, kind: str,
                   field: str = "masked") -> str:
    """Screening cache: ``.../{run}_{dataset}[_fullfield]_ordered_{kind}.npz``.

    ``run`` is ``"ensemble"`` or ``"member{i}"``; ``kind`` is ``"responses"`` or ``"indices"``.
    ``field`` is ``"masked"`` (RF-masked + L2-normed screening) or ``"full"`` (full-field: no mask,
    no L2 — the natural-image regime, e.g. for the population axis). ``masked`` keeps the original
    name so existing files still resolve.
    """
    regime = "" if field == "masked" else "fullfield_"
    return os.path.join(analysis_dir(area, backbone), f"{run}_{dataset}_{regime}ordered_{kind}.npz")


def dreamsim_embeddings_path(area: str, backbone: str, dataset: str) -> str:
    """DreamSim embeddings: ``.../dreamsim_{dataset}_embeddings.npz``."""
    return os.path.join(analysis_dir(area, backbone), f"dreamsim_{dataset}_embeddings.npz")


def dreamsim_indices_path(area: str, backbone: str) -> str:
    """ImageNet DreamSim subset indices: ``.../dreamsim_imagenet_indices.npy``."""
    return os.path.join(analysis_dir(area, backbone), "dreamsim_imagenet_indices.npy")


def similarity_path(area: str, backbone: str, dataset: str) -> str:
    """Similarity results: ``.../similarity_{dataset}.npz``."""
    return os.path.join(analysis_dir(area, backbone), f"similarity_{dataset}.npz")


# Synthesis methods: "free" is the original free ascent (dir kept as plain ``synthesis`` so the
# shipped MEIs/LEIs and everything downstream are untouched); "axis" is the population-axis
# (folded) method, written alongside in ``synthesis_axis``.
SYNTHESIS_VARIANTS = ("free", "axis")


def synthesis_dir(area: str, backbone: str, variant: str = "free") -> str:
    """MEI/LEI output dir: ``ANALYSIS_DIR/{area}/{backbone}/synthesis`` (free) or
    ``.../synthesis_{variant}`` for another synthesis method."""
    if variant not in SYNTHESIS_VARIANTS:
        raise ValueError(f"unknown synthesis variant {variant!r}; expected one of {SYNTHESIS_VARIANTS}")
    name = "synthesis" if variant == "free" else f"synthesis_{variant}"
    return os.path.join(analysis_dir(area, backbone), name)


def synthesis_neuron_path(area: str, backbone: str, neuron: int, variant: str = "free") -> str:
    """One neuron's MEI/LEI npz: ``.../{synthesis dir}/neuron{id:04d}.npz``."""
    return os.path.join(synthesis_dir(area, backbone, variant), f"neuron{int(neuron):04d}.npz")


# ---------------------------------------------------------------------------
#  Correlations + RF mask — staged (read-only) vs trained/regenerated
# ---------------------------------------------------------------------------

def correlations_path(area: str, backbone: str, weights_dir: Optional[str] = None) -> Optional[str]:
    """Path to this twin's ``correlations.npy``.

    ``weights_dir`` overrides; otherwise the staged (read-only) twin folder for a shipped twin, else
    ``TRAINED_MODELS_DIR/{area}/{backbone}/correlations.npy``.
    """
    if weights_dir is not None:
        return os.path.join(weights_dir, "correlations.npy")
    spec = resolve(area, backbone)
    if spec.staged_folder is not None:
        return os.path.join(_TWINS_DIR, spec.staged_folder, "correlations.npy")
    tm = env_dir("TRAINED_MODELS_DIR")
    return os.path.join(tm, area, backbone, "correlations.npy") if tm else None


def weights_dir(area: str, backbone: str) -> Optional[str]:
    """Default ensemble-weights dir for a twin, keyed off ``staged_folder`` (single switch, matching
    :func:`correlations_path`/:func:`mask_path`): ``None`` for a shipped/staged twin -- ``load_model``
    then reads the read-only ``twins/<folder>`` weights -- else ``TRAINED_MODELS_DIR/{area}/{backbone}``
    (the trained ensemble). Downstream loaders resolve their ``weights_dir`` through this so a user just
    picks ``(area, backbone)`` and gets matching weights + correlations + mask; pass one to override."""
    spec = resolve(area, backbone)
    if spec.staged_folder is not None:
        return None
    tm = env_dir("TRAINED_MODELS_DIR")
    return os.path.join(tm, area, backbone) if tm else None


def mask_path(area: str, backbone: str) -> str:
    """Authoritative RF mask to READ: a shipped twin's staged (read-only) ``mask.npy``, else the
    regenerated ``ANALYSIS_DIR/{area}/{backbone}/mask.npy`` (written by :mod:`dualneuron.synthesis.mask`)."""
    spec = resolve(area, backbone)
    if spec.staged_folder is not None:
        return os.path.join(_TWINS_DIR, spec.staged_folder, "mask.npy")
    return regenerated_mask_path(area, backbone)


def regenerated_mask_path(area: str, backbone: str) -> str:
    """Where :mod:`dualneuron.synthesis.mask` WRITES the mask built from this twin's MEIs/LEIs —
    always the non-staged ``ANALYSIS_DIR/{area}/{backbone}/mask.npy`` (staged masks are never overwritten)."""
    return os.path.join(analysis_dir(area, backbone), "mask.npy")


# ---------------------------------------------------------------------------
#  Neuron selection (correlation-to-average > threshold; sparse / non-sparse split)
# ---------------------------------------------------------------------------

def well_predicted_neurons(area: str, backbone: str, threshold: float = 0.4,
                           weights_dir: Optional[str] = None) -> np.ndarray:
    """Global indices of neurons whose correlation-to-average exceeds ``threshold`` for this twin."""
    corr = np.load(correlations_path(area, backbone, weights_dir))
    return np.where(corr > threshold)[0]


def sparse_split(area: str, backbone: str, threshold: float = 2.0,
                 responses_path: Optional[str] = None, weights_dir: Optional[str] = None) -> dict:
    """Split this twin's well-predicted neurons into sparse / non-sparse by ImageNet-screening skewness.

    Following Franke et al., a neuron's lifetime sparsity is the skewness of its predicted responses
    to the ImageNet screening set; skewness < ``threshold`` (2.0) is non-sparse. Restricted to the
    well-predicted neurons (correlation-to-average > 0.4).

    Args:
        area, backbone: The twin.
        threshold: Skewness cutoff; skewness < threshold is non-sparse.
        responses_path: ImageNet screening responses npz (keyed ``unit_{neuron}``); default the
            registry's ensemble ImageNet screening path for this twin.
        weights_dir: Passed to :func:`well_predicted_neurons` (trained-ensemble correlations).

    Returns:
        dict with ``neurons``, ``skewness``, ``non_sparse``, ``sparse`` (all np.ndarray).
    """
    neurons = well_predicted_neurons(area, backbone, weights_dir=weights_dir)
    if responses_path is None:
        responses_path = screening_path(area, backbone, "ensemble", "imagenet", "responses")
    if not os.path.exists(responses_path):
        raise FileNotFoundError(
            f"Screening responses not found: {responses_path}. Run the imagenet screening first: "
            f"python -m dualneuron.screening.run --area {area} --backbone {backbone} --dataset imagenet")
    responses = np.load(responses_path)

    from scipy.stats import skew
    skewness = np.array([skew(responses[f"unit_{int(n)}"]) for n in neurons])
    return {
        "neurons": neurons,
        "skewness": skewness,
        "non_sparse": neurons[skewness < threshold],
        "sparse": neurons[skewness >= threshold],
    }
