"""Central ``(area, backbone)`` twin registry — the single source of truth for the whole pipeline.

Every stage (training, screening, synthesis, DreamSim, similarity, figures) resolves a twin from an
``(area, backbone)`` pair here: which :func:`dualneuron.twins.nets.load_model` architecture to build,
its input geometry and normalization, its screening / synthesis constants, and where its analysis
artifacts, ``correlations.npy`` and RF ``mask.npy`` live.

**Storage convention** mirrors training: analysis outputs go under ``ANALYSIS_DIR/{area}/{backbone}/``
(folder-namespaced, bare filenames), exactly like ``FEATURES_DIR/{area}/{backbone}/`` and
``TRAINED_MODELS_DIR/{area}/{backbone}/``.

**Staged twins are read-only.** The shipped twins under ``dualneuron/twins/<folder>/``
(``V4ColorTaskDriven``, ``V4ColorDataDriven``, ``V1GrayTaskDriven``, ``V4GrayTaskDriven``) — their weights,
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
    train_upsample: Optional[int] = None   # optional pre-crop upsample side; None (all twins) -> off


# The twin catalog. Per area we have our trained backbone twin (v4 resnet / v1 convnext), the shipped
# read-only "staged" twin (V4ColorTaskDriven / V1GrayTaskDriven, the paper's original), and our trained
# dino; v4 additionally has the shipped read-only data-driven ensemble (V4ColorDataDriven), which is the
# data-driven counterpart of the staged task-driven twin on the same 394 neurons and the same geometry.
# staged_folder != None marks the shipped read-only twins; the others train into
# TRAINED_MODELS_DIR/{area}/{backbone}.
# Geometry (input_size/crop_size/channels/img_mean/img_std/n_neurons) and arch are verified. The
# screening / synthesis L2 constants (screen_norm, synth_target_norm, synth_values_range) are the
# established staged-twin values; the DINO twins inherit them as provisional starting points — NOT yet
# re-derived for the 224px DINO input, so revisit when the DINO synthesis/screening is actually run.
TWINS = {
    # v4: our trained resnet + the shipped staged twin (V4ColorTaskDriven, read-only) + our trained dino
    ("v4", "resnet"):   TwinSpec("v4", "resnet", "v4", 100, 200, 3, 113.5, 59.58, 394,
                                 40.0, "fourier", 40.0, (-1.9, 2.3), None),
    ("v4", "staged"):   TwinSpec("v4", "staged", "v4", 100, 200, 3, 113.5, 59.58, 394,
                                 40.0, "fourier", 40.0, (-1.9, 2.3), "V4ColorTaskDriven"),
    ("v4", "dino"):     TwinSpec("v4", "dino", "v4_dino", 224, 200, 3, 113.5, 59.58, 394,
                                 40.0, "fourier", 40.0, (-1.9, 2.3), None),
    ("v4", "data_driven"): TwinSpec("v4", "data_driven", "v4_data_driven", 100, 200, 3, 113.5, 59.58,
                                    394, 40.0, "fourier", 40.0, (-1.9, 2.3), "V4ColorDataDriven"),
    # v1: our trained convnext + the shipped staged twin (V1GrayTaskDriven, read-only) + our trained
    # dino. V1 stimulus transform = center-crop 93 of the stored 233x233 frame (= 233 - 2*70, the
    # nnvision config's crop: 70 per side); dino upsamples that 93 crop to its 224 input. The 93 crop
    # and the 167 screening crop are the same visual extent (the 233x233 frame stores a 420x420
    # field, so 93/(233/420) = 167).
    ("v1", "convnext"): TwinSpec("v1", "convnext", "v1", 93, 167, 1, 124.54466, 70.28, 458,
                                 12.0, "pixel", 12.0, (-1.77, 1.86), None, train_crop=93),
    ("v1", "staged"):   TwinSpec("v1", "staged", "v1", 93, 167, 1, 124.54466, 70.28, 458,
                                 12.0, "pixel", 12.0, (-1.77, 1.86), "V1GrayTaskDriven", train_crop=93),
    ("v1", "dino"):     TwinSpec("v1", "dino", "v1_dino", 224, 167, 1, 124.54466, 70.28, 458,
                                 12.0, "pixel", 12.0, (-1.77, 1.86), None, train_crop=93),
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
    """The backbones available for an area (e.g. ``['resnet', 'staged', 'dino']`` for v4)."""
    return [b for (a, b) in TWINS if a == area]


def check_pair(area: str, backbone: str, parser=None):
    """Validate that ``(area, backbone)`` is a real twin. With an argparse ``parser`` emit a clean
    ``parser.error`` (``--backbone`` choices span all areas, so a per-area pair must be checked after
    parsing); otherwise raise ``ValueError``."""
    if (area, backbone) not in TWINS:
        msg = f"{backbone!r} is not a backbone for area {area!r}; choose from {backbones_for(area)}"
        if parser is not None:
            parser.error(msg)
        raise ValueError(msg)


# ---------------------------------------------------------------------------
#  Artifact paths — ONE tree, THREE roots. Every artifact lives at
#  {ROOT}/{area}/{backbone}/<stage rel...>, ROOT = ANALYSIS_DIR (data) | LOGS_DIR
#  (logs) | PAPER_FIG_DIR (figures). A stage's relative path is IDENTICAL under all
#  three roots, so filenames stay constant and the FOLDER is the search axis.
#  Dataset-scoped stages nest under {dataset}/; model-intrinsic stages (synthesis +
#  its RF mask) sit at the twin root.
# ---------------------------------------------------------------------------

_ROOTS = {"analysis": "ANALYSIS_DIR", "log": "LOGS_DIR", "fig": "PAPER_FIG_DIR"}

SYNTHESIS_VARIANTS = ("free", "axis")   # free = plain activation ascent; axis = population-axis method


def artifact_dir(kind: str, area: str, backbone: str) -> Optional[str]:
    """``{ROOT}/{area}/{backbone}`` for ``kind`` in {"analysis","log","fig"} (None if the root unset)."""
    base = env_dir(_ROOTS[kind])
    return os.path.join(base, area, backbone) if base else None


def _art(kind, area, backbone, *rel):
    d = artifact_dir(kind, area, backbone)
    return os.path.join(d, *rel) if d else None


def analysis_dir(area: str, backbone: str) -> Optional[str]:
    """``ANALYSIS_DIR/{area}/{backbone}`` (None if ANALYSIS_DIR is unset)."""
    return artifact_dir("analysis", area, backbone)


# --- stage relative paths (shared by data / log / fig so the three mirror exactly) ---
def rel_screening(dataset, field="masked"):
    return (dataset, "screening", field)


def rel_dreamsim(dataset):
    return (dataset, "dreamsim")


def rel_synthesis(variant):
    if variant not in SYNTHESIS_VARIANTS:
        raise ValueError(f"unknown synthesis variant {variant!r}; expected one of {SYNTHESIS_VARIANTS}")
    return ("synthesis", variant)


def log_path(area, backbone, *rel):
    """LOGS_DIR mirror; ``rel`` = stage parts ending in the .log filename, e.g.
    ``log_path(area, backbone, *rel_screening(dataset, field), "screening.log")``."""
    return _art("log", area, backbone, *rel)


def fig_path(area, backbone, *rel):
    """PAPER_FIG_DIR mirror; ``rel`` = stage parts ending in the .pdf filename (or just the .pdf for a
    model-level figure, e.g. ``fig_path(area, backbone, "accuracy.pdf")``)."""
    return _art("fig", area, backbone, *rel)


def screening_path(area, backbone, dataset, kind, field="masked", run="ensemble"):
    """Screening cache ``.../{dataset}/screening/{field}/{kind}.npz``.

    ``kind`` is ``"responses"`` or ``"indices"``; ``field`` is ``"masked"`` (RF-masked + L2-normed) or
    ``"full"`` (full-field natural, no mask/L2). ``run="ensemble"`` (default) is unprefixed; a per-member
    run nests under ``member{i}/``.
    """
    rel = rel_screening(dataset, field)
    if run != "ensemble":
        rel = rel + (run,)
    return _art("analysis", area, backbone, *rel, f"{kind}.npz")


def dreamsim_embeddings_path(area, backbone, dataset):
    """DreamSim embeddings ``.../{dataset}/dreamsim/embeddings.npz``."""
    return _art("analysis", area, backbone, *rel_dreamsim(dataset), "embeddings.npz")


def dreamsim_indices_path(area, backbone, dataset="imagenet"):
    """DreamSim subset indices ``.../{dataset}/dreamsim/indices.npy`` (imagenet by default)."""
    return _art("analysis", area, backbone, *rel_dreamsim(dataset), "indices.npy")


def similarity_path(area, backbone, dataset):
    """Similarity results ``.../{dataset}/dreamsim/similarity.npz`` (Fig 6 / 9 / 10)."""
    return _art("analysis", area, backbone, *rel_dreamsim(dataset), "similarity.npz")


def synthesis_dir(area, backbone, variant="free"):
    """Variant synthesis dir ``.../synthesis/{variant}`` (self-contained: holds output/ + mask.npy)."""
    return _art("analysis", area, backbone, *rel_synthesis(variant))


def synthesis_output_dir(area, backbone, variant="free"):
    """Per-neuron MEI/LEI dir ``.../synthesis/{variant}/output``."""
    d = synthesis_dir(area, backbone, variant)
    return os.path.join(d, "output") if d else None


def synthesis_neuron_path(area, backbone, neuron, variant="free"):
    """One neuron's MEI/LEI npz ``.../synthesis/{variant}/output/neuron{id:04d}.npz``."""
    d = synthesis_output_dir(area, backbone, variant)
    return os.path.join(d, f"neuron{int(neuron):04d}.npz") if d else None


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


def mask_path(area: str, backbone: str, variant: str = "axis") -> Optional[str]:
    """Authoritative RF mask to READ, in precedence order: the regenerated
    ``.../synthesis/{variant}/mask.npy`` (default the ``axis`` variant -- the better RF estimate), then
    the regenerated ``free`` mask, then this twin's shipped ``twins/<staged_folder>/mask.npy``.
    Screening / DreamSim / neuron-strips read this.

    A regenerated mask always wins, so a staged twin's shipped mask serves only until that twin's own
    MEIs/LEIs exist and :mod:`dualneuron.synthesis.mask` has rebuilt one -- it then retires itself with
    no bookkeeping. A twin with no ``staged_folder`` (a trained one) has no shipped mask to fall back
    to, so it must regenerate its own; the returned path is then the canonical ``variant`` location,
    which does not exist yet, and the caller's ``np.load`` fails there pointing at it.
    """
    p = regenerated_mask_path(area, backbone, variant)
    if p is not None and os.path.exists(p):
        return p
    if variant == "axis":
        free = regenerated_mask_path(area, backbone, "free")
        if free is not None and os.path.exists(free):
            return free
    staged = staged_mask_path(area, backbone)
    if staged is not None and os.path.exists(staged):
        return staged
    return p


def staged_mask_path(area: str, backbone: str) -> Optional[str]:
    """This twin's shipped read-only RF mask, ``twins/<staged_folder>/mask.npy``, or ``None`` for a
    twin produced only by training (no ``staged_folder``). Read-only: :mod:`dualneuron.synthesis.mask`
    writes to :func:`regenerated_mask_path`, never here."""
    spec = resolve(area, backbone)
    if spec.staged_folder is None:
        return None
    return os.path.join(_TWINS_DIR, spec.staged_folder, "mask.npy")


def regenerated_mask_path(area: str, backbone: str, variant: str = "axis") -> Optional[str]:
    """Where :mod:`dualneuron.synthesis.mask` WRITES the RF mask built from a variant's MEIs/LEIs:
    ``.../synthesis/{variant}/mask.npy`` (beside that variant's output/)."""
    d = synthesis_dir(area, backbone, variant)
    return os.path.join(d, "mask.npy") if d else None


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
        responses_path = screening_path(area, backbone, "imagenet", "responses")
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
