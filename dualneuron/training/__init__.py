"""Digital-twin readout training, per ``(area, backbone)``.

Frozen-core features are extracted once to ``FEATURES_DIR``; the readout is then trained on them and
the ensemble saved to ``TRAINED_MODELS_DIR`` (loadable via ``twins.nets.load_model(weights_dir=...)``).
"""

# datajoint (imported transitively via nnfabrik/nnvision) replaces sys.excepthook with a handler
# that logs only "Uncaught exception" and hides the stack — catastrophic for debugging training
# subprocesses. Restore Python's default excepthook on import.
import sys as _sys
_sys.excepthook = _sys.__excepthook__

from dualneuron.training.config import TrainConfig, TWIN_SPECS, BACKBONES, AREAS
from dualneuron.training.features import extract_features, input_cache_path, load_features
from dualneuron.training.trainer import (
    train_ensemble,
    train_member,
    aggregate_ensemble,
    build_trainable_twin,
    poisson_loss,
    corr_per_neuron,
)

__all__ = [
    "TrainConfig", "TWIN_SPECS", "BACKBONES", "AREAS",
    "extract_features", "input_cache_path", "load_features",
    "train_ensemble", "train_member", "aggregate_ensemble", "build_trainable_twin",
    "poisson_loss", "corr_per_neuron",
]
