"""Configuration for digital-twin readout / fine-tuning training.

A twin is trained per ``(area, backbone)`` and saved to ``TRAINED_MODELS_DIR/{area}/{backbone}/``;
all locations are env-driven (see ``.env``). Two training regimes, selected by ``fine_tune``:

* ``fine_tune=False`` (V4 resnet, V4 dino) — the core is **frozen**, so its feature maps are extracted
  once to ``FEATURES_DIR`` and only the readout is trained (``cache_kind="features"``).
* ``fine_tune=True`` (V1 convnext, V1 dino) — the (truncated) backbone is **fine-tuned end-to-end**
  with the readout; the fixed *input images* are cached instead (``cache_kind="images"``).

``kind`` selects the backbone family: ``"nnvision"`` (resnet/convnext core+readout) or ``"dino"``
(DINOv3 + our Gaussian readout).
"""

import os
from dataclasses import dataclass, field
from typing import Optional

from dotenv import load_dotenv

from dualneuron.utils import env_dir
from dualneuron.twins import registry

load_dotenv()

# Training-specific spec per (area, backbone). The input geometry (input_size, crop_size, channels,
# img_mean/img_std, n_neurons) and the load_model arch come from the central registry
# (dualneuron.twins.registry) — the single source of truth shared with the analysis pipeline; only
# the training-specific fields live here:
#   kind        "nnvision" (resnet/convnext core+readout) or "dino" (DINOv3 + our readout)
#   fine_tune   False -> frozen core, cache features, train readout only
#               True  -> fine-tune the (truncated) backbone + readout on cached images
#   block       DINO transformer block to read out (V4=4, V1=1); layer_name for nnvision cores
#   readout_nonlin  post-BatchNorm, pre-readout nonlinearity; for the nnvision cores it is built into
#               the core (OutNonlin), for DINO it is applied explicitly. The task-driven cores use
#               ReLU in V4 and GELU in V1; both DINO twins use GELU, so the V4 pair is deliberately
#               NOT matched on this -- v4/resnet reads out after a ReLU, v4/dino after a GELU. Keep
#               that in mind when attributing a v4 resnet-vs-dino difference to the backbone.
#   elu_offset  output nonlinearity is ELU(x + elu_offset) + 1 (matches nnvision EncoderShifter).
#   gamma_readout  weight of the readout L1 term: gamma * sum|readout channel weights|, the summed
#               convention nnvision regularized under (see TrainableTwin.regularizer). V4=3.0,
#               V1=10.0 -- each area's nnvision config value. Overridable per run via --gamma_readout.
#   batch_size  training batch size (and the default for feature/image caching); V4=64, V1=128 --
#               each area's original nnvision dataset config. Overridable per run via --batch_size.
TWIN_SPECS = {
    ("v4", "resnet"):   dict(kind="nnvision", fine_tune=False,
                             model_name="resnet50_l2_eps0_1", layer_name="layer3.0",
                             feature_dim=None, spatial_size=None, block=None, readout_type=None,
                             readout_nonlin="relu", elu_offset=-1, gamma_readout=3.0, batch_size=64),
    ("v4", "dino"):     dict(kind="dino", fine_tune=False,
                             model_name="dinov3_vitb16", layer_name=None,
                             feature_dim=768, spatial_size=14, block=4, readout_type="fullgaussian2d",
                             readout_nonlin="gelu", elu_offset=-1, gamma_readout=3.0, batch_size=64),
    ("v1", "convnext"): dict(kind="nnvision", fine_tune=True,
                             model_name="facebook/convnextv2-atto-1k-224",
                             layer_name="convnextv2.encoder.stages.1.layers.0",
                             feature_dim=None, spatial_size=None, block=None, readout_type=None,
                             readout_nonlin="gelu", elu_offset=-1, gamma_readout=10.0, batch_size=128),
    ("v1", "dino"):     dict(kind="dino", fine_tune=True,
                             model_name="dinov3_vitb16", layer_name=None,
                             feature_dim=768, spatial_size=14, block=1, readout_type="fullgaussian2d",
                             readout_nonlin="gelu", elu_offset=-1, gamma_readout=10.0, batch_size=128),
}

AREAS = list(dict.fromkeys(a for (a, _) in TWIN_SPECS))
BACKBONES = sorted({bb for (_, bb) in TWIN_SPECS})


def _join(base: Optional[str], *parts: str) -> Optional[str]:
    """``os.path.join`` that propagates None when the env var is unset."""
    return os.path.join(base, *parts) if base else None


@dataclass
class TrainConfig:
    """Hyperparameters and resolved paths for one ``(area, backbone)`` training run."""

    area: str = "v4"
    backbone: str = "dino"

    # Readout / regularization. gamma_readout None -> the per-(area,backbone) spec value.
    gamma_readout: Optional[float] = None
    # Training batch size. None -> the per-(area,backbone) spec value (V4=64, V1=128).
    batch_size: Optional[int] = None
    init_mu_range: float = 0.4
    init_sigma_range: float = 0.6
    val_fraction: float = 0.2

    # Optimization (Adam + ReduceLROnPlateau, early-stop after lr_decay_steps reductions). These are
    # nnvision's trainer defaults, which is the regime the shipped twins were produced under:
    # lr_init 5e-3, factor 0.3, patience 5, an *absolute* plateau threshold of 1e-6, an LR floor of
    # 1e-4, 3 decays and at most 100 epochs. No gradient clipping, also as there.
    lr: float = 5e-3
    weight_decay: float = 0.0
    lr_decay_factor: float = 0.3
    lr_decay_patience: int = 5
    lr_decay_steps: int = 3
    lr_threshold: float = 1e-6
    min_lr: float = 1e-4
    max_epochs: int = 100
    # NOT the train/val split seed: each ensemble member is split (and initialized) with its OWN
    # member seed -- the 1..5 of ``--seeds``, passed straight to ``split_train_val`` in the trainer --
    # so the members differ in both init and split. Nothing reads this field.
    seed: int = 42

    # System.
    device: str = "cuda"
    num_workers: int = 0            # training loader (reads the in-RAM cache; 0 avoids worker copies)
    extract_num_workers: int = 4    # extraction/caching loader (parallel CIFS reads + GPU overlap)
    pin_memory: bool = True

    # Optional explicit overrides (else taken from the spec in __post_init__).
    block: Optional[int] = None
    readout_type: Optional[str] = None

    # Resolved at construction (not set by the caller).
    n_neurons: int = field(init=False)
    img_mean: float = field(init=False)
    img_std: float = field(init=False)
    kind: str = field(init=False)
    input_size: int = field(init=False)
    crop_size: int = field(init=False)
    train_crop: int = field(init=False)              # training-transform center-crop (defaults to crop_size)
    train_upsample: Optional[int] = field(init=False)  # optional pre-crop upsample side (V1: 420); None = off
    channels: int = field(init=False)
    fine_tune: bool = field(init=False)
    cache_kind: str = field(init=False)          # "features" (frozen) or "images" (fine-tuned)
    feature_dim: Optional[int] = field(init=False)
    spatial_size: Optional[int] = field(init=False)
    readout_nonlin: str = field(init=False)      # "relu" or "gelu": post-BN pre-readout nonlinearity
    elu_offset: float = field(init=False)        # output ELU(x + offset) + 1
    model_name: str = field(init=False)
    layer_name: Optional[str] = field(init=False)
    image_dir: Optional[str] = field(init=False)
    features_dir: Optional[str] = field(init=False)
    trained_dir: Optional[str] = field(init=False)
    dino_model_dir: Optional[str] = field(init=False)
    logs_dir: Optional[str] = field(init=False)

    def __post_init__(self):
        key = (self.area, self.backbone)
        if key not in TWIN_SPECS:
            raise ValueError(f"Unknown (area, backbone)={key}; expected one of {list(TWIN_SPECS)}.")

        spec = TWIN_SPECS[key]
        # Geometry + arch come from the central registry (single source of truth).
        r = registry.resolve(self.area, self.backbone)
        self.n_neurons = r.n_neurons
        self.img_mean, self.img_std = r.img_mean, r.img_std
        self.input_size = r.input_size
        self.crop_size = r.crop_size
        self.train_crop = r.train_crop or r.crop_size   # training crop (registry); falls back to screening crop
        self.train_upsample = r.train_upsample           # pre-crop upsample side (V1 only); None = off
        self.channels = r.channels
        self.kind = spec["kind"]
        self.fine_tune = spec["fine_tune"]
        self.cache_kind = "images" if self.fine_tune else "features"
        self.feature_dim = spec["feature_dim"]
        self.spatial_size = spec["spatial_size"]
        self.readout_nonlin = spec["readout_nonlin"]
        self.elu_offset = spec["elu_offset"]
        self.model_name = spec["model_name"]
        self.layer_name = spec["layer_name"]
        if self.block is None:
            self.block = spec["block"]
        if self.readout_type is None:
            self.readout_type = spec["readout_type"]
        if self.gamma_readout is None:
            self.gamma_readout = spec["gamma_readout"]
        if self.batch_size is None:
            self.batch_size = spec["batch_size"]

        # Env-driven locations (None if the corresponding env var is unset).
        self.image_dir = _join(env_dir("EXPERIMENT_DIR"), self.area, "images")
        self.features_dir = _join(env_dir("FEATURES_DIR"), self.area, self.backbone)
        self.trained_dir = _join(env_dir("TRAINED_MODELS_DIR"), self.area, self.backbone)
        self.dino_model_dir = _join(env_dir("MODELS_DIR"), "dinov3")
        self.logs_dir = env_dir("LOGS_DIR")
