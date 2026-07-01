"""DINOv3-backed digital-twin model classes.

A digital twin here is a **frozen** DINOv3 ViT backbone (a license-gated model loaded from a local
hubconf checkout) whose intermediate block feature map is read out by a trainable Gaussian readout,
with a trainable ``BatchNorm2d`` adapting the frozen features. The output passes through ``ELU + 1``
so predicted firing rates are positive — the same readout/nonlinearity contract as the nnvision
task-driven twins in :mod:`dualneuron.twins.nets`.

Two construction modes:

* **pretrained** (default): the backbone loads the locally-converted gated weights
  (see :mod:`dualneuron.training.convert_dinov3_weights`); this is the frozen core used for training
  and inference.
* **untrained**: the backbone is instantiated from the hubconf architecture with random weights
  (``pretrained=False`` + ``init_weights()``) — an architecture-only control that needs the hubconf
  repo but **not** the gated checkpoint.
"""

import math
import os
import warnings

warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
import torch.nn.functional as F

# Filename hash suffix the Meta hubconf factory appends to its .pth files; matches
# dinov3/hub/backbones.py. Add new entries when loading new variants. See
# dualneuron.training.convert_dinov3_weights for the one-time HF -> hubconf conversion.
DINOV3_WEIGHT_HASHES = {
    "dinov3_vitb16": "73cec8be",
}


def make_nonlinearity(name):
    """Post-BatchNorm, pre-readout nonlinearity, mirroring the nnvision cores' ``OutNonlin``.

    ``"relu"`` (V4 task-driven), ``"gelu"`` (V1 task-driven), or ``None`` -> identity.
    """
    if name is None:
        return nn.Identity()
    return {"relu": nn.ReLU(inplace=True), "gelu": nn.GELU()}[name]


class FullGaussian2dReadout(nn.Module):
    """Spatial readout matching the sinzlab ``FullGaussian2d`` convention.

    Each neuron has a learned position ``mu`` in ``[-1, 1]`` feature-map space. During training the
    sample position is drawn from ``N(mu, sigma)``; during eval it is fixed to ``mu`` (no noise).
    Features at that point are gathered with ``grid_sample`` and combined with per-neuron channel
    weights to give the raw (pre-nonlinearity) readout output; the twin applies the output
    ``ELU(x + offset) + 1`` (matching nnvision's ``EncoderShifter``).

    Args:
        n_neurons: Number of neurons to predict.
        feature_dim: Channel dimension of the backbone feature map.
        spatial_size: Feature-map side length (unused at runtime; kept for parity/introspection).
        init_mu_range: Uniform init range for the readout positions ``mu``.
        init_sigma: Initial value for the position-sampling ``sigma``.
        batch_sample: If True, draw an independent sample position per batch element.
        gauss_type: ``'isotropic'`` (default), ``'uncorrelated'`` or ``'full'``.
    """

    def __init__(
        self,
        n_neurons,
        feature_dim=768,
        spatial_size=14,
        init_mu_range=0.4,
        init_sigma=0.5,
        batch_sample=True,
        gauss_type="isotropic",
    ):
        super().__init__()
        self.n_neurons = n_neurons
        self.feature_dim = feature_dim
        self.spatial_size = spatial_size
        self.batch_sample = batch_sample
        self.gauss_type = gauss_type

        self.mu = nn.Parameter(torch.empty(1, n_neurons, 1, 2))
        nn.init.uniform_(self.mu, -init_mu_range, init_mu_range)

        if gauss_type == "isotropic":
            self.sigma = nn.Parameter(torch.full((1, n_neurons, 1, 1), init_sigma))
        elif gauss_type == "uncorrelated":
            self.sigma = nn.Parameter(torch.full((1, n_neurons, 1, 2), init_sigma))
        else:
            self.sigma = nn.Parameter(torch.empty(1, n_neurons, 2, 2))
            nn.init.uniform_(self.sigma, -init_sigma, init_sigma)

        self.features = nn.Parameter(torch.empty(1, feature_dim, 1, n_neurons))
        nn.init.normal_(self.features, std=1.0 / math.sqrt(feature_dim))

        self.bias = nn.Parameter(torch.zeros(n_neurons))

    def sample_grid(self, batch_size, sample=None):
        """Build the ``grid_sample`` positions: ``N(mu, sigma)`` in train, fixed ``mu`` in eval."""
        with torch.no_grad():
            self.mu.clamp_(min=-1, max=1)
            self.sigma.clamp_(min=0) if self.gauss_type != "full" else None

        grid_shape = (batch_size, self.n_neurons, 1, 2)
        sample = self.training if sample is None else sample

        if sample:
            noise = self.mu.new_empty(*grid_shape).normal_()
        else:
            noise = self.mu.new_zeros(*grid_shape)

        if self.gauss_type == "full":
            return torch.clamp(
                torch.einsum("ancd,bnid->bnic", self.sigma, noise) + self.mu,
                min=-1, max=1,
            )
        return torch.clamp(noise * self.sigma + self.mu, min=-1, max=1)

    def forward(self, feature_map):
        B = feature_map.shape[0]
        if self.batch_sample:
            grid = self.sample_grid(batch_size=B)
        else:
            grid = self.sample_grid(batch_size=1).expand(B, -1, -1, -1)

        y = F.grid_sample(feature_map, grid, align_corners=True).squeeze(-1)
        feat = self.features.squeeze(2)
        return (y * feat).sum(1) + self.bias

    @property
    def l1_regularization(self):
        return self.features.abs().mean()

    @property
    def receptive_fields(self):
        return self.mu.detach().squeeze()

    @property
    def rf_sizes(self):
        return self.sigma.detach().squeeze()


class GaussianReadout(nn.Module):
    """Gaussian spatial readout (softmax-pooled), an alternative to :class:`FullGaussian2dReadout`.

    Each neuron has a learned ``mu``, ``log_sigma``, channel features and bias; the spatial pool is a
    softmax over a fixed grid of an isotropic Gaussian centred at ``mu``. Returns the raw
    (pre-nonlinearity) readout output; the twin applies the output ``ELU(x + offset) + 1``.
    """

    def __init__(
        self,
        n_neurons,
        feature_dim=768,
        spatial_size=14,
        init_mu_range=0.4,
        init_sigma_range=0.6,
    ):
        super().__init__()
        self.n_neurons = n_neurons
        self.feature_dim = feature_dim
        self.spatial_size = spatial_size

        self.mu = nn.Parameter(torch.empty(n_neurons, 2))
        nn.init.uniform_(self.mu, -init_mu_range, init_mu_range)

        self.log_sigma = nn.Parameter(torch.empty(n_neurons))
        nn.init.uniform_(self.log_sigma, math.log(0.1), math.log(init_sigma_range))

        self.features = nn.Parameter(torch.empty(n_neurons, feature_dim))
        nn.init.normal_(self.features, std=1.0 / math.sqrt(feature_dim))
        self.bias = nn.Parameter(torch.zeros(n_neurons))

        self.register_buffer("grid", self._make_grid(spatial_size))

    @staticmethod
    def _make_grid(size):
        coords = torch.linspace(-1, 1, size)
        gy, gx = torch.meshgrid(coords, coords, indexing="ij")
        return torch.stack([gx, gy], dim=-1).reshape(-1, 2)

    def forward(self, feature_map):
        B, C, H, W = feature_map.shape
        sigma = F.softplus(self.log_sigma).unsqueeze(1)
        diff = self.grid.unsqueeze(0) - self.mu.unsqueeze(1)
        sq_dist = (diff ** 2).sum(dim=-1)
        log_weights = -sq_dist / (2 * sigma ** 2)
        weights = F.softmax(log_weights, dim=-1)

        flat = feature_map.reshape(B, C, H * W)
        pooled = torch.einsum("bcl,nl->bcn", flat, weights)
        return torch.einsum("bcn,nc->bn", pooled, self.features) + self.bias

    @property
    def l1_regularization(self):
        return self.features.abs().mean()

    @property
    def receptive_fields(self):
        return self.mu.detach()

    @property
    def rf_sizes(self):
        return F.softplus(self.log_sigma).detach()


class DINOv3Core(nn.Module):
    """Frozen DINOv3 ViT backbone that returns a spatial feature map, plus a trainable BatchNorm2d.

    The DINOv3 weights are license-gated, so the backbone is loaded from a local hubconf checkout
    under ``model_dir`` (``model_dir/facebookresearch_dinov3_main``) with locally-converted weights
    (``model_dir/checkpoints/...pth``). Features come from ``get_intermediate_layers(norm=False)`` so
    cached features are raw block outputs; the trainable ``BatchNorm2d`` is applied on top.

    Args:
        model_name: DINOv3 hubconf model (must be a key of :data:`DINOV3_WEIGHT_HASHES`).
        feature_dim: Backbone hidden dim (768 for ViT-B/16).
        block: Transformer block index to read (None = last block).
        model_dir: Directory holding the hubconf checkout and converted weights.
        untrained: If True, instantiate the architecture with random weights (no gated checkpoint
            required); otherwise load the converted gated weights.
        fine_tune: If True, enable gradients on the stem + blocks up to ``block`` and use a
            truncated forward (runs only those blocks) for end-to-end fine-tuning; later blocks stay
            frozen and are never run. If False, the whole backbone is frozen (the default).
    """

    def __init__(self, model_name="dinov3_vitb16", feature_dim=768, block=None,
                 model_dir=None, untrained=False, fine_tune=False):
        super().__init__()
        if model_dir is None:
            raise ValueError(
                "DINOv3Core requires model_dir (local hubconf checkout, e.g. MODELS_DIR/dinov3).")
        if model_name not in DINOV3_WEIGHT_HASHES:
            raise ValueError(
                f"Unknown DINOv3 variant {model_name!r}; add its hash to DINOV3_WEIGHT_HASHES.")

        repo_path = os.path.join(model_dir, "facebookresearch_dinov3_main")
        if untrained:
            self.backbone = torch.hub.load(repo_path, model_name, source="local", pretrained=False)
            self.backbone.init_weights()
        else:
            ckpt_name = (f"{model_name}_pretrain_lvd1689m-"
                         f"{DINOV3_WEIGHT_HASHES[model_name]}.pth")
            weights_path = os.path.join(model_dir, "checkpoints", ckpt_name)
            if not os.path.isfile(weights_path):
                raise FileNotFoundError(
                    f"DINOv3 converted weights not found at {weights_path}. "
                    f"Run: python -m dualneuron.training.convert_dinov3_weights")
            self.backbone = torch.hub.load(
                repo_path, model_name, source="local", weights=f"file://{weights_path}")

        self.block = block if block is not None else len(self.backbone.blocks) - 1
        self.fine_tune = fine_tune

        for p in self.backbone.parameters():
            p.requires_grad = False
        if fine_tune:
            # Enable grad only on what the truncated forward actually runs: the patch embed, the
            # RoPE embed, the cls/storage tokens, and blocks[0..block]. Later blocks stay frozen and
            # are never executed. The backbone is kept in eval mode (deterministic — no stochastic
            # depth), so fine-tuning updates weights without dropout/drop-path noise.
            tuned = [self.backbone.patch_embed, self.backbone.rope_embed]
            tuned += list(self.backbone.blocks[: self.block + 1])
            for m in tuned:
                if m is not None:
                    for p in m.parameters():
                        p.requires_grad = True
            for name in ("cls_token", "storage_tokens", "mask_token"):
                p = getattr(self.backbone, name, None)
                if isinstance(p, nn.Parameter):
                    p.requires_grad = True
        self.backbone.eval()

        self.norm = nn.BatchNorm2d(feature_dim, momentum=0.1)

    def _forward_truncated(self, x):
        """Run only patch-embed + blocks[0..block] (fine-tune path), returning a (B,C,H,W) map.

        Replicates ``get_intermediate_layers(n=[block], reshape=True, norm=False)`` but stops after
        ``block`` so no compute/grad is spent on later blocks.
        """
        bb = self.backbone
        B, _, in_h, in_w = x.shape
        tokens, (H, W) = bb.prepare_tokens_with_masks(x)
        for i in range(self.block + 1):
            rope = bb.rope_embed(H=H, W=W) if bb.rope_embed is not None else None
            tokens = bb.blocks[i](tokens, rope)
        patch = tokens[:, bb.n_storage_tokens + 1:]          # drop cls + storage tokens
        ph, pw = in_h // bb.patch_size, in_w // bb.patch_size
        return patch.reshape(B, ph, pw, -1).permute(0, 3, 1, 2).contiguous()

    def forward(self, x):
        if x.shape[1] == 1:                                  # grayscale (V1) -> replicate to 3ch
            x = x.repeat(1, 3, 1, 1)
        if self.fine_tune:
            return self.norm(self._forward_truncated(x))
        out = self.backbone.get_intermediate_layers(x, n=[self.block], reshape=True, norm=False)
        return self.norm(out[0])

    def train(self, mode=True):
        # Keep the backbone in eval mode (no stochastic depth / frozen norms) regardless of the
        # module's train/eval state; only the readout and self.norm should toggle. In the fine-tune
        # case its weights still receive gradients (eval mode affects dropout/drop-path, not autograd).
        super().train(mode)
        self.backbone.eval()
        return self


class DINONeuralPredictor(nn.Module):
    """Full DINOv3 twin: frozen :class:`DINOv3Core` + a trainable Gaussian readout.

    Two forward paths:

    * :meth:`forward` runs the whole pipeline (backbone + BatchNorm + readout) from images.
    * :meth:`forward_from_features` runs only BatchNorm + readout from a cached raw block feature
      map — used for fast readout training/eval on pre-extracted features.

    Args:
        n_neurons: Number of neurons to predict.
        model_name: DINOv3 hubconf model.
        feature_dim: Backbone hidden dim.
        spatial_size: Feature-map side length (14 for ViT-B/16 at 224px).
        init_mu_range, init_sigma_range: Readout init ranges.
        block: Transformer block index (None = last).
        model_dir: Directory with the hubconf checkout + converted weights.
        readout_type: ``'fullgaussian2d'`` (default) or ``'gaussian'``.
        readout_nonlin: Post-BatchNorm, pre-readout nonlinearity (``'relu'`` for V4, ``'gelu'`` for
            V1) so the DINO twin mirrors its area's task-driven head; ``None`` -> identity.
        elu_offset: Output nonlinearity is ``ELU(x + elu_offset) + 1`` (``-1`` matches nnvision).
        untrained: Forwarded to :class:`DINOv3Core` (random vs. gated-pretrained backbone).
    """

    def __init__(
        self,
        n_neurons,
        model_name="dinov3_vitb16",
        feature_dim=768,
        spatial_size=14,
        init_mu_range=0.4,
        init_sigma_range=0.6,
        block=None,
        model_dir=None,
        readout_type="fullgaussian2d",
        readout_nonlin="relu",
        elu_offset=-1,
        untrained=False,
        fine_tune=False,
    ):
        super().__init__()
        self.core = DINOv3Core(
            model_name, feature_dim, block=block, model_dir=model_dir,
            untrained=untrained, fine_tune=fine_tune)
        self.readout_nonlin = make_nonlinearity(readout_nonlin)
        self.offset = elu_offset

        if readout_type == "fullgaussian2d":
            self.readout = FullGaussian2dReadout(
                n_neurons=n_neurons,
                feature_dim=feature_dim,
                spatial_size=spatial_size,
                init_mu_range=init_mu_range,
                init_sigma=init_sigma_range,
            )
        else:
            self.readout = GaussianReadout(
                n_neurons=n_neurons,
                feature_dim=feature_dim,
                spatial_size=spatial_size,
                init_mu_range=init_mu_range,
                init_sigma_range=init_sigma_range,
            )

    def forward(self, x):
        y = self.readout(self.readout_nonlin(self.core(x)))
        return F.elu(y + self.offset) + 1

    def forward_from_features(self, feature_map):
        y = self.readout(self.readout_nonlin(self.core.norm(feature_map)))
        return F.elu(y + self.offset) + 1

    @property
    def readout_l1(self):
        return self.readout.l1_regularization
