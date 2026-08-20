"""Twin architecture: task-driven cores, the Gaussian readout, the encoder and the ensemble.

These classes are a transcription of the architecture the shipped twins were built and trained with,
which the repo previously reached through the ``nnvision`` install. Vendoring them removes that whole
git-dependency chain — ``nnvision`` pulled ``nnfabrik``, ``mei`` (and through it ``datajoint``),
``neuralpredictors``, ``ptrnets`` and ``CORnet``.

Every twin that was previously built through that chain is built bitwise identically here. That was
established by building each one both ways in one process and comparing state_dict keys, every
parameter tensor and the forward output — 38 builds, all identical. It cannot be re-checked once the
old packages are uninstalled, so :mod:`dualneuron.twins.verify` pins the structure instead. (The DINO
twins never went through that chain; their readout is deliberately changed — see
:mod:`dualneuron.twins.dino`.)

Upstream names are kept deliberately (``TaskDrivenCore3``, ``FullGaussian2d``, ``EncoderShifter``,
...) so a reader can diff these against the original package one-for-one. Provenance:

* :class:`FullGaussian2d` — ``neuralpredictors.layers.readouts`` (KonstantinWilleke fork, ``interview``)
* :class:`MultipleFullGaussian2d`, :class:`TaskDrivenCore3`, :class:`ConvNextCore`,
  :class:`ConvNextV2`, :class:`EncoderShifter` — ``nnvision.models.{readouts,cores,convnext_v2,encoders}``
* :class:`EnsembleModel` — ``mei.modules``
* :func:`clip_model` — ``ptrnets.utils.mlayer``
* :func:`get_module_output`, :func:`eval_state` — ``neuralpredictors.{utils,training}``

Branches that no configuration in :mod:`dualneuron.twins.nets` can reach are dropped rather than
carried as untestable code; each omission is recorded on the class it belongs to.
"""

import warnings
from collections import OrderedDict
from contextlib import contextmanager

import numpy as np
import torch
import torch.nn.functional as F
import torchvision
from torch import nn


# ---------------------------------------------------------------------------
#  Helpers
# ---------------------------------------------------------------------------

def set_random_seed(seed, deterministic=True):
    """Seed Python, NumPy and PyTorch (CPU + CUDA); optionally pin CUDNN to deterministic kernels.

    A best-effort reproducibility guarantee, not an absolute one: some CUDA kernels are inherently
    non-deterministic. Called twice by each builder — once before the core is constructed and once
    before the readout — so a given ``seed`` reproduces a given initialization.
    """
    import random

    random.seed(seed)
    np.random.seed(seed)
    if deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    torch.manual_seed(seed)                      # sets both the CPU and the CUDA seeds


@contextmanager
def eval_state(model):
    """Run the block with ``model`` in eval mode, restoring whatever mode it was in on exit."""
    training_status = model.training
    try:
        model.eval()
        yield model
    finally:
        model.train(training_status)


def get_module_output(model, input_shape, use_cuda=True):
    """Output shape of ``model`` for a zero input of ``input_shape``, as a ``torch.Size``.

    Used to size the readout against the core it sits on, so the feature-map geometry never has to be
    written down twice. The model is evaluated in eval mode and returned to its original device.
    """
    initial_device = next(iter(model.parameters())).device
    device = "cuda" if torch.cuda.is_available() and use_cuda else "cpu"
    with eval_state(model):
        with torch.no_grad():
            output = model.to(device)(torch.zeros(1, *input_shape[1:], device=device))
    model.to(initial_device)
    return output.shape


def unpack_data_info(data_info):
    """Split a ``data_info`` dict into ``(n_neurons_dict, in_shapes_dict, input_channels)``.

    ``data_info`` is how :mod:`dualneuron.twins.nets` describes a twin's input/output geometry without
    a dataloader: ``{session_key: {"input_dimensions", "input_channels", "output_dimension", ...}}``.
    """
    in_shapes_dict = {k: v["input_dimensions"] for k, v in data_info.items()}
    input_channels = [v["input_channels"] for v in data_info.values()]
    n_neurons_dict = {k: v["output_dimension"] for k, v in data_info.items()}
    return n_neurons_dict, in_shapes_dict, input_channels


def clip_model(model, layer_name):
    """Copy of ``model`` truncated after ``layer_name``, as an ``nn.Sequential``.

    Walks the named-module tree collecting siblings until ``layer_name`` is reached, descending into
    the branch that contains it. The returned Sequential holds the *same* layer objects as ``model``
    (not copies), and inherits its train/eval mode.

    Args:
        model: Module to truncate.
        layer_name: Dotted name of the last module to keep, e.g. ``'layer3.0'``.
    """
    assert layer_name in [n for n, _ in model.named_modules()], f"No module named {layer_name}"

    features = OrderedDict()
    nodes_iter = iter(layer_name.split("."))
    mode = model.training

    def recursive(module, node=next(nodes_iter), prefix=()):
        for name, layer in module.named_children():
            fullname = ".".join((*prefix, name))
            if name == node and fullname != layer_name:
                recursive(layer, node=next(nodes_iter), prefix=(fullname,))
                return
            features[name] = layer
            if fullname == layer_name:
                return

    recursive(model)
    clipped_model = nn.Sequential(features)
    return clipped_model.train() if mode else clipped_model.eval()


def clip_convnext_layers(model, layer_name):
    """Replace every module at or after ``layer_name`` with ``nn.Identity`` and return ``model``.

    Truncation in place, rather than the Sequential rebuild :func:`clip_model` does — this is how
    upstream cut the HuggingFace ConvNeXt models, and it is what the shipped V1 weights were produced
    under. Note it mutates ``model``.
    """
    names = np.array([name for name, _ in model.named_modules()])
    cut_layer_index = np.where(names == layer_name)[0].item()
    for name in names[cut_layer_index:]:
        parent, _, attr = name.rpartition(".")
        setattr(model.get_submodule(parent) if parent else model, attr, nn.Identity())
    return model


def _init_conv(m):
    """Xavier-normal every ``Conv2d`` weight, zero its bias — the cores' random-init rule."""
    if isinstance(m, nn.Conv2d):
        nn.init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            m.bias.data.fill_(0)


# ---------------------------------------------------------------------------
#  Readout
# ---------------------------------------------------------------------------

class FullGaussian2d(nn.Module):
    """Gaussian point readout: one learned position per neuron plus per-neuron channel weights.

    Each neuron carries a mean position ``mu`` in ``[-1, 1]`` feature-map coordinates and a spread
    ``sigma``. In train mode the read position is drawn from ``N(mu, sigma)`` — an independent draw
    per batch element when ``batch_sample`` — and in eval mode it is fixed at ``mu``. The feature
    vector at that position is gathered with ``grid_sample`` and contracted with the neuron's channel
    weights.

    ``features`` initializes to the constant ``1 / in_shape[0]``: every neuron starts as the
    unweighted channel mean, so ``mu`` is what breaks symmetry first. This is not interchangeable
    with a random init — it is what the shipped twins were trained under, and
    :mod:`dualneuron.training` depends on both backbone families starting from the same head.

    Dropped relative to upstream: the ``grid_mean_predictor`` path and feature/grid sharing across
    sessions, both of which the builders in :mod:`dualneuron.twins.builders` pass as ``None``
    (upstream annotates ``grid_mean_predictor`` "not relevant for monkey data"); and the ``multiplex``
    forward mode, which reads every neuron at every pixel and which nothing here calls.

    ``mu`` and ``features`` stay exposed as properties over the ``_mu`` / ``_features`` parameters
    because those are the names the shipped state_dicts are keyed by, and the properties are the read
    API upstream offered. One caller writes through ``mu`` — ``centered=True`` in
    :mod:`dualneuron.twins.nets` zeroes it.

    Args:
        in_shape: Core output shape ``(channels, height, width)``.
        outdims: Number of neurons.
        bias: If True, add a learned per-neuron output bias.
        init_mu_range: ``mu`` initializes uniform on ``[-init_mu_range, init_mu_range]``.
        init_sigma: Initial ``sigma`` — filled for ``'isotropic'``/``'uncorrelated'``, uniform on
            ``[-init_sigma, init_sigma]`` for ``'full'``.
        batch_sample: Draw an independent position per batch element rather than one per batch.
        align_corners: Passed through to ``grid_sample``.
        gauss_type: ``'isotropic'``, ``'uncorrelated'`` or ``'full'`` — sets the shape of ``sigma``.
    """

    def __init__(
        self,
        in_shape,
        outdims,
        bias,
        init_mu_range=0.1,
        init_sigma=1,
        batch_sample=True,
        align_corners=True,
        gauss_type="full",
        **kwargs,
    ):
        super().__init__()
        if init_mu_range > 1.0 or init_mu_range <= 0.0 or init_sigma <= 0.0:
            raise ValueError(
                "either init_mu_range doesn't belong to [0.0, 1.0] or init_sigma_range is non-positive")
        if gauss_type not in ("full", "uncorrelated", "isotropic"):
            raise ValueError(f'gauss_type "{gauss_type}" not known')

        self.in_shape = in_shape
        self.outdims = outdims
        self.batch_sample = batch_sample
        self.gauss_type = gauss_type
        self.grid_shape = (1, outdims, 1, 2)
        self.sigma_shape = {"full": (1, outdims, 2, 2),
                            "uncorrelated": (1, outdims, 1, 2),
                            "isotropic": (1, outdims, 1, 1)}[gauss_type]

        # Registration order fixes the state_dict key order; keep it as upstream had it.
        self._mu = nn.Parameter(torch.Tensor(*self.grid_shape))
        self.sigma = nn.Parameter(torch.Tensor(*self.sigma_shape))
        self._features = nn.Parameter(torch.Tensor(1, in_shape[0], 1, outdims))
        self.register_parameter("bias", nn.Parameter(torch.Tensor(outdims)) if bias else None)

        self.init_mu_range = init_mu_range
        self.init_sigma = init_sigma
        self.align_corners = align_corners
        self.initialize()

    @property
    def features(self):
        return self._features

    @property
    def mu(self):
        return self._mu

    @property
    def grid(self):
        return self.sample_grid(batch_size=1, sample=False)

    @property
    def mu_dispersion(self):
        """Spread of the learned positions — a regularizer pushing neurons to similar positions."""
        return self._mu.squeeze().std(0).sum()

    def feature_l1(self, average=True):
        """L1 norm of the channel weights: their mean if ``average``, else their sum."""
        return self._features.abs().mean() if average else self._features.abs().sum()

    def initialize(self):
        """(Re-)initialize ``mu``, ``sigma``, the channel weights and the bias."""
        self._mu.data.uniform_(-self.init_mu_range, self.init_mu_range)
        if self.gauss_type != "full":
            self.sigma.data.fill_(self.init_sigma)
        else:
            self.sigma.data.uniform_(-self.init_sigma, self.init_sigma)
        self._features.data.fill_(1 / self.in_shape[0])
        if self.bias is not None:
            self.bias.data.fill_(0)

    def sample_grid(self, batch_size, sample=None):
        """``grid_sample`` positions: drawn from ``N(mu, sigma)``, or fixed at ``mu``.

        Args:
            batch_size: Number of position sets to return.
            sample: Force sampling on/off; ``None`` follows the module's train/eval state.
        """
        with torch.no_grad():
            self.mu.clamp_(min=-1, max=1)     # only mu is read at eval time, so it must stay in range
            if self.gauss_type != "full":
                self.sigma.clamp_(min=0)      # a standard deviation is a positive quantity

        grid_shape = (batch_size,) + self.grid_shape[1:]
        sample = self.training if sample is None else sample
        norm = (self.mu.new_empty(*grid_shape).normal_() if sample
                else self.mu.new_zeros(*grid_shape))

        if self.gauss_type != "full":
            return torch.clamp(norm * self.sigma + self.mu, min=-1, max=1)
        return torch.clamp(
            torch.einsum("ancd,bnid->bnic", self.sigma, norm) + self.mu, min=-1, max=1)

    def forward(self, x, sample=None, shift=None, out_idx=None, **kwargs):
        """Read ``x`` at each neuron's position -> ``(batch, outdims)``.

        Args:
            x: Core output, ``(batch, channels, height, width)``.
            sample: Force position sampling on/off; ``None`` follows the train/eval state.
            shift: Per-example grid shift (eye-tracking); ``None`` for the fixed-fixation twins.
            out_idx: Restrict the output to a subset of neurons (indices or a boolean mask).
        """
        N, c, w, h = x.size()
        if tuple(self.in_shape) != (c, w, h):
            warnings.warn(
                "the specified feature map dimension is not the readout's expected input dimension")

        feat = self.features.view(1, c, self.outdims)
        bias = self.bias
        outdims = self.outdims

        if self.batch_sample:
            grid = self.sample_grid(batch_size=N, sample=sample)
        else:
            grid = self.sample_grid(batch_size=1, sample=sample).expand(N, outdims, 1, 2)

        if out_idx is not None:
            if isinstance(out_idx, np.ndarray) and out_idx.dtype == bool:
                out_idx = np.where(out_idx)[0]
            feat = feat[:, :, out_idx]
            grid = grid[:, out_idx]
            if bias is not None:
                bias = bias[out_idx]
            outdims = len(out_idx)

        if shift is not None:
            grid = grid + shift[:, None, None, :]

        y = F.grid_sample(x, grid, align_corners=self.align_corners)
        y = (y.squeeze(-1) * feat).sum(1).view(N, outdims)
        if bias is not None:
            y = y + bias
        return y

    def __repr__(self):
        c, w, h = self.in_shape
        r = f"{self.gauss_type} {self.__class__.__name__} ({c} x {w} x {h} -> {self.outdims})"
        if self.bias is not None:
            r += " with bias"
        return r


class MultipleFullGaussian2d(nn.ModuleDict):
    """One :class:`FullGaussian2d` per session key, sized against the core it reads from.

    The twins here are single-session (``'all_sessions'``), but the dict structure is what the shipped
    state_dicts are keyed by, so it stays. ``forward`` resolves ``data_key`` implicitly when there is
    only one session.

    ``regularizer`` is **sum**-based, as upstream: ``gamma_readout * sum|features|``. Note that
    :mod:`dualneuron.training` calls this rather than rolling its own, so the training objective and
    the architecture agree on the convention.

    Args:
        core: The core the readout reads from — used only to infer the feature-map geometry.
        in_shape_dict: Per-session model input shape.
        n_neurons_dict: Per-session neuron count.
        init_mu_range, init_sigma, bias, gauss_type: Forwarded to :class:`FullGaussian2d`.
        gamma_readout: Weight of the L1 term on the channel weights.
        gamma_grid_dispersion: Weight of the position-dispersion term (0 = off).
    """

    def __init__(
        self,
        core,
        in_shape_dict,
        n_neurons_dict,
        init_mu_range,
        init_sigma,
        bias,
        gamma_readout,
        gauss_type,
        gamma_grid_dispersion=0,
    ):
        super().__init__()
        for k in n_neurons_dict:
            self.add_module(k, FullGaussian2d(
                in_shape=get_module_output(core, in_shape_dict[k])[1:],
                outdims=n_neurons_dict[k],
                init_mu_range=init_mu_range,
                init_sigma=init_sigma,
                bias=bias,
                gauss_type=gauss_type,
            ))
        self.gamma_readout = gamma_readout
        self.gamma_grid_dispersion = gamma_grid_dispersion

    def forward(self, *args, data_key=None, **kwargs):
        if data_key is None and len(self) == 1:
            data_key = list(self.keys())[0]
        return self[data_key](*args, **kwargs)

    def regularizer(self, data_key):
        return (self[data_key].feature_l1(average=False) * self.gamma_readout
                + self[data_key].mu_dispersion * self.gamma_grid_dispersion)


# ---------------------------------------------------------------------------
#  Cores
# ---------------------------------------------------------------------------

# ptrnets' L2-robust ResNet50s. Each is a plain torchvision ResNet50 architecture; ptrnets differs
# only in the adversarially-trained checkpoint it downloads, which the twins never use (see
# _task_driven_backbone).
_ROBUST_RESNET50 = frozenset(
    f"resnet50_l2_eps{eps}" for eps in
    ("0", "0_01", "0_03", "0_05", "0_1", "0_25", "0_5", "1", "3", "5"))


def _task_driven_backbone(model_name, pretrained):
    """The ImageNet backbone a task-driven core is clipped out of.

    The V4 twins name ptrnets' robust ResNet50s, and every config in :mod:`dualneuron.twins.nets`
    builds them with ``pretrained=False`` — the twin's own weights arrive via ``state_dict``, so only
    the architecture is needed and the robust checkpoint never is. Asking for ``pretrained=True``
    raises rather than silently substituting the standard ImageNet weights, which would be a
    different backbone.
    """
    if model_name in _ROBUST_RESNET50:
        if pretrained:
            raise ValueError(
                f"{model_name!r} with pretrained=True needs ptrnets' robust checkpoint, which is not "
                "vendored; every twin config in dualneuron.twins.nets builds with pretrained=False.")
        return torchvision.models.resnet50()
    factory = getattr(torchvision.models, model_name, None)
    if factory is None:
        raise ValueError(f"unknown task-driven backbone {model_name!r}")
    return factory(weights="IMAGENET1K_V1" if pretrained else None)


class TaskDrivenCore3(nn.Module):
    """An ImageNet-pretrained network clipped after ``layer_name``, plus BatchNorm and a ReLU.

    The frozen trunk of the task-driven twins: ``features`` is
    ``TaskDriven -> OutBatchNorm -> OutNonlin``, and the state_dict is keyed by exactly those names,
    which is what the shipped weights expect and what :mod:`dualneuron.training` reaches into when it
    resets the head.

    Grayscale input is repeated to three channels, matching the RGB backbone.

    Dropped relative to upstream: the forward-hook "probe" fallback for layer names that cannot be
    clipped. ``layer3.0`` clips cleanly, so the fallback was unreachable — and building it registered
    a permanent forward hook (ptrnets' ``ModuleHook``) that did ``output.clone()`` on *every* pass and
    kept the result alive, which is numerically inert but pure overhead. The two-channel (stacked
    previous image) input branch is also dropped: no config here uses it, and it was the only thing in
    this module that needed ``einops``.

    Args:
        input_channels: Channels of the images the twin is fed (1 or 3).
        model_name: Backbone name — see :func:`_task_driven_backbone`.
        layer_name: Dotted module name to clip after, e.g. ``'layer3.0'``.
        pretrained: Load the backbone's ImageNet weights. False for every twin here.
        bias: If False, zero the clipped trunk's final bias.
        final_batchnorm: Append ``OutBatchNorm``.
        final_nonlinearity: Append ``OutNonlin`` (ReLU).
        momentum: ``OutBatchNorm`` momentum.
        fine_tune: If False, freeze the clipped trunk.
    """

    def __init__(
        self,
        input_channels,
        model_name,
        layer_name,
        pretrained=True,
        bias=False,
        final_batchnorm=True,
        final_nonlinearity=True,
        momentum=0.1,
        fine_tune=False,
        **kwargs,
    ):
        if kwargs:
            warnings.warn(
                f"Ignoring input {kwargs!r} when creating {self.__class__.__name__}", UserWarning)
        super().__init__()

        self.input_channels = input_channels
        self.momentum = momentum
        self.layer_name = layer_name
        self.pretrained = pretrained
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        model = _task_driven_backbone(model_name, pretrained).to(self.device)
        model.eval()
        model_clipped = clip_model(model, self.layer_name)

        if not bias and "bias" in model_clipped[-1]._parameters:
            if model_clipped[-1].bias is not None:
                model_clipped[-1].bias.data = torch.zeros_like(model_clipped[-1].bias)

        if not fine_tune:
            for param in model_clipped.parameters():
                param.requires_grad = False

        self.features = nn.Sequential()
        self.features.add_module("TaskDriven", model_clipped)
        if final_batchnorm:
            self.features.add_module(
                "OutBatchNorm", nn.BatchNorm2d(self.outchannels, momentum=self.momentum))
        if final_nonlinearity:
            self.features.add_module("OutNonlin", nn.ReLU(inplace=True))

    @property
    def outchannels(self):
        """Channel count of the clipped trunk's output, measured with a probe forward pass."""
        x = torch.randn(1, 3, 224, 224).to(self.device)
        return self.features.TaskDriven(x).shape[1]

    def forward(self, input_):
        if len(input_.shape) == 3:
            input_ = input_[:, None, ...]
        if self.input_channels == 1:
            input_ = input_.repeat(1, 3, 1, 1)
        return self.features(input_)

    def regularizer(self):
        return 0                                  # the core contributes nothing to the loss

    def initialize(self):
        """Randomize the trunk's convolutions when the backbone was not loaded pretrained."""
        if not self.pretrained:
            self.apply(_init_conv)


class ConvNextV2(nn.Module):
    """A HuggingFace ConvNeXtV2 classifier truncated after ``cutoff_layer``, head removed.

    Args:
        model_name: HuggingFace model id, e.g. ``'facebook/convnextv2-atto-1k-224'``.
        cutoff_layer: Dotted module name after which the network is neutralized; None keeps it whole.
        patch_embedding_stride: Override the stem's stride (default 4 in all ConvNeXt models).
        cut_classification_head: Drop the ImageNet-1k classifier, keeping the encoder.
    """

    def __init__(
        self,
        model_name,
        cutoff_layer=None,
        patch_embedding_stride=None,
        cut_classification_head=True,
    ):
        from transformers import ConvNextV2ForImageClassification

        super().__init__()
        self.cutoff_layer = cutoff_layer
        self.model = ConvNextV2ForImageClassification.from_pretrained(model_name)
        self.patch_embedding_stride = patch_embedding_stride

        if self.patch_embedding_stride is not None:
            self.replace_patch_embedding_stride()
        if self.cutoff_layer is not None:
            self.model = clip_convnext_layers(self.model, self.cutoff_layer)
        if cut_classification_head:
            self.model = self.model.convnextv2

    def replace_patch_embedding_stride(self):
        """Rebuild the stem convolution at ``patch_embedding_stride``, carrying its weights over."""
        original = self.model.convnextv2.embeddings.patch_embeddings
        replacement = nn.Conv2d(
            in_channels=original.in_channels,
            out_channels=original.out_channels,
            kernel_size=original.kernel_size,
            stride=self.patch_embedding_stride,
            dilation=original.dilation,
            groups=original.groups,
            padding=original.padding,
            padding_mode=original.padding_mode,
            bias=True,
        )
        replacement.weight.data = original.weight.data
        replacement.bias.data = original.bias.data
        self.model.convnextv2.embeddings.patch_embeddings = replacement

    def forward(self, input_):
        return self.model(input_)[0]


class ConvNextCore(nn.Module):
    """A truncated ConvNeXtV2 backbone, optionally followed by a norm and a nonlinearity.

    The V1 twin's core. Unlike :class:`TaskDrivenCore3` this one is fine-tuned end to end, and its
    normalization/nonlinearity are named — ``OutBatchNorm`` / ``OutNonlin`` — to match, so both
    families present the same head to :mod:`dualneuron.training`.

    Dropped relative to upstream: the ``stack`` option, which concatenated several layers' outputs via
    ``torchextractor`` (a package that is not even installed here), and the two-channel input branch.

    Args:
        model_name: HuggingFace ConvNeXt model id.
        layer_name: Dotted module name to truncate after.
        patch_embedding_stride: Stem stride override; None keeps the model's own.
        cut_classification_head: Drop the ImageNet classifier head.
        pretrained: Keep the HuggingFace weights (True) or randomize in :meth:`initialize` (False).
        fine_tune: If False, freeze the backbone.
        in_shapes_dict: Per-session input shapes; when given, the output geometry is measured from
            them rather than from a 224x224 probe.
        final_norm: ``'BatchNorm'``, ``'LayerNorm'`` or None — appended as ``OutBatchNorm``.
        momentum: ``OutBatchNorm`` momentum. Note the default differs from
            :class:`TaskDrivenCore3`'s; see :func:`dualneuron.twins.nets._v1_convnext_config`.
        final_nonlinearity: Name of a ``torch.nn`` class (e.g. ``'GELU'``) appended as ``OutNonlin``.
    """

    def __init__(
        self,
        model_name,
        layer_name,
        patch_embedding_stride=None,
        cut_classification_head=True,
        pretrained=True,
        fine_tune=False,
        in_shapes_dict=None,
        final_norm=None,
        final_nonlinearity=None,
        momentum=0.9,
        **kwargs,
    ):
        if kwargs:
            warnings.warn(
                f"Ignoring input {kwargs!r} when creating {self.__class__.__name__}", UserWarning)
        super().__init__()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        backbone = ConvNextV2(
            model_name=model_name,
            cutoff_layer=layer_name,
            patch_embedding_stride=patch_embedding_stride,
            cut_classification_head=cut_classification_head,
        )
        self.features = nn.Sequential()
        self.features.add_module("backbone", backbone)
        self.features.to(self.device)

        self.pretrained = pretrained
        self.in_shapes_dict = in_shapes_dict
        if in_shapes_dict is not None:
            self.in_shape = self.get_input_shape_from_dict()
            self.out_dim = self.get_out_dims()
            self.outchannels = self.out_dim[0]
        else:
            self.outchannels = self.get_out_channels()
        self.momentum = momentum

        if not fine_tune:
            for param in self.features.parameters():
                param.requires_grad = False

        if final_norm == "BatchNorm":
            self.features.add_module(
                "OutBatchNorm", nn.BatchNorm2d(self.outchannels, momentum=self.momentum))
        elif final_norm == "LayerNorm":
            self.features.add_module("OutBatchNorm", nn.LayerNorm(self.out_dim))
        elif final_norm is not None:
            raise ValueError("final normalization can only be BatchNorm or LayerNorm")

        if final_nonlinearity:
            nonlinearity = getattr(nn, final_nonlinearity)()
            nonlinearity.inplace = True if hasattr(nonlinearity, "inplace") else None
            self.features.add_module("OutNonlin", nonlinearity)

    def get_input_shape_from_dict(self):
        """The single input shape all sessions share, without the batch dimension."""
        all_shapes = [tuple(v[1:]) for v in self.in_shapes_dict.values()]
        assert all(s == all_shapes[0] for s in all_shapes[1:]), \
            "Sessions in dataloader dict have different shapes"
        return all_shapes[0]

    def get_out_dims(self):
        with torch.no_grad():
            return self(torch.ones(1, *self.in_shape).to(self.device)).shape[1:]

    def get_out_channels(self):
        with torch.no_grad():
            return self.features(torch.randn(1, 3, 224, 224).to(self.device)).shape[1]

    def regularizer(self):
        return 0                                  # the core contributes nothing to the loss

    def initialize(self):
        """Randomize the backbone's convolutions when it was not loaded pretrained."""
        if not self.pretrained:
            self.apply(_init_conv)

    def forward(self, input_):
        if len(input_.shape) == 3:
            input_ = input_[:, None, ...]
        if input_.shape[1] == 1:
            input_ = input_.repeat(1, 3, 1, 1)
        return self.features(input_)


# ---------------------------------------------------------------------------
#  Encoder + ensemble
# ---------------------------------------------------------------------------

class EncoderShifter(nn.Module):
    """Core + readout + the output nonlinearity that makes predicted rates positive.

    ``ELU(x + offset) + 1`` with ``offset = -1`` is the twins' rate nonlinearity; ``regularizer``
    sums the core's contribution (zero) and the readout's.

    The ``shifter`` (an eye-position-driven grid shift) is carried because it is part of the shipped
    architecture's signature, but every twin here is built with ``shifter=None``, so the branch that
    uses ``eye_position`` never runs.

    Args:
        core: Feature extractor.
        readout: Per-session readout, e.g. :class:`MultipleFullGaussian2d`.
        shifter: Per-session grid shifter, or None.
        elu_offset: Offset inside the output ELU.
        final_elu: If False, return the raw readout output.
    """

    def __init__(self, core, readout, shifter, elu_offset, final_elu=True):
        super().__init__()
        self.core = core
        self.readout = readout
        self.offset = elu_offset
        self.shifter = shifter
        self.final_elu = final_elu

    def forward(self, *args, data_key=None, eye_position=None, shift=None, **kwargs):
        x = self.core(args[0])
        if eye_position is not None and self.shifter is not None:
            eye_position = eye_position.to(x.device).to(dtype=x.dtype)
            shift = self.shifter[data_key](eye_position)

        x = self.readout(x, data_key=data_key, sample=kwargs.pop("sample", None), shift=shift,
                         **kwargs)
        return F.elu(x + self.offset) + 1 if self.final_elu else x

    def regularizer(self, data_key):
        return self.core.regularizer() + self.readout.regularizer(data_key=data_key)


class EnsembleModel(nn.Module):
    """Several twins evaluated together, their predictions averaged.

    Args:
        *members: The ensemble's member models.
    """

    def __init__(self, *members):
        super().__init__()
        self.members = nn.ModuleList(members)

    def __call__(self, x, *args, **kwargs):
        """Average the members' predictions, or stack them when ``avg=False``."""
        outputs = torch.stack([m(x, *args, **kwargs) for m in self.members], dim=0)
        return outputs if kwargs.get("avg") is False else outputs.mean(dim=0)

    def __repr__(self):
        return f"{self.__class__.__qualname__}({', '.join(repr(m) for m in self.members)})"
