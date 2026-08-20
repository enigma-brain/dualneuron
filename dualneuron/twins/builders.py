"""Assembling a twin from a config: core + Gaussian readout + rate nonlinearity.

:func:`build_model` and the two builders it dispatches to replace what the repo previously imported
from ``nnfabrik.builder``; the modules they assemble live in :mod:`dualneuron.twins.layers`. Together
those two modules are the whole reason ``nnvision``, ``nnfabrik``, ``mei`` and ``datajoint`` were ever
installed.

No call site in :mod:`dualneuron.twins.nets` passes dataloaders — every twin is described by a
``data_info`` dict instead — so the builders take ``data_info`` only. Upstream's dataloader branch
(which read the geometry off a batch and primed the readout bias from the training responses) is
dropped, as is the eye-position ``shifter``, which every config here passes as ``None``.

``seed`` is applied twice, as upstream did it: once before the core is constructed, and again before
``core.initialize()``. The second reset is what makes the initialization independent of how much
randomness *constructing* the core consumed. It does not isolate the readout from the core — with
``pretrained=False``, ``core.initialize()`` xavier-inits every convolution and so consumes a large,
architecture-dependent amount of randomness before the readout is built. The shipped weights were
produced under exactly this ordering, which is why it is preserved rather than tidied.
"""

from dualneuron.twins.layers import (
    ConvNextCore,
    EncoderShifter,
    MultipleFullGaussian2d,
    TaskDrivenCore3,
    set_random_seed,
    unpack_data_info,
)


# ---------------------------------------------------------------------------
#  Weight loading
# ---------------------------------------------------------------------------

def load_state_dict(model, state_dict):
    """Load ``state_dict`` into ``model``, requiring an exact match.

    Equivalent to ``model.load_state_dict(state_dict, strict=True)``, but it names the keys that are
    unused, missing or shape-mismatched rather than raising one combined error.

    Raises:
        RuntimeError: If the key sets differ or any tensor's shape does not match.
    """
    model_dict = model.state_dict()
    filtered = {k: v for k, v in state_dict.items() if k in model_dict}

    unused = sorted(set(state_dict) - set(filtered))
    if unused:
        raise RuntimeError("Error in loading state_dict: Unused keys:\n" + "\n".join(unused))
    missing = sorted(set(model_dict) - set(filtered))
    if missing:
        raise RuntimeError("Error in loading state_dict: Missing keys:\n" + "\n".join(missing))
    for k, v in filtered.items():
        if v.shape != model_dict[k].shape:
            raise RuntimeError(f"Error in loading state_dict: Shape-mismatch for key {k}")

    model.load_state_dict(filtered, strict=True)


def build_model(model_fn, model_config, seed, data_info, state_dict=None):
    """Build one twin and, if given, load its weights.

    Args:
        model_fn: A builder — :func:`task_core_gauss_readout` or :func:`convnext_core_gauss_readout`.
        model_config: Keyword arguments for the builder (the architecture).
        seed: Initialization seed, passed to the builder.
        data_info: Per-session input/output geometry; see :func:`.layers.unpack_data_info`.
        state_dict: Trained weights, or None to keep the initialization.

    Returns:
        torch.nn.Module: The assembled twin.
    """
    net = model_fn(seed=seed, data_info=data_info, **model_config)
    if state_dict is not None:
        load_state_dict(net, state_dict)
    return net


# ---------------------------------------------------------------------------
#  Builders
# ---------------------------------------------------------------------------

def task_core_gauss_readout(
    seed,
    data_info,
    input_channels=1,
    model_name="vgg19",
    layer_name="features.10",
    pretrained=True,
    bias=False,
    final_batchnorm=True,
    final_nonlinearity=True,
    momentum=0.1,
    fine_tune=False,
    init_mu_range=0.4,
    init_sigma_range=0.6,
    readout_bias=True,
    gamma_readout=0.01,
    gauss_type="isotropic",
    elu_offset=-1,
):
    """An ImageNet-pretrained core clipped after ``layer_name``, with a Gaussian readout.

    The architecture of the staged V4 twins. ``input_channels`` is accepted because the configs pass
    it, but ``data_info`` is authoritative and supersedes it — as upstream did.

    Args:
        seed: Initialization seed, applied before the core is built and again before
            ``core.initialize()`` — see the module docstring.
        data_info: Per-session input/output geometry.
        input_channels: Superseded by ``data_info``; kept for config compatibility.
        model_name: Backbone name — see :func:`.layers._task_driven_backbone`.
        layer_name: Dotted module name to clip the backbone after.
        pretrained: Load the backbone's ImageNet weights. False for every twin here.
        bias: If False, zero the clipped trunk's final bias.
        final_batchnorm, final_nonlinearity, momentum, fine_tune: See :class:`.layers.TaskDrivenCore3`.
        init_mu_range, init_sigma_range, readout_bias, gamma_readout, gauss_type: See
            :class:`.layers.FullGaussian2d`.
        elu_offset: Offset in the output ``ELU(x + offset) + 1``.

    Returns:
        torch.nn.Module: An :class:`.layers.EncoderShifter` wrapping the core and readout.
    """
    n_neurons_dict, in_shapes_dict, channels = unpack_data_info(data_info)

    set_random_seed(seed)
    core = TaskDrivenCore3(
        input_channels=channels[0],
        model_name=model_name,
        layer_name=layer_name,
        pretrained=pretrained,
        bias=bias,
        final_batchnorm=final_batchnorm,
        final_nonlinearity=final_nonlinearity,
        momentum=momentum,
        fine_tune=fine_tune,
    )

    set_random_seed(seed)
    core.initialize()

    readout = MultipleFullGaussian2d(
        core,
        in_shape_dict=in_shapes_dict,
        n_neurons_dict=n_neurons_dict,
        init_mu_range=init_mu_range,
        init_sigma=init_sigma_range,
        bias=readout_bias,
        gamma_readout=gamma_readout,
        gauss_type=gauss_type,
    )

    return EncoderShifter(core, readout, shifter=None, elu_offset=elu_offset)


def convnext_core_gauss_readout(
    seed,
    data_info,
    model_name="facebook/convnextv2-nano-1k-224",
    layer_name="convnextv2.encoder.stages.2",
    pretrained=True,
    fine_tune=False,
    patch_embedding_stride=None,
    cut_classification_head=True,
    final_norm=None,
    momentum=0.9,
    final_nonlinearity=None,
    init_mu_range=0.4,
    init_sigma_range=0.6,
    readout_bias=True,
    gamma_readout=0.01,
    gauss_type="isotropic",
    elu_offset=-1,
):
    """A truncated HuggingFace ConvNeXtV2 core with a Gaussian readout.

    The architecture of the staged V1 twin. Note ``momentum`` defaults to 0.9 here against 0.1 in
    :func:`task_core_gauss_readout`; the V1 config does not override it, so the V1 twin's
    ``OutBatchNorm`` tracks its running statistics far faster than the V4 twins' do. That asymmetry is
    inherited from upstream, not chosen — see :func:`dualneuron.twins.nets._v1_convnext_config`.

    Args:
        seed: Initialization seed, applied before the core is built and again before
            ``core.initialize()`` — see the module docstring.
        data_info: Per-session input/output geometry.
        model_name, layer_name, patch_embedding_stride, cut_classification_head, pretrained,
            fine_tune, final_norm, momentum, final_nonlinearity: See :class:`.layers.ConvNextCore`.
        init_mu_range, init_sigma_range, readout_bias, gamma_readout, gauss_type: See
            :class:`.layers.FullGaussian2d`.
        elu_offset: Offset in the output ``ELU(x + offset) + 1``.

    Returns:
        torch.nn.Module: An :class:`.layers.EncoderShifter` wrapping the core and readout.
    """
    n_neurons_dict, in_shapes_dict, _ = unpack_data_info(data_info)

    set_random_seed(seed)
    core = ConvNextCore(
        model_name=model_name,
        layer_name=layer_name,
        patch_embedding_stride=patch_embedding_stride,
        cut_classification_head=cut_classification_head,
        pretrained=pretrained,
        fine_tune=fine_tune,
        in_shapes_dict=in_shapes_dict,
        final_norm=final_norm,
        momentum=momentum,
        final_nonlinearity=final_nonlinearity,
    )

    set_random_seed(seed)
    core.initialize()

    readout = MultipleFullGaussian2d(
        core,
        in_shape_dict=in_shapes_dict,
        n_neurons_dict=n_neurons_dict,
        init_mu_range=init_mu_range,
        init_sigma=init_sigma_range,
        bias=readout_bias,
        gamma_readout=gamma_readout,
        gauss_type=gauss_type,
    )

    return EncoderShifter(core, readout, shifter=None, elu_offset=elu_offset)
