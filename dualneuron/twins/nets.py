"""
Neural network model loaders for visual cortex digital twins.

This module provides functions to load pretrained neural predictive models
that predict neural responses in macaque visual cortex (V1, V4). It also
supports loading standard ImageNet-trained models for comparison.

Available Models:
    - V4ColorTaskDriven: Color V4 model (3 channels, 100x100, 394 neurons)
    - V1GrayTaskDriven: Grayscale V1 model (1 channel, 93x93, 458 neurons)
    - V4GrayTaskDriven: Grayscale V4 model (1 channel, 100x100, 1244 neurons)
    - Standard torchvision models (vgg16, resnet50, vit_b_16, etc.)
"""
import warnings
warnings.filterwarnings('ignore')
import torch
import os
import numpy as np
import torchvision.models as models

from dualneuron.twins.builders import (
    build_model,
    convnext_core_gauss_readout,
    task_core_gauss_readout
)
from dualneuron.twins.layers import EnsembleModel
from dualneuron.twins.activations import (
    ActivationExtractor,
    count_units
)
from dualneuron.twins.dino import DINONeuralPredictor

from dotenv import load_dotenv
load_dotenv()

DATA_DIR = os.getenv("DATA_DIR")
MODELS_DIR = os.getenv("MODELS_DIR") or (
    os.path.join(DATA_DIR, "models") if DATA_DIR else "./models"
)
TRAINED_MODELS_DIR = os.getenv("TRAINED_MODELS_DIR") or (
    os.path.join(DATA_DIR, "trained_models") if DATA_DIR else "./trained_models"
)

# Directory holding the GitHub-staged twin weights (twins/<ModelFolder>/...).
_TWINS_DIR = os.path.dirname(os.path.abspath(__file__))


def _member_paths(weights_dir, model_folder, n_members):
    """Resolve the ensemble member weight files, named uniformly ``member_{i}.pth.tar`` (1-indexed).

    When ``weights_dir`` is None, use the GitHub-staged files in ``twins/<model_folder>``; otherwise
    use the same ``member_{i}.pth.tar`` names under ``weights_dir`` (the per-area/backbone trained dir
    written by ``dualneuron.training``). The folder resolves the twin, so the filename is constant.
    """
    base = os.path.join(_TWINS_DIR, model_folder) if weights_dir is None else weights_dir
    return [os.path.join(base, f"member_{i}.pth.tar") for i in range(1, n_members + 1)]


def V4ColorTaskDriven(
    ensemble=False,
    centered=False,
    untrained=False,
    weights_dir=None,
):
    """
    Load the color V4 neural predictive model.
    
    A ResNet50-based model trained to predict neural responses in macaque V4
    to color natural images. Uses an L2-robust pretrained backbone with a
    Gaussian readout for spatial pooling.
    
    Args:
        ensemble (bool): If True, returns an ensemble of 5 models with
            averaged predictions. If False, returns a single model.
            Default: False.
        centered (bool): If True, sets readout positions to image center,
            removing spatial selectivity. Useful for MEI synthesis.
            Default: False.
        untrained (bool): If True, returns model with random weights
            (architecture only). Default: False.
        weights_dir (str, optional): Directory of user-trained ensemble weights
            (named ``member_{i}.pth.tar``), e.g. ``TRAINED_MODELS_DIR/v4/resnet``.
            If None (default), loads the GitHub-staged weights.

    Returns:
        torch.nn.Module: The V4 model (single or ensemble).

    Model Details:
        - Input: (batch, 3, 100, 100) RGB images
        - Output: (batch, 394) predicted firing rates
        - Backbone: ResNet50 L2-robust (layer3.0)
        - Normalization: mean=113.5, std=59.58
    """
    
    model_fn = task_core_gauss_readout
    model_config = {
        'input_channels': 3,
        'model_name': 'resnet50_l2_eps0_1',
        'layer_name': 'layer3.0',
        'pretrained': False,
        'bias': False,
        'final_batchnorm': True,
        'final_nonlinearity': True,
        'momentum': 0.1,
        'fine_tune': False,
        'init_mu_range': 0.4,
        'init_sigma_range': 0.6,
        'readout_bias': True,
        'gamma_readout': 3.0,
        'gauss_type': 'isotropic',
        'elu_offset': -1
    }
    training_img_mean = 113.5
    training_img_std = 59.58
    
    data_info = {
        "all_sessions": {
            "input_dimensions": torch.Size([64, 3, 100, 100]),
            "input_channels": 3,
            "output_dimension": 394,
            "img_mean": training_img_mean,
            "img_std": training_img_std
        }
    }
    
    member_paths = _member_paths(weights_dir, 'V4ColorTaskDriven', 5)
    models_list = []
    for i, filename in enumerate(member_paths):
        state_dict = torch.load(filename, map_location='cpu')
        model = build_model(
            model_fn,
            model_config,
            seed=10,
            data_info=data_info,
            state_dict=None if untrained else state_dict
        )

        if centered:
            model.readout['all_sessions'].mu.data.fill_(0)

        models_list.append(model)
        if not ensemble and i==0: break
    
    if ensemble:
        model = EnsembleModel(*models_list)
        
    return model


def V4ColorDataDriven(
    ensemble=False,
    centered=False,
    untrained=False,
    weights_dir=None,
):
    """
    Load the data-driven color V4 neural predictive model.

    The data-driven counterpart of :func:`V4ColorTaskDriven`: the same architecture,
    input geometry and normalization, predicting the same 394 V4 neurons, but loaded
    from the ``V4ColorDataDriven`` ensemble weights.

    Args:
        ensemble (bool): If True, returns an ensemble of 5 models with
            averaged predictions. If False, returns a single model.
            Default: False.
        centered (bool): If True, sets readout positions to image center,
            removing spatial selectivity. Useful for MEI synthesis.
            Default: False.
        untrained (bool): If True, returns model with random weights
            (architecture only). Default: False.
        weights_dir (str, optional): Directory of user-trained ensemble weights
            (named ``member_{i}.pth.tar``). If None (default), loads the
            GitHub-staged weights.

    Returns:
        torch.nn.Module: The V4 model (single or ensemble).

    Model Details:
        - Input: (batch, 3, 100, 100) RGB images
        - Output: (batch, 394) predicted firing rates
        - Backbone: ResNet50 L2-robust (layer3.0)
        - Normalization: mean=113.5, std=59.58
    """

    model_fn = task_core_gauss_readout
    model_config = {
        'input_channels': 3,
        'model_name': 'resnet50_l2_eps0_1',
        'layer_name': 'layer3.0',
        'pretrained': False,
        'bias': False,
        'final_batchnorm': True,
        'final_nonlinearity': True,
        'momentum': 0.1,
        'fine_tune': False,
        'init_mu_range': 0.4,
        'init_sigma_range': 0.6,
        'readout_bias': True,
        'gamma_readout': 3.0,
        'gauss_type': 'isotropic',
        'elu_offset': -1
    }
    training_img_mean = 113.5
    training_img_std = 59.58

    data_info = {
        "all_sessions": {
            "input_dimensions": torch.Size([64, 3, 100, 100]),
            "input_channels": 3,
            "output_dimension": 394,
            "img_mean": training_img_mean,
            "img_std": training_img_std
        }
    }

    member_paths = _member_paths(weights_dir, 'V4ColorDataDriven', 5)
    models_list = []
    for i, filename in enumerate(member_paths):
        state_dict = torch.load(filename, map_location='cpu')
        model = build_model(
            model_fn,
            model_config,
            seed=10,
            data_info=data_info,
            state_dict=None if untrained else state_dict
        )

        if centered:
            model.readout['all_sessions'].mu.data.fill_(0)

        models_list.append(model)
        if not ensemble and i==0: break

    if ensemble:
        model = EnsembleModel(*models_list)

    return model


# Single source of truth for the V1 ConvNeXt architecture, shared by the loader
# (:func:`V1GrayTaskDriven`) and the training-start builder (:func:`build_convnext_trainable`), so
# weights trained by ``dualneuron.training`` load back through ``V1GrayTaskDriven(weights_dir=...)``.
_V1_CONVNEXT_MODEL_FN = convnext_core_gauss_readout


def _v1_convnext_config(pretrained):
    """Architecture config for the V1 ConvNeXt twin. ``pretrained`` keeps the ImageNet backbone
    (True) or re-randomizes it in ``ConvNextCore.initialize`` (False).

    ``momentum`` is deliberately not set: :func:`~dualneuron.twins.builders.convnext_core_gauss_readout`
    defaults it to 0.9, where the three task-driven configs pass 0.1 explicitly. So this twin's
    ``OutBatchNorm`` tracks its running statistics an order of magnitude faster than the V4 twins' —
    inherited from the architecture the shipped weights came from, not a choice made here. It is not
    a state_dict entry, so it affects retraining only, never loading."""
    return {
        'model_name': 'facebook/convnextv2-atto-1k-224',
        'layer_name': 'convnextv2.encoder.stages.1.layers.0',
        'patch_embedding_stride': None,
        'fine_tune': True,               # whole backbone trainable
        'pretrained': pretrained,
        'gamma_readout': 10,
        'final_norm': 'BatchNorm',       # OutBatchNorm
        'final_nonlinearity': 'GELU',    # OutNonlin (pre-readout)
    }


def _v1_convnext_data_info():
    return {"all_sessions": {"input_dimensions": torch.Size([512, 1, 93, 93]),
            "input_channels": 1, "output_dimension": 458,
            "img_mean": 124.54466, "img_std": 70.28}}


def build_convnext_trainable(seed, device):
    """Fresh V1 ConvNeXt for fine-tuning: ImageNet backbone (trainable) + fresh Gaussian readout.

    Built with ``pretrained=True`` so ``ConvNextCore.initialize`` keeps the ImageNet weights (a
    ``pretrained=False`` build re-randomizes them). Same architecture as :func:`V1GrayTaskDriven`, so
    the trained state_dict loads via ``V1GrayTaskDriven(weights_dir=...)``. ``seed`` varies the
    readout init across ensemble members.
    """
    model = build_model(_V1_CONVNEXT_MODEL_FN, _v1_convnext_config(pretrained=True),
                        seed=seed, data_info=_v1_convnext_data_info(), state_dict=None)
    return model.to(device)


def V1GrayTaskDriven(
    ensemble=False,
    centered=False,
    untrained=False,
    weights_dir=None,
):
    """
    Load the grayscale V1 neural predictive model.
    
    A ConvNeXt-based model trained to predict neural responses in macaque V1
    to grayscale natural images. Uses a ConvNeXtV2-Atto backbone with a
    Gaussian readout.
    
    Args:
        ensemble (bool): If True, returns an ensemble of 5 models with
            averaged predictions. If False, returns a single model.
            Default: False.
        centered (bool): If True, sets readout positions to image center,
            removing spatial selectivity. Useful for MEI synthesis.
            Default: False.
        untrained (bool): If True, returns model with random weights
            for both backbone and readout. Default: False.
        weights_dir (str, optional): Directory of user-trained ensemble weights
            (named ``member_{i}.pth.tar``), e.g. ``TRAINED_MODELS_DIR/v1/convnext``.
            If None (default), loads the GitHub-staged weights.

    Returns:
        torch.nn.Module: The V1 model (single or ensemble).
    
    Model Details:
        - Input: (batch, 1, 93, 93) grayscale images
        - Output: (batch, 458) predicted firing rates
        - Backbone: ConvNeXtV2-Atto (encoder.stages.1.layers.0)
        - Normalization: mean=124.54, std=70.28
    """

    model_fn = _V1_CONVNEXT_MODEL_FN
    model_config = _v1_convnext_config(pretrained=False)
    data_info = _v1_convnext_data_info()

    member_paths = _member_paths(weights_dir, 'V1GrayTaskDriven', 5)
    models_list = []
    for i, filename in enumerate(member_paths):
        torch.manual_seed(i)
        state_dict = torch.load(filename, map_location='cpu')

        model = build_model(
            model_fn, 
            model_config, 
            seed=10, 
            data_info=data_info, 
            state_dict=None if untrained else state_dict
        )
        
        if centered:
            model.readout['all_sessions'].mu.data.fill_(0)
            
        if untrained:
            dk = 'all_sessions'
            like = model.readout[dk].features.data
            model.readout[dk].features.data = torch.randn_like(like)
            
        models_list.append(model)
        if not ensemble and i==0: break
    
    if ensemble:
        model = EnsembleModel(*models_list)
        
    return model


def V4GrayTaskDriven(
    ensemble=False,
    centered=False,
    untrained=False,
    weights_dir=None,
):
    """
    Load the grayscale V4 neural predictive model.
    
    A ResNet50-based model trained to predict neural responses in macaque V4
    to grayscale natural images. Similar architecture to V4ColorTaskDriven
    but with single-channel input.
    
    Args:
        ensemble (bool): If True, returns an ensemble of 10 models with
            averaged predictions. If False, returns a single model.
            Default: False.
        centered (bool): If True, sets readout positions to image center,
            removing spatial selectivity. Useful for MEI synthesis.
            Default: False.
        untrained (bool): If True, returns model with random weights
            (architecture only). Default: False.
        weights_dir (str, optional): Directory of user-trained ensemble weights
            (named ``member_{i}.pth.tar``), e.g. ``TRAINED_MODELS_DIR/v4g/resnet``.
            If None (default), loads the GitHub-staged weights.

    Returns:
        torch.nn.Module: The V4 grayscale model (single or ensemble).
    
    Model Details:
        - Input: (batch, 1, 100, 100) grayscale images
        - Output: (batch, 1244) predicted firing rates
        - Backbone: ResNet50 L2-robust (layer3.0)
        - Normalization: mean=124.54, std=70.28
    
    Note:
        This model has more neurons (1244) and more ensemble members (10)
        than V4ColorTaskDriven.
    """
    
    model_fn = task_core_gauss_readout
    model_config = {
        'input_channels': 1,
        'model_name': 'resnet50_l2_eps0_1',
        'layer_name': 'layer3.0',
        'pretrained': False,
        'bias': False,
        'final_batchnorm': True,
        'final_nonlinearity': True,
        'momentum': 0.1,
        'fine_tune': True,
        'init_mu_range': 0.4,
        'init_sigma_range': 0.6,
        'readout_bias': True,
        'gamma_readout': 3.0,
        'gauss_type': 'isotropic',
        'elu_offset': -1,
    }

    data_info = {
        "all_sessions": {
            "input_dimensions": torch.Size([64, 1, 100, 100]),
            "input_channels": 1,
            "output_dimension": 1244,
            "img_mean": 124.54466,
            "img_std": 70.28,
        }
    }

    member_paths = _member_paths(weights_dir, 'V4GrayTaskDriven', 10)
    models_list = []
    for i, filename in enumerate(member_paths):
        torch.manual_seed(i)
        state_dict = torch.load(filename, map_location='cpu')

        model = build_model(
            model_fn, 
            model_config, 
            seed=10, 
            data_info=data_info, 
            state_dict=None if untrained else state_dict
        )
        
        if centered:
            model.readout['all_sessions'].mu.data.fill_(0)

        models_list.append(model)
        if not ensemble and i==0: break
    
    if ensemble:
        model = EnsembleModel(*models_list)

    return model


def _load_dino_member(filename, n_neurons, model_dir, block, readout_type, readout_nonlin,
                      elu_offset, fine_tune, untrained, train_hint=""):
    """Build one DINO twin member and load its weights (frozen or fine-tuned checkpoint format).

    When ``untrained`` is False, the architecture metadata (block, readout_type, readout_nonlin,
    elu_offset, fine_tune) is read from the checkpoint when present (falling back to the passed
    defaults), then the weights are loaded: a fine-tuned member stores the whole model under
    ``model_state_dict``; a frozen member stores only ``readout_state_dict`` + ``norm_state_dict``.
    """
    ckpt = None
    if not untrained:
        if not os.path.isfile(filename):
            raise FileNotFoundError(
                f"Trained DINO weights not found at {filename}. Train with "
                f"`python -m dualneuron.training.run {train_hint}`, or pass weights_dir=...")
        ckpt = torch.load(filename, map_location="cpu")
        block = ckpt.get("block", block)
        readout_type = ckpt.get("readout_type", readout_type)
        readout_nonlin = ckpt.get("readout_nonlin", readout_nonlin)
        elu_offset = ckpt.get("elu_offset", elu_offset)
        fine_tune = ckpt.get("fine_tune", fine_tune)

    model = DINONeuralPredictor(
        n_neurons=n_neurons, model_name="dinov3_vitb16", feature_dim=768, spatial_size=14,
        block=block, model_dir=model_dir, readout_type=readout_type, readout_nonlin=readout_nonlin,
        elu_offset=elu_offset, untrained=untrained, fine_tune=fine_tune)

    if ckpt is not None:
        if "model_state_dict" in ckpt:
            model.load_state_dict(ckpt["model_state_dict"])
        else:
            model.readout.load_state_dict(ckpt["readout_state_dict"])
            model.core.norm.load_state_dict(ckpt["norm_state_dict"])
    return model


def V4ColorDino(
    ensemble=False,
    centered=False,
    untrained=False,
    weights_dir=None,
    block=4,
    readout_type="fullgaussian2d",
    model_dir=None,
):
    """
    Load the DINOv3-based color V4 neural predictive model.

    A frozen, license-gated DINOv3 ViT-B/16 backbone (block ``block``) with a trainable Gaussian
    readout — the DINOv3 counterpart of :func:`V4ColorTaskDriven`. The interface mirrors the other
    twins (``ensemble``, ``centered``, ``untrained``) so :func:`load_model` treats it uniformly.

    Unlike the ResNet/ConvNeXt twins, the trained DINO weights are NOT staged in the repo: they are
    written by ``dualneuron.training`` to ``TRAINED_MODELS_DIR/v4/dino`` (named ``member_{i}.pth.tar``).
    The frozen DINOv3 backbone is license-gated and loaded from ``MODELS_DIR/dinov3``.

    Args:
        ensemble (bool): If True, average an ensemble of 5 members. Default: False.
        centered (bool): If True, set the readout positions to image center
            (``readout.mu = 0``), removing spatial selectivity. Default: False.
        untrained (bool): If True, random backbone + readout (architecture only); needs only the
            hubconf repo, not the gated checkpoint, and loads no trained weights. Default: False.
        weights_dir (str, optional): Directory of the trained ensemble. If None (default),
            ``TRAINED_MODELS_DIR/v4/dino``.
        block (int): DINOv3 transformer block to read out. Default: 4.
        readout_type (str): ``'fullgaussian2d'`` (default) or ``'gaussian'`` — must match training.
        model_dir (str, optional): DINOv3 hubconf+weights dir. If None, ``MODELS_DIR/dinov3``.

    Returns:
        torch.nn.Module: The DINOv3 V4 model (single or ensemble).

    Model Details:
        - Input: (batch, 3, 224, 224) RGB images
        - Output: (batch, 394) predicted firing rates
        - Backbone: frozen DINOv3 ViT-B/16, block 4; trainable BatchNorm2d + Gaussian readout
        - Normalization: mean=113.5, std=59.58 (applied in the training/eval transform)
    """
    n_neurons = 394
    if model_dir is None:
        model_dir = os.path.join(MODELS_DIR, "dinov3")
    if weights_dir is None:
        weights_dir = os.path.join(TRAINED_MODELS_DIR, "v4", "dino")

    models_list = []
    for i in range(1, 6):
        filename = os.path.join(weights_dir, f"member_{i}.pth.tar")
        model = _load_dino_member(
            filename, n_neurons=n_neurons, model_dir=model_dir, block=block,
            readout_type=readout_type, readout_nonlin="gelu", elu_offset=-1, fine_tune=False,
            untrained=untrained, train_hint="--area v4 --backbone dino")
        if centered:
            model.readout.mu.data.fill_(0)
        models_list.append(model)
        if not ensemble and i == 1:
            break

    if ensemble:
        model = EnsembleModel(*models_list)

    return model


def V1GrayDino(
    ensemble=False,
    centered=False,
    untrained=False,
    weights_dir=None,
    block=1,
    readout_type="fullgaussian2d",
    model_dir=None,
):
    """
    Load the DINOv3-based grayscale V1 neural predictive model (block-1 fine-tuned).

    The DINOv3 counterpart of :func:`V1GrayTaskDriven`: a DINOv3 ViT-B/16 whose stem + blocks 0..block
    are fine-tuned end-to-end, read out at ``block`` with the head ``BatchNorm2d -> GELU -> Gaussian
    readout -> ELU(x-1)+1`` (mirroring the V1 ConvNeXt task-driven head). Grayscale input is
    replicated to 3 channels at the stem.

    Trained weights are NOT staged in the repo (like V4 DINO): they are written by
    ``dualneuron.training`` to ``TRAINED_MODELS_DIR/v1/dino`` (``member_{i}.pth.tar``); the gated
    DINOv3 backbone is loaded from ``MODELS_DIR/dinov3``.

    Args:
        ensemble (bool): If True, average an ensemble of 5 members. Default: False.
        centered (bool): If True, set readout positions to image center. Default: False.
        untrained (bool): If True, random backbone + readout (architecture only). Default: False.
        weights_dir (str, optional): Trained ensemble dir. If None, ``TRAINED_MODELS_DIR/v1/dino``.
        block (int): DINOv3 block to read out / fine-tune up to. Default: 1.
        readout_type (str): ``'fullgaussian2d'`` (default) or ``'gaussian'`` — must match training.
        model_dir (str, optional): DINOv3 hubconf+weights dir. If None, ``MODELS_DIR/dinov3``.

    Returns:
        torch.nn.Module: The DINOv3 V1 model (single or ensemble).

    Model Details:
        - Input: (batch, 1, 224, 224) grayscale (replicated to 3ch at the stem)
        - Output: (batch, 458) predicted firing rates
        - Backbone: DINOv3 ViT-B/16, block 1, blocks 0-1 fine-tuned; BN2d + GELU + Gaussian readout
        - Normalization: mean=124.54, std=70.28 (applied in the training/eval transform)
    """
    n_neurons = 458
    if model_dir is None:
        model_dir = os.path.join(MODELS_DIR, "dinov3")
    if weights_dir is None:
        weights_dir = os.path.join(TRAINED_MODELS_DIR, "v1", "dino")

    models_list = []
    for i in range(1, 6):
        filename = os.path.join(weights_dir, f"member_{i}.pth.tar")
        model = _load_dino_member(
            filename, n_neurons=n_neurons, model_dir=model_dir, block=block,
            readout_type=readout_type, readout_nonlin="gelu", elu_offset=-1, fine_tune=True,
            untrained=untrained, train_hint="--area v1 --backbone dino")
        if centered:
            model.readout.mu.data.fill_(0)
        models_list.append(model)
        if not ensemble and i == 1:
            break

    if ensemble:
        model = EnsembleModel(*models_list)

    return model


def load_model(
    architecture='v4',
    layer=None,
    ensemble=False,
    centered=True,
    untrained=False,
    weights_dir=None,
    device='cuda',
    cache_dir=None,
    dreamsim_type="dino_vitb16"
):
    """
    Load a neural network model for activation extraction or prediction.
    
    Unified interface for loading neural predictive models (V1, V4) or
    standard ImageNet-trained models. Optionally wraps with ActivationExtractor
    for intermediate layer access.
    
    Args:
        architecture (str): Model architecture to load. Options:
            - 'v1': V1GrayTaskDriven (grayscale ConvNeXt, 93x93, 458 neurons)
            - 'v1_dino': V1GrayDino (grayscale DINOv3 block-1, 224x224, 458 neurons)
            - 'v4': V4ColorTaskDriven (color ResNet, 100x100, 394 neurons)
            - 'v4_data_driven': V4ColorDataDriven (data-driven color ResNet,
              100x100, 394 neurons)
            - 'v4_dino': V4ColorDino (color DINOv3, 224x224, 394 neurons)
            - 'v4g': V4GrayTaskDriven (grayscale, 100x100, 1244 neurons)
            - 'vgg16': VGG16 pretrained on ImageNet
            - 'vgg16_bn': VGG16 with batch normalization
            - 'resnet50': ResNet50 pretrained on ImageNet
            - 'vit_b_16': Vision Transformer B/16
            Default: 'v4'.
        layer (str, optional): Layer name to extract activations from.
            If None, returns the full model. If specified, wraps model
            with ActivationExtractor. Default: None.
        ensemble (bool): For neural predictive models, whether to use
            ensemble averaging. Ignored for ImageNet models. Default: False.
        centered (bool): For neural predictive models, whether to center
            readout positions. Ignored for ImageNet models. Default: True.
        untrained (bool): If True, returns model with random weights.
            Default: False.
        weights_dir (str, optional): For the V4 twins ('v4', 'v4_dino'), a
            directory of user-trained ensemble weights (see TRAINED_MODELS_DIR).
            If None (default), loads the GitHub-staged weights. Default: None.
        device (str or torch.device): Device to move model to.
            Default: 'cuda'.
        cache_dir (str, optional): Directory to cache downloaded models (e.g.
            DreamSim). Defaults to MODELS_DIR (DATA_DIR/models).
    
    Returns:
        torch.nn.Module or ActivationExtractor: The loaded model in eval mode.
            If layer is specified, returns an ActivationExtractor that can
            be called like a function.
    
    Raises:
        AssertionError: If architecture is not recognized.
    """
    assert architecture in [
        'v4', 'v4_data_driven', 'v4_dino', 'v1', 'v1_dino', 'v4g',
        'vgg16',
        'vgg16_bn',
        'resnet50',
        'vit_b_16',
        'dreamsim'
    ]
    if architecture == 'v4':
        model = V4ColorTaskDriven(
            ensemble=ensemble,
            centered=centered,
            untrained=untrained,
            weights_dir=weights_dir
        )
    elif architecture == 'v4_data_driven':
        model = V4ColorDataDriven(
            ensemble=ensemble,
            centered=centered,
            untrained=untrained,
            weights_dir=weights_dir
        )
    elif architecture == 'v4_dino':
        model = V4ColorDino(
            ensemble=ensemble,
            centered=centered,
            untrained=untrained,
            weights_dir=weights_dir
        )
    elif architecture == 'v1':
        model = V1GrayTaskDriven(
            ensemble=ensemble,
            centered=centered,
            untrained=untrained,
            weights_dir=weights_dir
        )
    elif architecture == 'v1_dino':
        model = V1GrayDino(
            ensemble=ensemble,
            centered=centered,
            untrained=untrained,
            weights_dir=weights_dir
        )
    elif architecture == 'v4g':
        model = V4GrayTaskDriven(
            ensemble=ensemble, 
            centered=centered, 
            untrained=untrained
        )
    elif architecture == 'dreamsim':
        from dreamsim import dreamsim
        model, _ = dreamsim(
            pretrained=True,
            device=device,
            cache_dir=cache_dir or MODELS_DIR,
            dreamsim_type=dreamsim_type
        )
    else:
        if untrained:
            model = getattr(models, architecture)(weights=None)
        else:
            model = getattr(models, architecture)(weights='IMAGENET1K_V1')

    model = model.eval().to(device)
    if layer is not None:
        model = ActivationExtractor(model=model, layer=layer)
    return model


def model_summary(architecture, input_size=(1, 3, 100, 100), device='cuda'):
    """
    Print a summary of all layers and their activation shapes.
    
    Runs a forward pass with dummy input and captures activations from
    all layers, printing their names, shapes, and unit counts.
    
    Args:
        architecture (str): Model architecture name (same options as load_model).
        input_size (tuple): Input tensor shape as (batch, channels, H, W).
            Default: (1, 3, 100, 100).
        device (str or torch.device): Device to run on. Default: 'cuda'.
    
    Returns:
        tuple: (model, activations)
            - model (torch.nn.Module): The loaded model.
            - activations (dict): Mapping from layer names to activation tensors.
    
    Note:
        Useful for exploring layer names to use with load_model(layer=...).
    """
    model = load_model(architecture=architecture, device=device)
    dummy_input = torch.randn(input_size).to(next(model.parameters()).device)
    extractor = ActivationExtractor(model)
    activations = extractor.get_all_activations(dummy_input)

    total_units = 0
    print(f"\n{'Layer':<60} {'Shape':<25} {'Units'}")
    print('-'*95)

    for name, activation in activations.items():
        shape = tuple(activation.shape)
        num_units = count_units(shape)
        if num_units:
            total_units += num_units
            print(f"{name:<60} {str(shape):<25} {num_units:>8,}")

    print(f"Total units: {total_units:,}")
    return model, activations