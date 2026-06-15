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
from mei.modules import EnsembleModel
from nnfabrik.builder import get_model
import torchvision.models as models

from dualneuron.twins.activations import (
    ActivationExtractor,
    count_units
)

from dotenv import load_dotenv
load_dotenv()

DATA_DIR = os.getenv("DATA_DIR")
MODELS_DIR = os.getenv("MODELS_DIR") or (
    os.path.join(DATA_DIR, "models") if DATA_DIR else "./models"
)


def V4ColorTaskDriven(
    ensemble=False, 
    centered=False, 
    untrained=False,
    base_dir=os.path.dirname(os.path.abspath(__file__))
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
        base_dir (str): Directory containing model weights. Default: module
            directory.
    
    Returns:
        torch.nn.Module: The V4 model (single or ensemble).
    
    Model Details:
        - Input: (batch, 3, 100, 100) RGB images
        - Output: (batch, 394) predicted firing rates
        - Backbone: ResNet50 L2-robust (layer3.0)
        - Normalization: mean=113.5, std=59.58
    """
    
    model_fn = 'nnvision.models.ptrmodels.task_core_gauss_readout'
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
    
    ensemble_names = [
        '33bd3a8c2c7dd6916c98ba7ad557eade.pth.tar',
        '44370def81b37c0588e260d6284610fe.pth.tar',
        'a1e5fa8957a5e802b51d70c31c87b62b.pth.tar',
        'ad6a12061d8a8ba02d04dd7b142ebc71.pth.tar',
        'c0f9f75fd8743c363df3f32dfbf88a7f.pth.tar'
    ]
    
    models_list = []
    for i, f in enumerate(ensemble_names):
        filename = os.path.join(base_dir, 'V4ColorTaskDriven', f)
        state_dict = torch.load(filename, map_location='cpu')
        model = get_model(
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


def V1GrayTaskDriven(
    ensemble=False, 
    centered=False, 
    untrained=False,
    base_dir=os.path.dirname(os.path.abspath(__file__))
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
        base_dir (str): Directory containing model weights. Default: module
            directory.
    
    Returns:
        torch.nn.Module: The V1 model (single or ensemble).
    
    Model Details:
        - Input: (batch, 1, 93, 93) grayscale images
        - Output: (batch, 458) predicted firing rates
        - Backbone: ConvNeXtV2-Atto (encoder.stages.1.layers.0)
        - Normalization: mean=124.54, std=70.28
    """

    model_fn = 'nnvision.models.ptrmodels.convnext_core_gauss_readout'
    model_config =  {
        'model_name': 'facebook/convnextv2-atto-1k-224',
        'layer_name': 'convnextv2.encoder.stages.1.layers.0',
        'patch_embedding_stride': None,
        'fine_tune': True,
        'pretrained': False,
        'gamma_readout': 10,
        'final_norm': 'BatchNorm',
        'final_nonlinearity': 'GELU'
    }
    
    data_info = {
        "all_sessions": {
            "input_dimensions": torch.Size([512, 1, 93, 93]),
            "input_channels": 1,
            "output_dimension": 458,
            "img_mean": 124.54466,
            "img_std": 70.28,
        },
    }
    
    ensemble_names = [
        'v1_convnext_1.pth.tar',
        'v1_convnext_2.pth.tar',
        'v1_convnext_3.pth.tar',
        'v1_convnext_4.pth.tar',
        'v1_convnext_5.pth.tar',
    ]
    
    models_list = []
    for i, f in enumerate(ensemble_names):
        torch.manual_seed(i)
        filename = os.path.join(base_dir, 'V1GrayTaskDriven', f)
        state_dict = torch.load(filename, map_location='cpu')
        
        model = get_model(
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
    base_dir=os.path.dirname(os.path.abspath(__file__))
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
        base_dir (str): Directory containing model weights. Default: module
            directory.
    
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
    
    model_fn = 'nnvision.models.ptrmodels.task_core_gauss_readout'
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

    ensemble_names = [
        'task_driven_ensemble_model_01.pth.tar',
        'task_driven_ensemble_model_02.pth.tar',
        'task_driven_ensemble_model_03.pth.tar',
        'task_driven_ensemble_model_04.pth.tar',
        'task_driven_ensemble_model_05.pth.tar',
        'task_driven_ensemble_model_06.pth.tar',
        'task_driven_ensemble_model_07.pth.tar',
        'task_driven_ensemble_model_08.pth.tar',
        'task_driven_ensemble_model_09.pth.tar',
        'task_driven_ensemble_model_10.pth.tar'
    ]

    models_list = []
    for i, f in enumerate(ensemble_names):
        torch.manual_seed(i)
        filename = os.path.join(base_dir, 'V4GrayTaskDriven', f)
        state_dict = torch.load(filename, map_location='cpu')

        model = get_model(
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


def load_model(
    architecture='v4', 
    layer=None, 
    ensemble=False, 
    centered=True, 
    untrained=False,
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
            - 'v1': V1GrayTaskDriven (grayscale, 93x93, 458 neurons)
            - 'v4': V4ColorTaskDriven (color, 100x100, 394 neurons)
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
        'v4', 'v1', 'v4g',
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
            untrained=untrained
        )
    elif architecture == 'v1':
        model = V1GrayTaskDriven(
            ensemble=ensemble, 
            centered=centered,
            untrained=untrained
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