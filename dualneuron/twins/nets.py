import warnings
warnings.filterwarnings('ignore')

import torch
import os
import numpy as np
from mei.modules import EnsembleModel
from nnfabrik.builder import get_model
import torchvision.models as models


def V4ColorTaskDriven(
    ensemble=False, 
    centered=False, 
    untrained=False,
    base_dir=os.path.dirname(os.path.abspath(__file__))
):
    
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
    
    models = []
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

        models.append(model)
        if not ensemble and i==0: break
    
    if ensemble:
        model = EnsembleModel(*models)
        
    return model


def V1GrayTaskDriven(
    ensemble=False, 
    centered=False, 
    untrained=False,
    base_dir=os.path.dirname(os.path.abspath(__file__))
):

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
    
    models = []
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
            
        models.append(model)
        if not ensemble and i==0: break
    
    if ensemble:
        model = EnsembleModel(*models)
        
    return model


def V4GrayTaskDriven(
    ensemble=False, 
    centered=False, 
    untrained=False,
    base_dir=os.path.dirname(os.path.abspath(__file__))
):
    
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

    models = []
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

        models.append(model)
        if not ensemble and i==0: break
    
    if ensemble:
        model = EnsembleModel(*models)

    return model


def resnet50():
    """
    Loads a torchvision ImageNet pretrained ResNet-50.
    """
    model = models.resnet50(weights='IMAGENET1K_V1')
    return model