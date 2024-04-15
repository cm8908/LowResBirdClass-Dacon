import os
import torch
from torchvision import models

def load_backbone_model(model_name, **kwargs):
    model_instances = {
        'resnet18': models.resnet18,
        'resnet50': models.resnet50,
        'resnet101': models.resnet101,
        'resnet152': models.resnet152,
        'densenet121': models.densenet121,
        'densenet169': models.densenet169,
        'densenet201': models.densenet201,
        'vith14': models.vit_h_14,
        'vitb16': models.vit_b_16,
        'vitb32': models.vit_b_32,
        'vitl16': models.vit_l_16,
        'vitl32': models.vit_l_32,
    }
    assert model_name in model_instances, f'Invalid model name: {model_name}'
    return model_instances[model_name](**kwargs)

