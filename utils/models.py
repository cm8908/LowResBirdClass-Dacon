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
    }
    assert model_name in model_instances, f'Invalid model name: {model_name}'
    return model_instances[model_name](**kwargs)

