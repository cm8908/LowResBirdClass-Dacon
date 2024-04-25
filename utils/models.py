import os
import torch
import transformers
from torchvision import models
from torch import nn

def load_backbone_model(model_name, **kwargs):
    def huggingface_wrapper(model):
        return lambda **kw: model.from_pretrained(kw['weights'])
    # models.로 시작하는 것들은 torchvision.models에 있는 모델들
    # huggingface_wrapper로 감싼 것들은 huggingface transformers에 있는 모델들
    # huggingface 모델들은 --pretrained_weights를 반드시 설정해줘야 합니다. (DEFAULT 설정 불가)
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
        'swinv2': huggingface_wrapper(transformers.Swinv2ForImageClassification),
        'vitl': huggingface_wrapper(transformers.ViTForImageClassification)
    }
    assert model_name in model_instances, f'Invalid model name: {model_name}'
    return model_instances[model_name](**kwargs)

def load_classifier(in_features, num_classes, cls_type='linear'):
    classifiers = {
        'linear': nn.Linear(in_features, num_classes),
        'mlp': nn.Sequential(
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Linear(512, num_classes)
        ),
        'tanh-linear': nn.Sequential(
            nn.Tanh(),
            nn.Linear(in_features, num_classes)
        ),
    }
    return classifiers[cls_type]