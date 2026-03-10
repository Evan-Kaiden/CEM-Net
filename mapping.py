from archetectures.vgg import VGG
from archetectures.resnet import *
from archetectures.mobilenetv2 import MobileNetV2

import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torchvision.models as tvm


map_arg = {
    "resnet18"    : lambda pretrained=False: _load_backbone(tvm.resnet18,    tvm.ResNet18_Weights.IMAGENET1K_V1,    pretrained),
    "resnet34"    : lambda pretrained=False: _load_backbone(tvm.resnet34,    tvm.ResNet34_Weights.IMAGENET1K_V1,    pretrained),
    "resnet50"    : lambda pretrained=False: _load_backbone(tvm.resnet50,    tvm.ResNet50_Weights.IMAGENET1K_V1,    pretrained),
    "resnet101"   : lambda pretrained=False: _load_backbone(tvm.resnet101,   tvm.ResNet101_Weights.IMAGENET1K_V1,   pretrained),
    "resnet152"   : lambda pretrained=False: _load_backbone(tvm.resnet152,   tvm.ResNet152_Weights.IMAGENET1K_V1,   pretrained),
    "vgg11"       : lambda pretrained=False: _load_backbone(tvm.vgg11,       tvm.VGG11_Weights.IMAGENET1K_V1,       pretrained),
    "vgg16"       : lambda pretrained=False: _load_backbone(tvm.vgg16,       tvm.VGG16_Weights.IMAGENET1K_V1,       pretrained),
    "vgg19"       : lambda pretrained=False: _load_backbone(tvm.vgg19,       tvm.VGG19_Weights.IMAGENET1K_V1,       pretrained),
    "mobilenetv2" : lambda pretrained=False: _load_backbone(tvm.mobilenet_v2, tvm.MobileNet_V2_Weights.IMAGENET1K_V1, pretrained),
    'mobilenetv2' : MobileNetV2(),
    'adam' : optim.Adam,
    'adamw' : optim.AdamW,
    'rmsprop' : optim.RMSprop,
    'sgd' : optim.SGD,
    'cosine' : lr_scheduler.CosineAnnealingLR, 
    'linear' : lr_scheduler.LinearLR, 
    'step' : lr_scheduler.StepLR, 
    'none' : None
}


def _load_backbone(model_fn, weights, pretrained):
    model = model_fn(weights=weights if pretrained else None)
    # strip final FC and avgpool — return spatial features
    return nn.Sequential(*list(model.children())[:-2])