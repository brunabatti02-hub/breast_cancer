import torch
from torch import nn
from torchvision import models


def build_histodx_torch(num_classes=2, pretrained=True):
    # EfficientNetV2-S is available in torchvision and avoids extra deps
    model = models.efficientnet_v2_s(weights=models.EfficientNet_V2_S_Weights.DEFAULT if pretrained else None)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)
    return model
