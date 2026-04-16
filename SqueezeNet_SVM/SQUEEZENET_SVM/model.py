import torch
from torch import nn
from torchvision import models


def build_squeezenet_feature_extractor(pretrained=True):
    model = models.squeezenet1_1(weights=models.SqueezeNet1_1_Weights.DEFAULT if pretrained else None)
    features = model.features
    pool = nn.AdaptiveAvgPool2d(1)

    class FeatureExtractor(nn.Module):
        def __init__(self, features, pool):
            super().__init__()
            self.features = features
            self.pool = pool

        def forward(self, x):
            x = self.features(x)
            x = self.pool(x)
            x = torch.flatten(x, 1)
            return x

    return FeatureExtractor(features, pool)
