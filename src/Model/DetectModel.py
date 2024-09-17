'''
 # @ author: cyq | bcy
 # @ date: 2024-09-17 02:04:15
 # @ license: MIT
 # @ description:
 '''

import torch
import torch.nn as nn

from BasicConv2DLayer import PreConv2DLayer
from MixedLayer import Mixed3aLayer, Mixed4aLayer, Mixed5aLayer
from InceptionLayer import InceptionALayer, InceptionBLayer, InceptionCLayer
from ReductionLayer import ReductionALayer, ReductionBLayer

class DetectModel(nn.Module):

    input_space: str = "RGB"
    input_size: tuple = (3, 299, 299)
    input_range: list = [0, 1]
    mean: list = [0.5, 0.5, 0.5]
    std: list = [0.5, 0.5, 0.5]

    def setInputSpace(self, input_space: str) -> None:
        self.input_space = input_space
        pass

    def setInputSize(self, input_size: list) -> None:
        self.input_size = input_size
        pass

    def setInputRange(self, input_range: list) -> None:
        self.input_range = input_range
        pass

    def setMean(self, mean: list) -> None:
        self.mean = mean
        pass

    def setStd(self, std: list) -> None:
        self.std = std
        pass

    def __init__(self, class_number: int) -> None:
        super(DetectModel, self).__init__()
        self.class_number: int = class_number
        self.features_layer: nn.Sequential = nn.Sequential(
            # pre conv2d
            PreConv2DLayer(),
            # mixed layers
            Mixed3aLayer(),
            Mixed4aLayer(),
            Mixed5aLayer(),
            # inception layers
            # 4 x Inception-A layers
            InceptionALayer(),
            InceptionALayer(),
            InceptionALayer(),
            InceptionALayer(),
            ReductionALayer(),
            # 7 x Inception-B layers
            InceptionBLayer(),
            InceptionBLayer(),
            InceptionBLayer(),
            InceptionBLayer(),
            InceptionBLayer(),
            InceptionBLayer(),
            InceptionBLayer(),
            ReductionBLayer(),
            # 3 x Inception-C layers
            InceptionCLayer(),
            InceptionCLayer(),
            InceptionCLayer(),
        )
        self.average_pool_layer: nn.AdaptiveAvgPool2d = nn.AdaptiveAvgPool2d(1)
        self.flatten_feature_layer: nn.Sequential = nn.Sequential(
            nn.BatchNorm1d(1536),
            nn.Dropout(0.5),
            nn.Linear(1536, self.class_number)
        )
        pass

    def logits(self, x: torch.Tensor) -> torch.Tensor:
        x: torch.Tensor = self.average_pool_layer(x)
        x: torch.Tensor = x.view(x.size(0), -1)
        x: torch.Tensor = self.flatten_feature_layer(x)
        return x
        pass

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x: torch.Tensor = self.features_layer(x)
        x: torch.Tensor = self.logits(x)
        return x
        pass

    pass