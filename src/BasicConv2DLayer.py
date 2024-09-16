'''
 # @ author: cyq | bcy
 # @ date: 2024-09-14 00:51:25
 # @ license: MIT
 # @ description:
 '''

import torch
import torch.nn as nn

class BasicConv2DLayer(nn.Module):
    
    def __init__(
        self,
        input_size: int,
        output_size: int,
        kernel_size: int,
        stride: int,
        padding: int,
        bias: bool = False,
        eps: float = 1e-3,
        momentum: float = 0.1
    ) -> None:
        super(BasicConv2DLayer, self).__init__()
        self.conv_2d: nn.Conv2d = nn.Conv2d(
            input_size, output_size,
            kernel_size = kernel_size,
            stride = stride,
            padding = padding,
            bias = bias
        )
        self.batch_norm_2d: nn.BatchNorm2d = nn.BatchNorm2d(
            eps = eps,
            momentum = momentum,
            affine = True
        )
        self.relu: nn.ReLU = nn.ReLU(inplace = True)
        pass
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x: torch.Tensor = self.conv_2d(x)
        x: torch.Tensor = self.batch_norm_2d(x)
        x: torch.Tensor = self.relu(x)
        return x
        pass
    
    pass

class PreConv2DLayer(nn.Module):
    
    def __init__(self) -> None:
        super(PreConv2DLayer, self).__init__()
        self.conv_sequnce: nn.Sequential = nn.Sequential(
            BasicConv2DLayer(3, 32, kernel_size = 3, stride = 2),
            BasicConv2DLayer(32, 32, kernel_size = 3, stride = 1),
            BasicConv2DLayer(32, 64, kernel_size = 3, stride = 1)
        )
        pass

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x: torch.Tensor = self.conv_sequnce(x)
        return x
        pass
    
    pass