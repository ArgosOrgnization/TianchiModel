'''
 # @ author: bella | bob
 # @ date: 2024-09-18 13:54:20
 # @ license: MIT
 # @ description:
 '''

import torch
import torch.nn as nn

from BasicConv2DLayer import BasicConv2DLayer

class BasicMixedLayer(nn.Module):
    
    def __init__(self, branch_1: nn.Module, branch_2: nn.Module) -> None:
        super(BasicMixedLayer, self).__init__()
        self.branch_1: nn.Module = branch_1
        self.branch_2: nn.Module = branch_2
        pass
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1: torch.Tensor = self.branch_1(x)
        x2: torch.Tensor = self.branch_2(x)
        x: torch.Tensor = torch.cat((x1, x2), dim=1)
        return x
        pass
    
    pass

class Mixed3aLayer(BasicMixedLayer):
    
    def __init__(self) -> None:
        branch_1: nn.MaxPool2d = nn.MaxPool2d(kernel_size = 3, stride = 2)
        branch_2: BasicConv2DLayer = BasicConv2DLayer(64, 96, kernel_size = 3, stride = 2)
        super(Mixed3aLayer, self).__init__(branch_1, branch_2)
        pass
    
    pass

class Mixed4aLayer(BasicMixedLayer):
    
    def __init__(self) -> None:
        branch_1: nn.Sequential = nn.Sequential(
            BasicConv2DLayer(160, 64, kernel_size = 1, stride = 1),
            BasicConv2DLayer(64, 96, kernel_size = 3, stride = 1)
        )
        branch_2: nn.Sequential = nn.Sequential(
            BasicConv2DLayer(160, 64, kernel_size = 1, stride = 1),
            BasicConv2DLayer(64, 64, kernel_size = (1, 7), stride = 1, padding = (0, 3)),
            BasicConv2DLayer(64, 64, kernel_size = (7, 1), stride = 1, padding = (3, 0)),
            BasicConv2DLayer(64, 96, kernel_size = (3, 3), stride = 1)
        )
        super(Mixed4aLayer, self).__init__(branch_1, branch_2)
        pass
    
    pass

class Mixed5aLayer(BasicMixedLayer):
    
    def __init__(self) -> None:
        branch_1: BasicConv2DLayer = BasicConv2DLayer(192, 192, kernel_size = 3, stride = 2)
        branch_2: nn.MaxPool2d = nn.MaxPool2d(kernel_size = 3, stride = 2)
        super(Mixed5aLayer, self).__init__(branch_1, branch_2)
        pass
    
    pass