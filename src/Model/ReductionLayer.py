'''
 # @ author: bella | bob
 # @ date: 2024-09-18 13:54:20
 # @ license: MIT
 # @ description:
 '''

import torch
import torch.nn as nn

from BasicConv2DLayer import BasicConv2DLayer

class BasicReductionLayer(nn.Module):

    def __init__(self, branch_1: nn.Module, branch_2: nn.Module, branch_3: nn.Module) -> None:
        super(BasicReductionLayer, self).__init__()
        self.branch_1: nn.Module = branch_1
        self.branch_2: nn.Module = branch_2
        self.branch_3: nn.Module = branch_3
        pass

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1: torch.Tensor = self.branch_1(x)
        x2: torch.Tensor = self.branch_2(x)
        x3: torch.Tensor = self.branch_3(x)
        x: torch.Tensor = torch.cat((x1, x2, x3), dim=1)
        return x
        pass

    pass

class ReductionALayer(BasicReductionLayer):

    def __init__(self) -> None:
        branch_1: BasicConv2DLayer = BasicConv2DLayer(384, 384, kernel_size = 3, stride = 2)
        branch_2: nn.Sequential = nn.Sequential(
            BasicConv2DLayer(384, 192, kernel_size = 1, stride = 1),
            BasicConv2DLayer(192, 224, kernel_size = 3, stride = 1, padding = 1),
            BasicConv2DLayer(224, 256, kernel_size = 3, stride = 2)
        )
        branch_3: nn.MaxPool2d = nn.MaxPool2d(kernel_size = 3, stride = 2)
        super(ReductionALayer, self).__init__(branch_1, branch_2, branch_3)
        pass

    pass

class ReductionBLayer(BasicReductionLayer):
    
    def __init__(self) -> None:
        branch_1: nn.Sequential = nn.Sequential(
            BasicConv2DLayer(1024, 192, kernel_size = 1, stride = 1),
            BasicConv2DLayer(192, 192, kernel_size = 3, stride = 2)
        )
        branch_2: nn.Sequential = nn.Sequential(
            BasicConv2DLayer(1024, 256, kernel_size = 1, stride = 1),
            BasicConv2DLayer(256, 256, kernel_size = (1, 7), stride = 1, padding = (0, 3)),
            BasicConv2DLayer(256, 320, kernel_size = (7, 1), stride = 1, padding = (3, 0)),
            BasicConv2DLayer(320, 320, kernel_size = 3, stride = 2)
        )
        branch_3: nn.MaxPool2d = nn.MaxPool2d(kernel_size = 3, stride = 2)
        super(ReductionBLayer, self).__init__(branch_1, branch_2, branch_3)
        pass
    
    pass