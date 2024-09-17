'''
 # @ author: cyq | bcy
 # @ date: 2024-09-16 23:05:23
 # @ license: MIT
 # @ description:
 '''

import torch
import torch.nn as nn

from BasicConv2DLayer import BasicConv2DLayer

class BasicInceptionLayer(nn.Module):
    
    def __init__(self, branch_1: nn.Module, branch_2: nn.Module, branch_3: nn.Module, branch_4: nn.Module) -> None:
        super(BasicInceptionLayer, self).__init__()
        self.branch_1: nn.Module = branch_1
        self.branch_2: nn.Module = branch_2
        self.branch_3: nn.Module = branch_3
        self.branch_4: nn.Module = branch_4
        pass
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1: torch.Tensor = self.branch_1(x)
        x2: torch.Tensor = self.branch_2(x)
        x3: torch.Tensor = self.branch_3(x)
        x4: torch.Tensor = self.branch_4(x)
        x: torch.Tensor = torch.cat((x1, x2, x3, x4), dim=1)
        return x
        pass
    
    pass

class InceptionALayer(BasicInceptionLayer):
    
    def __init__(self) -> None:
        branch_1: BasicConv2DLayer = BasicConv2DLayer(384, 96, kernel_size = 1, stride = 1)
        branch_2: nn.Sequential = nn.Sequential(
            BasicConv2DLayer(384, 64, kernel_size = 1, stride = 1),
            BasicConv2DLayer(64, 96, kernel_size = 3, stride = 1, padding = 1)
        )
        branch_3: nn.Sequential = nn.Sequential(
            BasicConv2DLayer(384, 64, kernel_size = 1, stride = 1),
            BasicConv2DLayer(64, 96, kernel_size = 3, stride = 1, padding = 1),
            BasicConv2DLayer(96, 96, kernel_size = 3, stride = 1, padding = 1)
        )
        branch_4: nn.Sequential = nn.Sequential(
            nn.AvgPool2d(kernel_size = 3, stride = 1, padding = 1, count_include_pad = False),
            BasicConv2DLayer(384, 96, kernel_size = 1, stride = 1)
        )
        super(InceptionALayer, self).__init__(branch_1, branch_2, branch_3, branch_4)
        pass
    
    pass

class InceptionBLayer(BasicInceptionLayer):

    def __init__(self) -> None:
        branch_1: BasicConv2DLayer = BasicConv2DLayer(1024, 384, kernel_size = 1, stride = 1)
        branch_2: nn.Sequential = nn.Sequential(
            BasicConv2DLayer(1024, 192, kernel_size = 1, stride = 1),
            BasicConv2DLayer(192, 224, kernel_size = (1, 7), stride = 1, padding = (0, 3)),
            BasicConv2DLayer(224, 256, kernel_size = (7, 1), stride = 1, padding = (3, 0))
        )
        branch_3: nn.Sequential = nn.Sequential(
            BasicConv2DLayer(1024, 192, kernel_size = 1, stride = 1),
            BasicConv2DLayer(192, 192, kernel_size = (7, 1), stride = 1, padding = (3, 0)),
            BasicConv2DLayer(192, 224, kernel_size = (1, 7), stride = 1, padding = (0, 3)),
            BasicConv2DLayer(224, 224, kernel_size = (7, 1), stride = 1, padding = (3, 0)),
            BasicConv2DLayer(224, 256, kernel_size = (1, 7), stride = 1, padding = (0, 3))
        )
        branch_4: nn.Sequential = nn.Sequential(
            nn.AvgPool2d(kernel_size = 3, stride = 1, padding = 1, count_include_pad = False),
            BasicConv2DLayer(1024, 128, kernel_size = 1, stride = 1)
        )
        super(InceptionBLayer, self).__init__(branch_1, branch_2, branch_3, branch_4)
        pass

    pass

class InceptionCLayer(nn.Module):

    def __init__(self) -> None:
        super(InceptionCLayer, self).__init__()
        # branch 1
        self.branch_1: BasicConv2DLayer = BasicConv2DLayer(1536, 256, kernel_size = 1, stride = 1)
        # branch 2
        self.branch_2_level_1: BasicConv2DLayer = BasicConv2DLayer(1536, 384, kernel_size = 1, stride = 1)
        self.branch_2_level_2_path_1: BasicConv2DLayer = BasicConv2DLayer(384, 256, kernel_size = (1, 3), stride = 1, padding = (0, 1))
        self.branch_2_level_2_path_2: BasicConv2DLayer = BasicConv2DLayer(384, 256, kernel_size = (3, 1), stride = 1, padding = (1, 0))
        # branch 3
        self.branch_3_level_1: BasicConv2DLayer = BasicConv2DLayer(1536, 384, kernel_size = 1, stride = 1)
        self.branch_3_level_2: BasicConv2DLayer = BasicConv2DLayer(384, 448, kernel_size = (3, 1), stride = 1, padding = (1, 0))
        self.branch_3_level_3: BasicConv2DLayer = BasicConv2DLayer(448, 512, kernel_size = (1, 3), stride = 1, padding = (0, 1))
        self.branch_3_level_4_path_1: BasicConv2DLayer = BasicConv2DLayer(512, 256, kernel_size = (1, 3), stride = 1, padding = (0, 1))
        self.branch_3_level_4_path_2: BasicConv2DLayer = BasicConv2DLayer(512, 256, kernel_size = (3, 1), stride = 1, padding = (1, 0))
        # branch 4
        self.branch_4: nn.Sequential = nn.Sequential(
            nn.AvgPool2d(kernel_size = 3, stride = 1, padding = 1, count_include_pad = False),
            BasicConv2DLayer(1536, 256, kernel_size = 1, stride = 1)
        )
        pass

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # branch 1
        x1: torch.Tensor = self.branch_1(x)
        # branch 2
        x2_level_1: torch.Tensor = self.branch_2_level_1(x)
        x2_level_2_path_1: torch.Tensor = self.branch_2_level_2_path_1(x2_level_1)
        x2_level_2_path_2: torch.Tensor = self.branch_2_level_2_path_2(x2_level_1)
        x2: torch.Tensor = torch.cat((x2_level_2_path_1, x2_level_2_path_2), dim=1)
        # branch 3
        x3_level_1: torch.Tensor = self.branch_3_level_1(x)
        x3_level_2: torch.Tensor = self.branch_3_level_2(x3_level_1)
        x3_level_3: torch.Tensor = self.branch_3_level_3(x3_level_2)
        x3_level_4_path_1: torch.Tensor = self.branch_3_level_4_path_1(x3_level_3)
        x3_level_4_path_2: torch.Tensor = self.branch_3_level_4_path_2(x3_level_3)
        x3: torch.Tensor = torch.cat((x3_level_4_path_1, x3_level_4_path_2), dim=1)
        # branch 4
        x4: torch.Tensor = self.branch_4(x)
        # concat
        return torch.cat((x1, x2, x3, x4), dim=1)
        pass

    pass