'''
 # @ author: bella | bob
 # @ date: 2024-09-24 21:02:30
 # @ license: MIT
 # @ description:
 '''

import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "."))
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import random
import numpy as np
import torch

from AverageMeter import AverageMeter
from Datasets import TrainDataset
from Model.DetectModel import DetectModel
from TrainRecipe import TrainRecipe

class TrainManager:
    
    def __init__(
        self,
        train_recipe_file_name: str,
    ) -> None:
        self.train_recipe: TrainRecipe = TrainRecipe(train_recipe_file_name)
        self.model: DetectModel = DetectModel(self.train_recipe.getClassNumber())
        self.model = torch.nn.DataParallel(self.model).cuda()
        self.checkWhetherResume()
        pass
    
    def adjustLearningRate(self) -> torch.optim.Adam:
        adam: torch.optim.Adam = torch.optim.Adam(
            self.model.parameters(),
            lr=self.train_recipe.getLearningRate(),
            weight_decay=self.train_recipe.getWeightDecay(),
            amsgrad=True
        )
        return adam
        pass
    
    def checkWhetherResume(self) -> None:
        if self.train_recipe.getWhetherResume() == True:
            checkpoint = torch.load(self.train_recipe.getResumeCheckpoint())
            self.train_recipe.setStartEpoch(checkpoint["epoch"] + 1)
            self.train_recipe.setBestPrecision(checkpoint["best_precision"])
            self.train_recipe.setLowestLoss(checkpoint["lowest_loss"])
            self.train_recipe.setStage(checkpoint["stage"])
            self.train_recipe.setLearningRate(checkpoint["lr"])
            self.model.load_state_dict(checkpoint["state_dict"])
            if self.train_recipe.getStartEpoch() in np.cumsum(self.train_recipe.getStageEpoches()):
                self.train_recipe.setStage(self.train_recipe.getStage() + 1)
                self.optimizer: torch.optim.Adam = self.adjustLearningRate()
                pass
            pass
        pass
    
    pass