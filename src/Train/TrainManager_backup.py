'''
 # @ author: bella | bob
 # @ date: 2024-09-19 21:01:36
 # @ license: MIT
 # @ description:
 '''

import os
import sys
import shutil
sys.path.append(os.path.join(os.path.dirname(__file__), "."))
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import time

import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

np.random.seed(666)
torch.manual_seed(666)
torch.cuda.manual_seed(666)
random.seed(666)

from AverageMeter import AverageMeter
from Datasets import TrainDataset
from Model.DetectModel import DetectModel

MODEL_PATH: str = "model"

def assurePathExists(path: str) -> None:
    if not os.path.exists(path):
        os.makedirs(path)
        pass
    pass
    
def assureFilePathExists(file_name: str) -> None:
    file_path: str = os.path.dirname(file_name)
    assurePathExists(file_path)
    pass

def getCurrentTimeStr() -> str:
    return time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())
    pass

assurePathExists(MODEL_PATH)

class TrainManager:
    
    def __init__(
        self,
        train_dataset: TrainDataset,
        validate_dataset: TrainDataset,
        test_dataset: TrainDataset,
        model: DetectModel,
        criterion: nn.CrossEntropyLoss,
        optimizer: torch.optim.Adam,
        epoch: int,
        print_frequency: int = 1,
        best_precision: float = 0,
        lowest_loss: float = 100,
        test_data_save_path: str = "result/test_data.csv",
        check_point_save_path: str = "check_points/check_point.pth.tar",
        learning_rate: float = 1e-4,
        learning_rate_decay: float = 5,
        weight_decay: float = 1e-4
    ) -> None:
        self.train_dataset: TrainDataset = train_dataset
        self.validate_dataset: TrainDataset = validate_dataset
        self.test_dataset: TrainDataset = test_dataset
        self.model: DetectModel = model
        self.criterion: nn.CrossEntropyLoss = criterion
        self.optimizer: torch.optim.Adam = optimizer
        self.epoch: int = epoch
        self.print_frequency: int = print_frequency
        self.best_precision: float = best_precision
        self.lowest_loss: float = lowest_loss
        self.time_stamp: str = getCurrentTimeStr()
        self.working_directory: str = os.path.join(MODEL_PATH, self.time_stamp)
        assureFilePathExists(self.working_directory)
        self.test_data_save_path: str = self.getSaveFileName(test_data_save_path)
        self.check_point_save_path: str = self.getSaveFileName(check_point_save_path)
        self.learning_rate: float = learning_rate
        self.learning_rate_decay: float = learning_rate_decay
        self.weight_decay: float = weight_decay
        pass
    
    def getSaveFileName(self, file_name: str) -> str:
        return os.path.join(self.working_directory, file_name)
        pass
    
    def calculateAccuracy(self, predict_labels: torch.Tensor, targets: torch.Tensor, topk: tuple = (1,)) -> tuple:
        final_accuracy: float = 0.0
        max_k: int = max(topk)
        predict_count: int = predict_labels.size(0)
        predict_correct_count: int = 0
        _, pred = predict_labels.topk(max_k, 1, True, True)
        for j in range(pred.size(0)):
            if int(targets[j]) == int(pred[j]):
                predict_correct_count += 1
                pass
            pass
        if predict_count != 0:
            final_accuracy = predict_correct_count / predict_count
            pass
        else:
            final_accuracy = 0.0
            pass
        return final_accuracy * 100, predict_count # percentage
        pass
    
    def train(self) -> None:
        batch_time: AverageMeter = AverageMeter()
        data_time: AverageMeter = AverageMeter() # evaluate the time it takes to load the data
        losses: AverageMeter = AverageMeter()
        accuracy: AverageMeter = AverageMeter()
        self.model.train()
        start_time: float = time.time()
        for index, (images, targets) in enumerate(self.train_dataset):
            # record time
            data_time.update(time.time() - start_time)
            # transfer data to GPU
            images_tensor: torch.Tensor = torch.tensor(images).cuda()
            # transfer target to GPU
            labels: torch.Tensor = torch.tensor(targets).cuda()
            # forward
            predict_labels: torch.Tensor = self.model(images_tensor)
            # calculate loss
            loss: torch.Tensor = self.criterion(predict_labels, labels)
            losses.update(loss.item(), images_tensor.size(0))
            # precison
            precision, predict_count = self.calculateAccuracy(predict_labels.data, targets, topk=(1, 1))
            accuracy.update(precision, predict_count)
            # backward
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            batch_time.update(time.time() - start_time)
            start_time = time.time()
            if index % self.print_frequency == 0:
                print(
                    f"Epoch: [{self.epoch}][{index}/{len(self.train_dataset)}]\t" +
                    f"Time {batch_time.value:.3f} ({batch_time.average:.3f})\t" +
                    f"Data {data_time.value:.3f} ({data_time.average:.3f})\t" +
                    f"Loss {losses.value:.4f} ({losses.average:.4f})\t" +
                    f"Accuracy {accuracy.value:.3f} ({accuracy.average:.3f})"
                )
                pass
            pass
        pass
    
    def validate(self) -> tuple:
        batch_time: AverageMeter = AverageMeter()
        losses: AverageMeter = AverageMeter()
        accuracy: AverageMeter = AverageMeter()
        self.model.eval()
        start_time: float = time.time()
        for index, (images, targets) in enumerate(self.validate_dataset):
            # transfer data to GPU
            images_tensor: torch.Tensor = torch.tensor(images).cuda()
            # transfer target to GPU
            labels: torch.Tensor = torch.tensor(targets).cuda()
            # forward
            with torch.no_grad():
                predict_labels: torch.Tensor = self.model(images_tensor)
                loss: torch.Tensor = self.criterion(predict_labels, labels)
                pass
            precision, predict_count = self.calculateAccuracy(predict_labels.data, targets, topk=(1, 1))
            losses.update(loss.item(), images_tensor.size(0))
            accuracy.update(precision, predict_count)
            batch_time.update(time.time() - start_time)
            start_time = time.time()
            if index % self.print_frequency == 0:
                print(
                    f"Epoch: [{self.epoch}][{index}/{len(self.validate_dataset)}]\t" +
                    f"Time {batch_time.value:.3f} ({batch_time.average:.3f})\t" +
                    f"Loss {losses.value:.4f} ({losses.average:.4f})\t" +
                    f"Accuracy {accuracy.value:.3f} ({accuracy.average:.3f})"
                )
                pass
            pass
        print(
            f" * Accuracy {accuracy.average:.3f} 'Previous Best Accuracy' {self.best_precision:.3f}\t" +
            f" * sLoss {losses.average:.4f} 'Previous Lowest Loss' {self.lowest_loss:.4f}"
        )
        return accuracy.average, losses.average
        pass
    
    def test(self, file_save_path: str = None) -> pd.DataFrame:
        self.model.eval()
        class_number: int = self.model.getClassNumber()
        columns_name: list = ["ImageFile"] + [f"Class{i}" for i in range(class_number)]
        df: pd.DataFrame = pd.DataFrame(columns=columns_name)
        for _, (images, image_file_names) in enumerate(self.test_dataset):
            image_file_names_base: str = [os.path.basename(image_file_name) for image_file_name in image_file_names]
            images_tensor: torch.Tensor = torch.tensor(images, requires_grad=False)
            with torch.no_grad():
                predict_labels: torch.Tensor = self.model(images_tensor)
                softmax: nn.Softmax = nn.Softmax(dim=1)
                soft_max_predict_labels: torch.Tensor = softmax(predict_labels)
                for i in range(len(image_file_names_base)):
                    row: list = [image_file_names_base[i]] + [f"{value:.6f}" for value in soft_max_predict_labels[i]]
                    df.loc[len(df)] = row
                    pass
                pass
            pass
        if file_save_path is not None:
            assureFilePathExists(file_save_path)
            df.to_csv(file_save_path, index=True)
            pass
        else:
            assureFilePathExists(self.test_data_save_path)
            df.to_csv(self.test_data_save_path, index=True)
            pass
        pass
    
    def saveCheckPoint(
        self,
        state: dict,
        is_best: bool,
        is_lowest_loss: bool,
        file_name: str
    ) -> None:
        assureFilePathExists(file_name)
        torch.save(state, file_name)
        base_name: str = os.path.basename(file_name)
        file_path: str = os.path.dirname(file_name)
        if is_best:
            shutil.copyfile(file_name, os.path.join(file_path, "best_" + base_name))
            pass
        if is_lowest_loss:
            shutil.copyfile(file_name, os.path.join(file_path, "lowest_loss_" + base_name))
            pass
        pass
    
    def adjustLearningRate(self) -> torch.optim.Adam:
        self.learning_rate /= self.learning_rate_decay
        adam: torch.optim.Adam = torch.optim.Adam(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
            amsgrad=True
        )
        return adam
        pass
    
    pass