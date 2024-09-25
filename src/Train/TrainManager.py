'''
 # @ author: bella | bob
 # @ date: 2024-09-24 21:02:30
 # @ license: MIT
 # @ description:
 '''

import os
import sys
import time
import shutil
import random
os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1"
sys.path.append(os.path.join(os.path.dirname(__file__), "."))
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split

from Utils.TimeRelated import getCurrentTimeStr
from AverageMeter import AverageMeter
from Datasets import TrainDataset, ValidateDataset, TestDataset
from Model.DetectModel import DetectModel
from TrainRecipe import TrainRecipe
from DataEnhancement import ImageRotator

class TrainManager:
    
    def __init__(
        self,
        train_recipe_file_name: str,
    ) -> None:
        self.train_recipe: TrainRecipe = TrainRecipe(train_recipe_file_name)
        self.model: DetectModel = DetectModel(self.train_recipe.getClassNumber())
        self.model = torch.nn.DataParallel(self.model).cuda()
        self.setRandomSeed()
        self.checkWhetherResume()
        self.loadData()
        self.initializeCriterionAndOptimizer()
        self.openLog()
        pass
    
    def __del__(self) -> None:
        self.log_file.close()
        self.best_accuracy_log_file.close()
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
    
    def setRandomSeed(self) -> None:
        np.random.seed(self.train_recipe.getRandomNumpySeed())
        torch.manual_seed(self.train_recipe.getRandomTorchManualSeed())
        torch.cuda.manual_seed_all(self.train_recipe.getRandomTorchCudaManualSeed())
        random.seed(self.train_recipe.getRandomSeed())
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
    
    def loadData(self) -> None:
        self.label_csv_file_data_frame: pd.DataFrame = pd.read_csv(self.train_recipe.getLabelCsvFile())
        self.train_data_list, self.validate_data_list = train_test_split(
            self.label_csv_file_data_frame,
            test_size=self.train_recipe.getValidateDatasetRatio(),
            random_state=self.train_recipe.getRandomState(),
            stratify=self.label_csv_file_data_frame["label"]
        )
        self.test_data_list = pd.read_csv(self.train_recipe.getTestCsvFile())
        self.normalize: transforms.Normalize = transforms.Normalize(
            mean=self.train_recipe.getMean(),
            std=self.train_recipe.getStd()
        )
        train_transform: transforms.Compose = transforms.Compose([
            transforms.Resize(self.train_recipe.getTrainTransformDict()["resize"]),
            transforms.ColorJitter(*self.train_recipe.getTrainTransformDict()["color jitter"]),
            transforms.RandomHorizontalFlip(),
            transforms.RandomGrayscale(),
            ImageRotator(self.train_recipe.getTrainTransformDict()["rotation angles"]),
            transforms.RandomCrop(self.train_recipe.getTrainTransformDict()["random crop"]),
            transforms.ToTensor(),
            self.normalize
        ])
        self.train_data: TrainDataset = TrainDataset(
            self.train_data_list,
            transform=train_transform,
        )
        validate_transform: transforms.Compose = transforms.Compose([
            transforms.Resize(self.train_recipe.getValidateTransformDict()["resize"]),
            transforms.CenterCrop(self.train_recipe.getValidateTransformDict()["center crop"]),
            transforms.ToTensor(),
            self.normalize
        ])
        self.validate_data: ValidateDataset = ValidateDataset(
            self.validate_data_list,
            transform=validate_transform,
        )
        test_transform: transforms.Compose = transforms.Compose([
            transforms.Resize(self.train_recipe.getTestTransformDict()["resize"]),
            transforms.CenterCrop(self.train_recipe.getTestTransformDict()["center crop"]),
            transforms.ToTensor(),
            self.normalize
        ])
        self.test_data: TestDataset = TestDataset(
            self.test_data_list,
            transform=test_transform,
        )
        self.train_data_loader: DataLoader = DataLoader(
            self.train_data,
            batch_size=self.train_recipe.getBatchSize(),
            shuffle=True,
            pin_memory=True,
            num_workers=self.train_recipe.getWorkerNumber()
        )
        self.validate_data_loader: DataLoader = DataLoader(
            self.validate_data,
            batch_size=self.train_recipe.getBatchSize() * 2,
            shuffle=True,
            pin_memory=True,
            num_workers=self.train_recipe.getWorkerNumber()
        )
        self.test_data_loader: DataLoader = DataLoader(
            self.test_data,
            batch_size=self.train_recipe.getBatchSize() * 2,
            shuffle=False,
            pin_memory=True,
            num_workers=self.train_recipe.getWorkerNumber()
        )
        pass
    
    def initializeCriterionAndOptimizer(self) -> None:
        self.criterion: nn.CrossEntropyLoss = nn.CrossEntropyLoss()
        self.optimizer: torch.optim.Adam = self.adjustLearningRate()
        pass
    
    def openLog(self) -> None:
        self.log_file = open(self.train_recipe.getLogFileName(), "w")
        self.best_accuracy_log_file = open(self.train_recipe.getBestAccuracyLogFileName(), "w")
        self.writeLog(
            f"time: {self.train_recipe.getTimeStamp()}\n" +
            f"recipe file: {self.train_recipe.getTrainRecipeFileName()}\n" +
            f"recipe name: {self.train_recipe.getRecipeName()}\n" +
            f"type: common log\n"
        )
        self.writeBestAccuracyLog(
            f"time: {self.train_recipe.getTimeStamp()}\n" +
            f"recipe file: {self.train_recipe.getTrainRecipeFileName()}\n" +
            f"recipe name: {self.train_recipe.getRecipeName()}\n" +
            f"type: best accuracy log\n"
        )
        pass
    
    def writeLog(self, log: str) -> None:
        self.log_file.write(log + "\n")
        pass
    
    def writeBestAccuracyLog(self, log: str) -> None:
        self.best_accuracy_log_file.write(log + "\n")
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
    
    def train(self, epoch: int) -> None:
        batch_time: AverageMeter = AverageMeter()
        data_time: AverageMeter = AverageMeter() # evaluate the time it takes to load the data
        losses: AverageMeter = AverageMeter()
        accuracy: AverageMeter = AverageMeter()
        self.model.train()
        start_time: float = time.time()
        for index, (images, targets) in enumerate(self.train_data_loader):
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
            # update time
            batch_time.update(time.time() - start_time)
            start_time = time.time()
            if index % self.train_recipe.getPrintFrequency() == 0:
                log: str = f"Epoch: [{epoch}][{index}/{len(self.train_data)}]\t" + \
                    f"Time {batch_time.value:.3f} ({batch_time.average:.3f})\t" + \
                    f"Data {data_time.value:.3f} ({data_time.average:.3f})\t" + \
                    f"Loss {losses.value:.4f} ({losses.average:.4f})\t" + \
                    f"Accuracy {accuracy.value:.3f} ({accuracy.average:.3f})"
                print(log)
                pass
            pass
        pass
    
    def validate(self) -> tuple:
        batch_time: AverageMeter = AverageMeter()
        losses: AverageMeter = AverageMeter()
        accuracy: AverageMeter = AverageMeter()
        self.model.eval()
        start_time: float = time.time()
        for index, (images, targets) in enumerate(self.validate_data_loader):
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
                log: str = f"TrainValidate: [{index}/{len(self.validate_data)}]\t" + \
                    f"Time {batch_time.value:.3f} ({batch_time.average:.3f})\t" + \
                    f"Loss {losses.value:.4f} ({losses.average:.4f})\t" + \
                    f"Accuracy {accuracy.value:.3f} ({accuracy.average:.3f})"
                print(log)
                pass
            pass
        log: str = f" * Accuracy {accuracy.average:.3f} 'Previous Best Accuracy' {self.train_recipe.getBestPrecision():.3f}\t" + \
            f" * Loss {losses.average:.4f} 'Previous Lowest Loss' {self.train_recipe.getLowestLoss():.4f}"
        print(log)
        return accuracy.average, losses.average
        pass
    
    def test(self) -> None:
        self.model.eval()
        class_number: int = self.model.getClassNumber()
        columns_name: list = ["ImageFile"] + [f"Class{i}" for i in range(class_number)]
        df: pd.DataFrame = pd.DataFrame(columns=columns_name)
        for _, (images, image_file_names) in enumerate(self.test_data_loader):
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
        df.to_csv(self.train_recipe.getResultFileName(), index=False)
        pass
    
    def saveCheckPoints(self, state: dict, is_best_accuracy: bool, is_lowest_loss: bool) -> None:
        torch.save(state, self.train_recipe.getCheckPointName())
        if is_best_accuracy == True:
            shutil.copyfile(self.train_recipe.getCheckPointName(), self.train_recipe.getBestModelName())
            pass
        if is_lowest_loss == True:
            shutil.copyfile(self.train_recipe.getCheckPointName(), self.train_recipe.getLowestLossName())
            pass
        pass
    
    def start(self) -> None:
        if self.train_recipe.getWheterEvaluate() == True:
            self.validate()
            return None
            pass
        for epoch in range(self.train_recipe.getStartEpoch(), self.train_recipe.getTotalEpoches()):
            self.train(epoch)
            validate_accuracy, validate_loss = self.validate()
            self.writeLog(
                f"Epoch: {epoch}, Precision: {validate_accuracy}, Loss: {validate_loss}"
            )
            is_best_accuracy: bool = validate_accuracy > self.train_recipe.getBestPrecision()
            is_lowest_loss: bool = validate_loss < self.train_recipe.getLowestLoss()
            best_precision: float = max(validate_accuracy, self.train_recipe.getBestPrecision())
            lowest_loss: float = min(validate_loss, self.train_recipe.getLowestLoss())
            self.train_recipe.setBestPrecision(best_precision)
            self.train_recipe.setLowestLoss(lowest_loss)
            state: dict = {
                "epoch": epoch,
                "state_dict": self.model.state_dict(),
                "best_precision": best_precision,
                "lowest_loss": lowest_loss,
                "stage": self.train_recipe.getStage(),
                "lr": self.train_recipe.getLearningRate()
            }
            self.saveCheckPoints(state, is_best_accuracy, is_lowest_loss)
            if (epoch + 1) in np.cumsum(self.train_recipe.getStageEpoches())[:-1]:
                self.setStage(self.train_recipe.getStage() + 1)
                self.optimizer = self.adjustLearningRate()
                self.model.load_state_dict(torch.load(self.train_recipe.getBestModelName())["state_dict"])
                log: str = f"==================== Step into next stage ===================="
                print(log)
                self.writeLog(log)
                pass
            pass
        log: str = f"* best accuracy: {self.train_recipe.getBestPrecision():.8f}, time: {getCurrentTimeStr()}"
        self.writeLog(log)
        self.writeBestAccuracyLog(log)
        # test
        best_model_state_dict: dict = torch.load(self.train_recipe.getBestModelName())
        self.model.load_state_dict(best_model_state_dict["state_dict"])
        self.test()
        # release GPU memory
        torch.cuda.empty_cache()
        pass
    
    pass

# train_manager: TrainManager = TrainManager("model/recipe/demo.json")
# print(train_manager.train_data_list)
# print(train_manager.validate_data_list)
# print(train_manager.test_data_list)