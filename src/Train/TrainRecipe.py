'''
 # @ author: bella | bob
 # @ date: 2024-09-24 17:55:48
 # @ license: MIT
 # @ description:
 '''

import os
import time
import sys
import json
import shutil
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

DEFAULT_TRAIN_RECIPE_FILE_NAME: str = "model/recipe/demo.json"
DEFAULT_MODEL_PATH: str = "model"

from Utils.TimeRelated import getCurrentTimeStr

def assurePathExists(path: str) -> None:
    if not os.path.exists(path):
        os.makedirs(path)
        pass
    pass

def assureFilePathExists(file_name: str) -> None:
    file_path: str = os.path.dirname(file_name)
    assurePathExists(file_path)
    pass

class TrainRecipe:
    
    def __init__(self, train_recipe_file_name: str = DEFAULT_TRAIN_RECIPE_FILE_NAME) -> None:
        self.setTrainRecipeFileName(train_recipe_file_name)
        pass
    
    def setTrainRecipeFileName(self, train_recipe_file_name: str) -> None:
        self.train_recipe_file_name: str = train_recipe_file_name
        self.load()
        pass
    
    def getTrainRecipeFileName(self) -> str:
        return self.train_recipe_file_name
        pass
    
    def load(self) -> dict:
        with open(self.train_recipe_file_name, "r", encoding="utf-8") as file:
            self.recipe_dict: dict = json.load(file)
            pass
        self.resolve()
        pass
    
    def resolveBasicInfomaion(self) -> None:
        self.recipe_name: str = self.recipe_dict["recipe name"]
        self.classes_dict: dict = self.recipe_dict["classes"]
        self.class_number: int = len(self.classes_dict)
        pass
    
    def resolveRandomSeed(self) -> None:
        self.random_numpy_seed: int = self.recipe_dict["random seed"]["numpy"]
        self.random_torch_manual_seed: int = self.recipe_dict["random seed"]["torch manual"]
        self.random_torhc_cuda_manual_seed: int = self.recipe_dict["random seed"]["torch cuda manual"]
        self.random_seed: int = self.recipe_dict["random seed"]["random"]
        pass
    
    def resolveDataParameter(self) -> None:
        self.label_csv_file: str = self.recipe_dict["data parameter"]["label csv file"]
        self.validate_dataset_ratio: float = self.recipe_dict["data parameter"]["validate dataset ratio"]
        self.random_state: int = self.recipe_dict["data parameter"]["random state"]
        self.test_csv_file: str = self.recipe_dict["data parameter"]["test csv file"]
        self.mean: list = self.recipe_dict["data parameter"]["mean"]
        self.std: list = self.recipe_dict["data parameter"]["std"]
        self.train_transform_dict: dict = self.recipe_dict["data parameter"]["train transform"]
        self.validate_transform_dict: dict = self.recipe_dict["data parameter"]["validate transform"]
        self.test_transform_dict: dict = self.recipe_dict["data parameter"]["test transform"]
        pass
    
    def resolveTrainParameter(self) -> None:
        self.batch_size: int = self.recipe_dict["train parameter"]["batch size"]
        self.worker_number: int = self.recipe_dict["train parameter"]["worker number"]
        self.stage_epoches: list = self.recipe_dict["train parameter"]["stage epoches"]
        self.total_epoches: int = sum(self.stage_epoches)
        self.learning_rate: float = self.recipe_dict["train parameter"]["learning rate"]
        self.learning_rate_decay: float = self.recipe_dict["train parameter"]["learning rate decay"]
        self.weight_decay: float = self.recipe_dict["train parameter"]["weight decay"]
        pass
    
    def resolveInitialParameter(self) -> None:
        self.stage: int = self.recipe_dict["initial parameter"]["stage"]
        self.start_epoch: int = self.recipe_dict["initial parameter"]["start epoch"]
        self.best_precision: float = self.recipe_dict["initial parameter"]["best precision"]
        self.lowest_loss: float = self.recipe_dict["initial parameter"]["lowest loss"]
        pass
    
    def getBestPrecision(self) -> float:
        return self.best_precision
        pass
    
    def setBestPrecision(self, best_precision: float) -> None:
        self.best_precision = best_precision
        pass
    
    def getLowestLoss(self) -> float:
        return self.lowest_loss
        pass
    
    def setLowestLoss(self, lowest_loss: float) -> None:
        self.lowest_loss = lowest_loss
        pass
    
    def resolveProgressSetup(self) -> None:
        self.print_frequency: int = self.recipe_dict["progress setup"]["print frequency"]
        self.whether_evaluate: bool = self.recipe_dict["progress setup"]["whether evaluate"]
        self.whether_resume: bool = self.recipe_dict["progress setup"]["whether resume"]
        self.resume_checkpoint: str = self.recipe_dict["progress setup"]["resume checkpoint"]
        pass
    
    def resolveModelPath(self) -> None:
        self.use_time_stamp: bool = self.recipe_dict["model path"]["use time stamp"]
        pass
    
    def resolve(self) -> None:
        self.resolveBasicInfomaion()
        self.resolveRandomSeed()
        self.resolveDataParameter()
        self.resolveTrainParameter()
        self.resolveInitialParameter()
        self.resolveProgressSetup()
        self.resolveModelPath()
        self.makeWorkingDirectory()
        pass
    
    def useTimeStamp(self) -> bool:
        return self.use_time_stamp
        pass
    
    def makeWorkingDirectory(self) -> str:
        self.working_directory: str = os.path.join(DEFAULT_MODEL_PATH, self.recipe_name)
        self.time_value: float = time.time()
        self.time_stamp: str = getCurrentTimeStr()
        if self.useTimeStamp():
            self.working_directory = os.path.join(self.working_directory, self.time_stamp)
            pass
        else:
            # count the number of directories in the working directory
            # then add 1 to the number
            folder_number: int = len(os.listdir(self.working_directory))
            self.working_directory = os.path.join(self.working_directory, str(folder_number + 1))
            pass
        assurePathExists(self.working_directory)
        shutil.copy(self.train_recipe_file_name, os.path.join(self.working_directory, os.path.basename(self.train_recipe_file_name)))
        self.result_directory: str = os.path.join(self.working_directory, self.recipe_dict["model path"]["result path"])
        self.check_points_directory: str = os.path.join(self.working_directory, self.recipe_dict["model path"]["check points path"])
        self.log_directory: str = os.path.join(self.working_directory, self.recipe_dict["model path"]["log path"])
        assurePathExists(self.result_directory)
        assurePathExists(self.check_points_directory)
        assurePathExists(self.log_directory)
        self.result_file_name: str = os.path.join(self.result_directory, self.recipe_dict["model path"]["result file name"])
        self.check_point_name: str = os.path.join(self.check_points_directory, self.recipe_dict["model path"]["check point name"])
        self.best_model_name: str = os.path.join(self.check_points_directory, self.recipe_dict["model path"]["best model name"])
        self.lowest_loss_name: str = os.path.join(self.check_points_directory, self.recipe_dict["model path"]["lowest loss name"])
        self.log_file_name: str = os.path.join(self.log_directory, self.recipe_dict["model path"]["log file name"])
        self.best_accuracy_log_file_name: str = os.path.join(self.log_directory, self.recipe_dict["model path"]["best accuracy log file name"])
        pass
    
    def getResultFileName(self) -> str:
        return self.result_file_name
        pass
    
    def getCheckPointName(self) -> str:
        return self.check_point_name
        pass
    
    def getBestModelName(self) -> str:
        return self.best_model_name
        pass
    
    def getLowestLossName(self) -> str:
        return self.lowest_loss_name
        pass
    
    def getLogFileName(self) -> str:
        return self.log_file_name
        pass
    
    def getBestAccuracyLogFileName(self) -> str:
        return self.best_accuracy_log_file_name
        pass
    
    def getClassNumber(self) -> int:
        return self.class_number
        pass
    
    def getWhetherResume(self) -> bool:
        return self.whether_resume
        pass
    
    def getWheterEvaluate(self) -> bool:
        return self.whether_evaluate
        pass
    
    def getResumeCheckpoint(self) -> str:
        return self.resume_checkpoint
        pass
    
    def setStartEpoch(self, start_epoch: int) -> None:
        self.start_epoch = start_epoch
        pass
    
    def getStartEpoch(self) -> int:
        return self.start_epoch
        pass
    
    def getTotalEpoches(self) -> int:
        return self.total_epoches
        pass
    
    def setBestPrecision(self, best_precision: float) -> None:
        self.best_precision = best_precision
        pass
    
    def setLowestLoss(self, lowest_loss: float) -> None:
        self.lowest_loss = lowest_loss
        pass
    
    def setStage(self, stage: int) -> None:
        self.stage = stage
        pass
    
    def getStage(self) -> int:
        return self.stage
        pass
    
    def getStageEpoches(self) -> list:
        return self.stage_epoches
        pass
    
    def setLearningRate(self, learning_rate: float) -> None:
        self.learning_rate = learning_rate
        pass
    
    def getWorkingDirectory(self) -> str:
        return self.working_directory
        pass
    
    def getResultDirectory(self) -> str:
        return self.result_directory
        pass
    
    def getCheckPointsDirectory(self) -> str:
        return self.check_points_directory
        pass
    
    def getLogDirectory(self) -> str:
        return self.log_directory
        pass
    
    def adjustLearningRate(self) -> None:
        self.learning_rate /= self.learning_rate_decay
        pass
    
    def getLearningRate(self) -> float:
        return self.learning_rate
        pass
    
    def getWeightDecay(self) -> float:
        return self.weight_decay
        pass
    
    def getLabelCsvFile(self) -> str:
        return self.label_csv_file
        pass
    
    def getValidateDatasetRatio(self) -> float:
        return self.validate_dataset_ratio
        pass
    
    def getRandomState(self) -> int:
        return self.random_state
        pass
    
    def getTestCsvFile(self) -> str:
        return self.test_csv_file
        pass
    
    def getTrainTransformDict(self) -> dict:
        return self.train_transform_dict
        pass
    
    def getValidateTransformDict(self) -> dict:
        return self.validate_transform_dict
        pass
    
    def getTestTransformDict(self) -> dict:
        return self.test_transform_dict
        pass
    
    def getMean(self) -> list:
        return self.mean
        pass
    
    def getStd(self) -> list:
        return self.std
        pass
    
    def getRandomNumpySeed(self) -> int:
        return self.random_numpy_seed
        pass
    
    def getRandomTorchManualSeed(self) -> int:
        return self.random_torch_manual_seed
        pass
    
    def getRandomTorchCudaManualSeed(self) -> int:
        return self.random_torhc_cuda_manual_seed
        pass
    
    def getRandomSeed(self) -> int:
        return self.random_seed
        pass
    
    def getRecipeName(self) -> str:
        return self.recipe_name
        pass
    
    def getTimeStamp(self) -> str:
        return self.time_stamp
        pass
    
    def getPrintFrequency(self) -> int:
        return self.print_frequency
        pass
    
    def getBatchSize(self) -> int:
        return self.batch_size
        pass
    
    def getWorkerNumber(self) -> int:
        return self.worker_number
        pass
    
    def getStage(self) -> int:
        return self.stage
        pass
    
    def setStage(self, stage: int) -> None:
        self.stage = stage
        pass
    
    pass