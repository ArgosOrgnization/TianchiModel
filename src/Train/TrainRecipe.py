'''
 # @ author: bella | bob
 # @ date: 2024-09-24 17:55:48
 # @ license: MIT
 # @ description:
 '''

import os
import time
import json

DEFAULT_TRAIN_RECIPE_FILE_NAME: str = "model/recipe/demo.json"
DEFAULT_MODEL_PATH: str = "model"

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

class TrainRecipe:
    
    def __init__(self, train_recipe_file_name: str = DEFAULT_TRAIN_RECIPE_FILE_NAME) -> None:
        self.setTrainRecipeFileName(train_recipe_file_name)
        pass
    
    def setTrainRecipeFileName(self, train_recipe_file_name: str) -> None:
        self.train_recipe_file_name: str = train_recipe_file_name
        self.load()
        pass
    
    def load(self) -> dict:
        with open(self.train_recipe_file_name, "r") as file:
            self.recipe_dict: dict = json.load(file)
            pass
        self.resolve()
        pass
    
    def resolveBasicInfomaion(self) -> None:
        self.recipe_name: str = self.recipe_dict["recipe name"]
        self.class_number: int = self.recipe_dict["basic information"]["class number"]
        pass
    
    def resolveRandomSeed(self) -> None:
        self.random_numpy_seed: int = self.recipe_dict["random seed"]["numpy"]
        self.random_torch_manual_seed: int = self.recipe_dict["random seed"]["torch manual"]
        self.random_torhc_cuda_manual_seed: int = self.recipe_dict["random seed"]["torch cuda manual"]
        self.random_seed: int = self.recipe_dict["random seed"]["random"]
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
        self.stage: int = 0
        self.start_epoch: int = 0
        self.best_precision: float = 0
        self.lowest_loss: float = 0
        pass
    
    def resolveProgressSetup(self) -> None:
        self.print_frequency: int = self.recipe_dict["progress setup"]["print frequency"]
        self.validate_dataset_ratio: float = self.recipe_dict["progress setup"]["validate dataset ratio"]
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
        self.resolveTrainParameter()
        self.resolveInitialParameter()
        self.resolveProgressSetup()
        self.resolveModelPath()
        self.makeWorkingDirectory()
        pass
    
    def useTimeStampe(self) -> bool:
        return self.use_time_stamp
        pass
    
    def makeWorkingDirectory(self) -> str:
        self.working_directory: str = os.path.join(DEFAULT_MODEL_PATH, self.recipe_name)
        self.time_value: float = time.time()
        self.time_stamp: str = getCurrentTimeStr()
        if self.useTimeStampe():
            self.working_directory = os.path.join(self.working_directory, self.time_stamp)
            pass
        else:
            # count the number of directories in the working directory
            # then add 1 to the number
            folder_number: int = len(os.listdir(self.working_directory))
            self.working_directory = os.path.join(self.working_directory, str(folder_number + 1))
            pass
        assurePathExists(self.working_directory)
        self.result_directory: str = os.path.join(self.working_directory, self.recipe_dict["model path"]["result path"])
        self.check_points_directory: str = os.path.join(self.working_directory, self.recipe_dict["model path"]["check points path"])
        assurePathExists(self.result_directory)
        assurePathExists(self.check_points_directory)
        pass
    
    def getClassNumber(self) -> int:
        return self.class_number
        pass
    
    def getWhetherResume(self) -> bool:
        return self.whether_resume
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
    
    def adjustLearningRate(self) -> None:
        self.learning_rate /= self.learning_rate_decay
        pass
    
    def getLearningRate(self) -> float:
        return self.learning_rate
        pass
    
    def getWeightDecay(self) -> float:
        return self.weight_decay
        pass
    
    pass