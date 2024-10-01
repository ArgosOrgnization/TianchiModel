'''
 # @ author: bella | bob
 # @ date: 2024-10-01 16:01:11
 # @ license: MIT
 # @ description:
 '''

import os
import json

class ModelLoader:

    def __init__(self, model_path: str) -> None:
        self.model_path: str = model_path
        self.checkModelPath()
        self.load()
        pass

    def __checkJson(self) -> bool:
        # search whether a json file exists in the model_path
        self.json_file_list: list = os.listdir(self.model_path)
        self.json_file_list = [file for file in self.json_file_list if file.endswith(".json")]
        return len(self.json_file_list) > 0
        pass

    def __checkModel(self) -> bool:
        self.model_check_point_path: str = os.path.join(self.model_path, "check_points")
        self.model_file_list: list = os.listdir(self.model_check_point_path)
        self.model_file_list = [os.path.join(self.model_check_point_path, file) for file in self.model_file_list if file.endswith(".pth.tar")]
        return len(self.model_file_list) == 3
        pass

    def __check(self) -> bool:
        return self.__checkJson() and self.__checkModel()
        pass

    def checkModelPath(self) -> None:
        if self.__check() == True:
            print("========== ModelLoader: Model path is valid. ==========")
            pass
        else:
            print("========== ModelLoader: Model path is invalid. ==========")
            pass
        pass

    def load(self) -> None:
        self.json_file_name: str = self.json_file_list[0]
        with open(os.path.join(self.model_path, self.json_file_name), "r", encoding="utf-8") as json_file:
            self.config_dict: dict = json.load(json_file)
            pass
        self.model_file_name: str = os.path.join(self.model_check_point_path, self.config_dict["model path"]["best model name"])
        self.classes_dict: dict = self.config_dict["classes"]
        self.class_number: int = len(self.classes_dict)
        keys, values = list(self.classes_dict.keys()), list(self.classes_dict.values())
        self.reverse_classes_dict: dict = dict(zip(values, keys))
        self.trasnform_resize: list = self.config_dict["data parameter"]["test transform"]["resize"]
        self.transform_center_crop: int = self.config_dict["data parameter"]["test transform"]["center crop"]
        self.transform_mean: list = self.config_dict["data parameter"]["mean"]
        self.transform_std: list = self.config_dict["data parameter"]["std"]
        pass

    def getClassNumber(self) -> int:
        return self.class_number
        pass

    def getModelFileName(self) -> str:
        return self.model_file_name
        pass

    def getTransformResize(self) -> list:
        return self.trasnform_resize
        pass

    def getTransformCenterCrop(self) -> int:
        return self.transform_center_crop
        pass

    def getTransformMean(self) -> list:
        return self.transform_mean
        pass

    def getTransformStd(self) -> list:
        return self.transform_std
        pass

    def getClassesDict(self) -> dict:
        return self.classes_dict
        pass

    def getClassName(self, class_index: int) -> str:
        return self.reverse_classes_dict[class_index]
        pass

    def getClassNames(self, class_index_list: list) -> list:
        return [self.reverse_classes_dict[index] for index in class_index_list]
        pass

    pass