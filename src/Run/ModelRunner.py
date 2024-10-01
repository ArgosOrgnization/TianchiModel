'''
 # @ author: bella | bob
 # @ date: 2024-10-01 02:55:21
 # @ license: MIT
 # @ description:
 '''

import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "."))
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from PIL import Image
import torch
import torchvision.transforms as transforms

from Model.DetectModel import DetectModel
from ModelLoader import ModelLoader
from Train.LoadImage import loadImageAsRGB

class ModelRunner:
    
    def __init__(
        self,
        device: str,
        model_path: str,
        load_function: callable = loadImageAsRGB
    ) -> None:
        self.device: str = device
        self.model_loader: ModelLoader = ModelLoader(model_path)
        self.model: DetectModel = DetectModel(self.model_loader.getClassNumber())
        self.model = torch.nn.DataParallel(self.model)
        self.loadModel()
        self.model.to(torch.device(self.device))
        self.model.eval()
        self.defineTransform()
        self.load_function: callable = load_function
        self.softmax: torch.nn.Softmax = torch.nn.Softmax(dim=1)
        pass

    def loadModel(self) -> None:
        self.model_file_name: str = self.model_loader.getModelFileName()
        checkpoint = torch.load(
            self.model_file_name,
            map_location=torch.device(self.device),
            weights_only=True
        )
        self.model.load_state_dict(checkpoint["state_dict"])
        pass

    def defineTransform(self) -> transforms.Compose:
        self.transform = transforms.Compose([
            transforms.Resize(self.model_loader.getTransformResize()),
            transforms.CenterCrop(self.model_loader.getTransformCenterCrop()),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=self.model_loader.getTransformMean(),
                std=self.model_loader.getTransformStd()
            )
        ])
        pass

    def singleCall(self, image_data: torch.Tensor) -> torch.Tensor:
        # add a batch dimension to the first axis of the image_data
        image_data = image_data.unsqueeze(0)
        return self.batchCall(image_data)[0]
        pass

    def batchCall(self, image_data: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            y: torch.Tensor =  self.model(image_data)
            pass
        return self.softmax(y)
        pass

    def singleCallByImage(self, image: Image.Image) -> torch.Tensor:
        image_data: torch.Tensor = self.transform(image)
        return self.singleCall(image_data)
        pass

    def batchCallByImage(self, image_list: list) -> torch.Tensor:
        image_data_list: list = [self.transform(image) for image in image_list]
        image_data: torch.Tensor = torch.stack(image_data_list)
        return self.batchCall(image_data)
        pass

    def singleCallByImagePath(self, image_path: str) -> torch.Tensor:
        image: Image.Image = self.load_function(image_path)
        return self.singleCallByImage(image)
        pass

    def batchCallByImagePath(self, image_path_list: list) -> torch.Tensor:
        image_list: list = [self.load_function(image_path) for image_path in image_path_list]
        return self.batchCallByImage(image_list)
        pass

    def singlePossibleClassIndex(self, output: torch.Tensor) -> int:
        return torch.argmax(output).item()
        pass

    def batchPossibleClassIndex(self, output: torch.Tensor) -> list:
        return torch.argmax(output).tolist()
        pass

    def singlePossibleClassName(self, index: int) -> str:
        return self.model_loader.getClassName(index)
        pass

    def batchPossibleClassName(self, index_list: list) -> list:
        return self.model_loader.getClassNames(index_list)
        pass

    pass

# if __name__ == "__main__":
#     model_device: str = "cuda" if torch.cuda.is_available() else "cpu"
#     model_path: str = "model/demo/2024_09_26_08_08_32/"
#     model_runner: ModelRunner = ModelRunner(model_device, model_path)
#     print(model_runner.batchCall(torch.randn(1, 3, 299, 299)))
#     print(model_runner.singleCall(torch.randn(3, 299, 299)))
#     print(model_runner.singlePossibleClassName(model_runner.singlePossibleClassIndex(model_runner.singleCallByImagePath("drafts/demo.png"))))
#     pass