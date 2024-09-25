'''
 # @ author: bella | bob
 # @ date: 2024-09-18 20:50:59
 # @ license: MIT
 # @ description:
 '''
 
import pandas as pd
from torch.utils.data import Dataset
from PIL import Image

from LoadImage import loadImageAsRGB

class BaseDataset(Dataset):
    
    def __init__(
        self,
        label_dataframe: pd.DataFrame,
        transform: object = None,
        target_transform: object = None,
        load_function: callable = loadImageAsRGB,
        image_path_tag: str = "image_path",
        label_tag: str = "label",
    ) -> None:
        self.image_path_tag: str = image_path_tag
        self.label_tag: str = label_tag
        self.label_dataframe: pd.DataFrame = label_dataframe
        self.transform: object = transform
        self.target_transform: object = target_transform
        self.load_function: callable = load_function
        pass
    
    def __getitem__(self, index: int) -> tuple:
        image_file_name, label = self.label_dataframe.iloc[index]
        image: Image = self.load_function(image_file_name)
        if self.transform is not None:
            image = self.transform(image)
            pass
        return image, label
        pass
    
    def __len__(self) -> int:
        return len(self.label_dataframe)
        pass
    
    pass

class TrainDataset(BaseDataset):
    
    def __init__(
        self,
        label_dataframe: pd.DataFrame,
        transform: object = None,
        target_transform: object = None,
        load_function: callable = loadImageAsRGB,
        image_path_tag: str = "image_path",
        label_tag: str = "label",
    ) -> None:
        super(TrainDataset, self).__init__(
            label_dataframe=label_dataframe,
            transform=transform,
            target_transform=target_transform,
            load_function=load_function,
            image_path_tag=image_path_tag,
            label_tag=label_tag,
        )
        pass
    
    pass

class ValidateDataset(BaseDataset):
    
    def __init__(
        self,
        label_dataframe: pd.DataFrame,
        transform: object = None,
        target_transform: object = None,
        load_function: callable = loadImageAsRGB,
        image_path_tag: str = "image_path",
        label_tag: str = "label",
    ) -> None:
        super(ValidateDataset, self).__init__(
            label_dataframe=label_dataframe,
            transform=transform,
            target_transform=target_transform,
            load_function=load_function,
            image_path_tag=image_path_tag,
            label_tag=label_tag,
        )
        pass
    
    pass

class TestDataset(BaseDataset):
    
    def __init__(
        self,
        label_dataframe: pd.DataFrame,
        transform: object = None,
        target_transform: object = None,
        load_function: callable = loadImageAsRGB,
        image_path_tag: str = "image_path",
        label_tag: str = "label",
    ) -> None:
        super(TestDataset, self).__init__(
            label_dataframe=label_dataframe,
            transform=transform,
            target_transform=target_transform,
            load_function=load_function,
            image_path_tag=image_path_tag,
            label_tag=label_tag,
        )
        pass
    
    def __getitem__(self, index: int) -> tuple:
        image_file_name = self.label_dataframe.iloc[index, 0]
        image: Image = self.load_function(image_file_name)
        if self.transform is not None:
            image = self.transform(image)
            pass
        return image, image_file_name
        pass
    
    pass