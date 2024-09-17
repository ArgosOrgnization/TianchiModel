'''
 # @ author: cyq | bcy
 # @ date: 2024-09-14 00:34:03
 # @ license: MIT
 # @ description:
 '''

# model_zoo is a module to load pre-trained models from the internet
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
# for certificate verification, we need to import ssl
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

from DetectModel import DetectModel

IMAGENET_URL = "http://data.lip6.fr/cadene/pretrainedmodels/inceptionv4-8e4777a0.pth"
IMAGENET_BACKGROUND_URL = "http://data.lip6.fr/cadene/pretrainedmodels/inceptionv4-8e4777a0.pth"

PRETRAINED_MODEL_SETTINGS = {
    "shared settings": {
        "input space": "RGB",
        "input size": (3, 299, 299),
        "input range": [0, 1],
        "mean": [0.5, 0.5, 0.5],
        "std": [0.5, 0.5, 0.5],
    },
    "imagenet": {
        "url": IMAGENET_URL,
        "class number": 1000,
    },
    "imagenet_background": {
        "url": IMAGENET_BACKGROUND_URL,
        "class number": 1001,
    },
}

def compareModelStateDict(model: DetectModel, state_dict: dict) -> bool:
    model_state_dict: dict = model.state_dict()
    for key, value in state_dict.items():
        if key not in model_state_dict:
            continue
            pass
        elif isinstance(value, nn.Parameter):
            value = value.data
            pass
        model_state_dict[key].copy_(value)
        pass
    pass

def loadPretraindModel(class_number: int, pretrained_tag: str = "imagenet") -> DetectModel:
    if pretrained_tag == None:
        model: DetectModel = DetectModel(class_number)
        return model
    elif pretrained_tag in PRETRAINED_MODEL_SETTINGS.keys():
        model: DetectModel = DetectModel(PRETRAINED_MODEL_SETTINGS[pretrained_tag]["class number"])
        state_dict: dict = model_zoo.load_url(PRETRAINED_MODEL_SETTINGS[pretrained_tag]["url"])
        compareModelStateDict(model, state_dict)
        model.setInputSpace(PRETRAINED_MODEL_SETTINGS["shared settings"]["input space"])
        model.setInputSize(PRETRAINED_MODEL_SETTINGS["shared settings"]["input size"])
        model.setInputRange(PRETRAINED_MODEL_SETTINGS["shared settings"]["input range"])
        model.setMean(PRETRAINED_MODEL_SETTINGS["shared settings"]["mean"])
        model.setStd(PRETRAINED_MODEL_SETTINGS["shared settings"]["std"])
        return model
        pass
    pass