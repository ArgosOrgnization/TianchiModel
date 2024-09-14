'''
 # @ author: cyq | bcy
 # @ date: 2024-09-14 00:34:03
 # @ license: MIT
 # @ description:
 '''

# model_zoo is a module to load pre-trained models from the internet
import torch.utils.model_zoo as model_zoo
# for certificate verification, we need to import ssl
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

IMAGENET_URL = "http://data.lip6.fr/cadene/pretrainedmodels/inceptionv4-8e4777a0.pth"
IMAGENET_BACKGROUND_URL = "http://data.lip6.fr/cadene/pretrainedmodels/inceptionv4-8e4777a0.pth"