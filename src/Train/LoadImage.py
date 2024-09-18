'''
 # @ author: bella | bob
 # @ date: 2024-09-18 20:45:53
 # @ license: MIT
 # @ description:
 '''

from PIL import Image

def loadImageAsRGB(image_path: str) -> Image:
    return Image.open(image_path).convert("RGB")
    pass