'''
 # @ author: bella | bob
 # @ date: 2024-09-18 21:17:39
 # @ license: MIT
 # @ description:
 '''

import random
from PIL import Image

class ImageRotator:
    
    def __init__(self, angles: list) -> None:
        self.angles: list = list(angles)
        self.angles_number: int = len(angles)
        pass
    
    def __call__(self, image: Image) -> Image:
        degrees: float = random.choice(self.angles)
        return self.rotate(image, degrees)
        pass
    
    def rotate(image: Image, degrees: float) -> Image:
        return image.rotate(degrees)
        pass
    
    pass