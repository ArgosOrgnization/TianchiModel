'''
 # @ author: bella | bob
 # @ date: 2024-09-24 13:55:19
 # @ license: MIT
 # @ description:
 '''

import os
import sys
os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1"
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from Train.TrainManager import TrainManager