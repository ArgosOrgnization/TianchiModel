'''
 # @ author: cyq | bcy
 # @ date: 2024-09-17 16:44:11
 # @ license: MIT
 # @ description: __init__.py file
 '''

import os
import sys

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(CURRENT_DIR, "."))
sys.path.append(os.path.join(CURRENT_DIR, ".."))