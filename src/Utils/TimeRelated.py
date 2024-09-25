'''
 # @ author: bella | bob
 # @ date: 2024-09-25 22:08:34
 # @ license: MIT
 # @ description:
 '''

import time

def getCurrentTimeStr() -> str:
    return time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())
    pass