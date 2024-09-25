'''
 # @ author: bella | bob
 # @ date: 2024-09-24 13:55:19
 # @ license: MIT
 # @ description:
 '''

import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from Train.TrainManager import TrainManager

def main() -> None:
    train_recipe_file_name: str = input("please enter the train recipe file name: ")
    train_manager: TrainManager = TrainManager(train_recipe_file_name)
    train_manager.start()
    pass

if __name__ == "__main__":
    main()
    pass