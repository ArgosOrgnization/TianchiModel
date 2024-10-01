'''
 # @ author: bella | bob
 # @ date: 2024-10-02 00:40:57
 # @ license: MIT
 # @ description:
 '''

import os

from PySide6 import QtCore
from PySide6.QtWidgets import QDialog
from PySide6.QtWidgets import QFileDialog
from PySide6.QtWidgets import QVBoxLayout, QHBoxLayout
from PySide6.QtWidgets import QPushButton, QLabel
from PySide6 import QtGui

class ModelChoiceDialog(QDialog):

    def __init__(self) -> None:
        super().__init__()
        self.setStyleSheet(
            """
            background-image: url('resources/images/login_background.png');
            """
        )
        self.initialize()
        pass

    def __initializeBasic(self) -> None:
        # window title
        self.setWindowTitle("训练模型选择窗")
        # window style
        self.setGeometry(400, 200, 600, 600)
        # icon
        self.setWindowIcon(QtGui.QIcon(os.path.join(os.path.dirname(__file__), "../../resources/images/logo.png")))
        self.setWindowOpacity(0.85)
        pass

    def __initializeLayout(self) -> None:
        self.main_layout: QVBoxLayout = QVBoxLayout()
        self.main_layout.setAlignment(QtCore.Qt.AlignCenter)
        self.main_layout.setSpacing(10)
        self.setLayout(self.main_layout)
        self.top_vertical_layout: QVBoxLayout = QVBoxLayout()
        self.bottom_horizontal_layout: QHBoxLayout = QHBoxLayout()
        self.main_layout.addLayout(self.top_vertical_layout)
        self.main_layout.addLayout(self.bottom_horizontal_layout)
        pass

    def __initializeTopLayout(self) -> None:
        # add some space
        self.top_vertical_layout.addSpacing(20)
        self.model_choice_button: QPushButton = QPushButton()
        self.model_choice_button.setText("选择模型路径")
        self.top_vertical_layout.addWidget(self.model_choice_button)
        self.model_choice_button.setStyleSheet("font-size: 20px; font-weight: bold; font-family: Microsoft YaHei;")
        # add some space
        self.top_vertical_layout.addSpacing(40)
        self.model_path_label: QLabel = QLabel()
        self.model_path_label.setText("模型路径: 保证存有 .json 配置文件和 check_points 文件夹")
        self.top_vertical_layout.addWidget(self.model_path_label)
        self.model_path_label.setStyleSheet("color: white; font-size: 15px; font-family: Microsoft YaHei; font-weight: bold;")
        pass

    def initialize(self) -> None:
        self.__initializeBasic()
        self.__initializeLayout()
        self.__initializeTopLayout()
        pass

    pass