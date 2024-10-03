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
from PySide6.QtWidgets import QPushButton
from PySide6.QtWidgets import QLineEdit
from PySide6.QtWidgets import QComboBox
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
        self.setLayout(self.main_layout)
        self.top_vertical_layout: QVBoxLayout = QVBoxLayout()
        self.bottom_horizontal_layout: QHBoxLayout = QHBoxLayout()
        self.main_layout.addLayout(self.top_vertical_layout)
        self.main_layout.addSpacing(50)
        self.main_layout.addLayout(self.bottom_horizontal_layout)
        pass

    def __initializeTopLayout(self) -> None:
        self.model_choice_button: QPushButton = QPushButton()
        self.model_choice_button.setText("选择模型路径")
        self.top_vertical_layout.addWidget(self.model_choice_button)
        self.model_choice_button.setStyleSheet("font-size: 20px; font-weight: bold; font-family: Microsoft YaHei;")
        # add some space
        self.top_vertical_layout.addSpacing(80)
        self.model_path_line_edit: QLineEdit = QLineEdit()
        self.model_path_line_edit.setText("模型路径: 保证存有 .json 配置文件和 check_points 文件夹")
        self.model_path_line_edit.setPlaceholderText("模型路径: 保证存有 .json 配置文件和 check_points 文件夹")
        self.top_vertical_layout.addWidget(self.model_path_line_edit)
        self.model_path_line_edit.setStyleSheet("font-size: 16px; font-weight: bold; font-family: Microsoft YaHei;")
        # add some space
        self.top_vertical_layout.addSpacing(50)
        self.model_runner_device_combobox: QComboBox = QComboBox()
        self.model_runner_device_combobox.addItem("cpu")
        self.model_runner_device_combobox.addItem("cuda")
        self.model_runner_device_combobox.setCurrentIndex(0)
        self.top_vertical_layout.addWidget(self.model_runner_device_combobox)
        self.model_runner_device_combobox.setStyleSheet("font-size: 16px; font-weight: bold; font-family: Microsoft YaHei;")
        pass

    def __initializeBottomLayout(self) -> None:
        # 确认
        self.confirm_button: QPushButton = QPushButton()
        self.confirm_button.setText("确认")
        self.bottom_horizontal_layout.addWidget(self.confirm_button)
        self.confirm_button.setStyleSheet("font-size: 20px; font-weight: bold; font-family: Microsoft YaHei;")
        # 取消
        self.cancel_button: QPushButton = QPushButton()
        self.cancel_button.setText("取消")
        self.bottom_horizontal_layout.addWidget(self.cancel_button)
        self.cancel_button.setStyleSheet("font-size: 20px; font-weight: bold; font-family: Microsoft YaHei;")
        pass

    def __onModelChoiceButtonClicked(self) -> None:
        # choose a model path, which is a directory
        model_path: str = QFileDialog().getExistingDirectory(
            self,
            "选择模型路径",
            os.path.join(os.path.dirname(__file__), "../../resources/models")
        )
        self.model_path_line_edit.setText(model_path)
        pass

    def __linkEvents(self) -> None:
        self.model_choice_button.clicked.connect(self.__onModelChoiceButtonClicked)
        self.confirm_button.clicked.connect(self.accept)
        self.cancel_button.clicked.connect(self.reject)
        pass

    def initialize(self) -> None:
        self.__initializeBasic()
        self.__initializeLayout()
        self.__initializeTopLayout()
        self.__initializeBottomLayout()
        self.__linkEvents()
        pass

    def getModelPath(self) -> str:
        return self.model_path_line_edit.text()
        pass

    def getModelRunnerDevice(self) -> str:
        return self.model_runner_device_combobox.currentText()
        pass

    pass