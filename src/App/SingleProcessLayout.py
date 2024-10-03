'''
 # @ author: bella | bob
 # @ date: 2024-10-03 00:53:04
 # @ license: MIT
 # @ description:
 '''

import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "."))
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import torch

from PySide6 import QtCore
from PySide6.QtWidgets import QHBoxLayout, QVBoxLayout
from PySide6.QtWidgets import QLabel
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import QPushButton
from PySide6.QtWidgets import QFileDialog
from PySide6.QtWidgets import QTableWidget
from PySide6.QtWidgets import QTableWidgetItem

from Run.ModelRunner import ModelRunner

class SingleProcessLayout(QHBoxLayout):

    # a pircture shown left side with a chosen box up
    # right is a table shown the ratio of detection
    # and a label shown the result of detection

    def __init__(self, model_runner: ModelRunner) -> None:
        super().__init__()
        self.model_runner: ModelRunner = model_runner
        self.picture_file_name: str = os.path.join(os.path.dirname(__file__), "../../resources/images/show.png")
        self.initialize()
        pass

    def getPicturFileName(self) -> str:
        return self.picture_file_name
        pass

    def setPictureFileName(self, picture_file_name: str) -> None:
        self.picture_file_name = picture_file_name
        pass

    def getPictureFileNameBase(self) -> str:
        return os.path.basename(self.picture_file_name)
        pass

    def __detect(self) -> None:
        result_tensor: torch.Tensor = self.model_runner.singleCallByImagePath(self.getPicturFileName())
        class_index: int = self.model_runner.singlePossibleClassIndex(result_tensor)
        class_name: str = self.model_runner.model_loader.getClassName(class_index)
        result_list = list(result_tensor.detach().numpy())
        for index, result in enumerate(result_list):
            self.result_table_widget.setItem(index, 1, QTableWidgetItem(str(result)))
            pass
        self.result_label.setText(f"检测结果: {class_name}")
        pass

    def __onPictureChosenButtonClicked(self) -> None:
        file_dialog: QFileDialog = QFileDialog()
        file_dialog.setFileMode(QFileDialog.ExistingFile)
        file_dialog.setNameFilter("Images (*.png *.xpm *.jpg)")
        file_dialog.setViewMode(QFileDialog.Detail)
        file_dialog.setDirectory(os.path.join(os.path.dirname(__file__), "../../"))
        if file_dialog.exec():
            file_path: str = file_dialog.selectedFiles()[0]
            self.setPictureFileName(file_path)
            self.picture_label.setPixmap(QPixmap(file_path))
            self.picture_chosen_button.setText(self.getPictureFileNameBase())
            self.__detect()
            pass
        pass

    def __makeLeftSide(self) -> None:
        self.left_verticle_layout: QVBoxLayout = QVBoxLayout()
        self.addLayout(self.left_verticle_layout)
        # picture chosen button
        self.picture_chosen_button: QPushButton = QPushButton("选择图片")
        self.picture_chosen_button.setStyleSheet("font-size: 16px; font-weight: bold; font-family: Microsoft YaHei;")
        self.picture_chosen_button.setMaximumWidth(800)
        self.picture_chosen_button.setMaximumHeight(100)
        self.left_verticle_layout.addWidget(self.picture_chosen_button)
        self.left_verticle_layout.setAlignment(QtCore.Qt.AlignCenter)
        # picture shown
        self.picture_label: QLabel = QLabel()
        self.picture_label.setPixmap(QPixmap(self.getPicturFileName()))
        self.picture_label.setMaximumWidth(800)
        self.picture_label.setMaximumHeight(600)
        # center alignment
        self.picture_label.setAlignment(QtCore.Qt.AlignCenter)
        self.left_verticle_layout.addWidget(self.picture_label)
        pass

    def __makeRightSide(self) -> None:
        self.right_verticle_layout: QVBoxLayout = QVBoxLayout()
        self.addLayout(self.right_verticle_layout)
        # label
        self.result_label: QLabel = QLabel("检测结果")
        self.result_label.setStyleSheet("font-size: 16px; font-weight: bold; font-family: Microsoft YaHei;")
        self.result_label.setMaximumWidth(800)
        self.result_label.setMaximumHeight(100)
        self.right_verticle_layout.addWidget(self.result_label)
        self.result_label.setAlignment(QtCore.Qt.AlignCenter)
        # table
        self.result_table_widget: QTableWidget = QTableWidget()
        self.result_table_widget.setRowCount(self.model_runner.model_loader.getClassNumber())
        self.result_table_widget.setColumnCount(2)
        self.result_table_widget.setHorizontalHeaderLabels(["检测类别名", "检测识别概率"])
        self.result_table_widget.setStyleSheet("font-size: 16px; font-weight: bold; font-family: Microsoft YaHei;")
        self.result_table_widget.setColumnWidth(0, 300)
        self.result_table_widget.setColumnWidth(1, 300)
        self.result_table_widget.setRowHeight(0, 50)
        self.result_table_widget.setRowHeight(1, 50)
        for index in range(self.model_runner.model_loader.getClassNumber()):
            self.result_table_widget.setItem(index, 0, QTableWidgetItem(self.model_runner.model_loader.getClassName(index)))
            self.result_table_widget.setItem(index, 1, QTableWidgetItem("0.0"))
            pass
        self.right_verticle_layout.addWidget(self.result_table_widget)
        pass

    def __linkEvents(self) -> None:
        self.picture_chosen_button.clicked.connect(self.__onPictureChosenButtonClicked)
        pass

    def initialize(self) -> None:
        self.__makeLeftSide()
        self.__makeRightSide()
        self.__linkEvents()
        pass

    pass