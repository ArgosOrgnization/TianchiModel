'''
 # @ author: bella | bob
 # @ date: 2024-10-02 23:59:58
 # @ license: MIT
 # @ description:
 '''

import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "."))
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from PySide6 import QtGui
from PySide6.QtWidgets import QWidget
from PySide6.QtWidgets import QMainWindow
from PySide6.QtWidgets import QTabWidget
from PySide6.QtWidgets import QVBoxLayout

from Run.ModelRunner import ModelRunner
from SingleProcessLayout import SingleProcessLayout

class MainWindow(QMainWindow):

    def __init__(self, model_runner_device: str, model_path: str) -> None:
        super().__init__()
        self.model_runner: ModelRunner = ModelRunner(model_runner_device, model_path)
        self.initialize()
        pass

    def __initializeBasic(self) -> None:
        self.setWindowTitle("HorusEye - 模型运行器")
        self.setGeometry(100, 100, 1500, 1000)
        self.setWindowIcon(QtGui.QIcon(os.path.join(os.path.dirname(__file__), "../../resources/images/logo.png")))
        self.setWindowOpacity(0.95)
        pass

    def __makeTabWidget(self) -> None:
        self.tab_widget: QTabWidget = QTabWidget()
        self.tab_widget.setStyleSheet("font-size: 16px; font-weight: bold; font-family: Microsoft YaHei;")
        self.tab_widget.setTabPosition(QTabWidget.TabPosition.North)
        self.tab_widget.setTabShape(QTabWidget.Triangular)
        self.tab_widget.setDocumentMode(True)
        self.tab_widget.setMovable(True)
        self.central_layout.addWidget(self.tab_widget)
        self.tab_name_list: list = ["单张图片检测",]
        self.tab_name_index_dict: dict = {tab_name: index for index, tab_name in enumerate(self.tab_name_list)}
        self.tab_index_name_dict: dict = {index: tab_name for index, tab_name in enumerate(self.tab_name_list)}
        self.tabs_list: list = []
        for tab_name in self.tab_name_list:
            self.tabs_list.append(QWidget())
            self.tab_widget.addTab(self.tabs_list[-1], tab_name)
            pass
        self.single_process_layout: SingleProcessLayout = SingleProcessLayout(self.model_runner)
        self.tabs_list[0].setLayout(self.single_process_layout)
        self.setCentralWidget(self.tab_widget)
        pass

    def initialize(self) -> None:
        self.central_widget: QWidget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.central_layout: QVBoxLayout = QVBoxLayout()
        self.central_widget.setLayout(self.central_layout)
        self.__initializeBasic()
        self.__makeTabWidget()
        pass

    pass