'''
 # @ author: bella | bob
 # @ date: 2024-10-02 00:39:49
 # @ license: MIT
 # @ description:
 '''

import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "."))
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from PySide6 import QtCore
from PySide6 import QtWidgets

from qt_material import apply_stylesheet

from ModelChoiceDialog import ModelChoiceDialog
from MainWindow import MainWindow

os.environ["QT_API"] = "pyside6"

def main() -> None:
    # below line seem aborted in PySide6
    # QtCore.QCoreApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling) # support high dpi scaling
    app: QtWidgets.QApplication = QtWidgets.QApplication(sys.argv)
    apply_stylesheet(app, theme="dark_teal.xml")
    model_choice_dialog: ModelChoiceDialog = ModelChoiceDialog()
    model_choice_dialog.show()
    if model_choice_dialog.exec() == QtWidgets.QDialog.Accepted:
        model_runner_device: str = model_choice_dialog.getModelRunnerDevice()
        model_path: str = model_choice_dialog.getModelPath()
        try:
            main_window: MainWindow = MainWindow(model_runner_device, model_path)
            main_window.show()
            pass
        except Exception as e:
            print(f"error: {e}")
            print(f"model path: {model_path} is invalid")
            warning_nessage_box: QtWidgets.QMessageBox = QtWidgets.QMessageBox()
            warning_nessage_box.setWindowTitle("警告")
            warning_nessage_box.setText("模型路径无效或遇错误, 请重新选择或者检查路径")
            warning_nessage_box.setStyleSheet("font-size: 16px; font-weight: bold; font-family: Microsoft YaHei;")
            warning_nessage_box.show()
            if warning_nessage_box.exec() == QtWidgets.QMessageBox.Ok:
                model_choice_dialog.show()
                pass
            pass
        pass
    else:
        sys.exit(0)
        pass
    sys.exit(app.exec())
    pass

'''
url: https://qt-material.readthedocs.io/en/latest/index.html

theme list:

['dark_amber.xml',
 'dark_blue.xml',
 'dark_cyan.xml',
 'dark_lightgreen.xml',
 'dark_pink.xml',
 'dark_purple.xml',
 'dark_red.xml',
 'dark_teal.xml',
 'dark_yellow.xml',
 'light_amber.xml',
 'light_blue.xml',
 'light_cyan.xml',
 'light_cyan_500.xml',
 'light_lightgreen.xml',
 'light_pink.xml',
 'light_purple.xml',
 'light_red.xml',
 'light_teal.xml',
 'light_yellow.xml']
'''