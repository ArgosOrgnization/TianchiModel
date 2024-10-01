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

os.environ["QT_API"] = "pyside6"

def main() -> None:
    QtCore.QCoreApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling) # support high dpi scaling
    app: QtWidgets.QApplication = QtWidgets.QApplication(sys.argv)
    apply_stylesheet(app, theme="light_teal.xml")
    model_choice_dialog: ModelChoiceDialog = ModelChoiceDialog()
    model_choice_dialog.show()
    sys.exit(app.exec())
    pass

main()

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