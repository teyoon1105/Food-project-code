from PyQt5 import QtCore, QtWidgets

class HomeWindow(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setup_ui()

    def setup_ui(self):
        self.setObjectName("homeWindow")
        self.resize(1200, 700)

        self.team_logo = QtWidgets.QLabel("Team Logo", self)
        self.team_logo.setGeometry(QtCore.QRect(260, 60, 880, 640))
        self.team_logo.setAlignment(QtCore.Qt.AlignCenter)

        self.btn_start = QtWidgets.QPushButton("Start", self)
        self.btn_start.setGeometry(QtCore.QRect(30, 50, 200, 100))

        self.btn_settings = QtWidgets.QPushButton("Settings", self)
        self.btn_settings.setGeometry(QtCore.QRect(30, 180, 200, 100))