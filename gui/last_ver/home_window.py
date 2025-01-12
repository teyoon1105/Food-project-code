from PyQt5 import QtCore, QtWidgets

class HomeWindow(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setup_ui()

    def setup_ui(self):
        self.setObjectName("homeWindow")
        self.resize(1200, 700)

        self.team_logo = QtWidgets.QLabel("MEALBOM", self)
        self.team_logo.setGeometry(QtCore.QRect(200, 30, 880, 640))
        self.team_logo.setAlignment(QtCore.Qt.AlignCenter)
        self.team_logo.setStyleSheet("font-size: 60pt; font-weight: bold;")

        self.btn_start = QtWidgets.QPushButton("식사시작", self)
        self.btn_start.setGeometry(QtCore.QRect(30, 50, 200, 100))

        self.btn_settings = QtWidgets.QPushButton("회원등록", self)
        self.btn_settings.setGeometry(QtCore.QRect(30, 180, 200, 100))