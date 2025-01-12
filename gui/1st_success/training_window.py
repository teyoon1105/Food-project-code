from PyQt5 import QtCore, QtWidgets

class TrainingWindow(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setup_ui()

    def setup_ui(self):
        self.setObjectName("window5")
        self.resize(1200, 700)

        self.camera_frame = QtWidgets.QLabel("Camera Training Frame", self)
        self.camera_frame.setGeometry(QtCore.QRect(30, 40, 880, 640))
        self.camera_frame.setAlignment(QtCore.Qt.AlignCenter)

        self.btn_home = QtWidgets.QPushButton("Home", self)
        self.btn_home.setGeometry(QtCore.QRect(950, 30, 200, 100))

        self.btn_next = QtWidgets.QPushButton("Next", self)
        self.btn_next.setGeometry(QtCore.QRect(950, 590, 200, 100))