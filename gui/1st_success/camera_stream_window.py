from PyQt5 import QtCore, QtWidgets

class CameraStreamWindow(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setup_ui()

    def setup_ui(self):
        self.setObjectName("window1")
        self.resize(1200, 700)

        self.camera_frame = QtWidgets.QLabel("Camera 1 Frame", self)
        self.camera_frame.setGeometry(QtCore.QRect(30, 20, 880, 640))
        self.camera_frame.setAlignment(QtCore.Qt.AlignCenter)

        self.btn_next = QtWidgets.QPushButton("Next", self)
        self.btn_next.setGeometry(QtCore.QRect(940, 570, 200, 100))

        self.btn_home = QtWidgets.QPushButton("Home", self)
        self.btn_home.setGeometry(QtCore.QRect(950, 40, 200, 100))