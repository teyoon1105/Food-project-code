from PyQt5 import QtCore, QtWidgets

class CameraStreamWindow(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setup_ui()

    def setup_ui(self):
        self.setObjectName("window1")
        self.resize(1200, 700)

        # Customer ID 입력 창
        self.label_customer_id = QtWidgets.QLabel("Enter Customer ID:", self)
        self.label_customer_id.setGeometry(QtCore.QRect(30, 20, 200, 30))

        self.input_customer_id = QtWidgets.QLineEdit(self)
        self.input_customer_id.setGeometry(QtCore.QRect(250, 20, 300, 30))

        self.btn_start_frame = QtWidgets.QPushButton("Start Frame", self)
        self.btn_start_frame.setGeometry(QtCore.QRect(580, 20, 150, 30))
        self.btn_start_frame.clicked.connect(self.start_frame)

        # Camera Frame Display
        self.camera_frame = QtWidgets.QLabel("Camera 1 Frame", self)
        self.camera_frame.setGeometry(QtCore.QRect(30, 70, 880, 590))
        self.camera_frame.setAlignment(QtCore.Qt.AlignCenter)

        # Navigation Buttons
        self.btn_next = QtWidgets.QPushButton("Next", self)
        self.btn_next.setGeometry(QtCore.QRect(940, 570, 200, 100))

        self.btn_home = QtWidgets.QPushButton("Home", self)
        self.btn_home.setGeometry(QtCore.QRect(950, 40, 200, 100))

    def start_frame(self):
        customer_id = self.input_customer_id.text()
        if customer_id:
            QtWidgets.QMessageBox.information(self, "Success", f"Frame started for Customer ID: {customer_id}")
            # 이후 frame 실행 코드를 여기에 추가
        else:
            QtWidgets.QMessageBox.warning(self, "Error", "Please enter a valid Customer ID.")
