from PyQt5 import QtCore, QtWidgets

class UserInfoWindow(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setup_ui()

    def setup_ui(self):
        self.setObjectName("window4")
        self.resize(1200, 700)

        # Name
        self.label_name = QtWidgets.QLabel("Name:", self)
        self.label_name.setGeometry(QtCore.QRect(10, 10, 60, 30))
        self.edit_name = QtWidgets.QLineEdit(self)
        self.edit_name.setGeometry(QtCore.QRect(80, 10, 200, 30))

        # Birth Date
        self.label_birth_date = QtWidgets.QLabel("Birth Date:", self)
        self.label_birth_date.setGeometry(QtCore.QRect(10, 50, 60, 30))
        self.edit_birth_date = QtWidgets.QLineEdit(self)
        self.edit_birth_date.setGeometry(QtCore.QRect(80, 50, 200, 30))

        # Gender
        self.label_gender = QtWidgets.QLabel("Gender:", self)
        self.label_gender.setGeometry(QtCore.QRect(10, 90, 60, 30))
        self.combo_gender = QtWidgets.QComboBox(self)
        self.combo_gender.setGeometry(QtCore.QRect(80, 90, 200, 30))
        self.combo_gender.addItems(["Male", "Female"])

        # Phone
        self.label_phone = QtWidgets.QLabel("Phone:", self)
        self.label_phone.setGeometry(QtCore.QRect(10, 130, 60, 30))
        self.edit_phone = QtWidgets.QLineEdit(self)
        self.edit_phone.setGeometry(QtCore.QRect(80, 130, 200, 30))

        # Address
        self.label_address = QtWidgets.QLabel("Address:", self)
        self.label_address.setGeometry(QtCore.QRect(10, 170, 60, 30))
        self.edit_address = QtWidgets.QLineEdit(self)
        self.edit_address.setGeometry(QtCore.QRect(80, 170, 200, 30))

        # Height
        self.label_height = QtWidgets.QLabel("Height:", self)
        self.label_height.setGeometry(QtCore.QRect(10, 210, 60, 30))
        self.edit_height = QtWidgets.QLineEdit(self)
        self.edit_height.setGeometry(QtCore.QRect(80, 210, 200, 30))

        # Weight
        self.label_weight = QtWidgets.QLabel("Weight:", self)
        self.label_weight.setGeometry(QtCore.QRect(10, 250, 60, 30))
        self.edit_weight = QtWidgets.QLineEdit(self)
        self.edit_weight.setGeometry(QtCore.QRect(80, 250, 200, 30))

        # Buttons
        self.btn_next = QtWidgets.QPushButton("Next", self)
        self.btn_next.setGeometry(QtCore.QRect(950, 590, 200, 100))

        self.btn_home = QtWidgets.QPushButton("Home", self)
        self.btn_home.setGeometry(QtCore.QRect(950, 30, 200, 100))
