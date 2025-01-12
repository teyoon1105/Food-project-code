from PyQt5 import QtCore, QtWidgets
from register import FaceRegistration
import os

class UserInfoWindow(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.face_registration = FaceRegistration()
        self.setup_ui()

    def setup_ui(self):
        self.setObjectName("UserInfoWindow")
        self.resize(1200, 700)

        # Title Label
        self.label_title = QtWidgets.QLabel("User Information", self)
        self.label_title.setGeometry(QtCore.QRect(20, 20, 400, 30))

        # Input Fields
        self.edit_name = QtWidgets.QLineEdit(self)
        self.edit_name.setGeometry(QtCore.QRect(20, 70, 400, 30))
        self.edit_name.setPlaceholderText("Enter Name")

        self.edit_birth_date = QtWidgets.QLineEdit(self)
        self.edit_birth_date.setGeometry(QtCore.QRect(20, 120, 400, 30))
        self.edit_birth_date.setPlaceholderText("Enter Birth Date (YYYYMMDD)")

        self.combo_gender = QtWidgets.QComboBox(self)
        self.combo_gender.setGeometry(QtCore.QRect(20, 170, 400, 30))
        self.combo_gender.addItems(["M", "F"])

        self.edit_phone = QtWidgets.QLineEdit(self)
        self.edit_phone.setGeometry(QtCore.QRect(20, 220, 400, 30))
        self.edit_phone.setPlaceholderText("Enter Phone Number")

        self.edit_address = QtWidgets.QLineEdit(self)
        self.edit_address.setGeometry(QtCore.QRect(20, 270, 400, 30))
        self.edit_address.setPlaceholderText("Enter Address")

        self.edit_height = QtWidgets.QLineEdit(self)
        self.edit_height.setGeometry(QtCore.QRect(20, 320, 400, 30))
        self.edit_height.setPlaceholderText("Enter Height (cm)")

        self.edit_weight = QtWidgets.QLineEdit(self)
        self.edit_weight.setGeometry(QtCore.QRect(20, 370, 400, 30))
        self.edit_weight.setPlaceholderText("Enter Weight (kg)")
        
        # 스타일시트 적용
        self.setStyleSheet("""
        QLineEdit {
            background-color: #2C2F33;  /* 배경 진한 회색 */
            color: white;              /* 글자 흰색 */
            border: 1px solid #555555; /* 테두리 색 */
            border-radius: 5px;        /* 둥근 모서리 */
            padding: 5px;
        }
        QLineEdit:focus {
            border: 1px solid #7289DA; /* 포커스 시 테두리 색 */
        }
        QComboBox {
            background-color: #2C2F33;  /* 배경 진한 회색 */
            color: white;              /* 글자 흰색 */
            border: 1px solid #555555; /* 테두리 색 */
            border-radius: 5px;        /* 둥근 모서리 */
            padding: 5px;
        }
        QComboBox QAbstractItemView {
            background-color: #2C2F33;  /* 드롭다운 배경 진한 회색 */
            color: white;              /* 드롭다운 글자 흰색 */
            selection-background-color: #7289DA; /* 선택한 항목 배경색 */
        }
        """)

        # Buttons
        self.btn_next = QtWidgets.QPushButton("다음", self)
        self.btn_next.setGeometry(QtCore.QRect(950, 580, 200, 100))
        self.btn_next.clicked.connect(self.save_user_info)

        self.btn_home = QtWidgets.QPushButton("홈", self)
        self.btn_home.setGeometry(QtCore.QRect(950, 30, 200, 100))

    def find_main_app(self):
        """Recurse through parents to find MainApp."""
        parent = self.parent()
        while parent is not None and not isinstance(parent, QtWidgets.QMainWindow):
            parent = parent.parent()
        return parent

    def save_user_info(self):
        """Save user information and proceed to the next step."""
        user_info = {
            "아이디": None,
            "이름": self.edit_name.text(),
            "생년월일": self.edit_birth_date.text(),
            "성별": self.combo_gender.currentText(),
            "연락처": self.edit_phone.text(),
            "주소": self.edit_address.text(),
            "키": self.edit_height.text(),
            "몸무게": self.edit_weight.text(),
        }

        # Validate required fields
        if not user_info["이름"] or not user_info["생년월일"] or not user_info["키"] or not user_info["몸무게"]:
            QtWidgets.QMessageBox.warning(self, "Error", "Name, Birth Date, Height, and Weight are required.")
            return

        # Generate folder and save to CSV
        user_id, folder_path = self.face_registration.generate_folder(user_info["이름"])
        user_info["아이디"] = user_id
        os.makedirs(folder_path, exist_ok=True)

        self.face_registration.save_csv(user_info)

        # Access TrainingWindow through MainApp
        main_app = self.find_main_app()
        if main_app and hasattr(main_app, "training_window"):
            main_app.training_window.set_training_folder(folder_path, user_info)
            main_app.switch_window(5)  # Switch to TrainingWindow
        else:
            QtWidgets.QMessageBox.critical(self, "Error", "Training window is not set.")
