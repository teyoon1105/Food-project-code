from PyQt5 import QtWidgets
from home_window import HomeWindow
from camera_stream_window import CameraStreamWindow
from list_display_window import ListDisplayWindow
from nutrition_dashboard import NutritionDashboard  # Import NutritionDashboard
from user_info_window import UserInfoWindow
from training_window import TrainingWindow
from summary import NutritionSummary


class MainApp(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.nutrition_summary = None  # 초기값 설정
        self.customer_id = None
        self.setup_ui()

    def setup_ui(self):
        self.setWindowTitle("Main Application")
        self.resize(1200, 700)

        self.stacked_widget = QtWidgets.QStackedWidget()
        self.setCentralWidget(self.stacked_widget)

        # File paths for NutritionDashboard
        self.user_csv_file_path = "user_info.csv"
        self.diet_csv_file_path = "FOOD_DB/customer_diet_detail.csv"
        # self.customer_id = ''
        
        # self.nutrition_summary = None

        # Instantiate and add windows to stacked widget
        self.home_window = HomeWindow()
        self.camera_stream_window = CameraStreamWindow()
        self.list_display_window = ListDisplayWindow()
        self.nutrition_dashboard = NutritionDashboard()  # Replace TextChartWindow
        self.user_info_window = UserInfoWindow()
        self.training_window = TrainingWindow()

        self.stacked_widget.addWidget(self.home_window)  # Index 0
        self.stacked_widget.addWidget(self.camera_stream_window)  # Index 1
        self.stacked_widget.addWidget(self.list_display_window)  # Index 2
        self.stacked_widget.addWidget(self.nutrition_dashboard)  # Index 3 (Replaced)
        self.stacked_widget.addWidget(self.user_info_window)  # Index 4
        self.stacked_widget.addWidget(self.training_window)  # Index 5

        # Connect signals for navigation (example)
        self.home_window.btn_start.clicked.connect(lambda: self.switch_window(1))
        self.home_window.btn_settings.clicked.connect(lambda: self.switch_window(4))
        
        # Modified: CameraStreamWindow's Start Frame Button
        self.camera_stream_window.btn_start_frame.clicked.connect(self.start_camera_frame)
        self.camera_stream_window.recognized_user_id.connect(self.handle_recognized_user)  # Handle recognized user ID
        print("Signal connected successfully in MainApp")
        self.camera_stream_window.btn_next.clicked.connect(lambda: self.switch_window(2))
        self.camera_stream_window.btn_home.clicked.connect(lambda: self.switch_window(0))
        
        
        
        self.list_display_window.btn_complete.clicked.connect(lambda: self.switch_window(3))
        self.nutrition_dashboard.go_home_signal.connect(lambda: self.switch_window(0))
        # self.nutrition_dashboard.btn_home.clicked.connect(lambda: self.switch_window(0))  # Updated for NutritionDashboard
        self.user_info_window.btn_next.clicked.connect(lambda: self.switch_window(5))
        self.user_info_window.btn_home.clicked.connect(lambda: self.switch_window(0))
        self.training_window.btn_next.clicked.connect(lambda: self.switch_window(0))
        self.training_window.btn_home.clicked.connect(lambda: self.switch_window(0))

    def switch_window(self, index):
        self.stacked_widget.setCurrentIndex(index)
        
    def start_camera_frame(self):
        self.camera_stream_window.start_recognition()
        print(f"Start recognition, waiting for user ID...")


    def handle_recognized_user(self, user_id):
        print(f"[DEBUG] Recognized User ID: {user_id}")

        # 유저 ID 확인 후 목록에 설정
        self.list_display_window.set_customer_id(user_id)

        # NutritionSummary 객체 생성
        try:
            self.nutrition_summary = NutritionSummary(
                self.user_csv_file_path,
                self.diet_csv_file_path,
                user_id  # 얼굴 인식으로 받은 user_id를 전달
            )

            # 대시보드 업데이트
            self.nutrition_dashboard.update_dashboard(self.nutrition_summary)

            # 화면 전환
            self.switch_window(2)

        except Exception as e:
            QtWidgets.QMessageBox.warning(self, "Error", f"Error loading customer data: {e}")

    # def run_main_loop(self):
    #     print(f"[DEBUG] run_main_loop에서 고객 ID 확인: {self.customer_id}")
    #     # 고객 ID가 설정되지 않으면 카메라 루프를 시작하지 않음
    #     if not self.customer_id:
    #         print("Error: Customer ID is not set!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!.")
    #         return
if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    main_app = MainApp()
    main_app.show()
    sys.exit(app.exec_())