from PyQt5 import QtWidgets
from home_window import HomeWindow
from camera_stream_window import CameraStreamWindow
from list_display_window import ListDisplayWindow
from text_chart_window import TextChartWindow
from user_info_window import UserInfoWindow
from training_window import TrainingWindow

class MainApp(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setup_ui()

    def setup_ui(self):
        self.setWindowTitle("Main Application")
        self.resize(1200, 700)

        self.stacked_widget = QtWidgets.QStackedWidget()
        self.setCentralWidget(self.stacked_widget)

        # Instantiate and add windows to stacked widget
        self.home_window = HomeWindow()
        self.camera_stream_window = CameraStreamWindow()
        self.list_display_window = ListDisplayWindow()
        self.text_chart_window = TextChartWindow()
        self.user_info_window = UserInfoWindow()
        self.training_window = TrainingWindow()

        self.stacked_widget.addWidget(self.home_window)  # Index 0
        self.stacked_widget.addWidget(self.camera_stream_window)  # Index 1
        self.stacked_widget.addWidget(self.list_display_window)  # Index 2
        self.stacked_widget.addWidget(self.text_chart_window)  # Index 3
        self.stacked_widget.addWidget(self.user_info_window)  # Index 4
        self.stacked_widget.addWidget(self.training_window)  # Index 5

        # Connect signals for navigation (example)
        self.home_window.btn_start.clicked.connect(lambda: self.switch_window(1))
        self.home_window.btn_settings.clicked.connect(lambda: self.switch_window(4))
        
        # Modified: CameraStreamWindow's Start Frame Button
        self.camera_stream_window.btn_start_frame.clicked.connect(self.start_camera_frame)

        self.camera_stream_window.btn_next.clicked.connect(lambda: self.switch_window(2))
        self.camera_stream_window.btn_home.clicked.connect(lambda: self.switch_window(0))
        self.list_display_window.btn_complete.clicked.connect(lambda: self.switch_window(3))
        self.text_chart_window.btn_home.clicked.connect(lambda: self.switch_window(0))
        self.user_info_window.btn_next.clicked.connect(lambda: self.switch_window(5))
        self.user_info_window.btn_home.clicked.connect(lambda: self.switch_window(0))
        self.training_window.btn_next.clicked.connect(lambda: self.switch_window(0))
        self.training_window.btn_home.clicked.connect(lambda: self.switch_window(0))

    def switch_window(self, index):
        self.stacked_widget.setCurrentIndex(index)
        

    def start_camera_frame(self):
        customer_id = self.camera_stream_window.input_customer_id.text()
        if customer_id:
            QtWidgets.QMessageBox.information(self, "Customer ID", f"Starting frame for Customer ID: {customer_id}")
            # 전달할 customer_id를 ListDisplayWindow에 설정
            self.list_display_window.customer_id = customer_id
            self.switch_window(2)  # Proceed to list_display_window
        else:
            QtWidgets.QMessageBox.warning(self, "Missing ID", "Please enter a valid Customer ID before proceeding.")


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    main_app = MainApp()
    main_app.show()
    sys.exit(app.exec_())
