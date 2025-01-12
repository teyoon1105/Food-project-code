from PyQt5 import QtCore, QtWidgets
from recognize import FaceRecognition  # 변환된 FaceRecognition 클래스 가져오기
import cv2
from PyQt5.QtGui import QImage, QPixmap

class CameraStreamWindow(QtWidgets.QWidget):
    recognized_user_id = QtCore.pyqtSignal(str)  # 사용자 ID를 다음 창으로 전달하기 위한 시그널

    def __init__(self, parent=None):
        super().__init__(parent)
        self.face_recognition = FaceRecognition()  # FaceRecognition 객체 생성
        self.cap = None  # 카메라 객체 초기화
        self.setup_ui()

    def setup_ui(self):
        self.setObjectName("camera_stream_window")
        self.resize(1200, 700)

        # Camera Frame Display
        self.camera_frame = QtWidgets.QLabel("Camera is not running.", self)
        self.camera_frame.setGeometry(QtCore.QRect(30, 70, 880, 590))
        self.camera_frame.setAlignment(QtCore.Qt.AlignCenter)

        # Start Button
        self.btn_start_frame = QtWidgets.QPushButton("Start Recognition", self)
        self.btn_start_frame.setGeometry(QtCore.QRect(950, 70, 200, 50))
        self.btn_start_frame.clicked.connect(self.start_recognition)

        # Home Button
        self.btn_home = QtWidgets.QPushButton("Home", self)
        self.btn_home.setGeometry(QtCore.QRect(950, 150, 200, 50))
        self.btn_home.clicked.connect(self.go_back)

        # Next Button
        self.btn_next = QtWidgets.QPushButton("Next", self)
        self.btn_next.setGeometry(QtCore.QRect(950, 230, 200, 50))
        self.btn_next.setEnabled(False)  # Initially disabled, enabled upon recognition

        # Timer for real-time frame updates
        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.update_frame)

    def start_recognition(self):
        try:
            if self.cap is None:
                self.cap = cv2.VideoCapture(0)  # 카메라 시작
                if not self.cap.isOpened():
                    QtWidgets.QMessageBox.critical(self, "Error", "Failed to open the camera.")
                    return

            # Create and train the model using FaceRecognition method
            self.face_recognition.create_unified_model()
            print("Model loaded successfully.")

            # Update UI
            self.btn_start_frame.setEnabled(False)
            self.camera_frame.setText("Initializing camera...")
            self.timer.start(30)  # 30ms마다 프레임 업데이트

        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Error", f"An error occurred: {str(e)}")

    def update_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            self.camera_frame.setText("Failed to capture frame.")
            print("Failed to capture camera frame.")
            return

        image, user_id = self.face_recognition.detect_and_recognize(frame)
        if user_id:
            print(f"Recognized User ID: {user_id}")
            self.recognized_user_id.emit(user_id)
            self.btn_next.setEnabled(True)
            self.stop_recognition()
            return

        # OpenCV -> RGB 변환 (최적화)
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        qt_image = QImage(rgb_image.data, w, h, ch * w, QImage.Format_RGB888)
        self.camera_frame.setPixmap(QPixmap.fromImage(qt_image))

    def stop_recognition(self):
        self.timer.stop()
        if hasattr(self, "cap") and self.cap.isOpened():
            self.cap.release()
            self.cap = None  # 카메라 객체 해제
        self.camera_frame.setText("Camera stopped.")
        self.btn_start_frame.setEnabled(True)

    def go_back(self):
        self.stop_recognition()
        self.close()  # 현재 창 닫기

if __name__ == "__main__":
    import sys

    app = QtWidgets.QApplication(sys.argv)
    window = CameraStreamWindow()

    def handle_recognized_user(user_id):
        print(f"User recognized: {user_id}")
        # 다음 창으로 넘어가는 로직 추가

    window.recognized_user_id.connect(handle_recognized_user)
    window.show()
    sys.exit(app.exec_())