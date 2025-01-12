from PyQt5.QtCore import QObject, pyqtSignal
from PyQt5 import QtGui
import cv2

class TrainingWorker(QObject):
    update_camera_frame = pyqtSignal(QtGui.QPixmap)  # QLabel 업데이트
    update_progress = pyqtSignal(int)               # 프로그레스 바 업데이트
    training_complete = pyqtSignal()               # 학습 완료 시그널

    def __init__(self, face_registration, training_folder, user_info):
        super().__init__()
        self.face_registration = face_registration
        self.training_folder = training_folder
        self.user_info = user_info
        self.running = True

    def run(self):
        """Perform the training process."""
        try:
            cap = cv2.VideoCapture(0)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)  # 줄인 카메라 프레임 크기
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

            if not cap.isOpened():
                print("Error: Unable to access the camera.")
                self.training_complete.emit()  # Signal training completion to the main thread
                return

            count = 0
            total_images = 200

            while self.running and count < total_images:
                ret, frame = cap.read()
                if not ret:
                    print("Warning: Failed to read frame.")
                    continue

                # 얼굴 추출
                faces = self.face_registration.face_extractor(frame)
                if faces:
                    for face in faces:
                        count += 1
                        face_resized = cv2.resize(face, (200, 200))  # 얼굴 크기 최적화
                        face_gray = cv2.cvtColor(face_resized, cv2.COLOR_BGR2GRAY)

                        # 이미지 저장
                        file_name_path = f"{self.training_folder}/{self.user_info['아이디']}_{count}.jpg"
                        cv2.imwrite(file_name_path, face_gray)

                        # QLabel 업데이트
                        height, width = face_gray.shape
                        q_image = QtGui.QImage(
                            face_gray.data, width, height, width, QtGui.QImage.Format_Grayscale8
                        )
                        pixmap = QtGui.QPixmap.fromImage(q_image)
                        self.update_camera_frame.emit(pixmap)

                        # 프로그레스 바 업데이트
                        progress = int((count / total_images) * 100)
                        self.update_progress.emit(progress)

            # 학습 완료
            cap.release()
            self.face_registration.train_model(self.training_folder)
            self.training_complete.emit()

        except Exception as e:
            print(f"Error during training: {e}")
        finally:
            cap.release()  # Ensure the camera is released
