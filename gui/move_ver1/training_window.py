from PyQt5 import QtWidgets, QtCore, QtGui
from register import FaceRegistration
import cv2
import threading
from training_worker import TrainingWorker

class TrainingWindow(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.face_registration = FaceRegistration()
        self.training_folder = None
        self.user_info = None
        self.worker_thread = None  # Add a QThread for the worker
        self.setup_ui()

    def setup_ui(self):
        self.setObjectName("TrainingWindow")
        self.resize(1200, 700)

        # Camera Frame Label
        self.camera_frame = QtWidgets.QLabel("Camera Training Frame", self)
        self.camera_frame.setGeometry(QtCore.QRect(30, 70, 880, 590))
        self.camera_frame.setAlignment(QtCore.Qt.AlignCenter)
        self.camera_frame.setStyleSheet("border: 1px solid black;")

        # Home Button
        self.btn_home = QtWidgets.QPushButton("Home", self)
        self.btn_home.setGeometry(QtCore.QRect(950, 150, 200, 50))

        # Start Training Button
        self.btn_start_training = QtWidgets.QPushButton("Start Training", self)
        self.btn_start_training.setGeometry(QtCore.QRect(950, 70, 200, 50))
        self.btn_start_training.clicked.connect(self.start_training)

        # Next Button
        self.btn_next = QtWidgets.QPushButton("Next", self)
        self.btn_next.setGeometry(QtCore.QRect(950, 230, 200, 50))
        self.btn_next.setEnabled(False)
        self.btn_next.clicked.connect(self.go_next)

        # Progress Bar
        self.progress_bar = QtWidgets.QProgressBar(self)
        self.progress_bar.setGeometry(QtCore.QRect(30, 680, 900, 20))
        self.progress_bar.setValue(0)

    def set_training_folder(self, folder_path, user_info):
        self.training_folder = folder_path
        self.user_info = user_info
        print(f"Training folder set to: {self.training_folder}")
        print(f"User information: {self.user_info}")

    def start_training(self):
        if not self.training_folder or not self.user_info:
            QtWidgets.QMessageBox.warning(self, "Error", "Training data is missing.")
            return

        # Create a QThread and TrainingWorker
        self.worker_thread = QtCore.QThread()
        self.worker = TrainingWorker(self.face_registration, self.training_folder, self.user_info)
        self.worker.moveToThread(self.worker_thread)

        # Connect signals
        self.worker.update_camera_frame.connect(self.set_camera_frame)
        self.worker.update_progress.connect(self.progress_bar.setValue)
        self.worker.training_complete.connect(self.on_training_complete)

        # Start the thread and worker
        self.worker_thread.started.connect(self.worker.run)
        self.worker_thread.start()

    def set_camera_frame(self, pixmap):
        self.camera_frame.setPixmap(pixmap)

    def on_training_complete(self):
        QtWidgets.QMessageBox.information(self, "Training Complete", "Training is complete!")
        self.btn_next.setEnabled(True)
        self.worker_thread.quit()
        self.worker_thread.wait()

    def go_next(self):
        parent = self.parent()
        while parent and not hasattr(parent, "switch_window"):
            parent = parent.parent()

        if parent and hasattr(parent, "switch_window"):
            parent.switch_window(0)  # Switch back to HomeWindow
        else:
            QtWidgets.QMessageBox.critical(self, "Error", "Unable to navigate to the next window.")

