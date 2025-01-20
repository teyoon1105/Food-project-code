import sys
import os
import numpy as np
import pyrealsense2 as rs
import cv2
import torch
from PySide6.QtWidgets import QApplication, QWidget, QListWidgetItem
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtCore import QThread, Signal
from ultralytics import YOLO
from pyside_exui import Ui_Form  # PySide6 UI 파일에서 변환된 클래스
import torch.nn.functional as F

class CameraThread(QThread):
    """카메라 프레임을 처리하는 스레드"""
    new_frame = Signal(np.ndarray)  # UI로 프레임 전송

    def __init__(self, calculator):
        super().__init__()
        self.calculator = calculator

    def run(self):
        """main_loop를 스레드에서 실행"""
        self.calculator.initialize_camera()

        if os.path.exists("save_depth.npy"):
            self.calculator.save_depth = np.load("save_depth.npy")
            print("Loaded saved depth data.")
        else:
            print("No saved depth data found. Please save depth data.")

        try:
            while True:
                depth_frame, color_frame = self.calculator.capture_frames()
                if not depth_frame or not color_frame:
                    print("Frames not available.")
                    continue

                depth_image, color_image = self.calculator.preprocess_images(depth_frame, color_frame)
                cropped_depth = self.calculator.apply_roi(depth_image)
                cropped_color = self.calculator.apply_roi(color_image)

                blended_image = cropped_color.copy()

                results = self.calculator.model(cropped_color)
                if results:
                    for result in results:
                        if result.masks is not None:
                            masks = result.masks.data
                            original_size = (cropped_color.shape[0], cropped_color.shape[1])  # 원본 크기 (높이, 너비)
                            # PyTorch를 사용한 리사이즈
                            resized_masks = F.interpolate(masks.unsqueeze(1), size=original_size, mode='bilinear', align_corners=False)
                            resized_masks = resized_masks.squeeze(1)
                            for i, mask in enumerate(resized_masks):
                                class_idx = int(result.boxes.cls[i].item())
                                conf = result.boxes.conf[i].item()

                                object_name, color = self.calculator.cls_name_color.get(
                                    self.calculator.model.names[class_idx], ("Unknown", (255, 255, 255))
                                )

                                mask_indices = torch.where(mask > 0.5)
                                mask_y, mask_x = mask_indices[0].cpu().numpy(), mask_indices[1].cpu().numpy()

                                depth_intrin = depth_frame.profile.as_video_stream_profile().intrinsics
                                volume = self.calculator.calculate_volume_on_gpu(cropped_depth, depth_intrin, (mask_y, mask_x))

                                blended_image[mask_y, mask_x] = (
                                    blended_image[mask_y, mask_x] * 0.5 + np.array(color) * 0.5
                                ).astype(np.uint8)

                                # 탐지된 객체 리스트 업데이트
                                self.calculator.update_list(object_name, volume)

                # 프레임 업데이트
                self.new_frame.emit(blended_image)

        finally:
            self.calculator.pipeline.stop()
            print("Pipeline stopped.")

class DepthVolumeCalculator(QWidget):
    def __init__(self, model_path, roi_points, cls_name_color):
        super().__init__()
        self.ui = Ui_Form()
        self.ui.setupUi(self)

        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
        self.config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
        self.align = rs.align(rs.stream.color)
        self.save_depth = None
        self.roi_points = roi_points
        self.cls_name_color = cls_name_color

        self.model_name = os.path.basename(model_path)
        try:
            self.model = YOLO(model_path)
            self.model.to("cuda" if torch.cuda.is_available() else "cpu")
            print(f"YOLO model '{self.model_name}' loaded successfully.")
        except Exception as e:
            print(f"Error loading YOLO model: {e}")
            exit(1)

        # CameraThread 생성 및 연결
        self.camera_thread = CameraThread(self)
        self.camera_thread.new_frame.connect(self.update_frame)
        self.camera_thread.start()
        print("Camera thread started.")

    def initialize_camera(self):
        try:
            self.pipeline.start(self.config)
            self.align = rs.align(rs.stream.color)
            print("Camera initialized successfully.")
        except Exception as e:
            print(f"Failed to initialize camera: {e}")

    def capture_frames(self):
        frames = self.pipeline.wait_for_frames()
        aligned_frames = self.align.process(frames)
        depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()
        return depth_frame, color_frame

    def preprocess_images(self, depth_frame, color_frame):
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())
        return cv2.flip(depth_image, -1), cv2.flip(color_image, -1)

    def apply_roi(self, image):
        x1, y1 = self.roi_points[0]
        x2, y2 = self.roi_points[1]
        return image[y1:y2, x1:x2]
    

    def calculate_volume_on_gpu(self, cropped_depth, depth_intrin, mask_indices):
        depth_tensor = torch.tensor(cropped_depth, device="cuda", dtype=torch.float32)
        mask_y, mask_x = mask_indices
        mask_tensor = (torch.tensor(mask_y, device="cuda"), torch.tensor(mask_x, device="cuda"))

        saved_depth_tensor = torch.tensor(self.save_depth, device="cuda", dtype=torch.float32)

        z_cm = depth_tensor[mask_tensor] / 10.0
        base_depth_cm = saved_depth_tensor[mask_tensor] / 10.0

        height_cm = torch.clamp(base_depth_cm - z_cm, min=0)
        pixel_area_cm2 = (z_cm ** 2) / (depth_intrin.fx * depth_intrin.fy)
        volume = torch.sum(height_cm * pixel_area_cm2).item()

        return volume

    def update_frame(self, blended_image):
        """OpenCV 이미지를 QLabel에 표시."""
        height, width, channel = blended_image.shape
        print(f"Updating Frame - Image Shape: {height}x{width}x{channel}")
        bytes_per_line = channel * width
        q_image = QImage(blended_image.data, width, height, bytes_per_line, QImage.Format_BGR888)
        pixmap = QPixmap.fromImage(q_image)
        if pixmap.isNull():
            print("Failed to create pixmap")
        self.ui.frame.setPixmap(pixmap)  # QLabel에 픽스맵 설정
        self.ui.frame.update()
        print("Frame updated successfully.")


    def update_list(self, object_name, volume):
        """탐지된 객체 리스트 업데이트."""
        item = QListWidgetItem(f"{object_name}: {volume:.1f} cm³")
        self.ui.listWidget.addItem(item)

if __name__ == "__main__":
    MODEL_PATH = os.path.join(os.getcwd(), "model", "path/your/model.pt")
    ROI_POINTS = [(175, 50), (1055, 690)]
    # class name for mapping
    CLS_NAME_COLOR = {
    '01011001': ('Rice', (255, 0, 255)), # 자주색
    '01012006': ('Black Rice', (255, 0, 255)),
    '01012002': ('Soy bean Rice', (255, 0, 255)),
    '03011011': ('Pumpkin soup', (255, 0, 255)),
    '04011005': ('Seaweed Soup', (0, 255, 255)),
    '04011007': ('Beef stew', (0, 255, 255)),
    '04017001': ('Soybean Soup', (0, 255, 255)), # 노란색
    '04011011': ('Fish cake soup', (0, 255, 255)),
    '06012004': ('Tteokgalbi', (0, 255, 0)), # 초록색
    '06012008': ('Beef Bulgogi', (0, 255, 0)),
    '07014001': ('EggRoll', (0, 0, 255)), # 빨간색
    '08011003': ('Stir-fried anchovies', (0, 0, 255)),
    '10012001': ('Chicken Gangjeong', (0, 0, 255)),
    '07013003': ('Kimchijeon', (0, 0, 255)),
    '08012001': ('Stir-fried Potatoes', (255,255,0)),
    '11013010': ('KongNamul', (255, 255, 0)),
    '11013002': ('Gosari', (255, 255, 0)),
    '11013007': ('Spinach', (255, 255, 0)), # 청록색
    '12011008': ('Kimchi', (100, 100, 100)),
    '12011003': ('Radish Kimchi', (100, 100, 100))
    }


    app = QApplication(sys.argv)
    window = DepthVolumeCalculator(MODEL_PATH, ROI_POINTS, CLS_NAME_COLOR)
    window.show()
    sys.exit(app.exec())
