import sys
import os
import cv2
import numpy as np
import pyrealsense2 as rs
from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QApplication, QLabel, QVBoxLayout, QWidget, QPushButton
from ultralytics import YOLO
import torch
import logging

# gpt를 이용해서
class DepthVolumeCalculatorApp(QWidget):
    def __init__(self, model_path, roi_points, brightness_increase, cls_name_color):
        super().__init__()
        self.setWindowTitle("Depth Volume Calculator")
        self.resize(1280, 720)

        # ---- PyQt Layout ----
        self.image_label = QLabel("Camera Feed")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.info_label = QLabel("Status: Initializing")
        self.info_label.setAlignment(Qt.AlignLeft)

        self.save_depth_button = QPushButton("Save Depth Data")
        self.save_depth_button.clicked.connect(self.save_depth_data)

        # 종료 버튼 추가
        self.exit_button = QPushButton("Exit Application")
        self.exit_button.clicked.connect(self.close_application)

        self.layout = QVBoxLayout()
        self.layout.addWidget(self.image_label)
        self.layout.addWidget(self.info_label)
        self.layout.addWidget(self.save_depth_button)
        self.layout.addWidget(self.exit_button)  # 종료 버튼 추가
        self.setLayout(self.layout)

        # ---- Camera & YOLO Initialization ----
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
        self.config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
        self.align = rs.align(rs.stream.color)

        self.pipeline_active = False  # 파이프라인 초기화 상태
        self.save_depth = None  # 기준 깊이 데이터를 저장
        self.roi_points = roi_points
        self.brightness_increase = brightness_increase
        self.cls_name_color = cls_name_color

        self.model = YOLO(model_path)
        self.model.to('cuda' if torch.cuda.is_available() else 'cpu')

        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)

        # 카메라 초기화
        self.initialize_camera()
        self.info_label.setText("Status: Camera and YOLO Model Initialized")

    def initialize_camera(self):
        """카메라 파이프라인 초기화"""
        if not self.pipeline_active:  # 이미 시작된 경우 중복 호출 방지
            try:
                self.pipeline.start(self.config)
                self.pipeline_active = True
                self.info_label.setText("Camera initialized successfully!")
            except Exception as e:
                self.info_label.setText(f"Failed to initialize camera: {e}")
                self.pipeline_active = False

    def close_application(self):
        """애플리케이션 종료"""
        self.timer.stop()  # 타이머 중지
        if self.pipeline_active:  # 파이프라인이 활성화된 경우 정지
            self.pipeline.stop()
            self.pipeline_active = False
        QApplication.quit()

    def update_frame(self):
        """카메라 프레임을 업데이트하고, 객체 탐지를 수행"""
        if not self.pipeline_active:
            self.info_label.setText("Error: Camera pipeline is not active!")
            return

        try:
            frames = self.pipeline.wait_for_frames()
            aligned_frames = self.align.process(frames)

            depth_frame = aligned_frames.get_depth_frame()
            color_frame = aligned_frames.get_color_frame()

            if not depth_frame or not color_frame:
                return

            depth_image, color_image = self.preprocess_images(depth_frame, color_frame)
            cropped_color = self.apply_roi(color_image)

            # 밝기 증가
            brightened_image = cv2.convertScaleAbs(cropped_color, alpha=1, beta=self.brightness_increase)
            results = self.model(brightened_image)

            all_colored_mask = np.zeros_like(brightened_image)
            blended_image = brightened_image.copy()

            detected_objects = []
            for result in results:
                if result.masks is not None:
                    masks = result.masks.data.cpu().numpy()
                    classes = result.boxes.cls.cpu().numpy()

                    for i, mask in enumerate(masks):
                        conf = result.boxes.conf[i]
                        class_key = self.model.names[int(classes[i])]
                        object_name, color = self.cls_name_color.get(class_key, ("Unknown", (255, 255, 255)))

                        # 컬러 마스크 생성
                        resized_mask = cv2.resize(mask, (brightened_image.shape[1], brightened_image.shape[0]))
                        color_mask = (resized_mask > 0.5).astype(np.uint8)
                        mask_indices = np.where(color_mask > 0)

                        # 부피 계산
                        if self.save_depth is not None:
                            volume = self.calculate_volume(depth_image, mask_indices, depth_frame.profile.as_video_stream_profile().intrinsics)
                            all_colored_mask[color_mask == 1] = color
                            blended_image = self.visualize_results(
                                brightened_image, all_colored_mask, object_name, volume, conf, color, mask_indices)

                        detected_objects.append((object_name, conf))

            # GUI에 업데이트
            blended_image = cv2.cvtColor(blended_image, cv2.COLOR_BGR2RGB)
            height, width, channel = blended_image.shape
            q_image = QImage(blended_image.data, width, height, width * channel, QImage.Format_RGB888)
            self.image_label.setPixmap(QPixmap.fromImage(q_image))

            # 탐지된 객체 정보 업데이트
            self.info_label.setText(f"Detected Objects: {len(detected_objects)}")

        except Exception as e:
            self.info_label.setText(f"Error: {e}")

    def save_depth_data(self):
        """ROI 영역의 깊이 데이터를 저장"""
        if not self.pipeline_active:
            self.info_label.setText("Error: Camera pipeline is not active!")
            return

        depth_frame = self.capture_depth_frame()
        if depth_frame is not None:
            cropped_depth = self.apply_roi(depth_frame)  # ROI 영역으로 크롭
            flip_depth = cv2.flip(cropped_depth, -1)
            self.save_depth = flip_depth.copy()  # ROI 영역만 저장
            self.info_label.setText("Depth data saved successfully!")
        else:
            self.info_label.setText("Failed to save depth data.")

    def capture_depth_frame(self):
        """현재 깊이 프레임을 가져옴"""
        if not self.pipeline_active:
            raise RuntimeError("Pipeline is not started. Initialize the camera first.")
        frames = self.pipeline.wait_for_frames()
        aligned_frames = self.align.process(frames)
        depth_frame = aligned_frames.get_depth_frame()
        return np.asanyarray(depth_frame.get_data()) if depth_frame else None

    def preprocess_images(self, depth_frame, color_frame):
        """깊이 및 컬러 프레임 전처리"""
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())
        return cv2.flip(depth_image, -1), cv2.flip(color_image, -1)

    def apply_roi(self, image):
        """이미지에 관심 영역(ROI) 적용"""
        x1, y1 = self.roi_points[0]
        x2, y2 = self.roi_points[1]
        return image[y1:y2, x1:x2]

    def calculate_volume(self, cropped_depth, mask_indices, depth_intrin, min_depth_cm=20):
        """객체의 부피 계산"""
        total_volume = 0
        y_indices, x_indices = mask_indices
        for y, x in zip(y_indices, x_indices):
            z_cm = cropped_depth[y, x] / 10
            base_depth_cm = self.save_depth[y, x] / 10
            if z_cm > min_depth_cm and base_depth_cm > 25:
                height_cm = base_depth_cm - z_cm
                pixel_area_cm2 = (z_cm ** 2) / (depth_intrin.fx * depth_intrin.fy)
                total_volume += pixel_area_cm2 * height_cm
        return total_volume

    def visualize_results(self, cropped_image, mask, object_name, volume, conf, color, mask_indices):
        """탐지 결과를 시각화"""
        y_indices, x_indices = mask_indices
        min_x, min_y = np.min(x_indices), np.min(y_indices)
        label_position = (min_x, min_y - 10)
        cv2.putText(mask, f"{object_name}: V:{volume:.0f}cm^3, C:{conf:.2f}", label_position,
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        return cv2.addWeighted(cropped_image, 0.85, mask, 0.15, 0)


if __name__ == "__main__":
    # 로그 레벨 설정 (INFO 메시지 비활성화)
    logging.getLogger("ultralytics").setLevel(logging.WARNING)

    MODEL_PATH = "C:/Users/SBA/teyoon_github/Food-project-code/depth_camera/test_model/model/1st_mix_scale_best.pt"
    ROI_POINTS = [(175, 50), (1055, 690)]
    BRIGHTNESS_INCREASE = 50
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
    '11014002': ('Gosari', (255, 255, 0)),
    '11013007': ('Spinach', (255, 255, 0)), # 청록색
    '12011008': ('Kimchi', (100, 100, 100)),
    '12011003': ('Radish Kimchi', (100, 100, 100))
}

    app = QApplication(sys.argv)
    window = DepthVolumeCalculatorApp(MODEL_PATH, ROI_POINTS, BRIGHTNESS_INCREASE, CLS_NAME_COLOR)
    window.show()
    sys.exit(app.exec_())
