import pyrealsense2 as rs
from ultralytics import YOLO
import numpy as np
import cv2
import os
import threading
import queue
import torch
import logging
import time


class DepthVolumeCalculator:
    def __init__(self, model_path, roi_points, brightness_increase, cls_name_color):
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
        self.config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
        self.align = None
        self.save_depth = None
        self.roi_points = roi_points
        self.brightness_increase = brightness_increase
        self.cls_name_color = cls_name_color
        self.model_path = model_path
        self.model_name = os.path.basename(model_path)
        self.running = True  # 스레드 실행 제어 플래그

        # 큐 생성
        self.frame_queue = queue.Queue(maxsize=5)
        self.result_queue = queue.Queue(maxsize=5)

        try:
            self.model = YOLO(model_path)
            self.model.to('cuda' if torch.cuda.is_available() else 'cpu')
            print(f"YOLO model '{self.model_name}' loaded successfully.")
        except Exception as e:
            print(f"Error loading YOLO model: {e}")
            exit(1)

    def initialize_camera(self):
        try:
            self.pipeline.start(self.config)
            self.align = rs.align(rs.stream.color)
            print("Camera initialized successfully.")
        except Exception as e:
            print(f"Failed to initialize camera: {e}")
            exit(1)

    def capture_frames(self):
        """RealSense 카메라에서 프레임 캡처 및 큐에 저장"""
        while self.running:
            try:
                frames = self.pipeline.wait_for_frames()
                aligned_frames = self.align.process(frames)
                depth_frame = aligned_frames.get_depth_frame()
                color_frame = aligned_frames.get_color_frame()
                if depth_frame and color_frame:
                    self.frame_queue.put((depth_frame, color_frame), timeout=1)
            except queue.Full:
                pass  # 큐가 가득 차면 다음 프레임을 기다림
            except Exception as e:
                print(f"Error capturing frames: {e}")

    def process_frames(self):
        """YOLO 모델로 추론 수행"""
        while self.running:
            try:
                if not self.frame_queue.empty():
                    depth_frame, color_frame = self.frame_queue.get(timeout=1)
                    depth_image = np.asanyarray(depth_frame.get_data())
                    color_image = np.asanyarray(color_frame.get_data())
                    cropped_depth = self.apply_roi(depth_image)
                    roi_color = self.apply_roi(color_image)

                    brightened_image = cv2.convertScaleAbs(roi_color, alpha=1.2, beta=self.brightness_increase)
                    results = self.model(brightened_image)
                    self.result_queue.put((cropped_depth, roi_color, results), timeout=1)
            except queue.Empty:
                pass  # 큐가 비었으면 다음 작업을 기다림
            except Exception as e:
                print(f"Error processing frames: {e}")

    def display_results(self):
        """결과를 시각화 및 출력"""
        while self.running:
            try:
                if not self.result_queue.empty():
                    cropped_depth, roi_color, results = self.result_queue.get(timeout=1)
                    for result in results:
                        if result.masks is not None:
                            masks = result.masks.data.cpu().numpy()
                            classes = result.boxes.cls.cpu().numpy()
                            for i, mask in enumerate(masks):
                                conf = result.boxes.conf[i]
                                color_mask = (mask > 0.5).astype(np.uint8)
                                class_key = self.model.names[int(classes[i])]
                                object_name, color = self.cls_name_color.get(class_key, ("Unknown", (255, 255, 255)))

                                # 결과 시각화
                                cv2.putText(roi_color, f"{object_name} {conf:.2f}", (10, 50 + i * 30),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                    cv2.imshow('Results', roi_color)
                    if cv2.waitKey(1) & 0xFF == 27:  # ESC 키로 종료
                        self.running = False
            except queue.Empty:
                pass
            except Exception as e:
                print(f"Error displaying results: {e}")

    def apply_roi(self, image):
        x1, y1 = self.roi_points[0]
        x2, y2 = self.roi_points[1]
        return image[y1:y2, x1:x2]

    def start(self):
        """스레드 시작 및 관리"""
        self.initialize_camera()

        capture_thread = threading.Thread(target=self.capture_frames)
        process_thread = threading.Thread(target=self.process_frames)
        display_thread = threading.Thread(target=self.display_results)

        capture_thread.start()
        process_thread.start()
        display_thread.start()

        capture_thread.join()
        process_thread.join()
        display_thread.join()

        self.pipeline.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    logging.getLogger("ultralytics").setLevel(logging.WARNING)

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


    MODEL_DIR = os.path.join(os.getcwd(), 'model')

    model_list = ['1st_0org_100scale_1000mix_200_32_a100.pt', 
                  '1st_100org_0scale_0mix_500_32_2080.pt', 
                  '1st_100org_0scale_1000mix_200_96_a1002.pt', 
                  '1st_100org_0scale_8000mix_200_96_a1002.pt', 
                  '1st_100org_50scale_0mix_500_32_a100.pt', 
                  '1st_100org_50scale_1000mix_500_32_a100.pt', 
                  '1st_50org_100scale_1000mix_blur_200_32_a100.pt', 
                  '1st_50org_100scale_1000mix_sharp_200_32_a100.pt'
                  ]

    model_name = model_list[5]
    MODEL_PATH = os.path.join(MODEL_DIR, model_name)
    ROI_POINTS = [(175, 50), (1055, 690)]
    BRIGHTNESS_INCREASE = 10

    calculator = DepthVolumeCalculator(MODEL_PATH, ROI_POINTS, BRIGHTNESS_INCREASE, CLS_NAME_COLOR)
    calculator.start()
