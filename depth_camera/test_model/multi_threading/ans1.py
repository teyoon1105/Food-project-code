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
        self.current_detected_objects = []
        self.roi_points = roi_points
        self.brightness_increase = brightness_increase
        self.cls_name_color = cls_name_color
        self.model_path = model_path
        self.model_name = os.path.basename(model_path)
        self.running = True  # 스레드 실행 제어 플래그
        self.results = None  # 초기화

        # 큐 생성
        self.frame_queue = queue.Queue(maxsize=100)
        self.result_queue = queue.Queue(maxsize=50)

        try:
            self.model = YOLO(model_path)
            self.model.to('cuda' if torch.cuda.is_available() else 'cpu')
            print(f"YOLO model '{self.model_name}' loaded successfully.")
        except Exception as e:
            print(f"Error loading YOLO model: {e}")
            exit(1)

    def initialize_camera(self):
        """카메라 초기화"""
        try:
            self.pipeline.start(self.config)
            self.align = rs.align(rs.stream.color)
            print("Camera initialized successfully.")

            if os.path.exists('save_depth.npy'):
                self.save_depth = np.load('save_depth.npy')
                print("Loaded saved depth data from file.")
            else:
                print("No saved depth data found. Please save depth data first.")
            
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
                    # 프레임 전처리 (상하좌우 반전, ROI, 밝기 증가)
                    depth_image = np.asanyarray(depth_frame.get_data())
                    color_image = np.asanyarray(color_frame.get_data())
                    depth_image = cv2.flip(depth_image, -1)
                    color_image = cv2.flip(color_image, -1)

                    # 밝기 증가 처리
                    roi_color = self.apply_roi(color_image)
                    brightened_image = cv2.convertScaleAbs(roi_color, alpha=1.2, beta=self.brightness_increase)

                    # ROI 적용 후 큐에 저장
                    cropped_depth = self.apply_roi(depth_image)
                    if self.frame_queue.full():
                        _ = self.frame_queue.get_nowait()  # 가장 오래된 프레임 삭제
                    self.frame_queue.put((cropped_depth, brightened_image), timeout=1)

                    cv2.imshow('Input ROI', brightened_image)
                    cv2.waitKey(1)
            
            except Exception as e:
                print(f"Error capturing frames: {e}")
                self.running = False  # 스레드 종료 플래그 설정

    def process_frames(self):
        """YOLO 모델로 추론 수행"""
        while self.running:
            try:

                if not self.frame_queue.empty():
                    cropped_depth, brightened_image = self.frame_queue.get(timeout=1)
                    

                    # ROI 내 픽셀 평균값 계산 (간단한 객체 감지)
                    roi_mean = np.mean(brightened_image)
                    if roi_mean < 50:  # 임계값 설정 (조정 가능)
                        print(f"Skipping frame: ROI mean {roi_mean:.2f} below threshold.")
                        continue
                
                    # YOLO 모델 추론
                    print("Running YOLO inference...")
                    results = self.model(brightened_image)  # 모델 추론
                    print("YOLO inference completed.")

                    # 결과 큐에 저장
                    if not self.result_queue.full():
                        self.result_queue.put((cropped_depth, brightened_image, results), timeout=1)
                        print(f"Result added to queue. Result queue size: {self.result_queue.qsize()}")
                    else:
                        print("Result queue is full. Skipping this result.")
            except queue.Empty:
                print("Frame queue is empty. Skipping inference.")
            except Exception as e:
                print(f"Error processing frames: {e}")
                self.running = False  # 스레드 종료 플래그 설정

    def display_results(self):
        """결과를 시각화 및 출력"""
        while self.running:
            try:
                if not self.result_queue.empty():
                    print(f"Processing result. Remaining queue size: {self.result_queue.qsize()}")
                    cropped_depth, brightened_image, results = self.result_queue.get(timeout=1)
                    detected_objects_in_frame = []
                    blended_image = brightened_image.copy()

                    # depth_intrin 값을 직접 가져오기
                    depth_intrin = self.pipeline.wait_for_frames().get_depth_frame().profile.as_video_stream_profile().intrinsics

                    for result in results:
                        if result.masks is not None:
                            masks = result.masks.data.cpu().numpy()
                            classes = result.boxes.cls.cpu().numpy()

                            for i, mask in enumerate(masks):
                                conf = result.boxes.conf[i]
                                resized_mask = cv2.resize(mask, (blended_image.shape[1], blended_image.shape[0]))
                                color_mask = (resized_mask > 0.5).astype(np.uint8)
                                class_key = self.model.names[int(classes[i])]
                                object_name, color = self.cls_name_color.get(class_key, ("Unknown", (255, 255, 255)))

                                # 객체 이름만 저장
                                if object_name not in detected_objects_in_frame:
                                    detected_objects_in_frame.append(object_name)

                                # 마스크 좌표 가져오기
                                mask_indices = np.where(color_mask > 0)

                                # 부피 계산 및 마스크 시각화
                                volume = self.calculate_volume(cropped_depth, mask_indices, depth_intrin, min_depth_cm=40)
                                blended_image = self.visualize_results(brightened_image, object_name, volume, conf, color, mask_indices, blended_image)

                    # 결과 이미지 표시
                    cv2.imshow('Detection Results', blended_image)

                    # 현재 프레임에서 탐지된 객체와 기존 리스트 비교
                    for obj in detected_objects_in_frame:
                        if obj not in self.current_detected_objects:  # 새로운 객체만 추가
                            print(f"New object detected: {obj}")
                            self.current_detected_objects.append(obj)

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

    def calculate_volume(self, cropped_depth, mask_indices, depth_intrin, min_depth_cm=20):
        """부피 계산"""
        total_volume = 0
        y_indices, x_indices = mask_indices
        for y, x in zip(y_indices, x_indices):
            z_cm = cropped_depth[y, x] / 10
            base_depth_cm = self.save_depth[y, x] / 10
            if z_cm > min_depth_cm and base_depth_cm > 25:
                height_cm = max(0, base_depth_cm - z_cm)
                pixel_area_cm2 = (z_cm ** 2) / (depth_intrin.fx * depth_intrin.fy)
                total_volume += pixel_area_cm2 * height_cm
        return total_volume

    def visualize_results(self, cropped_image, object_name, volume, conf, color, mask_indices, blended_image):
        """탐지된 객체 시각화"""
        y_indices, x_indices = mask_indices
        color_filled_image = np.zeros_like(cropped_image)
        color_filled_image[y_indices, x_indices] = color

        # putText 전처리
        y_indices, x_indices = mask_indices
        min_y, min_x = np.min(y_indices), np.min(x_indices)
        max_y, max_x = np.max(y_indices), np.max(x_indices)
        mask_height = max_y - min_y
        mask_width =  max_x - min_x
        spacing = 10

        # 텍스트 표시
        texts = [
            f"{object_name}",
            f"V:{volume:.0f}cm^3",
            f"C:{conf:.2f}"
        ]
        text_sizes = []
        total_height = 0
        
        for text in texts:
            (text_width, text_height), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            text_sizes.append((text_width, text_height))
            total_height += text_height + spacing

        # 첫 텍스트의 y 좌표 계산 (전체 텍스트 블록의 시작점)
        y = (mask_height - total_height) // 2 + min_y
        
        # 각 줄 그리기
        for text, (text_width, text_height) in zip(texts, text_sizes):
            x = (mask_width - text_width) // 2 + min_x
            y += text_height  
            cv2.putText(color_filled_image, text, (x,y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color=(255,255,255), thickness = 3)
            cv2.putText(color_filled_image, text, (x,y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color=(0,0,0), thickness = 2)
            y += spacing  # 줄 간격 추가

        return cv2.addWeighted(blended_image, 1, color_filled_image, 0.5, 0) # 이미지 합성
    
    def monitor_keyboard(self):
        """키보드 입력을 모니터링하여 ESC 입력 시 종료"""
        while self.running:
            key = cv2.waitKey(10) & 0xFF  # ESC 키 확인
            if key == 27:  # ESC 키
                print("ESC key pressed. Stopping threads.")
                self.running = False
                break

    def start(self):
        """스레드 시작 및 관리"""
        self.initialize_camera()

        # 1. Capture thread 먼저 시작
        capture_thread = threading.Thread(target=self.capture_frames)
        capture_thread.start()

        # 2. frame_queue에 데이터가 쌓일 때까지 대기
        while self.frame_queue.empty():
            print("Waiting for frames to be captured...")
            time.sleep(0.1)

        # 3. Process thread 시작
        process_thread = threading.Thread(target=self.process_frames)
        process_thread.start()

        # 4. result_queue에 데이터가 쌓일 때까지 대기
        while self.result_queue.empty():
            print("Waiting for results to be processed...")
            time.sleep(0.1)

        # 5. Display thread 시작
        display_thread = threading.Thread(target=self.display_results)
        display_thread.start()

        # 스레드 모니터링
        while self.running:
            print(f"Frame queue size: {self.frame_queue.qsize()}, Result queue size: {self.result_queue.qsize()}")
            time.sleep(1)
            
        # 메인 스레드에서 키보드 모니터링
        self.monitor_keyboard()

        # 모든 스레드 종료 후 정리
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
    '11013002': ('Gosari', (255, 255, 0)),
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
                  '1st_50org_100scale_1000mix_sharp_200_32_a100.pt', 
                  'total_0org_100scale_10000mix_200_32_a100_best.pt', 
                  'total_50org_100scale_10000mix_200_32_a100_best.pt']

    model_name = model_list[-1]

    MODEL_PATH = os.path.join(MODEL_DIR, model_name)
    ROI_POINTS = [(175, 50), (1055, 690)]
    BRIGHTNESS_INCREASE = 10

    calculator = DepthVolumeCalculator(MODEL_PATH, ROI_POINTS, BRIGHTNESS_INCREASE, CLS_NAME_COLOR)
    calculator.start()


