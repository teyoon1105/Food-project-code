import pyrealsense2 as rs
import numpy as np
import cv2
import os
from ultralytics import YOLO
import torch
import logging


class DepthVolumeCalculator:
    def __init__(self, model_path, roi_points, cls_name_color, persistence_threshold=5):
        """
        DepthVolumeCalculator 클래스 초기화
        """
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
        self.config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
        self.align = None
        self.save_depth = None
        self.roi_points = roi_points
        self.cls_name_color = cls_name_color
        self.model_path = model_path
        self.model_name = os.path.basename(model_path)

        # 객체 상태 관리 변수
        self.fixed_objects = {}  # FIX된 객체 저장
        self.current_object_name = None  # 현재 처리 중인 객체 이름
        self.current_object_data = {}  # 현재 객체의 데이터 (부피, 마스크 등)
        self.persistence_counter = 0  # 새로운 객체의 연속 감지 횟수
        self.persistence_threshold = persistence_threshold  # 객체 지속성 기준 프레임 수

        # YOLO 모델 로드
        try:
            self.model = YOLO(model_path)
            self.model.to('cuda' if torch.cuda.is_available() else 'cpu')
            print(f"YOLO model '{self.model_name}' loaded successfully.")
        except Exception as e:
            print(f"Error loading YOLO model: {e}")
            exit(1)

    def initialize_camera(self):
        """카메라 초기화"""
        self.pipeline.start(self.config)
        self.align = rs.align(rs.stream.color)
        print("Camera initialized successfully.")

    def capture_frames(self):
        """카메라에서 정렬된 프레임 캡처"""
        frames = self.pipeline.wait_for_frames()
        aligned_frames = self.align.process(frames)
        return aligned_frames.get_depth_frame(), aligned_frames.get_color_frame()

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

    def compute_volume(self, cropped_depth, mask_indices, depth_intrin):
        """부피 계산"""
        total_volume = 0
        y_indices, x_indices = mask_indices
        for y, x in zip(y_indices, x_indices):
            z_cm = cropped_depth[y, x] / 10  # 깊이 -> cm 변환
            base_depth_cm = self.save_depth[y, x] / 10
            if z_cm > 20 and base_depth_cm > 25:  # 유효 범위 필터링
                height_cm = max(0, base_depth_cm - z_cm)
                pixel_area_cm2 = (z_cm ** 2) / (depth_intrin.fx * depth_intrin.fy)
                total_volume += pixel_area_cm2 * height_cm
        return total_volume

    def process_objects(self, results, cropped_depth, depth_intrin, blended_image):
        """객체 처리 및 지속성 검사"""
        current_detected_objects = set()

        # 1. FIX된 객체 시각화
        for obj_name, obj_data in self.fixed_objects.items():
            blended_image = self.visualize_results(
                blended_image, obj_name, obj_data["volume"], obj_data["conf"], obj_data["color"], obj_data["mask_indices"]
            )

        # 2. 현재 프레임에서 객체 처리
        for i, box in enumerate(results[0].boxes):
            cls_id = int(box.cls[0])
            object_name = self.model.names[cls_id]
            current_detected_objects.add(object_name)

            # 객체 마스크 및 부피 계산
            mask = results[0].masks.data[i].cpu().numpy()
            resized_mask = cv2.resize(mask, (cropped_depth.shape[1], cropped_depth.shape[0]))
            mask_indices = np.where(resized_mask > 0.5)
            total_volume = self.compute_volume(cropped_depth, mask_indices, depth_intrin)
            confidence = box.conf[0].item()

            # 새로운 객체 감지 확인
            if self.current_object_name != object_name:
                # 새로운 객체 등장 시 카운터 초기화
                if self.persistence_counter == 0:
                    print(f"New object detected: {object_name}")
                self.persistence_counter += 1

                # 객체가 지속적으로 감지되면 이전 객체 FIX
                if self.persistence_counter >= self.persistence_threshold:
                    if self.current_object_name:
                        print(f"Fixing object: {self.current_object_name}")
                        self.fixed_objects[self.current_object_name] = self.current_object_data

                    # 새로운 객체를 현재 객체로 설정
                    print(f"Processing new object: {object_name}")
                    self.current_object_name = object_name
                    self.current_object_data = {
                        "volume": total_volume,
                        "conf": confidence,
                        "color": self.cls_name_color.get(object_name, (255, 255, 255))[1],
                        "mask_indices": mask_indices,
                    }
                    self.persistence_counter = 0  # 카운터 초기화
            else:
                # 현재 객체가 연속 감지되면 데이터 업데이트
                self.current_object_data = {
                    "volume": total_volume,
                    "conf": confidence,
                    "color": self.cls_name_color.get(object_name, (255, 255, 255))[1],
                    "mask_indices": mask_indices,
                }

            # 현재 객체 마스킹
            blended_image = self.visualize_results(
                blended_image, self.current_object_name, 
                self.current_object_data["volume"], 
                self.current_object_data["conf"],
                self.current_object_data["color"], 
                self.current_object_data["mask_indices"]
            )

        return blended_image


    def visualize_results(self, blended_image, object_name, volume, confidence, color, mask_indices):
        """탐지 결과 시각화"""
        y_indices, x_indices = mask_indices
        color_filled = np.zeros_like(blended_image)
        color_filled[y_indices, x_indices] = color
        blended_image = cv2.addWeighted(blended_image, 1, color_filled, 0.5, 0)

        text = f"{object_name}, V:{volume:.0f}cm^3, C:{confidence:.2f}"
        cv2.putText(blended_image, text, (x_indices.min(), y_indices.min()), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        return blended_image

    def main_loop(self):
        """메인 처리 루프"""
        self.initialize_camera()
        self.save_depth = np.load("save_depth.npy") if os.path.exists("save_depth.npy") else None
        print("Loaded saved depth data." if self.save_depth is not None else "No saved depth data found.")

        try:
            while True:
                depth_frame, color_frame = self.capture_frames()
                if not depth_frame or not color_frame:
                    continue

                depth_image, color_image = self.preprocess_images(depth_frame, color_frame)
                cropped_depth = self.apply_roi(depth_image)
                roi_color = self.apply_roi(color_image)
                results = self.model(roi_color)
                depth_intrin = depth_frame.profile.as_video_stream_profile().intrinsics

                blended_image = self.process_objects(results, cropped_depth, depth_intrin, roi_color.copy())

                cv2.imshow("Segmented Mask with Heights", blended_image)
                if cv2.waitKey(10) & 0xFF == 27:  # ESC 키 종료
                    break

        finally:
            self.pipeline.stop()
            cv2.destroyAllWindows()
if __name__ == "__main__":
    # 로그 레벨 설정 (INFO 메시지 비활성화)
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


    model_list = [
                '1st_0org_100scale_1000mix_200_32_a100.pt', 
                '1st_100org_0scale_0mix_500_32_2080.pt', 
                '1st_100org_0scale_1000mix_200_96_a1002.pt',
                '1st_100org_0scale_8000mix_200_96_a1002.pt', 
                '1st_100org_50scale_0mix_500_32_a100.pt',
                '1st_100org_50scale_1000mix_500_32_a100.pt',
                '1st_50org_100scale_1000mix_blur_200_32_a100.pt',
                '1st_50org_100scale_1000mix_sharp_200_32_a100.pt', 
                'total_0org_100scale_10000mix_200_32_a100_best.pt', 
                'total_50org_100scaled_10000mix_700_96_a1002_best.pt', 
                'total_50org_100scale_10000mix_200_32_a100_best.pt']   

    model_name = model_list[5]
    MODEL_PATH = os.path.join(MODEL_DIR, model_name)
    
    ROI_POINTS = [(175, 50), (1055, 690)]

    calculator = DepthVolumeCalculator(MODEL_PATH, ROI_POINTS, CLS_NAME_COLOR, persistence_threshold=5)
    calculator.main_loop()
