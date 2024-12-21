import pyrealsense2 as rs
import numpy as np
import cv2
import os
from ultralytics import YOLO
import torch
import logging
import torch.nn.functional as F


class DepthVolumeCalculator:
    def __init__(self, model_path, roi_points, cls_name_color):
        """
        초기화
        """
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
        self.config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
        self.align = None
        self.save_depth = None
        self.roi_points = roi_points
        self.cls_name_color = cls_name_color
        # 새로운 객체 확인용
        self.detected_names = set()  # 최종적으로 확인된 객체
        self.candidate_objects = {}  # 이름과 프레임 수 기록
        self.model_name = os.path.basename(model_path)  # 모델 이름만 추출
        
        # YOLO 모델 로드 및 예외처리
        try:
            self.model = YOLO(model_path) # YOLO 모델 초기화
            self.model.to('cuda' if torch.cuda.is_available() else 'cpu') # GPU 이용이 가능하면 GPU 사용
            print(f"YOLO model '{self.model_name}' loaded successfully.") # 연결 확인
        except Exception as e:
            print(f"Error loading YOLO model: {e}") 
            exit(1) # 모델 로드 실패시 종료(출력을 보고 모델 로드 실패 확인)

    def initialize_camera(self):
        """카메라 초기화"""
        self.pipeline.start(self.config)
        self.align = rs.align(rs.stream.color)
        print("Camera initialized.")

    def capture_frames(self):
        """정렬된 깊이 및 컬러 프레임 반환"""
        frames = self.pipeline.wait_for_frames()
        aligned_frames = self.align.process(frames)
        depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()
        return depth_frame, color_frame

    def preprocess_images(self, depth_frame, color_frame):
        """깊이 및 컬러 프레임 전처리"""
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())
        return cv2.flip(depth_image, -1), cv2.flip(color_image, -1)

    def apply_roi(self, image):
        """ROI 적용"""
        x1, y1 = self.roi_points[0]
        x2, y2 = self.roi_points[1]
        return image[y1:y2, x1:x2]
    

    def calculate_volume_on_gpu(self, cropped_depth, depth_intrin, mask_indices):
        """
        GPU를 사용한 부피 계산
        """
        # NumPy 데이터를 GPU로 전송
        depth_tensor = torch.tensor(cropped_depth, device='cuda', dtype=torch.float32)
        mask_y, mask_x = mask_indices
        mask_tensor = (torch.tensor(mask_y, device='cuda'), torch.tensor(mask_x, device='cuda'))

        # 저장된 깊이 데이터도 GPU로 전송
        saved_depth_tensor = torch.tensor(self.save_depth, device='cuda', dtype=torch.float32)

        # 마스크 좌표를 기반으로 높이 계산
        z_cm = depth_tensor[mask_tensor] / 10.0  # ROI 깊이 (cm)
        base_depth_cm = saved_depth_tensor[mask_tensor] / 10.0  # 기준 깊이 (cm)

        height_cm = torch.clamp(base_depth_cm - z_cm, min=0)  # 높이 계산
        pixel_area_cm2 = (z_cm ** 2) / (depth_intrin.fx * depth_intrin.fy)  # 픽셀 면적 (cm^2)
        volume = torch.sum(height_cm * pixel_area_cm2).item()  # 부피 계산 및 반환

        return volume

    def visualize_results(self, image, object_name, volume, conf, color, mask_indices):
        """결과 시각화"""
        mask_y, mask_x = mask_indices
        image[mask_y, mask_x] = (image[mask_y, mask_x] * 0.5 + np.array(color) * 0.5).astype(np.uint8)
        text = f"{object_name}: {volume:.1f}cm^3 ({conf:.2f})"
        cv2.putText(image, text, (mask_x.min(), mask_y.min() - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    def main_loop(self):
        """메인 처리 루프"""
        self.initialize_camera()

        if os.path.exists('save_depth.npy'):
            self.save_depth = np.load('save_depth.npy')
            print("Loaded saved depth data.")
        else:
            print("No saved depth data found. Please save depth data.")

        try:
            while True:
                depth_frame, color_frame = self.capture_frames()
                if not depth_frame or not color_frame:
                    continue

                # 내장 파라미터에서 초점거리 가져오기           
                depth_intrin = depth_frame.profile.as_video_stream_profile().intrinsics

                depth_image, color_image = self.preprocess_images(depth_frame, color_frame)
                cropped_depth = self.apply_roi(depth_image)
                cropped_color = self.apply_roi(color_image)

                results = self.model(cropped_color)

                current_frame_objects = set()  # 현재 프레임에서 탐지된 객체 이름

                for result in results:
                    if result.masks is not None:
                        masks = result.masks.data  # GPU 텐서로 유지
                        original_size = (cropped_color.shape[0], cropped_color.shape[1])  # 원본 크기 (높이, 너비)

                        # PyTorch를 사용한 리사이즈
                        resized_masks = F.interpolate(masks.unsqueeze(1), size=original_size, mode='bilinear', align_corners=False)
                        resized_masks = resized_masks.squeeze(1)  # 채널 축 제거
                        classes = result.boxes.cls  # GPU 텐서
                        confs = result.boxes.conf  # GPU 텐서

                        for i, mask in enumerate(resized_masks):
                            class_idx = int(classes[i].item())  # 클래스 인덱스
                            object_name, color = self.cls_name_color.get(
                                self.model.names[class_idx], ("Unknown", (255, 255, 255)))
                            conf = confs[i].item()  # 신뢰도
                            
                            # 현재 프레임에서 탐지된 객체 이름 추가
                            current_frame_objects.add(object_name)

                            # 마스크 좌표 추출 (GPU에서 바로 연산 가능)
                            mask_indices = torch.where(mask > 0.5)
                            mask_y = mask_indices[0].cpu().numpy()  # y 좌표
                            mask_x = mask_indices[1].cpu().numpy()  # x 좌표

                            
                            # 부피 계산 수행 (GPU에서 진행)
                            volume = self.calculate_volume_on_gpu(cropped_depth, depth_intrin, (mask_y, mask_x))

                            # 시각화 처리
                            self.visualize_results(cropped_color, object_name, volume, conf, color, (mask_y, mask_x))

                # 후보 객체 딕셔너리 업데이트
                for obj_name in current_frame_objects:
                    if obj_name in self.candidate_objects:
                        self.candidate_objects[obj_name] += 1
                    else:
                        self.candidate_objects[obj_name] = 1

                    # 15프레임 이상 탐지된 경우, 최종 객체로 등록
                    if self.candidate_objects[obj_name] >= 15 and obj_name not in self.detected_names:
                        print(f"New object confirmed: {obj_name}")
                        self.detected_names.add(obj_name)

                # 탐지되지 않은 객체는 후보에서 제거
                to_remove = [obj_name for obj_name in self.candidate_objects if obj_name not in current_frame_objects]
                for obj_name in to_remove:
                    del self.candidate_objects[obj_name]

                cv2.imshow("Results", cropped_color)
                key = cv2.waitKey(1)
                if key == 27:
                    break
                elif key == ord('s'):
                    self.save_depth = cropped_depth
                    np.save('save_depth.npy', self.save_depth)
                    print("Depth saved.")
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



    MODEL_PATH = os.path.join(os.getcwd(), 'model', "large_epoch300.pt")
    ROI_POINTS = [(175, 50), (1055, 690)]

    calculator = DepthVolumeCalculator(MODEL_PATH, ROI_POINTS, CLS_NAME_COLOR)
    calculator.main_loop()
