import pyrealsense2 as rs
import numpy as np
import cv2
import os
from ultralytics import YOLO
import torch
import logging


class DepthVolumeCalculator:
    def __init__(self, model_path, roi_points, cls_name_color):
        """
        DepthVolumeCalculator 클래스 초기화
        :param model_path: YOLO 모델 파일 경로
        :param roi_points: 관심 영역(ROI) 좌표 [(x1, y1), (x2, y2)]
        :param cls_name_color: 클래스 ID와 이름, 색상 매핑 딕셔너리
        """
        self.pipeline = rs.pipeline()  # RealSense 파이프라인
        self.config = rs.config()  # 설정 객체
        self.config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)  # 깊이 스트림 설정
        self.config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)  # 컬러 스트림 설정
        self.align = None  # 깊이 스트림을 컬러 스트림에 정렬하는 Align 객체
        self.save_depth = None  # 기준 깊이 데이터를 저장할 변수
        self.roi_points = roi_points  # 관심 영역 좌표
        self.cls_name_color = cls_name_color  # 클래스와 색상 매핑 정보 딕셔너리
        self.model_path = model_path  # 모델 경로
        self.model_name = os.path.basename(model_path)  # 모델 이름만 추출
        self.fixed_objects = {}  # 이미 고정된 객체의 정보를 저장하는 캐시
        self.current_processing_object = None  # 현재 연산 중인 객체 이름
        self.fixed_objects = {}  # FIX된 객체를 저장
        
        # YOLO 모델 로드 및 예외 처리
        try:
            self.model = YOLO(model_path)  # YOLO 모델 초기화
            self.model.to('cuda' if torch.cuda.is_available() else 'cpu')  # GPU 사용 가능 여부 확인
            print(f"YOLO model '{self.model_name}' loaded successfully.")
        except Exception as e:
            print(f"Error loading YOLO model: {e}")
            exit(1)  # 모델 로드 실패 시 종료

    def initialize_camera(self):
        """카메라 초기화"""
        try:
            self.pipeline.start(self.config)
            self.align = rs.align(rs.stream.color)  # 컬러 프레임에 맞게 깊이 프레임 정렬
            print("Camera initialized successfully.")
        except Exception as e:
            print(f"Failed to initialize camera: {e}")
            exit(1)

    def capture_frames(self):
        """카메라에서 정렬된 프레임 캡처"""
        try:
            frames = self.pipeline.wait_for_frames()
            aligned_frames = self.align.process(frames)
            return aligned_frames.get_depth_frame(), aligned_frames.get_color_frame()
        except Exception as e:
            print(f"Error capturing frames: {e}")
            return None, None

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
    
    
    def process_and_fix_objects(self, results, cropped_depth, depth_intrin, blended_image):
        """
        객체를 처리하고 새로운 객체가 일정 프레임 동안 감지되면 이전 객체를 고정 (FIX)
        """

        # 결과가 존재하는 경우에만 진행
        if len(results[0].boxes) > 0:
            try:
                # 객체 이름 가져오기
                cls_id = int(results[0].boxes.cls[0])  # 첫 번째 객체만 처리
                object_name = self.model.names[cls_id]

                # 새로운 객체 감지 시 연속 카운트 초기화
                if self.current_processing_object is None or self.current_processing_object != object_name:
                    print(f"New object detected: {object_name}, resetting count.")
                    self.current_processing_object = object_name
                    self.current_object_count = 0  # 연속 감지 프레임 수 초기화

                # 마스크 및 부피 계산 (항상 실행)
                mask = results[0].masks.data.cpu().numpy()[0]
                resized_mask = cv2.resize(mask, (cropped_depth.shape[1], cropped_depth.shape[0]))
                mask_indices = np.where(resized_mask > 0.5)

                # 부피 계산
                total_volume = self.compute_volume(cropped_depth, mask_indices, depth_intrin)

                # 연속 프레임 카운트 증가
                self.current_object_count += 1
                print(f"{object_name} detected for {self.current_object_count} frames - Volume: {total_volume:.0f} cm^3")

                # 30프레임 이상 감지 시 FIX 수행
                if self.current_object_count >= 30:
                    print(f"Fixing object: {object_name}")
                    self.fixed_objects[object_name] = {
                        "volume": total_volume,
                        "mask_indices": mask_indices,
                        "color": self.cls_name_color.get(object_name, ("Unknown", (255, 255, 255)))[1],
                        "conf": results[0].boxes.conf[0].item()
                    }
                    self.current_processing_object = None  # 현재 객체 처리 완료
                    self.current_object_count = 0  # 카운터 초기화

                # 현재 객체를 화면에 시각화 (연산 중에도 표시)
                blended_image = self.visualize_results(
                    blended_image,
                    object_name,
                    total_volume,  # FIX 전에도 실시간 부피 계산 값 표시
                    results[0].boxes.conf[0].item(),
                    self.cls_name_color.get(object_name, ("Unknown", (255, 255, 255)))[1],
                    mask_indices
                )

            except Exception as e:
                print(f"Error processing object: {e}")

        # FIX된 객체 표시
        for object_name, fixed_data in self.fixed_objects.items():
            blended_image = self.visualize_results(
                blended_image,
                object_name,
                fixed_data["volume"],  # FIX된 부피 값 사용
                fixed_data["conf"],
                fixed_data["color"],
                fixed_data["mask_indices"]
            )

        return blended_image

    def compute_volume(self, cropped_depth, mask_indices, depth_intrin):
        """
        부피 계산 로직
        :param cropped_depth: ROI로 크롭된 깊이 이미지
        :param mask_indices: 객체의 마스크 인덱스
        :param depth_intrin: 카메라 내부 파라미터
        :return: 계산된 부피 값
        """
        total_volume = 0
        y_indices, x_indices = mask_indices
        for y, x in zip(y_indices, x_indices):
            z_cm = cropped_depth[y, x] / 10  # 깊이를 cm로 변환
            base_depth_cm = self.save_depth[y, x] / 10  # 기준 깊이
            if z_cm > 20 and base_depth_cm > 25:  # 유효 범위 필터링
                height_cm = max(0, base_depth_cm - z_cm)  # 높이 계산
                pixel_area_cm2 = (z_cm ** 2) / (depth_intrin.fx * depth_intrin.fy)  # 픽셀 면적 계산
                total_volume += pixel_area_cm2 * height_cm  # 부피 누적
        return total_volume
    
    def process_and_fix_objects(self, results, cropped_depth, depth_intrin, blended_image):
        """
        객체를 처리하고 새로운 객체가 일정 프레임 동안 감지되면 FIX하며,
        이미 FIX된 객체는 항상 화면에 표시.
        """
        # 현재 감지된 객체 이름 저장 (동시 처리 가능)
        currently_detected_objects = set()
    
        # 결과가 존재하는 경우에만 진행
        if len(results[0].boxes) > 0:
            try:
                # YOLO 결과 순회: 모든 객체 처리
                for idx in range(len(results[0].boxes)):
                    cls_id = int(results[0].boxes.cls[idx])  # 객체 클래스 ID
                    object_name = self.model.names[cls_id]  # 객체 이름
                    currently_detected_objects.add(object_name)

                    # FIX된 객체는 다시 처리하지 않고 시각화만 수행
                    if object_name in self.fixed_objects:
                        fixed_data = self.fixed_objects[object_name]
                        blended_image = self.visualize_results(
                            blended_image,
                            object_name,
                            fixed_data["volume"],
                            fixed_data["conf"],
                            fixed_data["color"],
                            fixed_data["mask_indices"]
                        )
                        continue

                    # 새로운 객체 감지 시 연속 카운트 초기화
                    if self.current_processing_object is None or self.current_processing_object != object_name:
                        print(f"New object detected: {object_name}, resetting count.")
                        self.current_processing_object = object_name
                        self.current_object_count = 0  # 연속 감지 프레임 수 초기화
                        self.weight_list = []
                    
                    

                    # 마스크 및 부피 계산 (항상 실행)
                    mask = results[0].masks.data.cpu().numpy()[idx]
                    resized_mask = cv2.resize(mask, (cropped_depth.shape[1], cropped_depth.shape[0]))
                    mask_indices = np.where(resized_mask > 0.5)

                    # 부피 계산
                    total_volume = self.compute_volume(cropped_depth, mask_indices, depth_intrin)

                    # 연속 프레임 카운트 증가
                    self.current_object_count += 1
                    # print(f"{object_name} detected for {self.current_object_count} frames - Volume: {total_volume:.0f} cm^3")

                    # 30프레임 이상 감지 시 FIX 수행
                    if self.current_object_count >= 30:
                        print(f"Fixing object: {object_name}")
                        self.fixed_objects[object_name] = {
                            "volume": total_volume,
                            "mask_indices": mask_indices,
                            "color": self.cls_name_color.get(object_name, ("Unknown", (255, 255, 255)))[1],
                            "conf": results[0].boxes.conf[idx].item()
                        }
                        self.current_processing_object = None  # 현재 객체 처리 완료
                        self.current_object_count = 0  # 카운터 초기화

                    # 현재 객체를 화면에 시각화 (연산 중에도 표시)
                    blended_image = self.visualize_results(
                        blended_image,
                        object_name,
                        total_volume,  # FIX 전에도 실시간 부피 계산 값 표시
                        results[0].boxes.conf[idx].item(),
                        self.cls_name_color.get(object_name, ("Unknown", (255, 255, 255)))[1],
                        mask_indices
                    )

            except Exception as e:
                print(f"Error processing object: {e}")

        # FIX된 객체는 항상 화면에 시각화
        for object_name, fixed_data in self.fixed_objects.items():
            if object_name not in currently_detected_objects:  # FIX된 객체만 추가 시각화
                blended_image = self.visualize_results(
                    blended_image,
                    object_name,
                    fixed_data["volume"],  # FIX된 부피 값 사용
                    fixed_data["conf"],
                    fixed_data["color"],
                    fixed_data["mask_indices"]
                )

        return blended_image

    def visualize_results(self, blended_image, object_name, volume, conf, color, mask_indices):
        """탐지 결과를 시각화"""
        y_indices, x_indices = mask_indices
        color_filled_image = np.zeros_like(blended_image)
        color_filled_image[y_indices, x_indices] = color
        blended_image = cv2.addWeighted(blended_image, 1, color_filled_image, 0.5, 0)

        # 텍스트 추가
        text = f"{object_name}, V:{volume:.0f}cm^3, C:{conf:.2f}"
        cv2.putText(blended_image, text, (x_indices.min(), y_indices.min()), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        return blended_image


    def main_loop(self):
        """메인 처리 루프"""
        self.initialize_camera()

        if os.path.exists("save_depth.npy"):
            self.save_depth = np.load("save_depth.npy")
            print("Loaded saved depth data.")
        else:
            print("No saved depth data found. Please save depth data.")

        try:
            while True:
                depth_frame, color_frame = self.capture_frames()
                if not depth_frame or not color_frame:
                    continue

                depth_image, color_image = self.preprocess_images(depth_frame, color_frame)
                cropped_depth = self.apply_roi(depth_image)
                roi_color = self.apply_roi(color_image)
                blended_image = roi_color.copy()

                results = self.model(roi_color)
                depth_intrin = depth_frame.profile.as_video_stream_profile().intrinsics


                # 객체 처리 및 고정
                blended_image = self.process_and_fix_objects(results, cropped_depth, depth_intrin, blended_image)
                
            

                cv2.imshow("Segmented Mask with Heights", blended_image)
                key = cv2.waitKey(10) & 0xFF
                if key == 27:  # ESC 키
                    break
                elif key == ord("s"):  # 깊이 데이터 저장
                    self.save_depth = cropped_depth.copy()
                    np.save("save_depth.npy", self.save_depth)
                    print("Depth data saved!")

        except Exception as e:
            print(f"Unexpected error: {e}")
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

    model_name = model_list[-1]
    MODEL_PATH = os.path.join(MODEL_DIR, model_name)
    
    ROI_POINTS = [(175, 50), (1055, 690)]

    calculator = DepthVolumeCalculator(MODEL_PATH, ROI_POINTS, CLS_NAME_COLOR)
    calculator.main_loop()
