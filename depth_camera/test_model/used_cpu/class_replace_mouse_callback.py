
import pyrealsense2 as rs
import numpy as np
import cv2
import os
from ultralytics import YOLO
import torch
import logging


class DepthVolumeCalculator:
    def __init__(self, model_path, roi_points, brightness_increase, cls_name_color):
        """
        DepthVolumeCalculator 클래스 초기화
        :param model_path: YOLO 모델 파일 경로
        :param roi_points: 관심 영역(ROI) 좌표 [(x1, y1), (x2, y2)]
        :param brightness_increase: ROI 영역의 밝기 증가 값
        :param cls_name_color: 클래스 ID와 이름, 색상 매핑 딕셔너리
        """
        # 클래스 안의 인스턴스 변수 생성
        self.pipeline = rs.pipeline() # 파이프 라인 
        self.config = rs.config() # 설정 객체
        self.config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30) # 깊이 스트림 설정
        self.config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30) # 컬러 스트림 설정
        self.align = None  # 깊이 스트림을 컬러 스트림에 정렬하는 Align 객체
        self.save_depth = None  # 기준 깊이 데이터를 저장할 변수
        self.roi_points = roi_points # 관심 영역 좌표
        self.brightness_increase = brightness_increase # 밝기 증가 값
        self.cls_name_color = cls_name_color # 클래스와 색상 매핑 정보 딕셔너리
        self.model_path = model_path # 모델 경로
        self.model_name = os.path.basename(model_path)  # 모델 이름만 추출

        # YOLO 모델 로드 및 예외처리
        try:
            self.model = YOLO(model_path) # YOLO 모델 초기화
            self.model.to('cuda' if torch.cuda.is_available() else 'cpu') # GPU 이용이 가능하면 GPU 사용
            print(f"YOLO model '{self.model_name}' loaded successfully.") # 연결 확인
        except Exception as e:
            print(f"Error loading YOLO model: {e}") 
            exit(1) # 모델 로드 실패시 종료(출력을 보고 모델 로드 실패 확인)

    # realsense 카메라 로드 및 예외처리
    def initialize_camera(self):
        """카메라 초기화"""
        try:
            self.pipeline.start(self.config) # 파이프라인 시작
            self.align = rs.align(rs.stream.color) # 컬러 프레임을 정렬의 기준으로 설정
            print("Camera initialized successfully.")
        except Exception as e:
            print(f"Failed to initialize camera: {e}")
            exit(1) # 로드 실패 시 종료 및 원인 판단

    # 프레임 받고 프레임 정렬처리
    def capture_frames(self):
        """카메라에서 정렬된 프레임 캡처"""
        try:
            frames = self.pipeline.wait_for_frames() # 프레임 받아서 저장
            aligned_frames = self.align.process(frames) # 컬러 프레임에 맞게 깊이 프레임 정렬
            return aligned_frames.get_depth_frame(), aligned_frames.get_color_frame() # 깊이, 컬러 프레임 반환
        except Exception as e:
            print(f"Error capturing frames: {e}")
            return None, None # 실패시 None 반환 실패 원인 판단

    # 프레임 전처리
    def preprocess_images(self, depth_frame, color_frame):
        """
        깊이 및 컬러 프레임 전처리
        :param depth_frame: 깊이 데이터 프레임
        :param color_frame: 컬러 데이터 프레임
        :return: 상하좌우 반전된 깊이 및 컬러 이미지
        """
        depth_image = np.asanyarray(depth_frame.get_data()) # 깊이 데이터를 NumPy 배열로 변환
        color_image = np.asanyarray(color_frame.get_data()) # 컬러 데이터를 NumPy 배열로 변환
        return cv2.flip(depth_image, -1), cv2.flip(color_image, -1) # 반전된 깊이, 컬러 이미지 반환

    def apply_roi(self, image):
        """
        이미지에 관심 영역(ROI) 적용
        :param image: 입력 이미지
        :return: ROI로 크롭된 이미지
        """
        x1, y1 = self.roi_points[0]
        x2, y2 = self.roi_points[1]
        return image[y1:y2, x1:x2]

    def is_depth_saved(self):
        """깊이 데이터 저장 여부 확인"""
        return self.save_depth is not None

    # save_depth 없을 시 화면에 표기
    def display_message(self, image, message, position=(390, 370), color=(0, 0, 255)):
        """
        이미지에 메시지 표시
        :param image: 메시지를 표시할 이미지
        :param message: 표시할 메시지
        :param position: 메시지 위치 (기본값: 중앙)
        :param color: 메시지 색상 (기본값: 빨간색)
        """
        cv2.putText(image, message, position, cv2.FONT_HERSHEY_TRIPLEX, 1, color, 2)
        cv2.imshow('Color Image with ROI', image)
        cv2.waitKey(1)

    def calculate_volume(self, cropped_depth, mask_indices, depth_intrin, min_depth_cm=20):
        """
        객체의 부피 계산
        :param cropped_depth: ROI로 크롭된 깊이 이미지
        :param mask_indices: 객체 마스크 좌표
        :param depth_intrin: 깊이 카메라 내부 파라미터
        :param min_depth_cm: 최소 깊이 값(cm)
        :return: 객체의 부피(cm^3)
        """
        total_volume = 0 # 부피 초기화
        y_indices, x_indices = mask_indices # 마스크 좌표값

        for y, x in zip(y_indices, x_indices): # 각 좌표를 순회
            z_cm = cropped_depth[y, x] / 10 # 깊이 값을 cm로 변환
            base_depth_cm = self.save_depth[y, x] / 10 # 해당 좌표에 저장된 깊이

            if z_cm > min_depth_cm and base_depth_cm > 25: # 최소값 필터링
                height_cm = max(0, base_depth_cm - z_cm) # 높이 계산
                pixel_area_cm2 = (z_cm ** 2) / (depth_intrin.fx * depth_intrin.fy) # 픽셀의 면적 계산
                total_volume += pixel_area_cm2 * height_cm # 부피 계산

        return total_volume

    def visualize_results(self, cropped_image, mask, object_name, volume, conf, color, mask_indices):
        """
        탐지 결과를 시각화
        :param cropped_image: 크롭된 컬러 이미지
        :param mask: 객체 마스크
        :param object_name: 객체 이름
        :param volume: 객체 부피
        :param conf: 신뢰도 점수
        :param color: 객체 색상
        :param mask_indices: 객체 마스크 좌표
        :return: 결과 시각화 이미지
        """
        y_indices, x_indices = mask_indices
        min_x, min_y = np.min(x_indices), np.min(y_indices) # 마스크 좌표의 최소값
        label_position = (min_x, min_y - 10) # 라벨 위치 지정
        cv2.putText(mask, f"{object_name}: V:{volume:.0f}cm^3: C:{conf:.2f}", label_position,
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)  # 텍스트 추가
        return cv2.addWeighted(cropped_image, 0.85, mask, 0.15, 0) # 이미지 합성

    def main_loop(self):
        """메인 처리 루프"""
        self.initialize_camera()
        previous_detected_objects = []  # 이전 탐지된 객체를 저장하는 집합
        current_detected_objects = []  # 현재 탐지된 객체를 저장 (지속적으로 유지)
        try:
            while True:
                depth_frame, color_frame = self.capture_frames() # 깊이, 컬러 스트림 받아오기
                if not depth_frame or not color_frame: # 없으면 다음 반복
                    continue

                depth_image, color_image = self.preprocess_images(depth_frame, color_frame) # 깊이, 컬러 스트림 전처리
                cropped_depth = self.apply_roi(depth_image) # ROI 좌표에 맞게 자르기
                cropped_color = self.apply_roi(color_image) # ROI 좌표에 맞게 자르기

                # ROI 표시
                cv2.rectangle(color_image, self.roi_points[0], self.roi_points[1], (0, 0, 255), 2)
                cv2.imshow('Color Image with ROI', color_image)

                # 밝기 증가
                brightened_image = cv2.convertScaleAbs(cropped_color, alpha=1, beta=self.brightness_increase)
                results = self.model(brightened_image) # yolo에 ROI 영역, 밝기 증가시킨 이미지 입력
                all_colored_mask = np.zeros_like(brightened_image) # 마스크를 씌울 빈 넘파이 배열 생성
                blended_image = brightened_image.copy() # 마스크를 합성할 이미지 복사

                detected_objects_in_frame = []  # 현재 프레임에서 탐지된 객체만 임시 저장하기 위한 리스트

                for result in results:
                    if result.masks is not None:
                        # 탐지된 마스크들
                        masks = result.masks.data.cpu().numpy() 
                        classes = result.boxes.cls.cpu().numpy()
                        
                        # 탐지된 마스크를 순회하며 enumerate로 번호 매김
                        for i, mask in enumerate(masks):
                            conf = result.boxes.conf[i] # 해당 번호의 객체 conf 값 받아오기
                            resized_mask = cv2.resize(mask, (brightened_image.shape[1], brightened_image.shape[0])) # 결과값의 마스크는 원본 ROI 영역과 크기가 다름, 맞추기
                            color_mask = (resized_mask > 0.5).astype(np.uint8) # 넘파이 배열에서 0.5 이상이 되는 부분만 color mask로 배열 저장
                            class_key = self.model.names[int(classes[i])] # 객체 번호에 해당하는 객체 이름 가져오기
                            object_name, color = self.cls_name_color.get(class_key, ("Unknown", (255, 255, 255))) # 객체 이름을 키값으로 삼아 딕셔너리에서 이름과 컬러 가져오기

                            # 객체 이름만 저장
                            if object_name not in current_detected_objects:
                                detected_objects_in_frame.append(object_name)

                            # 컬러 마스크에 대항하는 마스크 좌표 가져오기
                            mask_indices = np.where(color_mask > 0)

                            # 저장된 깊이가 없다면 다음 반복
                            if not self.is_depth_saved():
                                self.display_message(color_image, "Save your depth first!")
                                continue
                            
                            # 마스크 좌표의 깊이 정보 확인
                            valid_depths = cropped_depth[mask_indices]
                            if len(valid_depths[valid_depths > 0]) == 0:
                                continue

                            # 내장 파라미터에서 초점거리 가져오기           
                            depth_intrin = depth_frame.profile.as_video_stream_profile().intrinsics
                            # 부피계산
                            volume = self.calculate_volume(cropped_depth, mask_indices, depth_intrin)
                            # 마스크 color로 채우기
                            all_colored_mask[color_mask == 1] = color
                            # 이미지와 마스크 합치지 및 후처리
                            blended_image = self.visualize_results(brightened_image, all_colored_mask, object_name,
                                                                   volume, conf, color, mask_indices)
                            
                # 현재 프레임에서 탐지된 객체와 기존 리스트 비교
                for obj in detected_objects_in_frame:
                    if obj not in current_detected_objects:  # 새로운 객체만 추가
                        print(f"New object detected: {obj}")
                        current_detected_objects.append(obj)
                        print(current_detected_objects)

                # 이전 상태 업데이트 (현재 프레임에서 유지되는 객체만 남김)
                previous_detected_objects = detected_objects_in_frame.copy()    

                # 결과 이미지 표시
                cv2.imshow('Segmented Mask with Heights', blended_image)

                key = cv2.waitKey(1) & 0xFF
                if key == 27:  # ESC 키
                    break

                elif key == ord('s'):  # 깊이 데이터 저장
                    self.save_depth = cropped_depth.copy()
                    print("Depth image saved!")

        except Exception as e:
            print(f"An unexpected error occurred in the main loop: {e}")

        finally:
            self.pipeline.stop()
            cv2.destroyAllWindows()


if __name__ == "__main__":
    # 로그 레벨 설정 (INFO 메시지 비활성화)
    logging.getLogger("ultralytics").setLevel(logging.WARNING)

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

    MODEL_DIR = os.path.join(os.getcwd(), 'model')

    # model pt file's list
    model_list = ['1st_0org_100scale_1000mix_200_32_a100.pt', 
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

    model_name = model_list[-2]
    MODEL_PATH = os.path.join(MODEL_DIR, model_name)
    
    ROI_POINTS = [(175, 50), (1055, 690)]
    BRIGHTNESS_INCREASE = 10

    calculator = DepthVolumeCalculator(MODEL_PATH, ROI_POINTS, BRIGHTNESS_INCREASE, CLS_NAME_COLOR)
    calculator.main_loop()
