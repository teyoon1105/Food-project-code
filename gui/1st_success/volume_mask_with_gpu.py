# 필요한 라이브러리 임포트
import pyrealsense2 as rs  # Intel RealSense 카메라 제어를 위한 라이브러리
import numpy as np  # 수치 연산을 위한 라이브러리
import cv2  # 컴퓨터 비전 처리를 위한 라이브러리
import os  # 파일/디렉토리 조작을 위한 라이브러리
from ultralytics import YOLO  # YOLO 객체 탐지 모델
import torch  # PyTorch 딥러닝 프레임워크
import logging  # 로깅을 위한 라이브러리
import torch.nn.functional as F  # PyTorch의 함수형 인터페이스


class DepthVolumeCalculator:
    def __init__(self, model_path, roi_points, cls_name_color):
        """
        초기화
        """

        logging.getLogger("ultralytics").setLevel(logging.WARNING)
        # RealSense 카메라 파이프라인 설정
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        
        # 카메라 스트림 설정: 1280x720 해상도, 30fps
        self.config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)  # 깊이 스트림
        self.config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)  # 컬러 스트림
        
        # 클래스 변수 초기화
        self.align = None  # 깊이와 컬러 프레임 정렬을 위한 변수
        self.save_depth = None  # 저장된 기준 깊이 데이터
        self.roi_points = roi_points  # 관심 영역(ROI) 좌표
        self.cls_name_color = cls_name_color  # 클래스별 이름과 색상 매핑
        self.detected_names = set()  # 확인된 객체 목록
        self.candidate_objects = {}  # 임시 감지된 객체와 프레임 카운트
        self.model_name = os.path.basename(model_path)

        # Tray(식판) 영역 정의: 6개 구역의 좌표 (x1, y1, x2, y2)
        self.tray_bboxes = [
            (10, 10, 240, 280),    # 구역 1
            (230, 10, 400, 280),   # 구역 2
            (390, 10, 570, 280),   # 구역 3
            (560, 10, 770, 280),   # 구역 4
            (10, 270, 430, 630),   # 구역 5
            (420, 270, 800, 630)   # 구역 6
        ]

        # YOLO 모델 로드 및 GPU 설정
        try:
            self.model = YOLO(model_path)
            self.model.to('cuda' if torch.cuda.is_available() else 'cpu')
            print(f"YOLO model '{self.model_name}' loaded successfully.")
        except Exception as e:
            print(f"Error loading YOLO model: {e}")
            exit(1)

    def initialize_camera(self):
        """카메라 초기화 및 스트림 시작"""
        self.pipeline.start(self.config)
        self.align = rs.align(rs.stream.color)  # 깊이 프레임을 컬러 프레임에 정렬
        print("Camera initialized.")

    def capture_frames(self):
        """프레임 캡처 및 정렬"""
        frames = self.pipeline.wait_for_frames()  # 프레임 대기
        aligned_frames = self.align.process(frames)  # 프레임 정렬
        depth_frame = aligned_frames.get_depth_frame()  # 깊이 프레임 추출
        color_frame = aligned_frames.get_color_frame()  # 컬러 프레임 추출
        return depth_frame, color_frame

    def preprocess_images(self, depth_frame, color_frame):
        """프레임을 numpy 배열로 변환하고 상하 반전"""
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())
        return cv2.flip(depth_image, -1), cv2.flip(color_image, -1)

    def apply_roi(self, image):
        """관심 영역(ROI) 크롭"""
        x1, y1 = self.roi_points[0]
        x2, y2 = self.roi_points[1]
        return image[y1:y2, x1:x2]

    def find_closest_tray_region(self, mask_indices):
        """
        객체가 위치한 식판 구역 찾기 (GPU 연산)
        """
        # 마스크의 x, y 좌표를 분리
        mask_y, mask_x = mask_indices
        
        # 마스크의 경계값(최소/최대 x,y 좌표) 계산
        min_x = torch.min(mask_x)
        max_x = torch.max(mask_x)
        min_y = torch.min(mask_y)
        max_y = torch.max(mask_y)

        # tray_bboxes를 GPU 텐서로 변환
        tray_bboxes = torch.tensor(self.tray_bboxes, device='cuda', dtype=torch.float32)
        
        # 객체가 각 구역 내부에 있는지 확인 (x축)
        inside_x = (min_x >= tray_bboxes[:, 0]) & (max_x <= tray_bboxes[:, 2])
        # 객체가 각 구역 내부에 있는지 확인 (y축)
        inside_y = (min_y >= tray_bboxes[:, 1]) & (max_y <= tray_bboxes[:, 3])
        # x축과 y축 모두 내부에 있는지 확인
        inside = inside_x & inside_y

        # 내부에 있는 구역이 있으면 그 중 첫 번째 구역 반환
        indices = torch.where(inside)[0]
        if len(indices) > 0:
            return indices[0].item() + 1

        # 내부에 없는 경우, 객체의 중심점 계산
        center_x = (min_x + max_x) / 2
        center_y = (min_y + max_y) / 2
        
        # 각 구역의 중심점 계산
        tray_centers = torch.tensor(
            [[(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2] for bbox in self.tray_bboxes],
            device='cuda',
            dtype=torch.float32
        )
        
        # 객체 중심점과 각 구역 중심점 사이의 거리 계산
        distances = torch.sqrt((tray_centers[:, 0] - center_x) ** 2 + (tray_centers[:, 1] - center_y) ** 2)
        # 가장 가까운 구역의 인덱스 찾기
        closest_index = torch.argmin(distances).item()
        # 구역 번호 반환 (인덱스 + 1)
        return closest_index + 1

    def calculate_volume_on_gpu(self, cropped_depth, depth_intrin, mask_indices):
        """
        객체의 부피 계산 (GPU 연산)
        """
        # 깊이 이미지를 GPU 텐서로 변환
        depth_tensor = torch.tensor(cropped_depth, device='cuda', dtype=torch.float32)
        
        # 마스크 좌표 분리하고 GPU 텐서로 변환
        mask_y, mask_x = mask_indices
        mask_tensor = (torch.as_tensor(mask_y, device='cuda'), torch.as_tensor(mask_x, device='cuda'))

        # 저장된 기준 깊이를 GPU 텐서로 변환
        saved_depth_tensor = torch.tensor(self.save_depth, device='cuda', dtype=torch.float32)

        # 마스크 영역의 현재 깊이값을 센티미터 단위로 변환
        z_cm = depth_tensor[mask_tensor] / 10.0
        # 마스크 영역의 기준 깊이값을 센티미터 단위로 변환
        base_depth_cm = saved_depth_tensor[mask_tensor] / 10.0
        # 기준 깊이와 현재 깊이의 차이로 높이 계산 (음수 방지)
        height_cm = torch.clamp(base_depth_cm - z_cm, min=0)

        # 픽셀당 실제 면적 계산 (깊이에 따른 보정)
        pixel_area_cm2 = (z_cm ** 2) / (depth_intrin.fx * depth_intrin.fy)
        # 총 부피 계산 (면적 * 높이의 합)
        volume = torch.sum(height_cm * pixel_area_cm2).item()
        return volume

    def save_cropped_object_with_bbox(self, image, bbox, save_path, object_name):
        """
        감지된 객체 이미지 저장
        """
        # 바운딩 박스 좌표 추출
        x1, y1, x2, y2 = bbox
        # 이미지에서 객체 영역만 크롭
        cropped = image[y1:y2, x1:x2]

        # 저장 경로의 기존 파일 목록 확인
        existing_files = os.listdir(save_path)
        # 동일 객체명으로 저장된 파일 수 계산
        count = sum(1 for file in existing_files if file.startswith(object_name.replace(' ', '_')) and file.endswith('.jpg'))

        # 새로운 파일명 생성 (객체명_번호.jpg)
        file_name = f"{object_name.replace(' ', '_')}_{count + 1}.jpg"
        file_path = os.path.join(save_path, file_name)

        # 크롭된 이미지 저장
        cv2.imwrite(file_path, cropped)
        print(f"Saved cropped object to: {file_path}")

    def save_detected_objects(self, image, detected_objects):
        """
        탐지된 객체들을 저장
        """
        # 기본 저장 경로 설정
        save_base_path = os.path.join(os.getcwd(), "detected_objects")
        os.makedirs(save_base_path, exist_ok=True)

        # 각 탐지된 객체에 대해 처리
        for obj_id, (obj_name, bbox, region, volume) in detected_objects.items():
            # obj_id를 폴더 이름으로 사용
            save_path = os.path.join(save_base_path, obj_id)
            os.makedirs(save_path, exist_ok=True)

            # 크롭 및 저장
            self.save_cropped_object_with_bbox(image, bbox, save_path, obj_name)
            print(f"Saved {obj_name} (ID {obj_id}) in Region {region} with Volume: {volume:.1f} cm^3")
        print("All detected objects have been saved.")

    def visualize_results(self, blend_image, object_name, region, volume, color, mask_indices):
        """
        결과 시각화: blend_image에 마스크를 적용하여 화면에 표시
        """
        # 마스크 좌표 분리
        mask_y, mask_x = mask_indices

        # GPU 텐서를 CPU로 이동 후 NumPy 배열로 변환
        mask_y = mask_y.cpu().numpy()
        mask_x = mask_x.cpu().numpy()

        # 마스크 부분을 강조하여 blend_image 시각화
        blend_image[mask_y, mask_x] = (blend_image[mask_y, mask_x] * 0.5 + np.array(color) * 0.5).astype(np.uint8)

        # 텍스트 정보 표시
        text = f"{object_name}: {volume:.1f}cm^3 ({region})"
        cv2.putText(blend_image, text, (mask_x.min(), mask_y.min() - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)


    def main_loop(self):
        """메인 처리 루프"""
        # 카메라 초기화 호출
        self.initialize_camera()

        # 저장된 기준 깊이 데이터가 있는지 확인하고 로드
        if os.path.exists('save_depth.npy'):
            self.save_depth = np.load('save_depth.npy')
            print("Loaded saved depth data.")
        else:
            print("No saved depth data found. Please save depth data.")

        try:
            while True:
                # 깊이와 컬러 프레임 캡처 
                depth_frame, color_frame = self.capture_frames()
                if not depth_frame or not color_frame:
                    continue

                # 깊이 카메라의 내부 파라미터 가져오기
                depth_intrin = depth_frame.profile.as_video_stream_profile().intrinsics
                
                # 프레임 전처리 (numpy 배열 변환 및 상하 반전)
                depth_image, color_image = self.preprocess_images(depth_frame, color_frame)
                
                # ROI 영역 크롭
                cropped_depth = self.apply_roi(depth_image)
                cropped_color = self.apply_roi(color_image)

                # 시각화를 위한 이미지 복사
                blend_image = cropped_color.copy()

                # YOLO 모델로 객체 감지
                results = self.model(cropped_color)
                
                # 현재 프레임에서 감지된 객체 저장을 위한 딕셔너리
                current_frame_objects = {}

                # 감지된 각 객체에 대한 처리
                for result in results:
                    if result.masks is not None:
                        # 마스크 데이터 추출
                        masks = result.masks.data
                        # 원본 이미지 크기로 마스크 크기 조정
                        original_size = (cropped_color.shape[0], cropped_color.shape[1])
                        resized_masks = F.interpolate(masks.unsqueeze(1), size=original_size, 
                                                    mode='bilinear', align_corners=False).squeeze(1)
                        # 클래스 정보 추출
                        classes = result.boxes.cls

                        # 각 마스크에 대한 처리
                        for i, mask in enumerate(resized_masks):
                            # 객체 ID 및 정보 가져오기
                            obj_id = self.model.names[int(classes[i].item())]
                            obj_name, color, _ = self.cls_name_color.get(obj_id, ("Unknown", (255, 255, 255), 999))

                            # 마스크 영역 좌표 추출
                            mask_indices = torch.where(mask > 0.5)
                            mask_y = mask_indices[0]
                            mask_x = mask_indices[1]

                            # 부피 계산 및 구역 판단
                            volume = self.calculate_volume_on_gpu(cropped_depth, depth_intrin, (mask_y, mask_x))
                            region = self.find_closest_tray_region((mask_y, mask_x))

                            # 구역이 판단된 경우 해당 구역의 bbox 사용, 아닌 경우 마스크 영역으로 bbox 생성
                            if region is not None:
                                bbox = self.tray_bboxes[region - 1]
                            else:
                                print(f"Object {obj_name} not within any predefined region.")
                                bbox = (mask_x.min().cpu().item(), mask_y.min().cpu().item(), 
                                        mask_x.max().cpu().item(), mask_y.max().cpu().item())

                            # 현재 프레임의 객체 정보 저장
                            current_frame_objects[obj_id] = (obj_name, bbox, region, volume)

                            # 결과 시각화
                            self.visualize_results(blend_image, obj_name, region, volume, color, mask_indices)

                # 객체 후보 목록 업데이트 및 확정
                for obj_id in current_frame_objects:
                    obj_name = current_frame_objects[obj_id][0]
                    # 프레임 카운트 증가 또는 초기화
                    self.candidate_objects[obj_name] = self.candidate_objects.get(obj_name, 0) + 1
                    # 15프레임 이상 감지된 경우 확정 객체로 등록
                    if self.candidate_objects[obj_name] >= 15 and obj_name not in self.detected_names:
                        print(f"New object confirmed: {obj_name}")
                        self.detected_names.add(obj_name)

                # 현재 프레임에서 감지되지 않은 객체는 후보 목록에서 제거
                to_remove = [obj_name for obj_name in self.candidate_objects 
                            if obj_name not in [v[0] for v in current_frame_objects.values()]]
                for obj_name in to_remove:
                    del self.candidate_objects[obj_name]

                yield blend_image, current_frame_objects

        finally:
            self.pipeline.stop()
                

if __name__ == "__main__":
    
    # 객체의 출력 이름과 실제 객체의 이름, 색 매핑
    CLS_NAME_COLOR = {
    '01011001': ('Rice', (255, 0, 255), 0),  # Steamed Rice
    '04017001': ('Soybean Soup', (0, 255, 255), 1),  # Soybean Paste Stew
    '06012004': ('Tteokgalbi', (0, 255, 0), 2),  # Grilled Short Rib Patties (Tteokgalbi)
    '07014001': ('Egg Roll', (0, 0, 255), 3),  # Rolled Omelette
    '11013007': ('Spinach Namul', (255, 255, 0), 4),  # Spinach Namul
    '12011008': ('Kimchi', (100, 100, 100), 5),  # Napa Cabbage Kimchi
    '01012006': ('Black Rice', (255, 0, 255), 6),  # Black Rice
    '04011005': ('Seaweed Soup', (0, 255, 255), 7),  # Seaweed Soup
    '04011007': ('Beef Stew', (0, 255, 255), 8),  # Beef Radish Soup
    '06012008': ('Beef Bulgogi', (0, 255, 0), 9),  # Beef Bulgogi
    '08011003': ('Stir-fried Anchovies', (0, 0, 255), 10),  # Stir-fried Anchovies
    '10012001': ('Chicken Gangjeong', (0, 0, 255), 11),  # Sweet and Spicy Fried Chicken
    '11013002': ('Fernbrake Namul', (255, 255, 0), 12),  # Fernbrake Namul
    '12011003': ('Radish Kimchi', (100, 100, 100), 13),  # Radish Kimchi (Kkakdugi)
    '01012002': ('Bean Rice', (255, 0, 255), 14),  # Soybean Rice
    '04011011': ('Fish Cake Soup', (0, 255, 255), 15),  # Fish Cake Soup
    '07013003': ('Kimchi Pancake', (0, 0, 255), 16),  # Kimchi Pancake
    '11013010': ('Bean Sprouts Namul', (255, 255, 0), 17),  # Bean Sprout Namul
    '03011011': ('Pumpkin Soup', (255, 0, 255), 18),  # Pumpkin Porridge
    '08012001': ('Stir-fried Potatoes', (255, 255, 0), 19)  # Stir-fried Potatoes
}
    # 모델 경로 설정
    MODEL_PATH = os.path.join(os.getcwd(), 'model', "large_epoch200.pt")

    # ROI 설정
    ROI_POINTS = [(130, 15), (1020, 665)]

    # 클래스 설정
    calculator = DepthVolumeCalculator(MODEL_PATH, ROI_POINTS, CLS_NAME_COLOR)
    # 클래스 내의 메인 루프 실행
    calculator.main_loop()



    