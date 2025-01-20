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
        self.detected_names = set()  # 최종적으로 확인된 객체
        self.candidate_objects = {}  # 이름과 프레임 수 기록
        self.model_name = os.path.basename(model_path)

        # Tray BBox 초기화
        self.tray_bboxes = [
            (10, 10, 240, 280),   # 구역 1
            (230, 10, 400, 280),  # 구역 2
            (390, 10, 570, 280),  # 구역 3
            (560, 10, 770, 280),  # 구역 4
            (10, 270, 430, 630),  # 구역 5
            (420, 270, 800, 630)  # 구역 6
        ]

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

    def find_closest_tray_region(self, mask_indices):
        """
        PyTorch를 사용하여 GPU에서 구역(BBox) 판단.
        """
        mask_y, mask_x = mask_indices
        min_x = torch.min(mask_x)
        max_x = torch.max(mask_x)
        min_y = torch.min(mask_y)
        max_y = torch.max(mask_y)

        tray_bboxes = torch.tensor(self.tray_bboxes, device='cuda', dtype=torch.float32)
        inside_x = (min_x >= tray_bboxes[:, 0]) & (max_x <= tray_bboxes[:, 2])
        inside_y = (min_y >= tray_bboxes[:, 1]) & (max_y <= tray_bboxes[:, 3])
        inside = inside_x & inside_y

        # 포함된 첫 번째 구역 반환
        indices = torch.where(inside)[0]
        if len(indices) > 0:
            return indices[0].item() + 1

        # 포함되지 않은 경우 가장 가까운 구역 계산
        center_x = (min_x + max_x) / 2
        center_y = (min_y + max_y) / 2
        tray_centers = torch.tensor(
            [[(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2] for bbox in self.tray_bboxes],
            device='cuda',
            dtype=torch.float32
        )
        distances = torch.sqrt((tray_centers[:, 0] - center_x) ** 2 + (tray_centers[:, 1] - center_y) ** 2)
        closest_index = torch.argmin(distances).item()
        return closest_index + 1  # 가장 가까운 구역 반환

    def calculate_volume_on_gpu(self, cropped_depth, depth_intrin, mask_indices):
        """
        GPU를 사용하여 객체의 부피를 계산
        """
        depth_tensor = torch.tensor(cropped_depth, device='cuda', dtype=torch.float32)
        mask_y, mask_x = mask_indices
        mask_tensor = (torch.tensor(mask_y, device='cuda'), torch.tensor(mask_x, device='cuda'))

        saved_depth_tensor = torch.tensor(self.save_depth, device='cuda', dtype=torch.float32)

        z_cm = depth_tensor[mask_tensor] / 10.0  # ROI 깊이 (cm)
        base_depth_cm = saved_depth_tensor[mask_tensor] / 10.0  # 기준 깊이 (cm)
        height_cm = torch.clamp(base_depth_cm - z_cm, min=0)

        pixel_area_cm2 = (z_cm ** 2) / (depth_intrin.fx * depth_intrin.fy)
        volume = torch.sum(height_cm * pixel_area_cm2).item()
        return volume
    

    def save_cropped_object_with_bbox(self, image, bbox, save_path, object_name):
        """
        지정된 BBox 영역을 크롭하고 저장. 파일 이름에 고유 번호를 추가.
        """
        x1, y1, x2, y2 = bbox
        cropped = image[y1:y2, x1:x2]

        # 현재 폴더 내의 파일 갯수 확인
        existing_files = os.listdir(save_path)
        count = sum(1 for file in existing_files if file.startswith(object_name.replace(' ', '_')) and file.endswith('.jpg'))

        # 파일 이름 생성
        file_name = f"{object_name.replace(' ', '_')}_{count + 1}.jpg"  # 고유 번호 추가
        file_path = os.path.join(save_path, file_name)

        # 크롭된 이미지 저장
        cv2.imwrite(file_path, cropped)
        print(f"Saved cropped object to: {file_path}")

    def save_detected_objects(self, image, detected_objects):
        """
        탐지된 객체를 저장. 폴더 이름은 모델 출력 ID를 그대로 사용.
        """
        save_base_path = os.path.join(os.getcwd(), "detected_objects")
        os.makedirs(save_base_path, exist_ok=True)

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
        시각화: blend_image에 마스크를 적용하여 화면에 표시.
        """
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

                depth_intrin = depth_frame.profile.as_video_stream_profile().intrinsics
                depth_image, color_image = self.preprocess_images(depth_frame, color_frame)
                cropped_depth = self.apply_roi(depth_image)
                cropped_color = self.apply_roi(color_image)

                # 블렌드 이미지 생성
                blend_image = cropped_color.copy()

                results = self.model(cropped_color)
                current_frame_objects = {}

                for result in results:
                    if result.masks is not None:
                        masks = result.masks.data
                        original_size = (cropped_color.shape[0], cropped_color.shape[1])
                        resized_masks = F.interpolate(masks.unsqueeze(1), size=original_size, mode='bilinear', align_corners=False).squeeze(1)
                        classes = result.boxes.cls

                        for i, mask in enumerate(resized_masks):
                            obj_id = self.model.names[int(classes[i].item())]
                            obj_name, color, _ = self.cls_name_color.get(obj_id, ("Unknown", (255, 255, 255), 999))

                            mask_indices = torch.where(mask > 0.5)
                            mask_y = mask_indices[0]
                            mask_x = mask_indices[1]

                            volume = self.calculate_volume_on_gpu(cropped_depth, depth_intrin, (mask_y, mask_x))
                            region = self.find_closest_tray_region((mask_y, mask_x))

                            if region is not None:
                                bbox = self.tray_bboxes[region - 1]
                            else:
                                print(f"Object {obj_name} not within any predefined region.")
                                bbox = (mask_x.min().cpu().item(), mask_y.min().cpu().item(), mask_x.max().cpu().item(), mask_y.max().cpu().item())

                            current_frame_objects[obj_id] = (obj_name, bbox, region, volume)

                            # 시각화는 blend_image에 적용
                            self.visualize_results(blend_image, obj_name, region, volume, color, mask_indices)

                for obj_id in current_frame_objects:
                    obj_name = current_frame_objects[obj_id][0]
                    self.candidate_objects[obj_name] = self.candidate_objects.get(obj_name, 0) + 1
                    if self.candidate_objects[obj_name] >= 15 and obj_name not in self.detected_names:
                        print(f"New object confirmed: {obj_name}")
                        self.detected_names.add(obj_name)

                to_remove = [obj_name for obj_name in self.candidate_objects if obj_name not in [v[0] for v in current_frame_objects.values()]]
                for obj_name in to_remove:
                    del self.candidate_objects[obj_name]

                cv2.imshow("Results", blend_image)  # blend_image를 화면에 표시
                key = cv2.waitKey(1)
                if key == 27:
                    break
                elif key == ord('s'):
                    if len(self.detected_names) == len(self.candidate_objects):
                        self.save_detected_objects(cropped_color, current_frame_objects)
                    else:
                        print("Not all objects have been confirmed. Please wait.")
                elif key == ord('f'):  # 'f' 키를 눌러 초기화
                    self.detected_names.clear()
                    self.candidate_objects.clear()
                    print("All detected objects and candidate objects have been reset.")
                    
        finally:
            self.pipeline.stop()
            cv2.destroyAllWindows()



if __name__ == "__main__":
    logging.getLogger("ultralytics").setLevel(logging.WARNING)

    # class name for mapping
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
    MODEL_PATH = os.path.join(os.getcwd(), 'model', "path/your/model.pt")
    ROI_POINTS = [(175, 50), (1055, 690)]
    calculator = DepthVolumeCalculator(MODEL_PATH, ROI_POINTS, CLS_NAME_COLOR)
    calculator.main_loop()
