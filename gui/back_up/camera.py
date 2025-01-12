import pyrealsense2 as rs
import numpy as np
import cv2
import os
from ultralytics import YOLO
import torch
import logging
import torch.nn.functional as F
import get_weight as gw  # 무게 관련 클래스 임포트
import time

class DepthVolumeCalculator:
    def __init__(self, model_path, roi_points, cls_name_color, food_processor, total_calories):
        """
        초기화
        """
        logging.getLogger("ultralytics").setLevel(logging.WARNING)
        # 무게 객체 생성
        self.GET_WEIGHT = gw.GetWeight()
        
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
        self.config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
        self.align = None
        self.save_depth = None
        self.roi_points = roi_points
        self.cls_name_color = cls_name_color

        self.last_confirmed_obj_id = None  # 가장 최근 확정 객체 id 초기화
        self.candidate_objects = {}  # 객체 이름 및 프레임 수 기록
        self.confirmed_objects = {}  # 확정된 객체
        self.food_processor = food_processor
        self.model_name = os.path.basename(model_path)
        self.total_calories = total_calories
        self.is_running = True  # 루프 상태 플래그
        self.latest_blend_image = None  # QLabel에 표시할 최신 이미지 저장

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
            print(f"YOLO model '{os.path.basename(model_path)}' loaded successfully.")
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

    # ========================== 디 버 그 코 드 2 ===================================== # 
    def save_results(self):
        """저장 로직"""
        try:
            plate_food_data = []
            for obj_id, obj_data in self.confirmed_objects.items():
                plate_food_data.append((
                    obj_id,
                    obj_data['weight'],
                    obj_data['volume'],
                    obj_data['region']
                ))

            if plate_food_data:
                # 음식 데이터 처리 및 카테고리 계산
                total_nutrients = self.food_processor.process_food_data(
                    plate_food_data, self.customer_id, self.food_processor.min_max_table
                )

                # 자른 이미지와 카테고리 정보 저장
                self.save_detected_objects(
                    image=self.cropped_color,
                    detected_objects=self.confirmed_objects,
                    q_categories={food_id: "Q1" for food_id, _, _, _ in plate_food_data}  # 필요에 따라 조정
                )

                # 고객 식단 정보 저장
                total_weight = sum(obj['weight'] for obj in self.confirmed_objects.values())
                self.food_processor.save_customer_diet_detail(
                    customer_id=self.customer_id,
                    total_weight=total_weight,
                    total_nutrients=total_nutrients
                )

                print("[INFO] Results saved successfully.")
                return True
            else:
                print("[INFO] No data to process for saving.")
                return False
        except Exception as e:
            print(f"[ERROR] Error during save_results: {e}")
            return False

    def save_detected_objects(self, image, detected_objects, q_categories):
        """탐지된 객체를 저장"""
        save_base_path = os.path.join(os.getcwd(), "detected_objects")
        os.makedirs(save_base_path, exist_ok=True)

        for obj_id, obj_data in detected_objects.items():
            obj_name = obj_data.get('obj_name', 'Unknown')
            region = obj_data.get('region', None)
            bbox = obj_data.get('bbox', None)
            volume = obj_data.get('volume', None)
            weight = obj_data.get('weight', None)

            if obj_id not in q_categories:
                print(f"[ERROR] Object ID {obj_id} not found in q_categories. Skipping.")
                continue

            category = q_categories.get(obj_id, 'Unknown')
            save_path = os.path.join(save_base_path, obj_id, category)
            os.makedirs(save_path, exist_ok=True)

            self.save_cropped_object_with_bbox(image, bbox, save_path, obj_name)

    def save_cropped_object_with_bbox(self, image, bbox, save_path, object_name):
        """지정된 BBox 영역을 크롭하고 저장"""
        try:
            x1, y1, x2, y2 = bbox
            cropped = image[y1:y2, x1:x2]

            existing_files = os.listdir(save_path)
            count = sum(1 for file in existing_files if file.startswith(object_name.replace(' ', '_')) and file.endswith('.jpg'))

            file_name = f"{object_name.replace(' ', '_')}_{count + 1}.jpg"
            file_path = os.path.join(save_path, file_name)

            cv2.imwrite(file_path, cropped)
            print(f"[DEBUG] Saved cropped image to: {file_path}")
        except Exception as e:
            print(f"[ERROR] Error saving cropped image: {e}")

    def calculate_volume_on_gpu(self, cropped_depth, depth_intrin, mask_indices):
        """GPU를 사용하여 객체의 부피를 계산"""
        depth_tensor = torch.tensor(cropped_depth, device='cuda', dtype=torch.float32)
        mask_y, mask_x = mask_indices
        mask_tensor = (torch.as_tensor(mask_y, device='cuda'), torch.as_tensor(mask_x, device='cuda'))

        saved_depth_tensor = torch.tensor(self.save_depth, device='cuda', dtype=torch.float32)

        z_cm = depth_tensor[mask_tensor] / 10.0
        base_depth_cm = saved_depth_tensor[mask_tensor] / 10.0
        height_cm = torch.clamp(base_depth_cm - z_cm, min=0)

        pixel_area_cm2 = (z_cm ** 2) / (depth_intrin.fx * depth_intrin.fy)
        volume = torch.sum(height_cm * pixel_area_cm2).item()
        return volume

    def find_closest_tray_region(self, mask_indices):
        """PyTorch를 사용하여 GPU에서 구역(BBox) 판단"""
        mask_y, mask_x = mask_indices
        min_x = torch.min(mask_x)
        max_x = torch.max(mask_x)
        min_y = torch.min(mask_y)
        max_y = torch.max(mask_y)

        tray_bboxes = torch.tensor(self.tray_bboxes, device='cuda', dtype=torch.float32)
        inside_x = (min_x >= tray_bboxes[:, 0]) & (max_x <= tray_bboxes[:, 2])
        inside_y = (min_y >= tray_bboxes[:, 1]) & (max_y <= tray_bboxes[:, 3])
        inside = inside_x & inside_y

        indices = torch.where(inside)[0]
        if len(indices) > 0:
            return indices[0].item() + 1

        center_x = (min_x + max_x) / 2
        center_y = (min_y + max_y) / 2
        tray_centers = torch.tensor(
            [[(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2] for bbox in self.tray_bboxes],
            device='cuda',
            dtype=torch.float32
        )
        distances = torch.sqrt((tray_centers[:, 0] - center_x) ** 2 + (tray_centers[:, 1] - center_y) ** 2)
        closest_index = torch.argmin(distances).item()
        return closest_index + 1
    
    

    def visualize_results(self, blend_image, object_name, region, volume, color, mask_indices):
        """시각화: blend_image에 마스크를 적용하여 화면에 표시"""
        mask_y, mask_x = mask_indices

        mask_y = mask_y.cpu().numpy()
        mask_x = mask_x.cpu().numpy()

        blend_image[mask_y, mask_x] = (blend_image[mask_y, mask_x] * 0.5 + np.array(color) * 0.5).astype(np.uint8)

        text = f"{object_name}: {volume:.1f}cm^3 ({region})"
        cv2.putText(blend_image, text, (mask_x.min(), mask_y.min() - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    def main_loop(self, customer_id):
        """메인 처리 루프"""
        self.customer_id = customer_id
        self.initialize_camera()

        if os.path.exists('save_depth.npy'):
            self.save_depth = np.load('save_depth.npy')
            print("Loaded saved depth data.")
        else:
            self.save_depth = None
            print("No saved depth data found. Please save depth data.")

        try:
            while self.is_running:
                depth_frame, color_frame = self.capture_frames()
                if not depth_frame or not color_frame:
                    continue

                depth_intrin = depth_frame.profile.as_video_stream_profile().intrinsics
                depth_image, color_image = self.preprocess_images(depth_frame, color_frame)
                cropped_depth = self.apply_roi(depth_image)
                self.cropped_color = self.apply_roi(color_image)

                blend_image = self.cropped_color.copy()
                results = self.model(self.cropped_color)

                # 객체 감지 및 무게 측정 로직 유지
                current_frame_objects = {}
                for result in results:
                    if result.masks is not None:
                        masks = result.masks.data
                        original_size = (self.cropped_color.shape[0], self.cropped_color.shape[1])
                        resized_masks = F.interpolate(masks.unsqueeze(1), size=original_size, mode='bilinear', align_corners=False).squeeze(1)
                        classes = result.boxes.cls

                        for i, mask in enumerate(resized_masks):
                            obj_id = self.model.names[int(classes[i].item())]
                            obj_name, color, _ = self.cls_name_color.get(obj_id, ("Unknown", (255, 255, 255), 999))

                            mask_indices = torch.where(mask > 0.5)
                            mask_y, mask_x = mask_indices[0], mask_indices[1]

                            volume = self.calculate_volume_on_gpu(cropped_depth, depth_intrin, (mask_y, mask_x))
                            region = self.find_closest_tray_region((mask_y, mask_x))

                            bbox = self.tray_bboxes[region - 1] if region is not None else (
                                mask_x.min().cpu().item(), mask_y.min().cpu().item(),
                                mask_x.max().cpu().item(), mask_y.max().cpu().item()
                            )

                            weight = None
                            current_frame_objects[obj_id] = (obj_name, region, volume, bbox, weight)

                            self.visualize_results(blend_image, obj_name, region, volume, color, mask_indices)

                for obj_id, obj_data in current_frame_objects.items():
                    obj_name, region, volume, bbox, weight = obj_data
                    if obj_id not in self.candidate_objects:
                        self.candidate_objects[obj_id] = 0
                    self.candidate_objects[obj_id] += 1

                    if self.candidate_objects[obj_id] == 3:
                        gw.first_object_weight = round(float(gw.recent_weight_str), 2)
                        print(f"[INFO] 초기화된 무게 (first_object_weight - 3프레임 이상 머물렀다는 의미): {gw.first_object_weight}")

                    if self.candidate_objects[obj_id] >= 15 and obj_id not in self.confirmed_objects:
                        print(f"새로운 객체 확정: {obj_name}")

                        self.confirmed_objects[obj_id] = {
                            "obj_name": obj_name,
                            "region": region,
                            "bbox": bbox,
                            "volume": round(volume, 2),
                        }

                        print(f"확정된 객체 정보: id : {obj_id}, name: {obj_name}, Volume: {volume:.1f} cm³, Region: {region}, weight : {weight}")

                        self.last_confirmed_obj_id = obj_id
                        print(f"last_confirm_obj_id : {self.last_confirmed_obj_id}")

                        time.sleep(3)

                        onair_recent_weight = round(float(gw.recent_weight_str), 2)
                        print(f"실시간 무게값 = {onair_recent_weight}")

                        object_weight = onair_recent_weight - gw.first_object_weight

                        print(f"나는 {obj_name} : {round(object_weight, 2)}!!!")

                        food_info = self.food_processor.get_food_info(obj_id)
                        if food_info:
                            base_weight = food_info['weight(g)']
                            base_calories = food_info['calories(kcal)']

                            calories = self.food_processor.calculate_nutrient(base_weight, base_calories, object_weight)
                            self.total_calories += calories
                            print(f"현재까지 총 칼로리: {self.total_calories:.2f} kcal")

                        if self.last_confirmed_obj_id is not None:
                            self.confirmed_objects[self.last_confirmed_obj_id]["weight"] = round(object_weight, 2)
                            gw.first_object_weight = round(float(gw.recent_weight_str), 2)
                            print(f"객체정보에 무게 추가: {self.confirmed_objects[self.last_confirmed_obj_id]}")
                            print(f"무게만 따로 확인하기 : {self.confirmed_objects[self.last_confirmed_obj_id]['weight']}")

                    if gw.first_object_weight is not None:
                        current_weight = round(float(gw.recent_weight_str), 2)
                        weight_difference = current_weight - gw.first_object_weight
                        if abs(weight_difference) >= 10:
                            print(f"[INFO] 무게 변화 감지: {self.confirmed_objects.get(self.last_confirmed_obj_id, {}).get('obj_name', 'Unknown')}, 무게 증가량: {weight_difference:.2f} g")

                            if self.last_confirmed_obj_id:
                                if self.last_confirmed_obj_id in current_frame_objects:
                                    current_volume = current_frame_objects[self.last_confirmed_obj_id][2]
                                    self.confirmed_objects[self.last_confirmed_obj_id]["volume"] = round(current_volume, 2)
                                    print(f"[INFO] Updated volume for object '{self.last_confirmed_obj_id}': {current_volume:.2f} cm³")
                                else:
                                    print(f"[WARNING] Object '{self.last_confirmed_obj_id}' not found. Resetting.")
                                    self.last_confirmed_obj_id = None
                                    if self.last_confirmed_obj_id in self.confirmed_objects:
                                        print(f"[INFO] Removing '{self.last_confirmed_obj_id}' from confirmed_objects.")
                                        del self.confirmed_objects[self.last_confirmed_obj_id]

                                update_weight = self.confirmed_objects[self.last_confirmed_obj_id].get("weight", 0) + weight_difference
                                self.confirmed_objects[self.last_confirmed_obj_id]["weight"] = round(update_weight, 2)
                                print(f"업데이트된 객체: id: {self.last_confirmed_obj_id}, name: {self.confirmed_objects[self.last_confirmed_obj_id]['obj_name']}, Volume: {self.confirmed_objects[self.last_confirmed_obj_id]['volume']:.1f} cm³, Updated Weight: {update_weight:.2f} g")

                                food_info = self.food_processor.get_food_info(self.last_confirmed_obj_id)
                                if food_info:
                                    base_weight = food_info['weight(g)']
                                    base_calories = food_info['calories(kcal)']

                                    new_calories = self.food_processor.calculate_nutrient(base_weight, base_calories, weight_difference)
                                    self.total_calories += new_calories
                                    print(f"[INFO] {self.confirmed_objects[self.last_confirmed_obj_id]['obj_name']} 칼로리 추가됨: {new_calories:.2f} kcal")
                                    print(f"현재까지 총 칼로리: {self.total_calories:.2f} kcal")

                            gw.first_object_weight = current_weight

                to_remove = [obj for obj in self.candidate_objects if obj not in current_frame_objects]
                for obj in to_remove:
                    del self.candidate_objects[obj]

                self.latest_blend_image = blend_image
                yield blend_image, current_frame_objects

        finally:
            self.pipeline.stop()
            print("Camera stopped.")
