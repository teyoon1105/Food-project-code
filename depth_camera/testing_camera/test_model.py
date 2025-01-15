import pyrealsense2 as rs
import numpy as np
import cv2
import os
from ultralytics import YOLO

# 현재 경로 및 모델 파일 경로
Now_Path = os.getcwd()
model_path = os.path.join(Now_Path, 'best.pt')  # YOLO 모델 파일 경로
model = YOLO(model_path)  # YOLO 모델 로드

# 이미지 경로
img_path = "path/your/image.jpg"
img = cv2.imread(img_path)

# 클래스 이름 정의
cls_name_color = {
    '01011001': ('Rice', (255, 0, 255)),  # 보라색
    '06012004': ('Tteokgalbi', (0, 255, 0)),  # 초록색
    '07014001': ('eggRoll', (0, 0, 255)),  # 빨간색
    '11013007': ('Spinach', (255, 255, 0)),  # 청록색
    '04017001': ('Doenjangjjigae', (0, 255, 255)),  # 노란색
    '12011008': ('Kimchi', (255, 100, 100))  # 파란색
}

# YOLO 모델을 사용하여 객체 탐지 수행
results = model(img)

# 좌표 저장 폴더 생성
coordinates_folder = "mask_coordinates"
os.makedirs(coordinates_folder, exist_ok=True)

for result in results:
    if result.masks is not None:  # 탐지된 객체에 마스크가 있는 경우
        masks = result.masks.data.cpu().numpy()  # 마스크 데이터 가져오기
        classes = result.boxes.cls.cpu().numpy()
        class_names = model.names  # 클래스 이름 매핑

        for i, mask in enumerate(masks):
            # 마스크를 크롭된 이미지 크기에 맞게 조정
            resized_mask = cv2.resize(mask, (img.shape[1], img.shape[0]))
            binary_mask = (resized_mask > 0.5).astype(np.uint8)  # 이진화

            # 좌표 추출 (비어 있지 않은 픽셀의 좌표)
            coordinates = np.column_stack(np.where(binary_mask > 0))

            # NumPy 좌표 저장
            object_name = class_names[int(classes[i])]
            npy_filename = os.path.join(coordinates_folder, f"{object_name}_{i}.npy")
            np.save(npy_filename, coordinates)

            print(f"Saved coordinates for {object_name}: {npy_filename}")

print("All coordinates saved.")
