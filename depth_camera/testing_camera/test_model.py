import pyrealsense2 as rs  # Intel RealSense 카메라를 위한 라이브러리
import numpy as np  # 수치 계산 및 배열 처리를 위한 라이브러리
import cv2  # OpenCV, 컴퓨터 비전 및 이미지 처리를 위한 라이브러리
import os  # 파일 및 경로 작업을 위한 라이브러리
from ultralytics import YOLO  # YOLO 객체 탐지 라이브러리
import logging  # 로그 메시지 출력을 위한 라이브러리
import torch

Now_Path = os.getcwd()
model_path = os.path.join(Now_Path, 'best.pt')  # YOLO 모델 파일 경로
model = YOLO(model_path)  # YOLO 모델 로드

img_path = "C:/Users/SBA/teyoon_github/Food-project-code/depth_camera/fourth_day/Image_20241202_203609_307.jpeg"

org_img = cv2.imread(img_path)

img = cv2.resize(org_img, (900, 1200))

cls_name_color = {
    '01011001': ('Rice', (255, 0, 255)),  # 보라색
    '06012004': ('Tteokgalbi', (0, 255, 0)),  # 초록색
    '07014001': ('eggRoll', (0, 0, 255)),  # 빨간색
    '11013007': ('Spinach', (255, 255, 0)),  # 청록색
    '04017001': ('Doenjangjjigae', (0, 255, 255)),  # 노란색
    '12011008': ('Kimchi', (255, 100, 100)) # 파란색

}

# YOLO 모델을 사용하여 객체 탐지 수행
results = model(img)

# 탐지된 객체 시각화를 위한 초기화
blended_image = img.copy()  # 크롭된 이미지를 복사
all_colored_mask = np.zeros_like(img)  # 빈 마스크 생성

for result in results:
    if result.masks is not None:  # 탐지된 객체에 마스크가 있는 경우
        masks = result.masks.data.cpu().numpy()  # 마스크 데이터 가져오기
        boxes = result.boxes.cpu().numpy()  # 박스 데이터 가져오기
        classes = result.boxes.cls.cpu().numpy()
        class_names = model.names  # 클래스 이름 매핑


        for i, mask in enumerate(masks):
            # 탐지 객체의 conf 출력
            conf = boxes.conf[i]
            
            # 마스크를 크롭된 이미지 크기에 맞게 조정
            resized_mask = cv2.resize(mask, (img.shape[1], img.shape[0]))
            color_mask = (resized_mask > 0.5).astype(np.uint8)  # 바이너리 마스크 생성

            # 클래스 이름과 색상 얻기
            key = class_names[int(classes[i])]
            object_name, color = cls_name_color.get(key, ("Unknown", (255, 255, 255)))

            # 마스크 영역에 색상 적용
            all_colored_mask[color_mask == 1] = color

            cv2.putText(all_colored_mask, f"{object_name}",(500,300), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

             # 마스크와 원본 이미지를 혼합하여 시각화
        blended_image = cv2.addWeighted(img, 0.7, all_colored_mask, 0.3, 0)

        # out.write(blended_image)

        # 결과 이미지 출력
        cv2.imshow('Segmented Mask with Heights', blended_image)

        # ESC 키 입력 시 루프 종료
        if cv2.waitKey() == 27:
            break

cv2.destroyAllWindows() 