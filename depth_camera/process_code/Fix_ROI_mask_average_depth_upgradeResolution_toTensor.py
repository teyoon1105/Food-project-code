import pyrealsense2 as rs
import numpy as np
import cv2
import os
from ultralytics import YOLO
import torch
import torchvision.transforms as transforms

# YOLO 모델 로드
Now_Path = os.getcwd()
model_path = os.path.join(Now_Path, 'best.pt')
model = YOLO(model_path)

# d435i 기준
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
pipeline.start(config)

# 깊이 이미지를 컬러 이미지에 정렬
align_to = rs.stream.color
align = rs.align(align_to)

# 고정 ROI 좌표 설정 (좌상단, 우하단)
roi_pts = [(410, 180), (870, 540)]

# 클래스별 이름과 시각화 색상 매핑
cls_name_color = {
   '01011001': ('Rice', (255, 0, 255)),
   '06012004': ('Tteokgalbi', (0, 255, 0)),
   '07014001': ('eggRoll', (0, 0, 255)), 
   '11013007': ('Spinach greens', (255, 255, 0)),
   '04017001': ('soy bean paste soup', (0, 255, 255))
}

try:
   while True:
       # 카메라에서 프레임 획득
       frames = pipeline.wait_for_frames()
       aligned_frames = align.process(frames)
       aligned_depth_frame = aligned_frames.get_depth_frame()
       color_frame = aligned_frames.get_color_frame()

       if not aligned_depth_frame or not color_frame:
           continue

       # 프레임을 numpy 배열로 변환 및 이미지 반전
       depth_image = np.asanyarray(aligned_depth_frame.get_data())
       img = np.asanyarray(color_frame.get_data())
       depth_image = cv2.flip(depth_image, -1)
       img = cv2.flip(img, -1)

       # ROI 영역 추출
       x1, y1 = roi_pts[0]
       x2, y2 = roi_pts[1]
       cropped_image = img[y1:y2, x1:x2]
       cropped_depth_image = depth_image[y1:y2, x1:x2]

       # 원본 이미지에 ROI 표시
       cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
       cv2.imshow('Color Image with ROI', img)

       # YOLO 모델 입력을 위한 이미지 전처리
       resized_image = cv2.resize(cropped_image, (640, 640), cv2.INTER_LANCZOS4)
       normalized_img = resized_image.astype(np.float32) / 255.0
       normalized_img = np.transpose(normalized_img, (2, 0, 1))
       normalized_img = torch.from_numpy(normalized_img)
       normalized_img = normalized_img.unsqueeze(0)

       # YOLO 모델로 객체 감지
       results = model(normalized_img)

       # 세그멘테이션 결과 시각화 준비
       blended_image = cropped_image.copy()
       all_colored_mask = np.zeros_like(cropped_image)

       for result in results:
           if result.masks is not None:
               masks = result.masks.data.cpu().numpy()
               classes = result.boxes.cls.cpu().numpy()
               class_names = model.names

               for i, mask in enumerate(masks):
                   # 마스크 크기를 원본 이미지에 맞게 조정
                   resized_mask = cv2.resize(mask, (cropped_image.shape[1], cropped_image.shape[0]))
                   color_mask = (resized_mask > 0.5).astype(np.uint8)

                   # 객체 클래스에 따른 이름과 색상 할당
                   key = class_names[int(classes[i])]
                   object_name, color = cls_name_color.get(key, ("Unknown", (255, 255, 255)))

                   # 마스크 영역에 색상 적용
                   all_colored_mask[color_mask == 1] = color

                   # 객체의 깊이 값 계산
                   mask_indices = np.where(color_mask > 0)
                   masked_depth_values = cropped_depth_image[mask_indices]
                   valid_depths = masked_depth_values[masked_depth_values > 0]

                   if len(valid_depths) == 0:
                       print("유효한 깊이 값 없음")
                       continue

                   # 평균 거리 계산 및 텍스트 표시
                   average_distance_mm = np.mean(valid_depths)
                   average_distance_cm = average_distance_mm / 10

                   y_indices, x_indices = mask_indices
                   min_x, max_x = np.min(x_indices), np.max(x_indices)
                   min_y, max_y = np.min(y_indices), np.max(y_indices)
                   label_x, label_y = min_x, min_y - 10
                   cv2.putText(all_colored_mask, f"{object_name}: {average_distance_cm:.2f}cm",
                               (label_x, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

       # 원본 이미지와 마스크 합성 및 결과 표시
       blended_image = cv2.addWeighted(cropped_image, 0.7, all_colored_mask, 0.3, 0)
       cv2.imshow('Segmented Mask with Heights', blended_image)

       if cv2.waitKey(1) == 27:
           break

finally:
   # 리소스 해제
   pipeline.stop()
   cv2.destroyAllWindows()