import pyrealsense2 as rs
import numpy as np
import cv2
import os
from ultralytics import YOLO

# YOLOv8 분할 모델 로드
Now_Path = os.getcwd()
model_path = os.path.join(Now_Path, 'best.pt')
model = YOLO(model_path)

# RealSense 파이프라인 설정 (깊이 및 컬러 스트림 활성화)
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
pipeline.start(config)

# 깊이 스트림을 컬러 스트림에 맞춰 정렬
align_to = rs.stream.color
align = rs.align(align_to)

# 관심 영역 (ROI) 좌표 설정
roi_pts = [(160, 120), (480, 360)]  # ROI의 좌상단 및 우하단 좌표

# 클래스 이름 매핑
cls_name_color = {
    '01011001': ('Rice', (255, 0, 0)),  # 빨강
    '06012004': ('Tteokgalbi', (0, 255, 0)),  # 초록
    '07014001': ('eggRoll', (0, 0, 255)),  # 파랑
    '11013007': ('Spinach greens', (255, 255, 0)),  # 노랑
    '04017001': ('soy bean paste soup', (0, 255, 255))  # 청록
}

try:
    while True:
        # RealSense 파이프라인에서 프레임 가져오기
        frames = pipeline.wait_for_frames()
        # 같은 픽셀 위치에서의 깊이 값과 컬러 정보를 일치시킴
        aligned_frames = align.process(frames)
        aligned_depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()

        # 프레임이 유효한지 확인
        if not aligned_depth_frame or not color_frame:
            continue

        # 프레임을 NumPy 배열로 변환
        depth_image = np.asanyarray(aligned_depth_frame.get_data())
        img = np.asanyarray(color_frame.get_data())

        # 이미지 상하좌우 반전 (필요에 따라)
        # depth_image = cv2.flip(depth_image, -1)
        # img = cv2.flip(img, -1)

        # ROI 영역 설정 및 crop
        x1, y1 = roi_pts[0]
        x2, y2 = roi_pts[1]
        cropped_image = img[y1:y2, x1:x2]
        cropped_depth_image = depth_image[y1:y2, x1:x2]

        # ROI를 컬러 이미지에 표시
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.imshow('Color Image with ROI', img)

        # YOLOv8 모델로 ROI 영역에서 객체 탐지 수행
        results = model(cropped_image)

        for result in results:
            if result.masks is not None:
                masks = result.masks.data.cpu().numpy()     # 모델의 객체 분할 마스크
                classes = result.boxes.cls.cpu().numpy()    # 탐지된 객체의 클래스 id
                class_names = model.names
                
                # 컬러 마스크 초기화 (전체 이미지 크기)
                colored_mask = np.zeros_like(cropped_image)

                for i, mask in enumerate(masks):
                    # 마스크크기를 roi 영역에 맞게 리사이즈
                    resized_mask = cv2.resize(mask, (cropped_image.shape[1], cropped_image.shape[0]))
                    # 마스크를 컬러 이미지에 색상 입히기
                    color_mask = (resized_mask > 0.5).astype(np.uint8)  # 이진화된 마스크 생성

                    # 클래스에 따라 색상 가져오기
                    key = class_names[int(classes[i])]
                    object_name, color = cls_name_color.get(key, ("Unknown", (255, 255, 255))) # 기본값 흰색


                    # colored_mask = np.zeros_like(cropped_image)
                    colored_mask[color_mask == 1] = color
                    # color 리스트 생성 후 클래스마다 색을 다르게 하면 됨
                    
                # 원본이미지(크랍된) 컬러 마스크 혼합
                blended_image = cv2.addWeighted(cropped_image, 0.8, colored_mask, 0.2, 0)

                for i, mask in enumerate(masks):
                    # 마스크 크기 조정
                    resized_mask = cv2.resize(mask, (cropped_image.shape[1], cropped_image.shape[0]))
                    color_mask = (resized_mask > 0.5).astype(np.uint8)

                    # 마스크 좌표를 ROI 기준으로 깊이 값 추출
                    mask_indices = np.where(color_mask > 0)
                    # 최초에 crop한 영역에서 탐지된 영역 좌표의 깊이값 가져오기
                    masked_depth_values = cropped_depth_image[mask_indices]

                    # 유효한 깊이 값 필터링
                    valid_depths = masked_depth_values[masked_depth_values > 0]

                    if len(valid_depths) == 0:
                        print("유효한 깊이 값 없음")
                        continue

                    # 평균 거리 계산
                    average_distance_mm = np.mean(valid_depths)
                    average_distance_cm = average_distance_mm / 10


                    # 텍스트로 클래스명과 평균 거리 표시
                    y_indices, x_indices = mask_indices  # 마스크 픽셀 좌표
                    min_x, max_x = np.min(x_indices), np.max(x_indices)
                    min_y, max_y = np.min(y_indices), np.max(y_indices)
                    label_x, label_y = min_x, min_y - 10  # 객체 위쪽에 텍스트 위치
                    cv2.putText(blended_image, f"{object_name}: {average_distance_cm:.2f}cm",
                                (label_x, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

                # 결과 이미지 출력
                cv2.imshow('Segmented Mask with Info', blended_image)

        # Esc 키를 누르면 반복문 종료
        if cv2.waitKey(1) == 27:
            break

finally:
    pipeline.stop()
    cv2.destroyAllWindows()