import pyrealsense2 as rs
import numpy as np
import cv2
import os
from ultralytics import YOLO

# YOLOv8 분할 모델 로드
Now_Path = os.getcwd()
model_path = os.path.join(Now_Path, 'best.pt')
model = YOLO(model_path)  # 학습된 YOLOv8 분할 모델 로드

# RealSense 파이프라인 설정 (깊이 및 컬러 스트림 활성화)
pipeline = rs.pipeline()
config = rs.config()

# 깊이 및 컬러 스트림 활성화 (해상도 640x480, 프레임 레이트 30fps)
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
pipeline.start(config)

# 깊이 스트림을 컬러 스트림에 맞춰 정렬
align_to = rs.stream.color
align = rs.align(align_to)

# 관심 영역 (ROI) 좌표 설정
roi_pts = [(160, 120), (480, 360)]  # ROI의 좌상단 및 우하단 좌표
frame_count = 0

# 비디오 저장 설정
# 필요시에 사용
# fourcc = cv2.VideoWriter_fourcc(*'XVID')  # 코덱 설정 (적절한 코덱 사용)
# fps = 30.0  # 프레임 레이트
# width = roi_pts[1][0] - roi_pts[0][0]  # ROI 너비
# height = roi_pts[1][1] - roi_pts[0][1] # ROI 높이
# out = cv2.VideoWriter('cropped_output.avi', fourcc, fps, (width, height)) # 파일명, 코덱, 프레임 레이트, 크기

try:
    while True:
        # RealSense 파이프라인에서 프레임 가져오기
        frames = pipeline.wait_for_frames()
        # 깊이 프레임을 컬러 프레임에 맞춰 정렬
        aligned_frames = align.process(frames)
        # 정렬된 깊이 및 컬러 프레임 가져오기
        aligned_depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()

        # 프레임이 유효한지 확인
        if not aligned_depth_frame or not color_frame:
            continue

        # 프레임을 NumPy 배열로 변환
        depth_image = np.asanyarray(aligned_depth_frame.get_data())
        img = np.asanyarray(color_frame.get_data())

        # 이미지 상하좌우 반전 (필요에 따라)
        #depth_image = cv2.flip(depth_image, -1)
        #img = cv2.flip(img, -1)

        # 컬러 이미지를 표시할 창 생성
        cv2.namedWindow('Color Image')

        # ROI 사각형을 컬러 이미지에 그리기
        x1, y1 = roi_pts[0]
        x2, y2 = roi_pts[1]
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.imshow('Color Image', img)


        # 이미지를 ROI 영역으로 자르기
        cropped_image = img[y1:y2, x1:x2]
        cv2.imshow('cropped_image', cropped_image)

        # YOLOv8 모델로 객체 탐지 수행
        results = model(cropped_image)

        # 클래스 이름 사전 정의 (모델 출력과 일치하도록 설정)
        cls_name = {'01011001': 'Rice', '06012004': 'Tteokgalbi', '07014001': 'eggRoll', '11013007': 'Spinach greens', '04017001' : 'soy bean paste soup'}

        for result in results:
            if result.masks is not None:
                masks = result.masks.data.cpu().numpy()
                classes = result.boxes.cls.cpu().numpy()
                class_names = model.names

                for i, mask in enumerate(masks):
                    try:
                        # 모델 출력에서 클래스 이름 가져오고 사용자 친화적인 이름으로 매핑
                        key = class_names[int(classes[i])]
                        object_name = cls_name.get(key, "Unknown")

                        # 바운딩 박스 좌표 가져오기
                        boxes = result.boxes.cpu().numpy()
                        box = result.boxes.xyxy[i].cpu().numpy()
                        cx1, cy1, cx2, cy2 = map(int, box)
                        cv2.rectangle(cropped_image, (cx1, cy1), (cx2, cy2), (255, 0, 0), 2)

                        # 바운딩 박스 좌표를 원본 이미지 좌표로 변환
                        real_pt = [(cx1 + x1, cy1 + y1), (cx2 + x1, cy2 + y1)]

                        # 깊이 계산을 위한 마스크 생성
                        mask = np.zeros(depth_image.shape, dtype=np.uint8)
                        cv2.rectangle(mask, real_pt[0], real_pt[1], 255, -1)

                        # 마스크를 깊이 이미지에 적용
                        masked_depth_image = cv2.bitwise_and(depth_image, depth_image, mask=mask)

                        # 바운딩 박스 내 평균 깊이 계산
                        box_depth = depth_image[masked_depth_image > 0]
                        average_distance_mm = np.mean(box_depth)
                        average_distance_cm = average_distance_mm / 10


                        # 거리를 이미지에 표시
                        cv2.putText(cropped_image, f"{object_name}: {average_distance_cm:.2f}cm", (cx1, cy1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

                        # 자른 이미지를 비디오 파일에 쓰기, 필요시에 사용
                        # out.write(cropped_image)

                        # 창에 띄우기
                        cv2.imshow('cropped_image', cropped_image)

                    except (IndexError, ValueError) as e:
                        print(f"깊이 계산 중 오류: {e}")

        # Esc 키를 누르면 반복문 종료
        if cv2.waitKey(1) == 27:
            break

finally:
    pipeline.stop()
    cv2.destroyAllWindows()
    # out.release() # 비디오 writer 해제