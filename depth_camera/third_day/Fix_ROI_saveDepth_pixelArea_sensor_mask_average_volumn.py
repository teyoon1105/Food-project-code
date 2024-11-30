import pyrealsense2 as rs
import numpy as np
import cv2
import os
from ultralytics import YOLO
import logging

# 로그 레벨 설정
logging.getLogger("ultralytics").setLevel(logging.WARNING)

# ... (모델 로드, RealSense 설정, ROI 설정, 색상 매핑 등 이전 코드와 동일)
Now_Path = os.getcwd()
model_path = os.path.join(Now_Path, 'best.pt')
model = YOLO(model_path)

pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
pipeline.start(config)

align_to = rs.stream.color
align = rs.align(align_to)

roi_pts = [(410, 180), (870, 540)]

# 비디오 저장 설정
# 필요시에 사용
# fourcc = cv2.VideoWriter_fourcc(*'XVID')  # 코덱 설정 (적절한 코덱 사용)
# fps = 30.0  # 프레임 레이트
# width = roi_pts[1][0] - roi_pts[0][0]  # ROI 너비
# height = roi_pts[1][1] - roi_pts[0][1] # ROI 높이
# out = cv2.VideoWriter('cropped_output.avi', fourcc, fps, (width, height)) # 파일명, 코덱, 프레임 레이트, 크기

 
cls_name_color = {
    '01011001': ('Rice', (255, 0, 255)),
    '06012004': ('Tteokgalbi', (0, 255, 0)), # 초록색
    '07014001': ('eggRoll', (0, 0, 255)), # 빨간색
    '11013007': ('Spinach', (255, 255, 0)), # 하늘색
    '04017001': ('Doenjangjjigae', (0, 255, 255)) # 노란색
}

save_depth = None  # 초기 상태를 비움으로 설정

# 마우스 콜백 함수
def mouse_callback(event, x, y, flags, param):
    global save_depth
    if event == cv2.EVENT_LBUTTONDOWN:
        save_depth = cropped_depth_image.copy()
        print("Baseline depth image saved.")

try:
    while True:
        # 프레임 가져오기 및 정렬
        frames = pipeline.wait_for_frames()
        aligned_frames = align.process(frames)
        aligned_depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()

        if not aligned_depth_frame or not color_frame:
            continue

        # 깊이 및 컬러 이미지 변환 (numpy 배열)
        depth_image = np.asanyarray(aligned_depth_frame.get_data())
        img = np.asanyarray(color_frame.get_data())

        # 이미지 반전 (필요시)
        depth_image = cv2.flip(depth_image, -1)
        img = cv2.flip(img, -1)

        # ROI 적용
        x1, y1 = roi_pts[0]
        x2, y2 = roi_pts[1]
        cropped_image = img[y1:y2, x1:x2]
        cropped_depth_image = depth_image[y1:y2, x1:x2]
        cropped_depth_image = cv2.medianBlur(cropped_depth_image, 5) # 깊이 이미지 블러 처리 추가

        # ROI 표시 및 마우스 콜백 설정
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.imshow('Color Image with ROI', img)
        cv2.setMouseCallback("Color Image with ROI", mouse_callback)

        # 객체 감지
        results = model(cropped_image)

        # 시각화 이미지 초기화
        blended_image = cropped_image.copy()
        all_colored_mask = np.zeros_like(cropped_image)

        for result in results:
            if result.masks is not None:
                masks = result.masks.data.cpu().numpy()
                classes = result.boxes.cls.cpu().numpy()
                class_names = model.names

                for i, mask in enumerate(masks):
                    resized_mask = cv2.resize(mask, (cropped_image.shape[1], cropped_image.shape[0]))
                    color_mask = (resized_mask > 0.5).astype(np.uint8)

                    key = class_names[int(classes[i])]
                    object_name, color = cls_name_color.get(key, ("Unknown", (255, 255, 255)))
                    all_colored_mask[color_mask == 1] = color

                    mask_indices = np.where(color_mask > 0)
                    y_indices, x_indices = mask_indices

                    if save_depth is None:
                        print("Baseline depth not set. Please click to set it.")
                        continue

                    # 깊이 정보 가져오기 (cm 단위) - cropped_depth_image 사용
                    z_cm_array = cropped_depth_image[mask_indices] / 10
                    base_depth_cm_array = save_depth[mask_indices] / 10

                    # 유효한 값 필터링
                    valid_indices = np.where((z_cm_array > 0.1) & (base_depth_cm_array > 0) & np.isfinite(z_cm_array) & np.isfinite(base_depth_cm_array))
                    valid_z_cm = z_cm_array[valid_indices]
                    valid_base_depth_cm = base_depth_cm_array[valid_indices]
                    valid_height_cm = valid_base_depth_cm - valid_z_cm

                    # 면적 및 부피 계산
                    depth_intrin = aligned_depth_frame.profile.as_video_stream_profile().intrinsics
                    # D455 컬러 센서 크기 (mm)
                    sensor_width_mm = 5.6
                    sensor_height_mm = 4.2

                    # 픽셀 크기 (cm) 계산
                    pixel_width_cm = (sensor_width_mm / depth_intrin.width) * 0.1
                    pixel_height_cm = (sensor_height_mm / depth_intrin.height) * 0.1

                    total_area_cm2 = 0
                    total_volume_cm3 = 0
                    
                    if valid_z_cm.size > 0:
                        individual_pixel_areas = pixel_width_cm * pixel_height_cm * np.ones_like(valid_z_cm) # 모든 픽셀 면적이 같도록 수정
        
                        total_area_cm2 = np.sum(individual_pixel_areas)
                        total_volume_cm3 = np.sum(individual_pixel_areas * valid_height_cm)
                        average_height_cm = np.mean(valid_height_cm)
                    else:
                        average_height_cm = 0
                        total_area_cm2 = 0
                        total_volume_cm3 = 0


                    # 결과 표시 (ROI 기준)
                    min_x, max_x = np.min(x_indices), np.max(x_indices)
                    min_y, max_y = np.min(y_indices), np.max(y_indices)
                    label_x = min_x
                    label_y = min(max_y + 10, color_mask.shape[0] - 10)
                    cv2.putText(all_colored_mask, f": V={total_volume_cm3:.2f}cm_3,S={total_area_cm2:.2f}cm_2,H={average_height_cm:.2f}cm", (label_x, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)


       # 마스크와 원본 이미지 혼합
        blended_image = cv2.addWeighted(cropped_image, 0.7, all_colored_mask, 0.3, 0)

        # 결과 출력
        cv2.imshow('Segmented Mask with Heights', blended_image)

        # out.write(blended_image)

        if cv2.waitKey(1) == 27:
            break

finally:
    pipeline.stop()
    cv2.destroyAllWindows()
    # out.release() # 비디오 writer 해제