import pyrealsense2 as rs
import numpy as np
import cv2
import os
from ultralytics import YOLO
import logging

# 로그 레벨을 WARNING 이상으로 설정 (INFO 메시지 비활성화)
logging.getLogger("ultralytics").setLevel(logging.WARNING)

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


brightness_increase = 50  # 밝기를 증가시킬 값 (0~255 범위)
 
cls_name_color = {
    '01011001': ('Rice', (255, 0, 255)),
    '06012004': ('Tteokgalbi', (0, 255, 0)), # 초록색
    '07014001': ('eggRoll', (0, 0, 255)), # 빨간색
    '11013007': ('Spinach greens', (255, 255, 0)), # 하늘색
    '04017001': ('soy bean paste soup', (0, 255, 255)) # 노란색
}

save_depth = None  # 초기 상태를 비움으로 설정

# 마우스 콜백 함수
def mouse_callback(event, x, y, flags, param):
    global save_depth

    if event == cv2.EVENT_LBUTTONDOWN:  # 좌클릭 시
        save_depth = cropped_depth_image
        print("Depth image saved!")


try:
    while True:
        frames = pipeline.wait_for_frames()
        aligned_frames = align.process(frames)
        aligned_depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()

        if not aligned_depth_frame or not color_frame:
            continue

        depth_image = np.asanyarray(aligned_depth_frame.get_data())
        img = np.asanyarray(color_frame.get_data())
        # 이미지 상하좌우 반전 (필요에 따라)
        depth_image = cv2.flip(depth_image, -1)
        img = cv2.flip(img, -1)

        x1, y1 = roi_pts[0]
        x2, y2 = roi_pts[1]
        cropped_image = img[y1:y2, x1:x2]
        cropped_depth_image = depth_image[y1:y2, x1:x2]

        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.imshow('Color Image with ROI', img)
        cv2.setMouseCallback("Color Image with ROI", mouse_callback)
        

        brighter_image = cv2.convertScaleAbs(cropped_image, alpha=1, beta=brightness_increase)

        results = model(brighter_image)

        # 시각화할 이미지 초기화
        blended_image = brighter_image.copy()
        all_colored_mask = np.zeros_like(brighter_image)

        for result in results:
            if result.masks is not None:
                masks = result.masks.data.cpu().numpy()
                classes = result.boxes.cls.cpu().numpy()
                class_names = model.names


                for i, mask in enumerate(masks):
                    resized_mask = cv2.resize(mask, (brighter_image.shape[1], brighter_image.shape[0]))
                    color_mask = (resized_mask > 0.5).astype(np.uint8)

                    key = class_names[int(classes[i])]
                    object_name, color = cls_name_color.get(key, ("Unknown", (255, 255, 255)))

                    all_colored_mask[color_mask == 1] = color


                    mask_indices = np.where(color_mask > 0)

                    # ** 좌표 변환 (크롭된 이미지 -> 원본 이미지) **
                    y_indices, x_indices = mask_indices
                    original_y_indices = y_indices + y1
                    original_x_indices = x_indices + x1
                    
                    masked_depth_values = depth_image[original_y_indices, original_x_indices]
                    valid_depths = masked_depth_values[masked_depth_values > 0]

                    if len(valid_depths) == 0:
                        print("유효한 깊이 값 없음")
                        continue
                    
                    tray_mask_depth = save_depth[mask_indices]
                    tray_valid_depths = tray_mask_depth[tray_mask_depth > 0]

                    tray_average_mm = np.mean(tray_valid_depths)
                    tray_average_cm = tray_average_mm / 10

                    average_distance_mm = np.mean(valid_depths)
                    average_distance_cm = average_distance_mm / 10

                    average_height_cm = tray_average_cm - average_distance_cm

                    y_indices, x_indices = mask_indices
                    min_x, max_x = np.min(x_indices), np.max(x_indices)
                    min_y, max_y = np.min(y_indices), np.max(y_indices)
                    label_x, label_y = min_x, min_y - 10
                    cv2.putText(all_colored_mask, f"{object_name}: {average_height_cm:.2f}cm",
                                (label_x, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                    # 자른 이미지를 비디오 파일에 쓰기, 필요시에 사용
                    

        # 마스크와 원본 이미지 혼합
        blended_image = cv2.addWeighted(brighter_image, 0.7, all_colored_mask, 0.3, 0)

        # 결과 출력
        cv2.imshow('Segmented Mask with Heights', blended_image)

        # out.write(blended_image)

        if cv2.waitKey(1) == 27:
            break

finally:
    pipeline.stop()
    cv2.destroyAllWindows()
    # out.release() # 비디오 writer 해제