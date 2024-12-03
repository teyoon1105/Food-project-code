import pyrealsense2 as rs  # Intel RealSense 카메라 라이브러리
import numpy as np  # 수치 계산 및 배열 처리
import cv2  # OpenCV 라이브러리
import os  # 파일 및 경로 작업
from ultralytics import YOLO  # YOLO 객체 탐지 라이브러리
import logging  # 로그 메시지 관리

# 로그 레벨 설정 (INFO 메시지 비활성화)
logging.getLogger("ultralytics").setLevel(logging.WARNING)

# ---- 설정 ----
# YOLO 모델 경로 및 초기화
MODEL_PATH = os.path.join(os.getcwd(), 'best.pt')
model = YOLO(MODEL_PATH)

# Intel RealSense 카메라 설정
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)

# ROI 설정
ROI_POINTS = [(175, 50), (1055, 690)]
BRIGHTNESS_INCREASE = 50

# 클래스 ID와 이름, 색상을 매핑
CLS_NAME_COLOR = {
    '01011001': ('Rice', (255, 0, 255)),
    '06012004': ('Tteokgalbi', (0, 255, 0)),
    '07014001': ('EggRoll', (0, 0, 255)),
    '11013007': ('Spinach', (255, 255, 0)),
    '04017001': ('Doenjangjjigae', (0, 255, 255))
}

# ---- 전역 변수 ----
save_depth = None  # 기준 깊이 데이터 저장 변수


# ---- 함수 정의 ----
def initialize_camera():
    """카메라 파이프라인 초기화 및 시작"""
    pipeline.start(config)
    align_to = rs.stream.color
    return rs.align(align_to)


def capture_frames(align):
    """카메라에서 프레임 캡처 및 정렬"""
    frames = pipeline.wait_for_frames()
    aligned_frames = align.process(frames)
    return aligned_frames.get_depth_frame(), aligned_frames.get_color_frame()


def preprocess_images(depth_frame, color_frame):
    """깊이 및 컬러 프레임 전처리"""
    depth_image = np.asanyarray(depth_frame.get_data())
    color_image = np.asanyarray(color_frame.get_data())
    depth_image = cv2.flip(depth_image, -1)
    color_image = cv2.flip(color_image, -1)
    return depth_image, color_image


def crop_roi(image, roi_points):
    """이미지에서 관심 영역(ROI) 크롭"""
    x1, y1 = roi_points[0]
    x2, y2 = roi_points[1]
    return image[y1:y2, x1:x2]


def mouse_callback(event, x, y, flags, param):
    """마우스 콜백 함수로 깊이 데이터 저장"""
    global save_depth
    if event == cv2.EVENT_LBUTTONDOWN:
        save_depth = param.copy()
        print("Depth image saved!")


def calculate_volume(cropped_depth, save_depth, mask_indices, depth_intrin, min_depth_cm=0.1):
    """깊이 데이터를 이용하여 부피 계산"""
    total_volume = 0
    y_indices, x_indices = mask_indices

    for pixel_y, pixel_x in zip(y_indices, x_indices):
        z_cm = cropped_depth[pixel_y, pixel_x] / 10  # 현재 깊이 (cm)
        base_depth_cm = save_depth[pixel_y, pixel_x] / 10  # 기준 깊이 (cm)

        if z_cm > min_depth_cm and base_depth_cm > 0:
            height_cm = base_depth_cm - z_cm
            pixel_area_cm2 = (z_cm ** 2) / (depth_intrin.fx * depth_intrin.fy)
            total_volume += pixel_area_cm2 * height_cm

    return total_volume


def visualize_results(cropped_image, all_colored_mask, object_name, total_volume, color, mask_indices):
    """탐지 결과를 시각화"""
    y_indices, x_indices = mask_indices
    min_x, max_x = np.min(x_indices), np.max(x_indices)
    min_y = np.min(y_indices)
    label_position = (min_x, min_y - 10)

    # 마스크 색상 적용 및 텍스트 표시
    cv2.putText(all_colored_mask, f"{object_name}: V:{total_volume:.2f}cm_3",
                label_position, cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    return cv2.addWeighted(cropped_image, 0.7, all_colored_mask, 0.3, 0)


# ---- 메인 처리 루프 ----
def main():
    global save_depth
    align = initialize_camera()

    try:
        while True:
            depth_frame, color_frame = capture_frames(align)

            if not depth_frame or not color_frame:
                continue

            depth_image, color_image = preprocess_images(depth_frame, color_frame)
            cropped_color = crop_roi(color_image, ROI_POINTS)
            cropped_depth = crop_roi(depth_image, ROI_POINTS)

            # ROI 표시
            cv2.rectangle(color_image, ROI_POINTS[0], ROI_POINTS[1], (0, 0, 255), 2)
            cv2.imshow('Color Image with ROI', color_image)
            cv2.setMouseCallback("Color Image with ROI", mouse_callback, cropped_depth)

            brightened_image = cv2.convertScaleAbs(cropped_color, alpha=1, beta=BRIGHTNESS_INCREASE)

            # 객체 탐지 수행
            results = model(brightened_image)
            all_colored_mask = np.zeros_like(brightened_image)
            blended_image = brightened_image.copy()  # 기본값으로 초기화

            for result in results:
                if result.masks is not None:
                    masks = result.masks.data.cpu().numpy()
                    classes = result.boxes.cls.cpu().numpy()

                    for i, mask in enumerate(masks):
                        resized_mask = cv2.resize(mask, (brightened_image.shape[1], brightened_image.shape[0]))
                        color_mask = (resized_mask > 0.5).astype(np.uint8)

                        class_key = model.names[int(classes[i])]
                        object_name, color = CLS_NAME_COLOR.get(class_key, ("Unknown", (255, 255, 255)))

                        mask_indices = np.where(color_mask > 0)

                        if save_depth is None:
                            cv2.putText(color_image, "Save your depth first!", (390,370), cv2.FONT_HERSHEY_TRIPLEX, 1, (0,0,255), 2)
                            cv2.imshow('Color Image with ROI', color_image)
                            continue

                        valid_depths = cropped_depth[mask_indices]
                        if len(valid_depths[valid_depths > 0]) == 0:
                            continue

                        # 카메라 내부 파라미터
                        depth_intrin = depth_frame.profile.as_video_stream_profile().intrinsics

                        total_volume = calculate_volume(cropped_depth, save_depth, mask_indices, depth_intrin)

                        # 개별 마스크에 색상 적용
                        all_colored_mask[color_mask == 1] = color

                        blended_image = visualize_results(brightened_image, all_colored_mask, object_name, total_volume, color, mask_indices)

            # 결과 이미지 표시
            cv2.imshow('Segmented Mask with Heights', blended_image)

            if cv2.waitKey(1) == 27:  # ESC 키로 종료
                break
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()