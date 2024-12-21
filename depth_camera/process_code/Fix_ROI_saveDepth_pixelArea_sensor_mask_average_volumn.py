import pyrealsense2 as rs  # Intel RealSense 카메라를 위한 라이브러리
import numpy as np  # 수치 계산 및 배열 처리를 위한 라이브러리
import cv2  # OpenCV, 컴퓨터 비전 및 이미지 처리를 위한 라이브러리
import os  # 파일 및 경로 작업을 위한 라이브러리
from ultralytics import YOLO  # YOLO 객체 탐지 라이브러리
import logging  # 로그 메시지 출력을 위한 라이브러리

# 실패 코드

# 로그 레벨 설정 (INFO 메시지 비활성화)
logging.getLogger("ultralytics").setLevel(logging.WARNING)

# 현재 작업 디렉토리를 얻고 YOLO 모델 파일의 경로를 설정
Now_Path = os.getcwd()
model_path = os.path.join(Now_Path, 'best.pt')  # YOLO 모델 파일 경로
model = YOLO(model_path)  # YOLO 모델 로드

# Intel RealSense 카메라 스트림을 설정
pipeline = rs.pipeline()  # 파이프라인 생성
config = rs.config()  # 설정 객체 생성
config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)  # 깊이 스트림 설정
config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)  # 컬러 스트림 설정
pipeline.start(config)  # 스트림 시작

# 깊이 데이터를 컬러 이미지와 정렬
align_to = rs.stream.color
align = rs.align(align_to)

# 관심 영역 (ROI) 좌표 설정
roi_pts = [(410, 180), (870, 540)]

# 클래스 ID와 이름, 색상을 매핑
cls_name_color = {
    '01011001': ('Rice', (255, 0, 255)),  # 보라색
    '06012004': ('Tteokgalbi', (0, 255, 0)),  # 초록색
    '07014001': ('eggRoll', (0, 0, 255)),  # 빨간색
    '11013007': ('Spinach', (255, 255, 0)),  # 하늘색
    '04017001': ('Doenjangjjigae', (0, 255, 255))  # 노란색
}

# 기준 깊이 데이터를 저장할 변수 초기화
save_depth = None

# 마우스 콜백 함수
def mouse_callback(event, x, y, flags, param):
    """
    마우스 클릭 시 기준 깊이 이미지를 저장.
    """
    global save_depth  # 전역 변수 사용 선언
    if event == cv2.EVENT_LBUTTONDOWN:  # 마우스 왼쪽 버튼 클릭 이벤트
        save_depth = cropped_depth_image.copy()  # 현재 크롭된 깊이 이미지를 복사하여 저장
        print("Baseline depth image saved.")  # 저장 완료 메시지 출력

# 메인 루프
try:
    while True:
        # RealSense 카메라로부터 프레임 가져오기
        frames = pipeline.wait_for_frames()
        aligned_frames = align.process(frames)  # 깊이 데이터를 컬러 이미지에 정렬
        aligned_depth_frame = aligned_frames.get_depth_frame()  # 정렬된 깊이 프레임 가져오기
        color_frame = aligned_frames.get_color_frame()  # 정렬된 컬러 프레임 가져오기

        # 유효한 프레임이 없으면 건너뜀
        if not aligned_depth_frame or not color_frame:
            continue

        # 깊이 프레임과 컬러 프레임을 numpy 배열로 변환
        depth_image = np.asanyarray(aligned_depth_frame.get_data())
        img = np.asanyarray(color_frame.get_data())

        # 깊이와 컬러 이미지를 상하좌우로 반전
        depth_image = cv2.flip(depth_image, -1)
        img = cv2.flip(img, -1)

        # 관심 영역(ROI) 좌표를 사용해 이미지를 크롭
        x1, y1 = roi_pts[0]
        x2, y2 = roi_pts[1]
        cropped_image = img[y1:y2, x1:x2]  # 컬러 이미지에서 ROI 추출
        cropped_depth_image = depth_image[y1:y2, x1:x2]  # 깊이 이미지에서 ROI 추출

        # 깊이 이미지를 Median Blur로 부드럽게 처리
        cropped_depth_image = cv2.medianBlur(cropped_depth_image, 5)

        # 원본 이미지에 ROI 영역 표시
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)  # 빨간색 사각형으로 표시
        cv2.imshow('Color Image with ROI', img)  # ROI가 표시된 컬러 이미지 출력
        cv2.setMouseCallback("Color Image with ROI", mouse_callback)  # 마우스 콜백 함수 등록

        # YOLO 모델을 사용하여 객체 탐지 수행
        results = model(cropped_image)

        # 탐지된 객체 시각화를 위한 초기화
        blended_image = cropped_image.copy()  # 크롭된 이미지를 복사
        all_colored_mask = np.zeros_like(cropped_image)  # 빈 마스크 생성

        for result in results:
            if result.masks is not None:  # 탐지된 객체에 마스크가 있는 경우
                masks = result.masks.data.cpu().numpy()  # 마스크 데이터 가져오기
                classes = result.boxes.cls.cpu().numpy()  # 클래스 ID 가져오기
                class_names = model.names  # 클래스 이름 매핑

                for i, mask in enumerate(masks):
                    # 마스크를 크롭된 이미지 크기에 맞게 조정
                    resized_mask = cv2.resize(mask, (cropped_image.shape[1], cropped_image.shape[0]))
                    color_mask = (resized_mask > 0.5).astype(np.uint8)  # 바이너리 마스크 생성

                    # 클래스 이름과 색상 얻기
                    key = class_names[int(classes[i])]
                    object_name, color = cls_name_color.get(key, ("Unknown", (255, 255, 255)))

                    # 마스크 영역에 색상 적용
                    all_colored_mask[color_mask == 1] = color

                    mask_indices = np.where(color_mask > 0)
                    y_indices, x_indices = mask_indices

                    # 기준 깊이 이미지가 설정되지 않은 경우 건너뜀
                    if save_depth is None:
                        print("Baseline depth not set. Please click to set it.")
                        continue

                    # 깊이 정보 가져오기
                    z_cm_array = cropped_depth_image[mask_indices] / 10  # 현재 깊이 (cm 단위)
                    base_depth_cm_array = save_depth[mask_indices] / 10  # 기준 깊이 (cm 단위)

                    # 유효한 깊이 값 필터링
                    valid_indices = np.where((z_cm_array > 0.1) & (base_depth_cm_array > 0))
                    valid_z_cm = z_cm_array[valid_indices]
                    valid_base_depth_cm = base_depth_cm_array[valid_indices]
                    valid_height_cm = valid_base_depth_cm - valid_z_cm

                    # 픽셀 크기 계산
                    depth_intrin = aligned_depth_frame.profile.as_video_stream_profile().intrinsics
                    sensor_width_mm = 5.6  # 센서 너비 (mm)
                    sensor_height_mm = 4.2  # 센서 높이 (mm)
                    pixel_width_cm = (sensor_width_mm / depth_intrin.width) * 0.1
                    pixel_height_cm = (sensor_height_mm / depth_intrin.height) * 0.1

                    # 총 면적 및 부피 계산
                    if valid_z_cm.size > 0:
                        pixel_areas = pixel_width_cm * pixel_height_cm
                        total_area_cm2 = pixel_areas * len(valid_z_cm)
                        total_volume_cm3 = np.sum(pixel_areas * valid_height_cm)
                        average_height_cm = np.mean(valid_height_cm)
                    else:
                        total_area_cm2 = 0
                        total_volume_cm3 = 0
                        average_height_cm = 0

                    # 결과 표시
                    min_x, max_x = np.min(x_indices), np.max(x_indices)
                    min_y, max_y = np.min(y_indices), np.max(y_indices)
                    label_x = min_x
                    label_y = min(max_y + 10, color_mask.shape[0] - 10)
                    cv2.putText(
                        all_colored_mask,
                        f"V={total_volume_cm3:.2f}cm3, S={total_area_cm2:.2f}cm2, H={average_height_cm:.2f}cm",
                        (label_x, label_y),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        color,
                        2,
                    )

        # 마스크와 원본 이미지를 혼합
        blended_image = cv2.addWeighted(cropped_image, 0.7, all_colored_mask, 0.3, 0)

        # 결과 출력
        cv2.imshow('Segmented Mask with Heights', blended_image)

        # ESC 키 입력 시 루프 종료
        if cv2.waitKey(1) == 27:
            break

# 종료 시 리소스 해제
finally:
    pipeline.stop()  # RealSense 파이프라인 정지
    cv2.destroyAllWindows()  # 모든 OpenCV 창 닫기
