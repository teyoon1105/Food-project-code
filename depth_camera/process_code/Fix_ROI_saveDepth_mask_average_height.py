import pyrealsense2 as rs  # Intel RealSense 카메라 라이브러리
import numpy as np  # 수치 연산 라이브러리
import cv2  # OpenCV, 컴퓨터 비전 및 이미지 처리 라이브러리
import os  # 운영 체제 관련 작업 라이브러리
from ultralytics import YOLO  # YOLO 객체 탐지 라이브러리
import logging  # 로깅을 위한 라이브러리

# 로그 레벨을 WARNING 이상으로 설정하여 INFO 메시지를 비활성화
logging.getLogger("ultralytics").setLevel(logging.WARNING)

# 현재 작업 디렉토리를 얻고 모델 파일 경로를 설정
Now_Path = os.getcwd()
model_path = os.path.join(Now_Path, 'best.pt')
model = YOLO(model_path)  # YOLO 모델 로드

# RealSense 카메라 스트림 설정
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)  # 깊이 스트림 설정
config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)  # 컬러 스트림 설정
pipeline.start(config)  # 스트림 시작

align_to = rs.stream.color  # 깊이 데이터를 컬러 이미지에 정렬
align = rs.align(align_to)

# 관심 영역 (ROI) 좌표 설정
roi_pts = [(410, 180), (870, 540)]

# 비디오 저장 설정 (필요시 사용 가능)
# 코덱 설정, 프레임 속도 및 ROI 크기 설정
# fourcc = cv2.VideoWriter_fourcc(*'XVID')
# fps = 30.0
# width = roi_pts[1][0] - roi_pts[0][0]
# height = roi_pts[1][1] - roi_pts[0][1]
# out = cv2.VideoWriter('cropped_output.avi', fourcc, fps, (width, height))

brightness_increase = 50  # 밝기 증가 값 (0~255)

# 클래스 이름과 색상을 매핑
cls_name_color = {
    '01011001': ('Rice', (255, 0, 255)),  # 보라색
    '06012004': ('Tteokgalbi', (0, 255, 0)),  # 초록색
    '07014001': ('eggRoll', (0, 0, 255)),  # 빨간색
    '11013007': ('Spinach greens', (255, 255, 0)),  # 노란색
    '04017001': ('soy bean paste soup', (0, 255, 255))  # 하늘색
}

save_depth = None  # 트레이 깊이 데이터를 저장하기 위한 변수 초기화

# 마우스 이벤트 콜백 함수
def mouse_callback(event, x, y, flags, param):
    global save_depth  # 전역 변수 사용 선언

    if event == cv2.EVENT_LBUTTONDOWN:  # 마우스 좌클릭 이벤트
        save_depth = cropped_depth_image  # 현재 깊이 이미지를 저장
        print("Depth image saved!")  # 저장 알림

# 메인 실행 루프
try:
    while True:
        # RealSense에서 프레임을 가져옴
        frames = pipeline.wait_for_frames()
        aligned_frames = align.process(frames)  # 깊이와 컬러를 정렬
        aligned_depth_frame = aligned_frames.get_depth_frame()  # 깊이 프레임 가져오기
        color_frame = aligned_frames.get_color_frame()  # 컬러 프레임 가져오기

        if not aligned_depth_frame or not color_frame:
            continue  # 프레임이 없으면 다음 루프로 이동

        # 프레임 데이터를 numpy 배열로 변환
        depth_image = np.asanyarray(aligned_depth_frame.get_data())
        img = np.asanyarray(color_frame.get_data())

        # 이미지 상하좌우 반전 (필요 시 활성화)
        depth_image = cv2.flip(depth_image, -1)
        img = cv2.flip(img, -1)

        # ROI 설정
        x1, y1 = roi_pts[0]
        x2, y2 = roi_pts[1]
        cropped_image = img[y1:y2, x1:x2]  # 컬러 이미지에서 ROI 추출
        cropped_depth_image = depth_image[y1:y2, x1:x2]  # 깊이 이미지에서 ROI 추출

        # ROI 영역을 직사각형으로 표시
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.imshow('Color Image with ROI', img)  # 직사각형이 들어간 이미지를 원본 화면에 표시
        cv2.setMouseCallback("Color Image with ROI", mouse_callback)  # 원본 화면 창에 마우스 콜백 설정

        # 밝기를 증가시킨 이미지 생성
        brighter_image = cv2.convertScaleAbs(cropped_image, alpha=1, beta=brightness_increase)

        # YOLO 모델을 사용하여 객체 탐지 수행
        results = model(brighter_image)

        # 시각화를 위한 초기화
        blended_image = brighter_image.copy()  # 원본 밝기 증가 이미지를 복사, 마스크를 시각화하기 위해
        all_colored_mask = np.zeros_like(brighter_image)  # 마스크 초기화

        for result in results:
            if result.masks is not None:  # 탐지된 객체의 마스크가 있는 경우
                masks = result.masks.data.cpu().numpy()  # 마스크 데이터 추출
                classes = result.boxes.cls.cpu().numpy()  # 클래스 ID 추출
                class_names = model.names  # 클래스 이름 매핑

                for i, mask in enumerate(masks):
                    # 마스크 크기를 ROI 크기에 맞게 조정
                    resized_mask = cv2.resize(mask, (brighter_image.shape[1], brighter_image.shape[0]))
                    color_mask = (resized_mask > 0.5).astype(np.uint8)  # 바이너리 마스크 생성

                    # 클래스 ID로 객체 이름과 색상 얻기
                    key = class_names[int(classes[i])]
                    # 만들어준 영어이름과 색의 딕셔너리에서 key에 맞는(라벨) 값 가져와서 맵핑
                    object_name, color = cls_name_color.get(key, ("Unknown", (255, 255, 255)))

                    # ROI에서 마스크 영역에 색상 적용
                    all_colored_mask[color_mask == 1] = color

                    # 마스크의 유효 픽셀 좌표와 깊이 데이터 계산
                    mask_indices = np.where(color_mask > 0)
                    y_indices, x_indices = mask_indices
                    original_y_indices = y_indices + y1  # 원본 이미지 좌표로 변환
                    original_x_indices = x_indices + x1

                    # 깊이 데이터 추출 및 유효 깊이 계산
                    # 원본에서 깊이값을 가져옴
                    # 때문에 ROI의 좌표를 원본 좌표로 바꾸는 과정이 필요함
                    masked_depth_values = depth_image[original_y_indices, original_x_indices]
                    valid_depths = masked_depth_values[masked_depth_values > 0]

                    if len(valid_depths) == 0:  # 유효 깊이 값이 없으면 건너뜀
                        print("유효한 깊이 값 없음")
                        continue

                    # 트레이와 객체 높이 계산
                    # ROI의 배열 중 마스크 배열 부분의 깊이 정보만 가져옴
                    tray_mask_depth = save_depth[mask_indices]
                    # 해당 깊이 값이 0 보다 큰 애들만 가져옴
                    tray_valid_depths = tray_mask_depth[tray_mask_depth > 0]

                    tray_average_mm = np.mean(tray_valid_depths)  # 트레이 평균 깊이 (mm)
                    tray_average_cm = tray_average_mm / 10  # 트레이 평균 깊이 (cm)

                    average_distance_mm = np.mean(valid_depths)  # 객체 평균 깊이 (mm)
                    average_distance_cm = average_distance_mm / 10  # 객체 평균 깊이 (cm)

                    average_height_cm = tray_average_cm - average_distance_cm  # 객체 높이 (cm)

                    # 객체 이름과 높이를 이미지에 표시
                    min_x, max_x = np.min(x_indices), np.max(x_indices)
                    min_y, max_y = np.min(y_indices), np.max(y_indices)
                    label_x, label_y = min_x, min_y - 10
                    cv2.putText(all_colored_mask, f"{object_name}: {average_height_cm:.2f}cm",
                                (label_x, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # 원본 이미지와 마스크 혼합
        blended_image = cv2.addWeighted(brighter_image, 0.7, all_colored_mask, 0.3, 0)

        # 결과 이미지 출력
        cv2.imshow('Segmented Mask with Heights', blended_image)

        # ESC 키 입력 시 루프 종료
        if cv2.waitKey(1) == 27:
            break

# 종료 시 리소스 해제
finally:
    pipeline.stop()
    cv2.destroyAllWindows()
    # out.release()  # 비디오 writer 해제
