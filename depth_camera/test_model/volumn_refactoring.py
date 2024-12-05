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
Now_path = os.getcwd()
model_folder = os.path.join(Now_path, 'model')

model_list = ['1st_500_best.pt',
              '1st_contrast_best.pt',
              '1st_mix_scaled_best.pt', 
              '1st_scaled_best.pt', 
              '1st_sharpning_best.pt'
            ]

model_name = model_list[3]
MODEL_PATH = os.path.join(model_folder, model_name)
model = YOLO(MODEL_PATH)

# Intel RealSense 카메라 설정
pipeline = rs.pipeline() # 파이프라인 생성
config = rs.config() # 카메라 설정 객체 생성
config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30) # 깊이 스트림 설정
config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30) # 컬러 스트림 설정

# ROI 설정
ROI_POINTS = [(175, 50), (1055, 690)] 
BRIGHTNESS_INCREASE = 50 # ROI 영역 컬러 프레임의 밝기를 높일 값

# 비디오 저장 설정
# 필요시에 사용
fourcc = cv2.VideoWriter_fourcc(*'XVID')  # 코덱 설정 (적절한 코덱 사용)
fps = 30.0  # 프레임 레이트
width = ROI_POINTS[1][0] - ROI_POINTS[0][0]  # ROI 너비
height = ROI_POINTS[1][1] - ROI_POINTS[0][1] # ROI 높이
video_name = 'test' + model_name + '.avi'
out = cv2.VideoWriter(video_name, fourcc, fps, (width, height)) # 파일명, 코덱, 프레임 레이트, 크기


# 클래스 ID와 이름, 색상을 매핑
CLS_NAME_COLOR = {
    '01011001': ('Rice', (255, 0, 255)), # 자주색
    '01012006': ('Black Rice', (255, 0, 255)),
    '04011005': ('Seaweed Soup', (0, 255, 255)),
    '04011008': ('Beef stew', (0, 255, 255)),
    '04017001': ('Soybean Soup', (0, 255, 255)), # 노란색
    '06012004': ('Tteokgalbi', (0, 255, 0)), # 초록색
    '06012008': ('Beef Bulgogi', (0, 255, 0)),
    '07014001': ('EggRoll', (0, 0, 255)), # 빨간색
    '08011003': ('Stir-fried anchovies', (0, 0, 255)),
    '10012001': ('Chicken Gangjeong', (0, 0, 255)),
    '11014002': ('Gosari', (255, 255, 0)),
    '11013007': ('Spinach', (255, 255, 0)), # 청록색
    '12011008': ('Kimchi', (100, 100, 100)),
    '12011003': ('Radish Kimchi', (100, 100, 100))
}

# ---- 전역 변수 ----
save_depth = None  # 기준 깊이 데이터 저장 변수


# ---- 함수 정의 ----
def initialize_camera():
    """카메라 파이프라인 초기화 및 시작"""
    pipeline.start(config) # 파이프 라인 시작(프레임 받아옴)
    align_to = rs.stream.color # 깊이 스트림을 컬러 스트림에 맞게 정렬
    return rs.align(align_to) # 정렬 객체 반환


def capture_frames(align):
    """카메라에서 프레임 캡처 및 정렬"""
    frames = pipeline.wait_for_frames() # 파이프 라인을 통해 받아온 프레임 대기
    aligned_frames = align.process(frames) # 깊이 스트림을 컬러 프레임에 정렬
    return aligned_frames.get_depth_frame(), aligned_frames.get_color_frame() # 깊이 스트림, 컬러 스트림 반환


def preprocess_images(depth_frame, color_frame):
    """깊이 및 컬러 프레임 전처리"""
    depth_image = np.asanyarray(depth_frame.get_data()) # 깊이 프레임을 넘파이 배열로 변환
    color_image = np.asanyarray(color_frame.get_data()) # 컬러 프레임을 넘파이 배열로 변환
    depth_image = cv2.flip(depth_image, -1) # 깊이 넘파이 배열을 상하좌우 반전
    color_image = cv2.flip(color_image, -1) # 컬러 넘파이 배열을 상하좌우 반전
    return depth_image, color_image # 반전된 깊이, 컬러 넘파이 배열 반환


def crop_roi(image, roi_points):
    """이미지에서 관심 영역(ROI) 크롭"""
    x1, y1 = roi_points[0] # ROI 좌상단 좌표
    x2, y2 = roi_points[1] # ROI 우하단 좌표
    return image[y1:y2, x1:x2] # 컬러 넘파이 배열을 ROI 영역만큼 자른 값 반환


def mouse_callback(event, x, y, flags, param):
    """마우스 콜백 함수로 깊이 데이터 저장"""
    global save_depth # 전역 변수 save_depth를 사용(안하면 지역변수 됨)
    if event == cv2.EVENT_LBUTTONDOWN: # 마우스 클릭 시
        save_depth = param.copy() # param 여기선 ROI 영역의 깊이 배열 정보를 저장
        print("Depth image saved!") # 저장 완료 terminal에 출력


def calculate_volume(cropped_depth, save_depth, mask_indices, depth_intrin, min_depth_cm=20):
    """깊이 데이터를 이용하여 부피 계산"""
    total_volume = 0 # 총 부피를 저장한 값 초기화
    y_indices, x_indices = mask_indices # 모델이 예측한 마스크 좌표들 저장, 넘파이라 y(행)먼저

    for pixel_y, pixel_x in zip(y_indices, x_indices): # 좌표를 하나씩 zip을 통해 받아옴
        z_cm = cropped_depth[pixel_y, pixel_x] / 10  # 가져온 좌표의 깊이 cm로 변환
        base_depth_cm = save_depth[pixel_y, pixel_x] / 10  # 전역변수로 저장된 깊이 값에서 현재 좌표 깊이 cm로 변환

        if z_cm > min_depth_cm and base_depth_cm > 25: # 실제 높이값은 약 40cm정도 > 각각 20, 25 cm보다 작은 값이면 아예 받질 않는다(잘못된 값 필터링)
            height_cm = base_depth_cm - z_cm # 저장 깊이 - 현재 깊이 > 각 픽셀의 높이
            pixel_area_cm2 = (z_cm ** 2) / (depth_intrin.fx * depth_intrin.fy) # 공식을 통해 픽셀 넓이 추정
            total_volume += pixel_area_cm2 * height_cm # 한 픽셀을 직육면체라고 가정하고 부피 값을 구하고, 해당 값을 총 부피에 저장

    return total_volume # 총 부피 반환


def visualize_results(cropped_image, all_colored_mask, object_name, total_volume, conf, color, mask_indices):
    """탐지 결과를 시각화"""
    y_indices, x_indices = mask_indices # 모델이 예측한 마스크 좌표를 가져옴
    min_x = np.min(x_indices) # 우 상단에 필요한 최소 x 좌표
    min_y = np.min(y_indices) # 우 상단에 필요한 최소 y 좌표
    label_position = (min_x, min_y - 10) # 마스크에 겹치지 않게 약간 윗쪽으로

    # 마스크 색상 적용 및 텍스트 표시
    cv2.putText(all_colored_mask, f"{object_name}:V:{total_volume:.2f}cm_3,C:{conf:.2f}", # 인자로 받은 컬러 마스크 위에 객체 이름과 부피값을 텍스트로 작성
                label_position, cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    return cv2.addWeighted(cropped_image, 0.7, all_colored_mask, 0.3, 0) # 텍스트가 작성된 컬러 마스크와 roi 영역 합치기


# ---- 메인 처리 루프 ----
def main():
    global save_depth
    align = initialize_camera()

    try:
        while True:
            depth_frame, color_frame = capture_frames(align) # 카메라 정렬

            if not depth_frame or not color_frame: # 프레임이 컬러나 깊이가 아니면 다음 반복
                continue

            depth_image, color_image = preprocess_images(depth_frame, color_frame) # 컬러 프레임에 맞게 깊이 프레임 정렬, 전처리
            cropped_color = crop_roi(color_image, ROI_POINTS) # 컬러 넘파이 배열 ROI 영역 생성
            cropped_depth = crop_roi(depth_image, ROI_POINTS) # 깊이 넘파이 배열 ROI 영역 생성

            # ROI 표시
            cv2.rectangle(color_image, ROI_POINTS[0], ROI_POINTS[1], (0, 0, 255), 2)
            cv2.imshow('Color Image with ROI', color_image)
            cv2.setMouseCallback("Color Image with ROI", mouse_callback, cropped_depth) # param으로 ROI 영역의 깊이 넘파이 배열을 전달

            # ROI 컬러 넘파이 이미지에 밝기 증가
            brightened_image = cv2.convertScaleAbs(cropped_color, alpha=1, beta=BRIGHTNESS_INCREASE) 

            # 객체 탐지 수행
            results = model(brightened_image) # 모델에 밝기 높인 이미지 입력
            all_colored_mask = np.zeros_like(brightened_image) # 마스크를 만들기 위한 빈 넘파이 배열
            blended_image = brightened_image.copy()  # 기본값으로 초기화

            for result in results: # 모델의 결과값
                if result.masks is not None: # 마스크 결과값이 있으면 
                    masks = result.masks.data.cpu().numpy() # 결과값 마스크 데이터를 넘파이 형으로
                    classes = result.boxes.cls.cpu().numpy() # 클래스 id 값을 넘파이 형으로

                    for i, mask in enumerate(masks): # 마스크와 i값을 enumerate로 가져오기
                        # 탐지 객체의 conf 출력
                        conf = result.boxes.conf[i]
                        resized_mask = cv2.resize(mask, (brightened_image.shape[1], brightened_image.shape[0])) # 모델이 뱉은 크기랑 이미지랑 맞추기
                        color_mask = (resized_mask > 0.5).astype(np.uint8) # 마스크 배열의 확률이 0.5 보다 큰 놈들만 컬러마스크 배열로 저장

                        class_key = model.names[int(classes[i])] # 모델의 객체 이름에서 클래스 id에 해당하는 애들 가져와서
                        object_name, color = CLS_NAME_COLOR.get(class_key, ("Unknown", (255, 255, 255))) # 해당 객체 이름을 키값으로 하는 밸류값과, 색상값 가져오기

                        mask_indices = np.where(color_mask > 0) # 컬러마스크에서 0보다 큰 값들을 넘파이 좌표값으로 설정

                        if save_depth is None: # 저장되지 않으면 실행 안되게
                            cv2.putText(color_image, "Save your depth first!", (390,370), cv2.FONT_HERSHEY_TRIPLEX, 1, (0,0,255), 2)
                            cv2.imshow('Color Image with ROI', color_image)
                            continue

                        valid_depths = cropped_depth[mask_indices] # 마스크 좌표에 해당하는 좌표들을 깊이 이미지에서 슬라이싱
                        if len(valid_depths[valid_depths > 0]) == 0:
                            continue

                        # 카메라 내부 파라미터
                        depth_intrin = depth_frame.profile.as_video_stream_profile().intrinsics

                        # 부피 구하기
                        total_volume = calculate_volume(cropped_depth, save_depth, mask_indices, depth_intrin)

                        # 개별 마스크에 색상 적용
                        all_colored_mask[color_mask == 1] = color
                        
                        # 색상 마스크와 컬러 이미지 합치기
                        blended_image = visualize_results(brightened_image, all_colored_mask, object_name, total_volume, conf, color, mask_indices)

                        out.write(blended_image)

            # 결과 이미지 표시
            cv2.imshow('Segmented Mask with Heights', blended_image)

            if cv2.waitKey(1) == 27:  # ESC 키로 종료
                break
    finally:
        pipeline.stop() # 파이프 라인 종료
        cv2.destroyAllWindows() # 창 닫기
        out.release() # 비디오 writer 해제

if __name__ == "__main__": # 코드를 실행하면
    main() # 메인함수 실행