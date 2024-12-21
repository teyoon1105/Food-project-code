import pyrealsense2 as rs  # Intel RealSense 카메라 라이브러리
import numpy as np  # 배열 및 수치 계산을 위한 라이브러리
import cv2  # OpenCV: 이미지 및 비디오 처리 라이브러리
import os  # 파일 및 경로 작업을 위한 라이브러리
from ultralytics import YOLO  # YOLO 모델을 사용하기 위한 라이브러리
import torch  # 딥러닝 모델 처리를 위한 PyTorch 라이브러리
import multiprocessing as mp  # 병렬 처리를 위한 멀티프로세싱 라이브러리
import logging  # 로깅 기능 제공 라이브러리

# 초점 거리(Focal Length) 상수 설정 (카메라의 고정값)
FOCAL_LENGTH_X = 642.9123515625  # X축 초점 거리
FOCAL_LENGTH_Y = 642.9123515625  # Y축 초점 거리

# 관심 영역(ROI) 좌표 설정: 좌상단과 우하단 좌표
ROI_POINTS = [(175, 50), (1055, 690)]  # 관심 영역의 범위를 지정

# 객체 이름과 색상 매핑 딕셔너리
CLS_NAME_COLOR = {
    '01011001': ('Rice', (255, 0, 255)),  # 자주색
    '01012006': ('Black Rice', (255, 0, 255)),
    '01012002': ('Soy bean Rice', (255, 0, 255)),
    '03011011': ('Pumpkin soup', (255, 0, 255)),
    '04011005': ('Seaweed Soup', (0, 255, 255)),  # 노란색
    '04011007': ('Beef stew', (0, 255, 255)),
    '06012004': ('Tteokgalbi', (0, 255, 0)),  # 초록색
    '07014001': ('EggRoll', (0, 0, 255)),  # 빨간색
    '08012001': ('Stir-fried Potatoes', (255, 255, 0)),  # 노랑
    '12011008': ('Kimchi', (100, 100, 100)),  # 회색
}

def frame_preprocessing(frame_queue, stop_event, save_depth):
    """
    카메라에서 프레임을 가져와 전처리한 후 큐에 삽입하는 함수.
    """
    pipeline = rs.pipeline()  # Intel RealSense 파이프라인 생성
    config = rs.config()  # 카메라 설정을 위한 Config 객체 생성
    config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)  # 깊이 스트림 활성화
    config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)  # 컬러 스트림 활성화
    align = rs.align(rs.stream.color)  # 깊이 프레임을 컬러 프레임에 정렬
    pipeline.start(config)  # 파이프라인 시작

    try:
        while not stop_event.is_set():  # 종료 이벤트가 설정되지 않은 동안 계속 실행
            frames = pipeline.wait_for_frames()  # 새 프레임 가져오기
            aligned_frames = align.process(frames)  # 컬러 프레임에 맞게 정렬
            depth_frame = aligned_frames.get_depth_frame()  # 정렬된 깊이 프레임 추출
            color_frame = aligned_frames.get_color_frame()  # 정렬된 컬러 프레임 추출

            if not depth_frame or not color_frame:  # 프레임 유효성 확인
                continue  # 유효하지 않으면 다음 반복으로

            depth_image = np.asanyarray(depth_frame.get_data())  # 깊이 데이터를 NumPy 배열로 변환
            color_image = np.asanyarray(color_frame.get_data())  # 컬러 데이터를 NumPy 배열로 변환

            depth_image = cv2.flip(depth_image, -1)  # 깊이 이미지를 좌우+상하 반전
            color_image = cv2.flip(color_image, -1)  # 컬러 이미지를 좌우+상하 반전

            x1, y1 = ROI_POINTS[0]  # ROI의 좌상단 좌표
            x2, y2 = ROI_POINTS[1]  # ROI의 우하단 좌표

            if save_depth is None:  # 초기 깊이 데이터가 없는 경우
                cv2.putText(color_image, "Save depth first!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                cv2.imshow("Frame", color_image)  # 메시지를 화면에 표시
                key = cv2.waitKey(10)  # 키 입력 대기
                if key == ord('s'):  # 's' 키를 누르면 깊이 데이터 저장
                    save_depth[:] = depth_image  # 깊이 데이터를 저장
                    np.save('save_depth.npy', depth_image)  # 파일로 저장
                    print("Saved depth image.")
            else:
                roi_depth = depth_image[y1:y2, x1:x2]  # ROI 영역 자르기
                roi_color = color_image[y1:y2, x1:x2]  # ROI 영역 자르기
                frame_queue.put((roi_depth, roi_color))  # 큐에 깊이와 컬러 프레임 삽입

            cv2.rectangle(color_image, (x1, y1), (x2, y2), (0, 0, 255), 2)  # ROI 영역 사각형 그리기
            cv2.imshow("Color Frame with ROI", color_image)  # 컬러 프레임 출력
            cv2.waitKey(10)  # 키 입력 대기
    finally:
        pipeline.stop()  # 파이프라인 정지

def gpu_inference(frame_queue, result_queue, model_path, stop_event):
    """
    YOLO 모델을 사용하여 객체를 탐지한 후 결과를 큐에 넣는 함수.
    """
    model = YOLO(model_path).to('cuda' if torch.cuda.is_available() else 'cpu')  # YOLO 모델 로드 (GPU 또는 CPU)
    print(f"YOLO 모델이 GPU에서 실행 중입니다.")

    while not stop_event.is_set():  # 종료 이벤트가 설정되지 않은 동안 실행
        if not frame_queue.empty():  # 프레임 큐에 데이터가 있으면
            roi_depth, roi_color = frame_queue.get()  # ROI 깊이 및 컬러 프레임 가져오기
            results = model(roi_color)  # YOLO 모델 추론
            output_data = []  # 탐지 결과를 저장할 리스트

            for result in results:
                if result.masks is not None:  # 탐지된 객체가 있는 경우
                    masks = result.masks.data.cpu().numpy()  # 마스크 데이터 가져오기
                    classes = result.boxes.cls.cpu().numpy()  # 클래스 ID 가져오기
                    confs = result.boxes.conf.cpu().numpy()  # 신뢰도 가져오기

                    for i, mask in enumerate(masks):  # 각 객체에 대해 반복
                        class_key = model.names[int(classes[i])]  # 클래스 ID로 클래스 이름 찾기
                        object_name, color = CLS_NAME_COLOR.get(class_key, ("Unknown", (255, 255, 255)))  # 이름과 색상 매핑
                        conf = confs[i]  # 신뢰도 가져오기
                        resized_mask = cv2.resize(mask, (roi_color.shape[1], roi_color.shape[0]), interpolation=cv2.INTER_NEAREST)  # 마스크 크기 조정
                        output_data.append((object_name, color, conf, resized_mask))  # 결과 저장

            result_queue.put((roi_depth, roi_color, output_data))  # 결과 큐에 삽입

def calculate_volume(result_queue, volume_queue, stop_event, save_depth):
    """
    탐지된 객체의 부피를 계산한 후 결과를 큐에 넣는 함수.
    """
    while not stop_event.is_set():  # 종료 이벤트가 설정되지 않은 동안 실행
        if not result_queue.empty():  # 결과 큐에 데이터가 있으면
            roi_depth, roi_color, output_data = result_queue.get()  # 데이터 가져오기
            volume_data = []  # 부피 계산 결과를 저장할 리스트

            for object_name, color, conf, mask in output_data:  # 탐지된 객체에 대해 반복
                mask_indices = np.where(mask > 0.5)  # 마스크에서 활성화된 픽셀 좌표 가져오기
                total_volume = 0  # 부피 초기화

                for y, x in zip(*mask_indices):  # 마스크 픽셀 순회
                    z_cm = roi_depth[y, x] / 10  # 깊이 값을 cm로 변환
                    base_depth_cm = save_depth[y, x] / 10  # 기준 깊이 값을 cm로 변환
                    if z_cm > 25 and base_depth_cm > 25:  # 유효한 깊이 값만 사용
                        height_cm = max(0, base_depth_cm - z_cm)  # 높이 계산
                        pixel_area_cm2 = (z_cm ** 2) / (FOCAL_LENGTH_X * FOCAL_LENGTH_Y)  # 픽셀의 면적 계산
                        total_volume += pixel_area_cm2 * height_cm  # 부피 누적

                volume_data.append((object_name, total_volume, conf, color, mask_indices))  # 결과 저장

            volume_queue.put((roi_color, volume_data))  # 부피 결과를 큐에 삽입

def visualize_results(volume_queue, stop_event):
    """
    객체 탐지 결과를 시각화하는 함수.
    """
    while not stop_event.is_set():  # 종료 이벤트가 설정되지 않은 동안 실행
        if not volume_queue.empty():  # 부피 큐에 데이터가 있으면
            roi_color, volume_data = volume_queue.get()  # 데이터 가져오기
            for object_name, volume, conf, color, mask_indices in volume_data:  # 결과 반복
                y_indices, x_indices = mask_indices

                # 마스크 영역을 색상으로 강조
                for y, x in zip(y_indices, x_indices):
                    roi_color[y, x] = roi_color[y, x] * 0.5 + np.array(color) * 0.5

                # 마스크의 최소 좌표에 텍스트 표시
                min_y, min_x = np.min(y_indices), np.min(x_indices)
                text = f"{object_name} {volume:.1f}cm^3 {conf:.2f}"  # 이름, 부피, 신뢰도
                cv2.putText(roi_color, text, (min_x, min_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            cv2.imshow("Visualized Results", roi_color)  # 시각화된 결과 출력
            if cv2.waitKey(10) == 27:  # ESC 키 입력 시 종료
                stop_event.set()
                break

def main():
    """
    프로그램의 메인 함수.
    """
    model_path = os.path.join(os.getcwd(), 'model', 'total_50org_100scaled_10000mix_700_96_a1002_best.pt')  # YOLO 모델 경로
    save_depth = np.load('save_depth.npy') if os.path.exists('save_depth.npy') else None  # 저장된 깊이 데이터 로드

    frame_queue = mp.Queue(maxsize=5)  # 프레임 큐 생성
    result_queue = mp.Queue(maxsize=5)  # 결과 큐 생성
    volume_queue = mp.Queue(maxsize=5)  # 부피 큐 생성
    stop_event = mp.Event()  # 종료 이벤트 생성

    processes = [  # 병렬 처리 프로세스 생성
        mp.Process(target=frame_preprocessing, args=(frame_queue, stop_event, save_depth)),
        mp.Process(target=gpu_inference, args=(frame_queue, result_queue, model_path, stop_event)),
        mp.Process(target=calculate_volume, args=(result_queue, volume_queue, stop_event, save_depth)),
        mp.Process(target=visualize_results, args=(volume_queue, stop_event))
    ]

    for process in processes:  # 프로세스 시작
        process.start()

    try:
        for process in processes:  # 프로세스 종료 대기
            process.join()
    except KeyboardInterrupt:  # 키보드 인터럽트 시 처리
        print("Program interrupted. Cleaning up...")
        stop_event.set()  # 종료 이벤트 설정
        for process in processes:
            process.terminate()  # 프로세스 강제 종료
            process.join()
    finally:
        cv2.destroyAllWindows()  # OpenCV 창 닫기

if __name__ == "__main__":
    logging.getLogger("ultralytics").setLevel(logging.WARNING)  # YOLO 라이브러리 로그 수준 설정
    main()  # 메인 함수 실행
