import pyrealsense2 as rs  # Intel RealSense 카메라 제어를 위한 라이브러리
import numpy as np  # 수치 연산을 위한 라이브러리
import cv2  # 컴퓨터 비전 처리를 위한 라이브러리
import os  # 파일 및 디렉토리 작업을 위한 라이브러리
from ultralytics import YOLO  # YOLO 객체 탐지 모델
import torch  # PyTorch 딥러닝 프레임워크
import multiprocessing as mp  # 병렬 처리를 위한 멀티프로세싱
from multiprocessing.synchronize import Event
import logging  # 로깅 기능
import queue  # 프로세스 간 데이터 통신을 위한 큐
# from typing import Tuple, List, Dict  # 타입 힌팅을 위한 타입 정의
import time  # 시간 측정을 위한 모듈

# 카메라 내부 파라미터 (초점 거리) 설정
FOCAL_LENGTH_X = 642.9123515625  # X축 초점 거리 (픽셀 단위)
FOCAL_LENGTH_Y = 642.9123515625  # Y축 초점 거리 (픽셀 단위)

# 관심 영역(ROI) 좌표 설정: [(좌상단 x,y), (우하단 x,y)]
ROI_POINTS = [(175, 50), (1055, 690)]

# 클래스 ID와 해당하는 이름, 시각화 색상 매핑
CLS_NAME_COLOR = {
    '01011001': ('Rice', (255, 0, 255)), # 자주색
    '01012006': ('Black Rice', (255, 0, 255)),
    '01012002': ('Soy bean Rice', (255, 0, 255)),
    '03011011': ('Pumpkin soup', (255, 0, 255)),
    '04011005': ('Seaweed Soup', (0, 255, 255)),
    '04011007': ('Beef stew', (0, 255, 255)),
    '04017001': ('Soybean Soup', (0, 255, 255)), # 노란색
    '04011011': ('Fish cake soup', (0, 255, 255)),
    '06012004': ('Tteokgalbi', (0, 255, 0)), # 초록색
    '06012008': ('Beef Bulgogi', (0, 255, 0)),
    '07014001': ('EggRoll', (0, 0, 255)), # 빨간색
    '08011003': ('Stir-fried anchovies', (0, 0, 255)),
    '10012001': ('Chicken Gangjeong', (0, 0, 255)),
    '07013003': ('Kimchijeon', (0, 0, 255)),
    '08012001': ('Stir-fried Potatoes', (255,255,0)),
    '11013010': ('KongNamul', (255, 255, 0)),
    '11013002': ('Gosari', (255, 255, 0)),
    '11013007': ('Spinach', (255, 255, 0)), # 청록색
    '12011008': ('Kimchi', (100, 100, 100)),
    '12011003': ('Radish Kimchi', (100, 100, 100))
}

def frame_preprocessing(frame_queue: mp.Queue, stop_event: Event, save_depth: np.ndarray):
    """
    카메라에서 프레임을 획득하고 전처리하는 프로세스
    
    Args:
        frame_queue: 처리된 프레임을 다음 프로세스로 전달하는 큐
        stop_event: 프로세스 종료를 위한 이벤트 플래그
        save_depth: 저장된 기준 깊이 데이터 (ROI 영역만)
    """
    # RealSense 카메라 초기화
    pipeline = rs.pipeline()
    config = rs.config()
    
    # 스트림 설정 (1280x720, 30fps)
    config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
    
    # 깊이-컬러 프레임 정렬을 위한 객체
    align = rs.align(rs.stream.color)
    
    try:
        pipeline.start(config)  # 카메라 시작
        
        while not stop_event.is_set():
            # 프레임 획득
            start_time = time.time()  # 시작 시간 기록
            frames = pipeline.wait_for_frames()
            aligned_frames = align.process(frames)  # 깊이-컬러 프레임 정렬
            depth_frame = aligned_frames.get_depth_frame()
            color_frame = aligned_frames.get_color_frame()

            # 유효한 프레임 확인
            if not depth_frame or not color_frame:
                continue

            # 프레임을 NumPy 배열로 변환
            depth_image = np.asanyarray(depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())

            # 이미지 상하좌우 반전
            depth_image = cv2.flip(depth_image, -1)
            color_image = cv2.flip(color_image, -1)

            # ROI 좌표
            x1, y1 = ROI_POINTS[0]
            x2, y2 = ROI_POINTS[1]

            # ROI 영역 추출
            roi_depth = depth_image[y1:y2, x1:x2]
            roi_color = color_image[y1:y2, x1:x2]

            # 기준 깊이가 없는 경우
            if save_depth is None:
                cv2.putText(color_image, "Save depth first!", (50, 50), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                cv2.imshow("Frame", color_image)
                key = cv2.waitKey(10)
                if key == ord('s'):  # 's' 키로 ROI 영역의 깊이 저장
                    save_depth[:] = roi_depth  # ROI 영역만 저장
                    np.save('save_depth.npy', roi_depth)
                    print("Saved ROI depth image.")
            else:
                try:
                    # 큐에 프레임 전달 (타임아웃 0.1초)
                    frame_queue.put((roi_depth, roi_color), timeout=0.1)
                except queue.Full:
                    continue

            # 전체 프레임에 ROI 영역 표시
            # cv2.rectangle(color_image, (x1, y1), (x2, y2), (0, 0, 255), 2)
            # cv2.imshow("Color Frame with ROI", color_image)
            # cv2.waitKey(10)

            end_time = time.time()  # 종료 시간 기록
            print(f"[Frame Preprocessing] Processing time: {end_time - start_time:.4f} seconds")

    finally:
        pipeline.stop()  # 카메라 종료

def gpu_inference(frame_queue: mp.Queue, result_queue: mp.Queue, 
                 volume_control_queue: mp.Queue, model_path: str, stop_event: Event):
    """
    GPU를 사용하여 객체 탐지를 수행하는 프로세스
    
    Args:
        frame_queue: 전처리된 프레임을 받는 큐
        result_queue: 탐지 결과를 전달하는 큐
        volume_control_queue: 볼륨 계산 프로세스 제어를 위한 큐
        model_path: YOLO 모델 파일 경로
        stop_event: 프로세스 종료를 위한 이벤트 플래그
    """
    # YOLO 모델 로드 (가능한 경우 GPU 사용)
    model = YOLO(model_path).to('cuda' if torch.cuda.is_available() else 'cpu')
    print("YOLO 모델이 GPU에서 실행 중입니다.")

    while not stop_event.is_set():
        try:
            if not frame_queue.empty():
                # 프레임 획득
                start_time = time.time()  # 시작 시간 기록
                roi_depth, roi_color = frame_queue.get(timeout=0.1)
                # 객체 탐지 수행
                results = model(roi_color)
                output_data = []

                # 탐지된 객체 수 확인 및 제어 신호 전송
                num_objects = len(results[0].boxes) if results[0].boxes is not None else 0
                volume_control_queue.put(num_objects)

                # 탐지 결과 처리
                for result in results:
                    if result.masks is not None:
                        masks = result.masks.data.cpu().numpy()
                        classes = result.boxes.cls.cpu().numpy()
                        confs = result.boxes.conf.cpu().numpy()

                        # 각 탐지 객체에 대한 정보 추출
                        for i, mask in enumerate(masks):
                            class_key = model.names[int(classes[i])]
                            object_name, color = CLS_NAME_COLOR.get(class_key, ("Unknown", (255, 255, 255)))
                            conf = confs[i]
                            # 마스크 크기를 ROI 크기에 맞게 조정
                            resized_mask = cv2.resize(mask, (roi_color.shape[1], roi_color.shape[0]))
                            output_data.append((object_name, color, conf, resized_mask))

                # 결과 전달
                result_queue.put((roi_depth, roi_color, output_data))
                end_time = time.time()  # 종료 시간 기록
                print(f"[Model Inference] Inference time: {end_time - start_time:.4f} seconds")
        except queue.Empty:
            continue
        except Exception as e:
            print(f"Error in GPU inference: {e}")
            continue

def calculate_volume_worker(worker_id: int, result_queue: mp.Queue, volume_queue: mp.Queue, 
                          stop_event: Event, save_depth: np.ndarray):
    """
    객체의 부피를 계산하는 워커 프로세스
    
    Args:
        worker_id: 워커 프로세스 식별자
        result_queue: 탐지 결과를 받는 큐
        volume_queue: 계산된 부피를 전달하는 큐
        stop_event: 프로세스 종료를 위한 이벤트 플래그
        save_depth: 저장된 기준 깊이 데이터
    """
    while not stop_event.is_set():
        try:
            # 결과 데이터 획득
            start_time = time.time()  # 시작 시간 기록
            data = result_queue.get(timeout=1)
            if data is None:  # 종료 신호
                break

            roi_depth, roi_color, output_data = data
            volume_data = []

            # 각 탐지 객체에 대한 부피 계산
            for object_name, color, conf, mask in output_data:
                # 마스크된 영역의 좌표 획득
                mask_indices = np.where(mask > 0.5)
                total_volume = 0

                # 픽셀별 부피 계산
                for y, x in zip(*mask_indices):
                    z_cm = roi_depth[y, x] / 10  # mm를 cm로 변환
                    base_depth_cm = save_depth[y, x] / 10  # 기준 깊이도 cm로 변환
                    
                    # 유효한 깊이값에 대해서만 계산
                    if z_cm > 25 and base_depth_cm > 25:
                        # 높이 계산
                        height_cm = max(0, base_depth_cm - z_cm)
                        # 픽셀의 실제 면적 계산
                        pixel_area_cm2 = (z_cm ** 2) / (FOCAL_LENGTH_X * FOCAL_LENGTH_Y)
                        # 부피 누적
                        total_volume += pixel_area_cm2 * height_cm

                volume_data.append((object_name, total_volume, conf, color, mask_indices))

            # 계산 결과 전달
            volume_queue.put((roi_color, volume_data))
            end_time = time.time()  # 종료 시간 기록
            print(f"[Volume Calculation Worker {worker_id}] Processing time: {end_time - start_time:.4f} seconds")

        except queue.Empty:
            continue
        except Exception as e:
            print(f"Error in volume worker {worker_id}: {e}")
            continue

def manage_volume_calculation(result_queue: mp.Queue, volume_queue: mp.Queue, 
                            volume_control_queue: mp.Queue, stop_event: Event, save_depth: np.ndarray):
    """
    볼륨 계산 프로세스들을 관리하는 프로세스
    
    Args:
        result_queue: 탐지 결과를 받는 큐
        volume_queue: 계산된 부피를 전달하는 큐
        volume_control_queue: 워커 수 제어를 위한 큐
        stop_event: 프로세스 종료를 위한 이벤트 플래그
        save_depth: 저장된 기준 깊이 데이터
    """
    workers = []  # 워커 프로세스 리스트
    active_workers = 0  # 현재 활성화된 워커 수
    max_workers = 2  # 최대 워커 수

    # 초기 워커 프로세스 생성
    for i in range(max_workers):
        worker = mp.Process(target=calculate_volume_worker,
                          args=(i, result_queue, volume_queue, stop_event, save_depth))
        workers.append(worker)

    while not stop_event.is_set():
        try:
            start_time = time.time()  # 시작 시간 기록
            # 탐지된 객체 수 확인
            num_objects = volume_control_queue.get(timeout=1)
            # 필요한 워커 수 결정 (4개 이상: 2개, 그 외: 1개)
            required_workers = 2 if num_objects >= 4 else 1

            # 워커 수 조정
            if required_workers > active_workers:
                # 워커 추가
                for i in range(active_workers, required_workers):
                    if not workers[i].is_alive():
                        workers[i] = mp.Process(target=calculate_volume_worker,
                                             args=(i, result_queue, volume_queue, stop_event, save_depth))
                        workers[i].start()
                active_workers = required_workers
            elif required_workers < active_workers:
                # 워커 감소
                for i in range(required_workers, active_workers):
                    if workers[i].is_alive():
                        result_queue.put(None)  # 종료 신호 전송
                        workers[i].join(timeout=1)
                        if workers[i].is_alive():
                            workers[i].terminate()
                active_workers = required_workers

            end_time = time.time()  # 종료 시간 기록
            print(f"[Volume Management] Management time: {end_time - start_time:.4f} seconds")

        except queue.Empty:
            continue
        except Exception as e:
            print(f"Error om volume calculateion manager: {e}")
            continue

def visualize_results(volume_queue: mp.Queue, stop_event: Event):
    """
    계산된 부피와 탐지 결과를 시각화하는 프로세스
    
    Args:
        volume_queue: 계산된 부피 데이터를 받는 큐
        stop_event: 프로세스 종료를 위한 이벤트 플래그
    """
    while not stop_event.is_set():
        try:
            start_time = time.time()  # 시작 시간 기록
            if not volume_queue.empty():
                # 볼륨 데이터 획득
                roi_color, volume_data = volume_queue.get(timeout=0.1)
                
                # 각 탐지 객체에 대한 시각화
                for object_name, volume, conf, color, mask_indices in volume_data:
                    y_indices, x_indices = mask_indices
                    
                    # 마스크 영역 컬러 오버레이
                    for y, x in zip(y_indices, x_indices):
                        # 원본 이미지와 마스크 색상을 50:50 비율로 혼합
                        roi_color[y, x] = roi_color[y, x] * 0.5 + np.array(color) * 0.5

                    # 객체 정보 텍스트 표시 위치 계산
                    min_y = np.min(y_indices) if len(y_indices) > 0 else 0
                    min_x = np.min(x_indices) if len(x_indices) > 0 else 0
                    
                    # 객체 정보 텍스트 출력 (이름, 부피, 신뢰도)
                    text = f"{object_name} {volume:.1f}cm^3 {conf:.2f}"
                    cv2.putText(roi_color, text, (min_x, min_y),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)


                # 결과 화면 출력
                cv2.imshow("Visualized Results", roi_color)
                if cv2.waitKey(10) == 27:  # ESC 키 입력 시 종료
                    stop_event.set()
                    break
                end_time = time.time()  # 종료 시간 기록
                print(f"[Visualization] Visualization time: {end_time - start_time:.4f} seconds")

        except queue.Empty:
            continue
        except Exception as e:
            print(f"Error in visualization: {e}")
            continue

def main():
    """
    메인 함수: 전체 프로그램의 실행을 관리
    """
    # YOLO 모델 파일 경로 설정
    model_path = os.path.join(os.getcwd(), 'model', 
                             "total_50org_100scaled_10000mix_700_96_a1002_best.pt")
    
    # 저장된 ROI 깊이 데이터 로드 (없으면 None)
    save_depth = np.load('save_depth.npy') if os.path.exists('save_depth.npy') else None
    
    # 프로세스 간 통신을 위한 큐 생성
    frame_queue = mp.Queue(maxsize=5)  # 프레임 전달
    result_queue = mp.Queue(maxsize=5)  # 탐지 결과 전달
    volume_queue = mp.Queue(maxsize=5)  # 부피 계산 결과 전달
    volume_control_queue = mp.Queue(maxsize=5)  # 볼륨 계산 프로세스 제어
    stop_event = mp.Event()  # 프로세스 종료 제어

    # 실행할 프로세스 리스트 생성
    processes = [
        mp.Process(target=frame_preprocessing, 
                  args=(frame_queue, stop_event, save_depth)),
        mp.Process(target=gpu_inference, 
                  args=(frame_queue, result_queue, volume_control_queue, model_path, stop_event)),
        mp.Process(target=manage_volume_calculation, 
                  args=(result_queue, volume_queue, volume_control_queue, stop_event, save_depth)),
        mp.Process(target=visualize_results, 
                  args=(volume_queue, stop_event))
    ]
    start_time = time.time()  # 시작 시간

    try:
        # 모든 프로세스 시작
        for process in processes:
            process.start()
        
        # 프로세스 종료 대기
        for process in processes:
            process.join()
            
    except KeyboardInterrupt:
        print("\nProgram interrupted by user. Cleaning up...")
        stop_event.set()  # 종료 신호 설정
        
        # 모든 프로세스 강제 종료
        for process in processes:
            process.terminate()
            process.join()
            
    finally:
        cv2.destroyAllWindows()  # OpenCV 창 정리
        print("Program terminated successfully.")
        end_time = time.time()  # 종료 시간
        print(f"[Main] Total processing time per frame: {end_time - start_time:.4f} seconds")

if __name__ == "__main__":
    # YOLO 로깅 레벨 설정
    logging.getLogger("ultralytics").setLevel(logging.WARNING)
    
    # Windows에서 멀티프로세싱 지원
    mp.freeze_support()
    
    # 메인 함수 실행
    main()