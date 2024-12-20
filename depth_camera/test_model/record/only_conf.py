import pyrealsense2 as rs  # Intel RealSense 카메라 라이브러리
import numpy as np  # 수치 계산 및 배열 처리
import cv2  # OpenCV 라이브러리
import os  # 파일 및 경로 작업

# Intel RealSense 카메라 설정
pipeline = rs.pipeline() # 파이프라인 생성
config = rs.config() # 카메라 설정 객체 생성
config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30) # 컬러 스트림 설정

# ROI 설정
ROI_POINTS = [(175, 50), (1055, 690)] 

# ---- 전역 변수 ----
output_video_path = 'test_' + 'model' + '.avi' # 비디오 제목

fourcc = cv2.VideoWriter_fourcc(*'XVID')  # 코덱 설정 (적절한 코덱 사용)
fps = 30.0  # 프레임 레이트
width = ROI_POINTS[1][0] - ROI_POINTS[0][0]  # ROI 너비
height = ROI_POINTS[1][1] - ROI_POINTS[0][1] # ROI 높이
out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))


# ---- 함수 정의 ----
def initialize_camera():
    """카메라 파이프라인 초기화 및 시작"""
    pipeline.start(config) # 파이프 라인 시작(프레임 받아옴)

def capture_frames():
    """카메라에서 프레임 캡처 및 정렬"""
    frames = pipeline.wait_for_frames() # 파이프 라인을 통해 받아온 프레임 대기    
    return frames.get_color_frame() # 깊이 스트림, 컬러 스트림 반환

def preprocess_images(color_frame):
    """깊이 및 컬러 프레임 전처리"""
    color_image = np.asanyarray(color_frame.get_data()) # 컬러 프레임을 넘파이 배열로 변환
    color_image = cv2.flip(color_image, -1) # 컬러 넘파이 배열을 상하좌우 반전
    return color_image # 반전된 깊이, 컬러 넘파이 배열 반환

def crop_roi(image, roi_points):
    """이미지에서 관심 영역(ROI) 크롭"""
    x1, y1 = roi_points[0] # ROI 좌상단 좌표
    x2, y2 = roi_points[1] # ROI 우하단 좌표
    return image[y1:y2, x1:x2] # 컬러 넘파이 배열을 ROI 영역만큼 자른 값 반환


def save_video_with_timestamps(frames, timestamps, output_path):
    """프레임 간 타임스탬프를 기반으로 비디오 저장"""
    if len(frames) == 0:
        print("No frames to save.")
        return

    height, width, _ = frames[0].shape # 저장한 프레임 리스트에서 첫번째 프레임의 shape을 확인
    fourcc = cv2.VideoWriter_fourcc(*'XVID') # 비디오 녹화 코덱 설정
    out = cv2.VideoWriter(output_path, fourcc, 30, (width, height))  # 기본 FPS로 초기화

    for i in range(len(frames) - 1): # 기록한 프레임들의 길이만큼 반복(한 프레임씩 확인)
        # 각 프레임 사이의 시간 간격 계산
        frame_interval = timestamps[i + 1] - timestamps[i] # 가져온 프레임 처리 시간 확인
        frame_count = int(frame_interval * 30)  # 간격에 맞는 프레임 수 계산

        # 프레임을 여러 번 쓰기 (FPS 동기화)
        for _ in range(max(1, frame_count)): 
            out.write(frames[i])

    # 마지막 프레임 추가
    out.write(frames[-1])
    out.release()
    print(f"Video saved at {output_path}")


# ---- 메인 처리 루프 ----
def main():
    global save_depth
    initialize_camera()

    try:
        while True:
            color_frame = capture_frames() # 카메라 정렬

            if not color_frame: # 프레임이 컬러나 깊이가 아니면 다음 반복
                continue

            color_image = preprocess_images(color_frame) # 컬러 프레임에 맞게 깊이 프레임 정렬, 전처리
            cropped_color = crop_roi(color_image, ROI_POINTS) # 컬러 넘파이 배열 ROI 영역 생성

            # 결과 이미지 표시
            cv2.imshow('Segmented Mask with Heights', cropped_color)
            out.write(cropped_color)

            if cv2.waitKey(1) == 27:  # ESC 키로 종료
                break
    finally:
        
        pipeline.stop() # 파이프 라인 종료
        cv2.destroyAllWindows() # 창 닫기
        out.release()

if __name__ == "__main__": # 코드를 실행하면
    main() # 메인함수 실행
