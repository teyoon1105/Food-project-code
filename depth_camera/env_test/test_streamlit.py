# 필요한 라이브러리 임포트
import streamlit as st  # 웹 애플리케이션 구축을 위한 라이브러리
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration  # 웹캠 스트리밍을 위한 라이브러리
import pyrealsense2 as rs  # Intel RealSense 카메라 제어 라이브러리
import numpy as np  # 수치 계산을 위한 라이브러리
import cv2  # 컴퓨터 비전 라이브러리
import av  # 비디오 처리 라이브러리
from ultralytics import YOLO  # YOLO 객체 검출 모델
import torch  # PyTorch 딥러닝 프레임워크

# YOLO 모델 초기화
MODEL_PATH = "path/your/model.pt"  # YOLO 모델 파일 경로
model = YOLO(MODEL_PATH)  # YOLO 모델 로드
model.to("cuda" if torch.cuda.is_available() else "cpu")  # GPU 사용 가능시 GPU로, 아니면 CPU로 모델 이동

class RealSenseProcessor:
    """RealSense 카메라 처리를 위한 클래스"""
    def __init__(self):
        # RealSense 카메라 초기화 및 설정
        self.pipeline = rs.pipeline()  # RealSense 파이프라인 생성
        self.config = rs.config()  # 카메라 설정 객체 생성
        # 컬러 스트림 활성화 (1280x720, 30fps)
        self.config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
        # 깊이 스트림 활성화 (1280x720, 30fps)
        self.config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
        self.align = rs.align(rs.stream.color)  # 컬러와 깊이 프레임 정렬을 위한 객체
        self.pipeline.start(self.config)  # 카메라 스트리밍 시작
        self.save_depth = None  # 기준 깊이 데이터 저장 변수

    def get_frames(self):
        """카메라로부터 컬러와 깊이 프레임을 가져오는 메소드"""
        frames = self.pipeline.wait_for_frames()  # 프레임 획득
        aligned_frames = self.align.process(frames)  # 프레임 정렬
        color_frame = aligned_frames.get_color_frame()  # 컬러 프레임 추출
        depth_frame = aligned_frames.get_depth_frame()  # 깊이 프레임 추출

        # 프레임이 유효하지 않으면 None 반환
        if not color_frame or not depth_frame:
            return None, None

        # 프레임을 numpy 배열로 변환
        color_image = np.asanyarray(color_frame.get_data())
        depth_image = np.asanyarray(depth_frame.get_data())
        return color_image, depth_image

class VideoProcessor:
    """WebRTC 비디오 처리를 위한 클래스"""
    def __init__(self):
        self.realsense_processor = RealSenseProcessor()  # RealSense 프로세서 초기화

    def recv(self, frame):
        """프레임을 받아서 처리하는 메소드"""
        # RealSense 카메라에서 프레임 획득
        color_image, depth_image = self.realsense_processor.get_frames()
        
        # 프레임이 없으면 빈 프레임 반환
        if color_image is None or depth_image is None:
            return av.VideoFrame.from_ndarray(np.zeros((720, 1280, 3), dtype=np.uint8), format="bgr24")

        # YOLO 모델로 객체 검출
        results = model(color_image)
        predictions = results.pandas().xyxy[0]  # 예측 결과를 pandas DataFrame으로 변환

        # 각 검출된 객체에 대해 처리
        for _, row in predictions.iterrows():
            x1, y1, x2, y2, conf, cls = map(int, row[:6])  # 바운딩 박스 좌표와 신뢰도 추출
            label = f"{model.names[cls]} {conf:.2f}"  # 레이블 문자열 생성

            # 저장된 깊이 데이터가 있으면 높이 계산
            if self.realsense_processor.save_depth is not None:
                roi_depth = depth_image[y1:y2, x1:x2]  # 객체 영역의 깊이값
                height_cm = np.mean(roi_depth) / 10  # 밀리미터를 센티미터로 변환
                label += f", H: {height_cm:.1f}cm"  # 레이블에 높이 정보 추가

            # 검출 결과를 이미지에 시각화
            cv2.rectangle(color_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(color_image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        return av.VideoFrame.from_ndarray(color_image, format="bgr24")

# WebRTC 연결 설정
RTC_CONFIGURATION = RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})

# Streamlit UI 구성
st.title("RealSense Depth and YOLO Object Detection")

# WebRTC 스트리머 시작
webrtc_ctx = webrtc_streamer(
    key="realsense-yolo-stream",
    mode=WebRtcMode.SENDRECV,
    rtc_configuration=RTC_CONFIGURATION,
    video_processor_factory=VideoProcessor,
    media_stream_constraints={"video": True, "audio": False},
)

# 초기 깊이 데이터 저장 버튼 추가
if webrtc_ctx.video_processor:
    if st.button("Save Initial Depth"):
        _, depth_frame = webrtc_ctx.video_processor.realsense_processor.get_frames()
        if depth_frame is not None:
            webrtc_ctx.video_processor.realsense_processor.save_initial_depth(depth_frame)
            st.success("Initial depth data saved!")
        else:
            st.error("Failed to capture depth data.")