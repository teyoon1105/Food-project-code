import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
import pyrealsense2 as rs
import numpy as np
import cv2
import av
from ultralytics import YOLO
import torch

# YOLO 모델 설정
MODEL_PATH = "C:/Users/SBA/teyoon_github/best.pt/1st_mix_scaled_best.pt"  # 모델 경로를 설정하세요
model = YOLO(MODEL_PATH)
model.to("cuda" if torch.cuda.is_available() else "cpu")

# RealSense 프로세서 클래스
class RealSenseProcessor:
    def __init__(self):
        # RealSense 카메라 초기화
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
        self.config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
        self.align = rs.align(rs.stream.color)  # 정렬(Align) 객체 생성
        self.pipeline.start(self.config)
        self.save_depth = None  # 초기 깊이 데이터를 저장하기 위한 변수

    def get_frames(self):
        """RealSense 카메라에서 프레임 가져오기"""
        frames = self.pipeline.wait_for_frames()
        aligned_frames = self.align.process(frames)
        color_frame = aligned_frames.get_color_frame()
        depth_frame = aligned_frames.get_depth_frame()

        if not color_frame or not depth_frame:
            return None, None

        color_image = np.asanyarray(color_frame.get_data())
        depth_image = np.asanyarray(depth_frame.get_data())
        return color_image, depth_image

    def save_initial_depth(self, depth_frame):
        """초기 깊이 데이터 저장"""
        self.save_depth = np.asanyarray(depth_frame)

    def stop(self):
        """RealSense 카메라 정지"""
        self.pipeline.stop()

# WebRTC 비디오 프로세서 클래스
class VideoProcessor:
    def __init__(self):
        self.realsense_processor = RealSenseProcessor()

    def recv(self, frame):
        # RealSense 데이터 가져오기
        color_image, depth_image = self.realsense_processor.get_frames()
        if color_image is None or depth_image is None:
            return av.VideoFrame.from_ndarray(np.zeros((720, 1280, 3), dtype=np.uint8), format="bgr24")

        # 모델 예측
        results = model(color_image)
        predictions = results.pandas().xyxy[0]

        for _, row in predictions.iterrows():
            x1, y1, x2, y2, conf, cls = map(int, row[:6])
            label = f"{model.names[cls]} {conf:.2f}"

            # 깊이 데이터 활용
            if self.realsense_processor.save_depth is not None:
                roi_depth = depth_image[y1:y2, x1:x2]  # 객체의 ROI에 해당하는 깊이값
                height_cm = np.mean(roi_depth) / 10  # 깊이를 센티미터 단위로 변환
                label += f", H: {height_cm:.1f}cm"

            # 예측 결과를 이미지에 그리기
            cv2.rectangle(color_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(color_image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        return av.VideoFrame.from_ndarray(color_image, format="bgr24")

# WebRTC 설정
RTC_CONFIGURATION = RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})

# Streamlit UI
st.title("RealSense Depth and YOLO Object Detection")

# WebRTC 시작
webrtc_ctx = webrtc_streamer(
    key="realsense-yolo-stream",
    mode=WebRtcMode.SENDRECV,
    rtc_configuration=RTC_CONFIGURATION,
    video_processor_factory=VideoProcessor,
    media_stream_constraints={"video": True, "audio": False},
)

# 초기 깊이 데이터 저장 버튼
if webrtc_ctx.video_processor:
    if st.button("Save Initial Depth"):
        _, depth_frame = webrtc_ctx.video_processor.realsense_processor.get_frames()
        if depth_frame is not None:
            webrtc_ctx.video_processor.realsense_processor.save_initial_depth(depth_frame)
            st.success("Initial depth data saved!")
        else:
            st.error("Failed to capture depth data.")
