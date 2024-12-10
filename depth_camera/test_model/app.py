import streamlit as st
import pyrealsense2 as rs
import numpy as np
import cv2
import os
from ultralytics import YOLO
import torch
import logging
import time

# DepthVolumeCalculator 클래스 정의
class DepthVolumeCalculator:
    def __init__(self, model_path, roi_points, brightness_increase, cls_name_color):
        self.pipeline = None
        self.config = rs.config()
        self.config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
        self.config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
        self.align = rs.align(rs.stream.color)
        self.save_depth = None  # 저장된 깊이 데이터
        self.roi_points = roi_points
        self.brightness_increase = brightness_increase
        self.cls_name_color = cls_name_color
        self.model_path = model_path
        self.model_name = os.path.basename(model_path)

        try:
            self.model = YOLO(model_path)
            self.model.to("cuda" if torch.cuda.is_available() else "cpu")
        except Exception as e:
            st.error(f"Error loading YOLO model: {e}")
            exit(1)

    def initialize_camera(self):
        if self.pipeline is None:  # 이미 초기화된 경우 방지
            self.pipeline = rs.pipeline()  # Pipeline 초기화
        try:
            self.pipeline.start(self.config)
            st.session_state["camera_initialized"] = True  # 카메라 상태 업데이트
            st.success("Camera initialized successfully!")
        except Exception as e:
            st.error(f"Failed to initialize camera: {e}")

    def stop_camera(self):
        """카메라 종료"""
        if self.pipeline and st.session_state.get("camera_initialized", False):
            self.pipeline.stop()
            st.session_state["camera_initialized"] = False
            st.success("Camera stopped successfully!")
        else:
            st.warning("Camera is not running. Nothing to stop.")


    def capture_frames(self):
        """프레임 캡처"""
        if not st.session_state.get("camera_initialized", False):
            st.error("Camera is not initialized. Please click 'Initialize Camera' first.")
            return None, None
        try:
            frames = self.pipeline.wait_for_frames()
            aligned_frames = self.align.process(frames)
            return aligned_frames.get_depth_frame(), aligned_frames.get_color_frame()
        except Exception as e:
            st.error(f"Error capturing frames: {e}")
            return None, None
        
    def preprocess_images(self, depth_frame, color_frame):
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())
        return cv2.flip(depth_image, -1), cv2.flip(color_image, -1)

    def apply_roi(self, image):
        x1, y1 = self.roi_points[0]
        x2, y2 = self.roi_points[1]
        return image[y1:y2, x1:x2]

    def calculate_volume(self, cropped_depth, mask_indices, depth_intrin):
        total_volume = 0
        y_indices, x_indices = mask_indices

        for y, x in zip(y_indices, x_indices):
            z_cm = cropped_depth[y, x] / 10
            base_depth_cm = self.save_depth[y, x] / 10

            if z_cm > 20 and base_depth_cm > 25:
                height_cm = max(0, base_depth_cm - z_cm)
                pixel_area_cm2 = (z_cm ** 2) / (depth_intrin.fx * depth_intrin.fy)
                total_volume += pixel_area_cm2 * height_cm

        return total_volume


# Streamlit 애플리케이션 정의
def main():
    st.title("YOLO Segmentation and Volume Estimation")

    # Sidebar 설정
    st.sidebar.title("Settings")

    # 모델 설정
    MODEL_DIR = "C:/Users/SBA/teyoon_github/Food-project-code/depth_camera/test_model/model"
    model_files = [f for f in os.listdir(MODEL_DIR) if f.endswith(".pt")]
    selected_model = st.sidebar.selectbox("Select a YOLO Model", model_files)
    # model_path = os.path.join(MODEL_DIR, selected_model)

    # ROI 설정
    # roi_x1 = st.sidebar.number_input("ROI X1", value=175, step=5)
    # roi_y1 = st.sidebar.number_input("ROI Y1", value=50, step=5)
    # roi_x2 = st.sidebar.number_input("ROI X2", value=1055, step=5)
    # roi_y2 = st.sidebar.number_input("ROI Y2", value=690, step=5)
    # roi_points = [(roi_x1, roi_y1), (roi_x2, roi_y2)]

    # 밝기 증가 설정
    # brightness_increase = st.sidebar.slider("Brightness Increase", min_value=0, max_value=100, value=50)

    # Custom Classes
    # CLS_NAME_COLOR = {
    # '01011001': ('Rice', (255, 0, 255)), # 자주색
    # '01012006': ('Black Rice', (255, 0, 255)),
    # '04011005': ('Seaweed Soup', (0, 255, 255)),
    # '04011008': ('Beef stew', (0, 255, 255)),
    # '04017001': ('Soybean Soup', (0, 255, 255)), # 노란색
    # '06012004': ('Tteokgalbi', (0, 255, 0)), # 초록색
    # '06012008': ('Beef Bulgogi', (0, 255, 0)),
    # '07014001': ('EggRoll', (0, 0, 255)), # 빨간색
    # '08011003': ('Stir-fried anchovies', (0, 0, 255)),
    # '10012001': ('Chicken Gangjeong', (0, 0, 255)),
    # '11014002': ('Gosari', (255, 255, 0)),
    # '11013007': ('Spinach', (255, 255, 0)), # 청록색
    # '12011008': ('Kimchi', (100, 100, 100)),
    # '12011003': ('Radish Kimchi', (100, 100, 100))
    # }

    # 설정 기본값
    default_roi_points = [(175, 50), (1055, 690)]
    default_brightness_increase = 50
    default_cls_name_color = {
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

    # DepthVolumeCalculator 상태 관리
    if "calculator" not in st.session_state:
        # 초기 인스턴스 생성
        st.session_state["calculator"] = DepthVolumeCalculator(
            model_path=os.path.join(MODEL_DIR, selected_model),
            roi_points=default_roi_points,
            brightness_increase=default_brightness_increase,
            cls_name_color=default_cls_name_color,
        )
    else:
        # 모델 변경 감지 및 재로드
        if st.session_state["calculator"].model_path != os.path.join(MODEL_DIR, selected_model):
            st.session_state["calculator"].model = YOLO(os.path.join(MODEL_DIR, selected_model))
            st.session_state["calculator"].model_path = os.path.join(MODEL_DIR, selected_model)
            st.success(f"Model '{selected_model}' loaded successfully!")

    calculator = st.session_state["calculator"]

    # Sidebar에서 ROI와 Brightness 설정 변경
    roi_x1 = st.sidebar.number_input("ROI X1", value=calculator.roi_points[0][0], step=5)
    roi_y1 = st.sidebar.number_input("ROI Y1", value=calculator.roi_points[0][1], step=5)
    roi_x2 = st.sidebar.number_input("ROI X2", value=calculator.roi_points[1][0], step=5)
    roi_y2 = st.sidebar.number_input("ROI Y2", value=calculator.roi_points[1][1], step=5)
    calculator.roi_points = [(roi_x1, roi_y1), (roi_x2, roi_y2)]

    calculator.brightness_increase = st.sidebar.slider(
        "Brightness Increase", min_value=0, max_value=100, value=calculator.brightness_increase
    )

   # DepthVolumeCalculator 인스턴스를 세션 상태에 저장
    # if "calculator" not in st.session_state:
    #     st.session_state["calculator"] = DepthVolumeCalculator(model_path, roi_points, brightness_increase, CLS_NAME_COLOR)

    # calculator = st.session_state["calculator"]

    # 카메라 초기화
    if st.sidebar.button("Initialize Camera"):
        if st.session_state.get("camera_initialized", False):
            st.warning("Camera is already initialized.")
        else:
            calculator.initialize_camera()
            st.session_state["camera_initialized"] = True

    # Save Depth Data 버튼
    if st.sidebar.button("Save Depth Data"):
        if not st.session_state.get("camera_initialized", False):
            st.warning("Camera is not initialized. Please initialize the camera first.")
        else:
            st.session_state["save_depth_requested"] = True  # 저장 요청 상태 업데이트

    # Start Processing 버튼
    if st.sidebar.button("Start Processing"):
        if not st.session_state.get("camera_initialized", False):
            st.warning("Camera is not initialized. Please initialize the camera first.")
        else:
            try:
                stframe = st.empty()
                while st.session_state.get("processing", True):
                    depth_frame, color_frame = calculator.capture_frames()
                    if not depth_frame or not color_frame:
                        st.warning("No frames available. Check the camera connection.")
                        break

                    depth_image, color_image = calculator.preprocess_images(depth_frame, color_frame)
                    cropped_depth = calculator.apply_roi(depth_image)
                    cropped_color = calculator.apply_roi(color_image)

                    brightened_image = cv2.convertScaleAbs(cropped_color, alpha=1, beta=calculator.brightness_increase)
                    results = calculator.model(brightened_image)
                    all_colored_mask = np.zeros_like(brightened_image)
                    blended_image = brightened_image.copy()

                    for result in results:
                        if result.masks is not None:
                            masks = result.masks.data.cpu().numpy()
                            classes = result.boxes.cls.cpu().numpy()

                            for i, mask in enumerate(masks):
                                conf = result.boxes.conf[i]
                                resized_mask = cv2.resize(mask, (brightened_image.shape[1], brightened_image.shape[0]))
                                color_mask = (resized_mask > 0.5).astype(np.uint8)
                                class_key = calculator.model.names[int(classes[i])]
                                object_name, color = calculator.cls_name_color.get(class_key, ("Unknown", (255, 255, 255)))

                                mask_indices = np.where(color_mask > 0)

                                if calculator.save_depth is None:
                                    st.warning("Depth data not saved. Please press 'Save Depth Data' before processing.")
                                    continue  # save_depth가 설정되지 않으면 스킵

                                depth_intrin = depth_frame.profile.as_video_stream_profile().intrinsics
                                volume = calculator.calculate_volume(cropped_depth, mask_indices, depth_intrin)

                                # 개별 마스크에 색상 적용
                                all_colored_mask[color_mask == 1] = color

                                label_position = (mask_indices[1][0], mask_indices[0][0] - 10)
                                cv2.putText(all_colored_mask, f"{object_name}: {volume:.0f}cm^3, C:{conf:.2f}", label_position,
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                                blended_image = cv2.addWeighted(brightened_image, 0.7, all_colored_mask, 0.3, 0)

                    stframe.image(blended_image, channels="BGR", caption="Segmented Results")

                    # Save Depth Data 요청 처리
                    if st.session_state.get("save_depth_requested", False):
                        calculator.save_depth = cropped_depth.copy()
                        st.session_state["depth_saved"] = True
                        st.session_state["save_depth_requested"] = False  # 요청 처리 완료
                        st.success("Depth data saved successfully!")

            except Exception as e:
                st.error(f"Error during processing: {e}")
            finally:
                calculator.stop_camera()
                st.session_state["camera_initialized"] = False



if __name__ == "__main__":
    logging.getLogger("ultralytics").setLevel(logging.WARNING)
    if "depth_saved" not in st.session_state:
        st.session_state["depth_saved"] = False  # 초기 상태 설정
    main()
