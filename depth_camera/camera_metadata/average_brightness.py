import pyrealsense2 as rs
import cv2
import numpy as np
import time

# 카메라의 노출 및 밝기등 계산
# 파이프라인 설정
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)

# 스트림 시작
pipeline.start(config)

last_time = time.time()

try:
    while True:
        # 프레임 수집
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()

        # 프레임이 유효한지 확인
        if not color_frame:
            continue

        # 센서 노출 및 이득 정보 확인
        sensor = pipeline.get_active_profile().get_device().first_color_sensor()
        exposure = sensor.get_option(rs.option.exposure)  # 노출 시간
        gain = sensor.get_option(rs.option.gain)          # 이득

        # 컬러 프레임에서 밝기 계산
        color_image = np.asanyarray(color_frame.get_data())
        gray_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
        average_brightness = np.mean(gray_image)

        # 1초마다 정보 출력
        if time.time() - last_time > 1:
            print(f"Average Brightness: {average_brightness}")
            print(f"Exposure: {exposure}, Gain: {gain}")
            last_time = time.time()

        # 이미지 표시
        resized_image = cv2.resize(color_image, (320, 240))
        cv2.imshow('img', resized_image)

        # ESC 키를 눌러 종료
        key = cv2.waitKey(10)
        if key == 27:  # ESC
            break

finally:
    # 스트림 정리
    pipeline.stop()
    cv2.destroyAllWindows()