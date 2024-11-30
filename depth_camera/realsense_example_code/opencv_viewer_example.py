## License: Apache 2.0. See LICENSE file in root directory.
## Copyright(c) 2015-2017 Intel Corporation. All Rights Reserved.

###############################################
##      Open CV and Numpy integration        ##
###############################################

# 필요 라이브러리
# 카메라 제어
# 이미지 데이터 처리
# 이미지 처리, 디스플레이
import pyrealsense2 as rs
import numpy as np
import cv2

# 카메라 설정을 위한 기본 객체들 생성
# Configure depth and color streams
pipeline = rs.pipeline()
config = rs.config()

# 연결된 디파이스 정보
# Get device product line for setting a supporting resolution
# 파이프 라인 전체 감싸는 단계
pipeline_wrapper = rs.pipeline_wrapper(pipeline)
# config.resolve : 연결된 카메라 능력, 설정 확인
# 스트림 시작 전 카메라 지원기능 확인하는 과정
pipeline_profile = config.resolve(pipeline_wrapper)
# 기능이 저장된 프로파일로 부터 실제 카메라 객체를 가져옴
# device 객체로 카메라 정보 접근 가능
device = pipeline_profile.get_device()
# 다양한 정보 가져옴
device_product_line = str(device.get_info(rs.camera_info.product_line))

# 디바이스에 RGB카메라 센서가 있는 지 확인, 없다면 프로그램 종료
found_rgb = False
# 디바이스 객체로 카메라 센서에 접근, 순회
# 센서에 rgb 센서가 있는 지 확인
for s in device.sensors:
    if s.get_info(rs.camera_info.name) == 'RGB Camera':
        found_rgb = True
        break
if not found_rgb:
    print("The demo requires Depth camera with Color sensor")
    exit(0)

# 두 가지 스트림 생성
# 깊이 스트림, 해상도, 16비트 깊이 데이터, 30프레임
# z16포멧 > 0~65535(2의 16승) 사이의 값을 지님

# 컬러 스트림, 해상도, bgr 컬러 포맷, 30프레임
# bgr8 채널, 8비트 채널로 0~255 값을 지님
# 일반적인 컬러 이미지와 동일
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# 스트림 시작
# Start streaming
pipeline.start(config)

try:
    # 무한 루프 돌면서
    while True:
        
        # 깊이와 컬러 프레임을 동시에 가져옴
        # Wait for a coherent pair of frames: depth and color
        frames = pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        if not depth_frame or not color_frame:
            continue
        
        # 프레임 데이터를 넘파이 배열로 변환
        # Convert images to numpy arrays
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        # 깊이 이미지를 시각화하기 위해 JET 컬러맵 적용
        # Apply colormap on depth image (image must be converted to 8-bit per pixel first)
        # 깊이 값을 0~255로 스케일링(1mm당 0.03의 컬러맵)
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)

        # 깊이 맵과 컬러맵의 크기를 불러옴
        depth_colormap_dim = depth_colormap.shape
        color_colormap_dim = color_image.shape

        # 컬러맵과 깊이맵의 크기가 다르면 같게 하는 코드
        # 같게 만들고 두 이미지를 가로로 붙여서 하나의 이미지로 만든다
        # If depth and color resolutions are different, resize color image to match depth image for display
        if depth_colormap_dim != color_colormap_dim:
            resized_color_image = cv2.resize(color_image, dsize=(depth_colormap_dim[1], depth_colormap_dim[0]), interpolation=cv2.INTER_AREA)
            images = np.hstack((resized_color_image, depth_colormap))
        else:
            images = np.hstack((color_image, depth_colormap))

        # 리얼센스 윈도우 만들고
        # 이미지를 띄운다
        # Show images
        cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
        cv2.imshow('RealSense', images)
        key =cv2.waitKey(0)

        if key == 27:
            break

# 프로그램 종료시
# 스트림 종료
finally:

    # Stop streaming
    pipeline.stop()
