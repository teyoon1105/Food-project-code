## License: Apache 2.0. See LICENSE file in root directory.
## Copyright(c) 2017 Intel Corporation. All Rights Reserved.

#####################################################
##              Align Depth to Color               ##
## 깊이 영상과 컬러 영상을 정렬하고 배경을 제거하는 코드
#####################################################

# 스트림 제어
# 이미지 데이터 처리
# 이미지 처리 및 디스플레이
# First import the library
import pyrealsense2 as rs
# Import Numpy for easy array manipulation
import numpy as np
# Import OpenCV for easy image rendering
import cv2

# 파이트 라인 생성
# Create a pipeline
pipeline = rs.pipeline()


# Create a config and configure the pipeline to stream
#  different resolutions of color and depth streams
config = rs.config()

# 마찬가지로 파이프 라인 감싸는 코드
# 연결된 디바이스의 기능, 설정을 가져와
# 접근할 수 있는 디바이스 객체를 만들고
# 해당 디바이스의 정보를 가져옴
# Get device product line for setting a supporting resolution
pipeline_wrapper = rs.pipeline_wrapper(pipeline)
pipeline_profile = config.resolve(pipeline_wrapper)
device = pipeline_profile.get_device()
device_product_line = str(device.get_info(rs.camera_info.product_line))

# 디바이스 객체를 통해 센서를 순회하여
# 디바이스에 rgb 센서가 있는 지 확인
# 없으면 스트림 종료
found_rgb = False
for s in device.sensors:
    if s.get_info(rs.camera_info.name) == 'RGB Camera':
        found_rgb = True
        break
if not found_rgb:
    print("The demo requires Depth camera with Color sensor")
    exit(0)

# 깊이 스트림과 컬러 스트림 생성
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# 스트림 시작
# Start streaming
profile = pipeline.start(config)

# 디바이스 객체를 통해 깊이 센서의 스케일 값을 가져옴
# 목적 : 센서의 raw 데이터를 실제 거리로 변환하는데 사용
# Getting the depth sensor's depth scale (see rs-align example for explanation)
depth_sensor = profile.get_device().first_depth_sensor()
depth_scale = depth_sensor.get_depth_scale()
print("Depth Scale is: " , depth_scale)

# 배경 제거를 위한 임계 거리 값(1m) 설정
# 1m 보다 멀면 배경으로 간주
# depth_scale로 나누어 실제 센서 값으로 변환
# We will be removing the background of objects more than
#  clipping_distance_in_meters meters away
clipping_distance_in_meters = 1 #1 meter
clipping_distance = clipping_distance_in_meters / depth_scale

# 깊이 프레임을 컬러 프레임에 정렬하기 위해 정렬 객체 생성
# 깊이 영상과 컬러 영상의 픽셀을 1대1로 매칭하기 위함
# Create an align object
# rs.align allows us to perform alignment of depth frames to others frames
# The "align_to" is the stream type to which we plan to align depth frames.
align_to = rs.stream.color
align = rs.align(align_to)

# Streaming loop
try:
    # 무한루프 돌면서
    while True:
        # 깊이와 컬러 프레임을 가져옴
        # Get frameset of color and depth
        frames = pipeline.wait_for_frames()
        # frames.get_depth_frame() is a 640x360 depth image

        # 가져온 프레임들을 정렬
        # 컬러 프레임을 기준으로 깊이 프레임을 정렬
        # Align the depth frame to color frame
        aligned_frames = align.process(frames)

        
        # Get aligned frames
        aligned_depth_frame = aligned_frames.get_depth_frame() # aligned_depth_frame is a 640x480 depth image
        color_frame = aligned_frames.get_color_frame()

        # Validate that both frames are valid
        if not aligned_depth_frame or not color_frame:
            continue
        
        # 가져온 정렬 프레임을 넘파이 배열로 변환
        depth_image = np.asanyarray(aligned_depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        # 깊이 이미지를 rgb 다중 채널로 확장
        # 설정한 거리보다 먼 픽셀이나 유효하지 않은 픽셀은 회색으로
        # 나머지 픽셀은 원본 컬러
        # Remove background - Set pixels further than clipping_distance to grey
        grey_color = 153
        depth_image_3d = np.dstack((depth_image,depth_image,depth_image)) #depth image is 1 channel, color is 3 channels
        bg_removed = np.where((depth_image_3d > clipping_distance) | (depth_image_3d <= 0), grey_color, color_image)


        # 깊이 이미지에 컬러맵을 적용하여 시각화
        # 배경이 제거된 컬러 이미지와 깊이 이미지를 나란히 표시

        # Render images:
        #   depth align to color on left
        #   depth on right
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
        images = np.hstack((bg_removed, depth_colormap))

        cv2.namedWindow('Align Example', cv2.WINDOW_NORMAL)
        cv2.imshow('Align Example', images)
        key = cv2.waitKey(1)
        # Press esc or 'q' to close the image window
        if key & 0xFF == ord('q') or key == 27:
            cv2.destroyAllWindows()
            break
finally:
    pipeline.stop()
