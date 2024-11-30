import pyrealsense2 as rs
import numpy as np
import cv2

# 파이프라인 설정 (깊이 스트림 활성화)
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
pipeline.start(config)

# 관심 영역 (ROI) 좌표 설정 (예: 왼쪽 상단 모서리 (100, 100), 너비 200, 높이 150)
roi_x = 100
roi_y = 100
roi_width = 200
roi_height = 150

try:
    while True:
        frames = pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        if not depth_frame:
            continue

        # 관심 영역 (ROI) 내의 픽셀들을 순회하며 거리 계산
        for y in range(roi_y, roi_y + roi_height):
            for x in range(roi_x, roi_x + roi_width):
                distance_meters = depth_frame.get_distance(x, y)
                distance_cm = distance_meters * 100
                print(f"픽셀 ({x}, {y}) 거리: {distance_cm:.2f} cm")

        # 깊이 이미지를 컬러맵으로 변환하여 표시 (선택 사항)
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(np.asanyarray(depth_frame.get_data()), alpha=0.03), cv2.COLORMAP_JET)

        # 관심 영역 표시 (선택 사항)
        cv2.rectangle(depth_colormap, (roi_x, roi_y), (roi_x + roi_width, roi_y + roi_height), (0, 255, 0), 2)

        cv2.imshow('Depth Image', depth_colormap)

        key =cv2.waitKey(1)  # ESC 키를 누르면 종료
        
        if key == 27:
            break

finally:
    pipeline.stop()  # 스트리밍 중지
    cv2.destroyAllWindows()