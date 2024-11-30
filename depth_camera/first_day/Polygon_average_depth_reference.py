import pyrealsense2 as rs
import numpy as np
import cv2

# 파이프라인 설정 (깊이 및 컬러 스트림 활성화)
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
pipeline.start(config)

align_to = rs.stream.color
align = rs.align(align_to)


# TODO
# 그림(마우스 콜백)으로 관심 영역 설정하기

# 관심 영역 (ROI) 좌표 설정 (polygon 형식
roi_pts = np.array([[160, 120], [480, 120], [480, 360], [160, 360]], np.int32)  # 예시 polygon 좌표
roi_pts = roi_pts.reshape((-1, 1, 2))  # cv2.polylines() 함수에 필요한 형태로 변환


# 프레임 카운트 설정(1초당 1프레임)
frame_count = 0

try:
    while True:
        # TODO
        # color frame은 가져온다.
        # color frame은 cv2.imshow를 하고
        # 원하는 구역을 설정하면
        # 해당 구역을 
        frames = pipeline.wait_for_frames()
        # 가져온 프레임을 컬러 프레임에 맞게 정렬
        aligned_frames = align.process(frames)

        
        # 정렬된 프레임에서 깊이 프레임 가져오기    
        aligned_depth_frame = aligned_frames.get_depth_frame()
        # 컬러 프레임 가져오기
        color_frame = aligned_frames.get_color_frame()

        frame_count += 1

        if not aligned_depth_frame or not color_frame:
            continue
        

        # 깊이 프레임을 NumPy 배열로 변환
        depth_np_image = np.asanyarray(aligned_depth_frame.get_data())
        # 반전
        depth_image = cv2.flip(depth_np_image, -1)

        # 컬러 프레임을 NumPy 배열로 변환
        color_np_image = np.asanyarray(color_frame.get_data())
        color_image = cv2.flip(color_np_image, -1)
        

        # 컬러 이미지에 polygon ROI 그리기
        cv2.polylines(color_image, [roi_pts], True, (0, 255, 0), 2)

        # ROI 내부의 픽셀만 마스킹하여 거리 계산
        mask = np.zeros(depth_image.shape, dtype=np.uint8)
        cv2.fillPoly(mask, [roi_pts], 255)  # ROI 내부를 흰색(255)로 채움

        masked_depth_image = cv2.bitwise_and(depth_image, depth_image, mask=mask)

        if frame_count % 30 == 0:
            # 마스크된 깊이 이미지에서 거리 계산 (예: 평균 거리)
            roi_depth = masked_depth_image[masked_depth_image > 0]  # 0이 아닌 유효한 거리 값만 선택
            if roi_depth.size > 0:  # ROI에 유효한 깊이 값이 있는지 확인
                average_distance_mm = np.mean(roi_depth)
                average_distance_cm = average_distance_mm / 10
                average_distance_m = average_distance_cm / 100
                print(f"관심 영역 평균 거리: {average_distance_mm:.2f} mm")
                print(f"관심 영역 평균 거리: {average_distance_cm:.2f} cm")
                print(f"관심 영역 평균 거리: {average_distance_m:.2f} m")


        # 컬러 이미지 표시 (ROI 포함)
        cv2.imshow('Color Image', color_image)

        if cv2.waitKey(1) == 27:  # ESC 키를 누르면 종료
            break

finally:
    pipeline.stop()  # 스트리밍 중지
    cv2.destroyAllWindows()