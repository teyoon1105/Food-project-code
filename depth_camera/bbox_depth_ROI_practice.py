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

# ROI 설정 관련 변수
roi_pts = []  # 최종 ROI 꼭짓점 저장
temp_roi_pts = []  # 드래그 중 임시 ROI 꼭짓점 저장
drawing = False

def mouse_callback(event, x, y, flags, param):
    global roi_pts, temp_roi_pts, drawing, color_image # color_image 전역변수로 선언

    # 마우스 왼클릭 시
    # 그림 그리지 시작
    # 클릭한 좌표 temp_roi_pt에 저장
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        temp_roi_pts = [(x, y)]

    # 마우스를 움직이면
    elif event == cv2.EVENT_MOUSEMOVE:
        # 그리고 마우스 왼클을 통해 그림 그리기 시작함을 알리는 draging 값을 True로 받아왔다면
        if drawing:
            # 움직이는 좌표 좌표를 temp_roi_pts에 저장
            temp_roi_pts.append((x, y))
            # 지금 프레임을 복사(해당 복사본 프레임 위에 직사각형을 그릴 예정)
            # temp_color_image = color_image.copy()
            # temp_roi_pts 좌표가 2개 이상(직사각형을 그릴 수 있으면)
            if len(temp_roi_pts) > 1:
                # 복사본 이미지 위에, 좌표 2개(좌클릭시 얻은 좌표와, 움직임을 멈춰서 얻은 좌표), 초록색, 선두깨는 2로 직사각형 그리기
                cv2.rectangle(color_image, temp_roi_pts[0], temp_roi_pts[-1], (0, 255, 0), 2)
                cv2.imshow('Color Image', color_image)

    # 마우스 왼클릭을 떼었을 때
    elif event == cv2.EVENT_LBUTTONUP:
        # 더 이상 그리는 단계가 아니니(마우스 무빙은 해도 해당 if문 안에 안들어가게 하기 위해)
        drawing = False
        # 마우스 무빙하면서 저장한 모든 좌표 roi_pts리스트로 저장
        roi_pts = list(temp_roi_pts.copy()) # copy()를 사용하여 값 복사
        
        # 마찬가지로 2개 이상이면
        if len(roi_pts) > 1:  # 두 점 이상 선택되었는지 확인
            # 클릭 시작때 얻은 좌표와 클릭을 떼었을 때 얻은 좌표, 차별화를 두기 위해 빨간색 선
            cv2.rectangle(color_image, roi_pts[0], roi_pts[-1], (0, 0, 255), 2)
            cv2.imshow('Color Image', color_image)

        # 다음을 위해 삭제    
        temp_roi_pts.clear()
        

 # 프레임 카운트 설정(1초당 1프레임)
frame_count = 0


try:
    while True:
        frames = pipeline.wait_for_frames()
        # 가져온 프레임을 컬러 프레임에 맞게 정렬
        aligned_frames = align.process(frames)

        
        # 정렬된 프레임에서 깊이 프레임 가져오기    
        aligned_depth_frame = aligned_frames.get_depth_frame()
        # 정렬된 컬러 프레임 가져오기
        color_frame = aligned_frames.get_color_frame()

        # 프레임을 가져올 때마다 +1
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

        cv2.namedWindow('Color Image')
        cv2.setMouseCallback('Color Image', mouse_callback)

        cv2.imshow('Color Image', color_image)
        

        if len(roi_pts) > 1:

            if frame_count % 30 == 0:

                x1, y1 = roi_pts[0]
                x2, y2 = roi_pts[1]
                x1, x2 = min(x1, x2), max(x1, x2)
                y1, y2 = min(y1, y2), max(y1, y2)

                print(x1, x2, y1, y2)

                cv2.rectangle(color_image, (x1, y1), (x2, y2), (0,0,255), 2)

                roi_depth = depth_image[y1:y2, x1:x2]
                roi_depth = roi_depth[roi_depth > 0]

                if roi_depth.size > 0:  # ROI에 유효한 깊이 값이 있는지 확인
                    average_distance_mm = np.mean(roi_depth)
                    average_distance_cm = average_distance_mm / 10
                    average_distance_m = average_distance_cm / 100
                    print(f"관심 영역 평균 거리: {average_distance_mm:.2f} mm")
                    print(f"관심 영역 평균 거리: {average_distance_cm:.2f} cm")
                    print(f"관심 영역 평균 거리: {average_distance_m:.2f} m")



        if cv2.waitKey(1) == 27:  # ESC 키를 누르면 종료
            break

finally:
    pipeline.stop()  # 스트리밍 중지
    cv2.destroyAllWindows()