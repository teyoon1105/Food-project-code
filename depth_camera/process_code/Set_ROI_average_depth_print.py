import pyrealsense2 as rs
import numpy as np
import cv2

# 파이프라인 설정 (깊이 및 컬러 스트림 활성화)
pipeline = rs.pipeline()
config = rs.config()

# 두 스트림 모두 너비, 높이를 640, 480로 받아옴
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
# 해당 설정으로 파이프라인 시작
pipeline.start(config)

# 컬러  스트림을 기준으로 정렬
align_to = rs.stream.color
# 정렬 객체 생성
align = rs.align(align_to)

# ROI 설정 관련 변수
# 마우스 콜백 함수로 저장할 좌표
roi_pts = []
# 그리기 상태 플래그
drawing = False


# 마우스 콜백 함수
def mouse_callback(event, x, y, flags, param):
    global roi_pts, drawing, img

    # 좌클릭시
    if event == cv2.EVENT_LBUTTONDOWN:
        # 그리기 플래그 on
        drawing = True
        # 현 마우스 포인터 좌표 저장
        roi_pts = [(x, y)]
    
    # 움직일 시
    elif event == cv2.EVENT_MOUSEMOVE:
        # 그리기 플래그 on된 상태일 때
        if drawing:
            # 현재 프레임 복사 후 
            img_copy = img.copy() 
            # 복사본 이미지 위에 ROI 그리고
            cv2.rectangle(img_copy, roi_pts[0], (x, y), (0, 255, 0), 2)
            # 그려진 복사본 이미지를 imshow
            cv2.imshow('Color Image', img_copy)
    # 좌클릭 뗄 때
    elif event == cv2.EVENT_LBUTTONUP:
        # 이젠 그리기 플래그 off(안하면 움지일 때 마다 콜백 됨)
        drawing = False
        # 마지막 좌표를 저장
        roi_pts.append((x, y))

# 프레임 30번 당 1번 거리를 출력하기 위한 counter 변수
frame_count = 0

try:
    # 무한 루프 돌면서
    while True:
        # 프레임값들을 파이프 라인에서 받아옴
        frames = pipeline.wait_for_frames()
        # 프레임들을 컬러 스트림에 맞게 정렬하고
        aligned_frames = align.process(frames)
        # 깊이 스트림 프레임값을 가져옴
        aligned_depth_frame = aligned_frames.get_depth_frame()
        # 컬러 스트림 프레임값을 가져옴
        color_frame = aligned_frames.get_color_frame()

        # 프레임 가져왔으니 카운터 +1
        frame_count += 1

        # 가져온 깊이 프레임, 컬러 프레임이 없다면 다음 반복
        if not aligned_depth_frame or not color_frame:
            continue
        
        # 깊이 프레임 값을 넘파이 배열로 바꿈
        depth_image = np.asanyarray(aligned_depth_frame.get_data())
        # 컬러 프레임 값을 넘파이 배열로 바꿈
        img = np.asanyarray(color_frame.get_data())
        # 깊이, 컬러 넘파이 배열을 180도 회전(상하좌우)
        depth_image = cv2.flip(depth_image, -1) #필요하다면
        img = cv2.flip(img, -1) #필요하다면

        # 프레임을 띄울 창을 생성
        cv2.namedWindow('Color Image')
        # 해당 창에서 마우스 콜백함수 사용 가능하게 설정
        cv2.setMouseCallback('Color Image', mouse_callback)

        # 만약 두 점이라면, 즉 직사각형을 그릴 수 있는 상황이라면
        if len(roi_pts) == 2:  # 두 점이 선택되었으면 ROI 계산 및 표시
            
            # 해당 좌표 값을 정렬
            x1, y1 = roi_pts[0]
            x2, y2 = roi_pts[1]
            x1, x2 = min(x1, x2), max(x1, x2)
            y1, y2 = min(y1, y2), max(y1, y2)

            # 정렬된(작은 값, 큰 값) 좌표로 직사각형 만들어서 컬러 이미지에 그리기
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)

            # 깊이 넘파이 배열에서 ROI 좌표 안에 있는 배열 값만 가져와서
            roi_depth = depth_image[y1:y2, x1:x2]
            # 유의미 한 값들만 가져오기
            roi_depth = roi_depth[roi_depth > 0]

            # 프레임 30회당
            if frame_count % 30 == 0:
                # 깊이 배열값의 평균을 mm, cm, m 값으로 각각 변수에 저장해서
                # 해당 값을 출력
                average_distance_mm = np.mean(roi_depth)
                average_distance_cm = average_distance_mm / 10
                average_distance_m = average_distance_cm / 100
                print(f"관심 영역 평균 거리: {average_distance_mm:.2f} mm")
                print(f"관심 영역 평균 거리: {average_distance_cm:.2f} cm")
                print(f"관심 영역 평균 거리: {average_distance_m:.2f} m")


            cropped_image = img[y1:y2, x1:x2]
            cv2.imshow('Cropped Image', cropped_image) #새로운 창 생성
            
        # 처음에는 컬러이미지, 마우스 콜백 함수 실행되면 직사각형 있는 컬러 이미지
        cv2.imshow('Color Image', img)

        # esc 눌리면 반복문 탈출
        if cv2.waitKey(1) == 27:
            break

finally:
    # 파이프 라인 멈추고
    pipeline.stop()
    # 창을 닫는다
    cv2.destroyAllWindows()
