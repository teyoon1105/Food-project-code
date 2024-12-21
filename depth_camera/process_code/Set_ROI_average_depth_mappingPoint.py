import pyrealsense2 as rs
import numpy as np
import cv2
from ultralytics import YOLO
import os

# YOLO Segmentation 모델 로드
Now_path = os.getcwd()
model_path = os.path.join(Now_path, 'best.pt')
model = YOLO(model_path)  # 학습된 YOLO Segmentation 모델 사용

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
# frame_count = 0

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
        # frame_count += 1

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
        
        cv2.imshow('Color Image', img)

        # YOLO Segmentation 모델 실행
        results = model(img)
        
        for result in results:
            if result.masks is not None:
                masks = result.masks.data.cpu().numpy()
                classes = result.boxes.cls.cpu().numpy()
                class_names = model.names

                for i, mask in enumerate(masks):
                    binary_mask = (mask > 0.5).astype(np.uint8)
                    mask_indices = np.where(binary_mask > 0)

                    try:
                        object_depths = depth_image[mask_indices]
                        object_depths = object_depths[object_depths > 0]  # 유효하지 않은 깊이 값 (0) 필터링
                        average_depth_mm = np.mean(object_depths)
                        average_depth_cm = average_depth_mm / 10  # 밀리미터에서 센티미터로 변환
                    except IndexError:
                        print("에러: 객체에 대한 깊이 데이터를 추출할 수 없습니다.")
                        average_depth_cm = 0  # 또는 적절하게 에러 처리

                    # 바운딩 박스와 클래스 레이블 그리기
                    box = result.boxes.xyxy[i].cpu().numpy()
                    x1, y1, x2, y2 = map(int, box)
                    cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
                    cv2.putText(img, f"{class_names[int(classes[i])]}: {average_depth_cm:.2f} cm", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

        # 만약 두 점이라면, 즉 직사각형을 그릴 수 있는 상황이라면
        if len(roi_pts) == 2:  # 두 점이 선택되었으면 ROI 계산 및 표시
            
            x1, y1 = roi_pts[0]
            x2, y2 = roi_pts[1]
            x1, x2 = min(x1, x2), max(x1, x2)
            y1, y2 = min(y1, y2), max(y1, y2)
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.imshow('Color Image', img)
            
            cropped_image = img[y1:y2, x1:x2]
            crob_width = cropped_image.shape[1]
            width_ratio = crob_width/640
            crob_height = cropped_image.shape[0]
            height_ratio = crob_height/480
            resize_crob_img = cv2.resize(cropped_image, (640, 480), interpolation=cv2.INTER_LANCZOS4)

            # YOLO Segmentation 모델 실행
            results = model(resize_crob_img)
        
            for result in results:
                if result.masks is not None:
                    masks = result.masks.data.cpu().numpy()
                    classes = result.boxes.cls.cpu().numpy()
                    class_names = model.names

                    for i, mask in enumerate(masks):
                        binary_mask = (mask > 0.5).astype(np.uint8)
                        mask_indices = np.where(binary_mask > 0)

                        try:
                            # resize > crob으로 좌표값 변환
                            full_mask_indices = (mask_indices[0]*height_ratio.astype(int), mask_indices[1]*width_ratio.astype(int))
                            # crob > 원본으로 좌표값 변환
                            indices = (full_mask_indices[0]+y1, full_mask_indices[1]+x1)
                            object_depths = depth_image[indices]
                            object_depths = object_depths[object_depths > 0]
                            average_depth_mm = np.mean(object_depths)
                            average_depth_cm = average_depth_mm / 10

                            box = result.boxes.xyxy[i].cpu().numpy()
                            cx1, cy1, cx2, cy2 = map(int, box)
                            cv2.rectangle(resize_crob_img, (cx1, cy1), (cx2, cy2), (255,0,0), 2)
                            cv2.putText(resize_crob_img, f"{class_names[int(classes[i])]}: {average_depth_cm:.2f} cm", (cx1, cy1 -10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,0,0), 2)
                            cv2.imshow('model_predict', resize_crob_img)
                            
                        except (IndexError, ValueError) as e:
                            print(f"깊이 계산 중 오류: {e}")

                        
            cv2.imshow('model_predict', resize_crob_img)

        # esc 눌리면 반복문 탈출
        if cv2.waitKey(1) == 27:
            break

finally:
    # 파이프 라인 멈추고
    pipeline.stop()
    # 창을 닫는다
    cv2.destroyAllWindows()
