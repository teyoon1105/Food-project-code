import pyrealsense2 as rs
import numpy as np
import cv2
from ultralytics import YOLO

# YOLO Segmentation 모델 로드
model_path = "C:/Users/SBA/teyoon_github/Food-project-code/depth_camera/best.pt"
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

# ROI 고정
roi_pts = [(160, 120), (480,360)]
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

    
        # 해당 좌표 값을 정렬
        x1, y1 = roi_pts[0]
        x2, y2 = roi_pts[1]

        # 정렬된(작은 값, 큰 값) 좌표로 직사각형 만들어서 컬러 이미지에 그리기
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.imshow('Color Image', img)
        
        cropped_image = img[y1:y2, x1:x2]
        
    
        cv2.imshow('model_predict', cropped_image)
        # YOLO Segmentation 모델 실행
        results = model(cropped_image)
            
    
        for result in results:
                if result.masks is not None:
                    masks = result.masks.data.cpu().numpy()
                    classes = result.boxes.cls.cpu().numpy()
                    class_names = model.names

                    for i, mask in enumerate(masks):
                        binary_mask = (mask > 0.5).astype(np.uint8)
                        mask_indices = np.where(binary_mask > 0)

                        try:
                            # TODO
                            # roi 평균 높이를 잘 구하는지 확인
                            # 마스크도 위에 imshow하기  
                            # # crob 이미지 좌표를 원본 이미지 좌표로 변경
                            # full_mask_indices = (mask_indices[0]+y1, mask_indices[1]+x1)
                            
                            # object_depths = depth_image[full_mask_indices]
                            # object_depths = object_depths[object_depths > 0]
                            # average_depth_mm = np.mean(object_depths)
                            # average_depth_cm = average_depth_mm / 10

                            box = result.boxes.xyxy[i].cpu().numpy()
                            cx1, cy1, cx2, cy2 = map(int, box)
                            cv2.rectangle(cropped_image, (cx1, cy1), (cx2, cy2), (255,0,0), 2)

                            # bbox의 좌표를 원본이미지 좌표로 변환
                            real_pt = [(cx1+x1, cy1+y1), (cx2+x1, cy2+y1)]
                            
                            # 원본이미지 위에 bbox 좌표의 마스크 넘파이 배열을 만들기
                            mask = np.zeros(depth_image.shape, dtype=np.uint8)
                            
                            cv2.rectangle(mask, real_pt[0], real_pt[1], 255, -1)
                            
                            masked_depth_image = cv2.bitwise_and(depth_image, depth_image, mask=mask)
                            
                            box_depth = depth_image[masked_depth_image>0]
                            
                            average_distance_mm = np.mean(box_depth)
                            average_distance_cm = average_distance_mm / 10
                            
                            cv2.putText(cropped_image, f"{class_names[int(classes[i])]}:{average_distance_cm:.2f}cm", (cx1, cy1 -10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 2)
                            cv2.imshow('model_predict', cropped_image)
                            
                        except (IndexError, ValueError) as e:
                            print(f"깊이 계산 중 오류: {e}")

        # esc 눌리면 반복문 탈출
        if cv2.waitKey(1) == 27:
            break

finally:
    # 파이프 라인 멈추고
    pipeline.stop()
    # 창을 닫는다
    cv2.destroyAllWindows()
