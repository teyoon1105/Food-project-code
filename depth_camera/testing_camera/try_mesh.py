import pyrealsense2 as rs  # Intel RealSense 카메라를 위한 라이브러리
import numpy as np  # 수치 계산 및 배열 처리를 위한 라이브러리
import cv2  # OpenCV, 컴퓨터 비전 및 이미지 처리를 위한 라이브러리
import os  # 파일 및 경로 작업을 위한 라이브러리
from ultralytics import YOLO  # YOLO 객체 탐지 라이브러리
import logging  # 로그 메시지 출력을 위한 라이브러리
from scipy.spatial import ConvexHull
import open3d as o3d

# 로그 레벨 설정 (INFO 메시지 비활성화)
logging.getLogger("ultralytics").setLevel(logging.WARNING)

# 현재 작업 디렉토리를 얻고 YOLO 모델 파일의 경로를 설정
Now_Path = os.getcwd()
model_path = os.path.join(Now_Path, 'best.pt')  # YOLO 모델 파일 경로
model = YOLO(model_path)  # YOLO 모델 로드

# Intel RealSense 카메라 스트림을 설정
pipeline = rs.pipeline()  # 파이프라인 생성
config = rs.config()  # 설정 객체 생성
config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)  # 깊이 스트림 설정
config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)  # 컬러 스트림 설정
pipeline.start(config)  # 스트림 시작

# 깊이 데이터를 컬러 이미지와 정렬
align_to = rs.stream.color
align = rs.align(align_to)

# 관심 영역 (ROI) 좌표 설정
roi_pts = [(410, 180), (870, 540)]

# 클래스 ID와 이름, 색상을 매핑
cls_name_color = {
    '01011001': ('Rice', (255, 0, 255)),  # 보라색
    '06012004': ('Tteokgalbi', (0, 255, 0)),  # 초록색
    '07014001': ('eggRoll', (0, 0, 255)),  # 빨간색
    '11013007': ('Spinach', (255, 255, 0)),  # 노란색
    '04017001': ('Doenjangjjigae', (0, 255, 255))  # 하늘색
}

# 기준 트레이 깊이 데이터를 저장할 변수 초기화
save_depth = None

# 마우스 콜백 함수
def mouse_callback(event, x, y, flags, param):
    global save_depth  # 전역 변수 사용 선언

    # 마우스 왼쪽 버튼 클릭 이벤트 처리
    if event == cv2.EVENT_LBUTTONDOWN:
        save_depth = cropped_depth_image  # 현재 깊이 이미지를 저장
        print("Depth image saved!")  # 저장 완료 메시지 출력

def calculate_volume(hull):
    total_volume = 0
    for simplex in hull.simplices:  # 삼각형의 꼭짓점 인덱스
        vertices = hull.points[simplex]  # 삼각형 꼭짓점 (3개)
        a, b, c = vertices[0], vertices[1], vertices[2]
        volume = np.dot(a, np.cross(b, c)) / 6  # 삼각뿔 부피 공식
        total_volume += volume

    return abs(total_volume)  # 부피는 절댓값으로 반환

# 메인 처리 루프
try:
    while True:
        # 카메라에서 프레임 읽기
        frames = pipeline.wait_for_frames()
        aligned_frames = align.process(frames)  # 깊이 데이터를 컬러 이미지에 정렬
        aligned_depth_frame = aligned_frames.get_depth_frame()  # 정렬된 깊이 프레임 가져오기
        color_frame = aligned_frames.get_color_frame()  # 정렬된 컬러 프레임 가져오기

        # 유효한 프레임이 없으면 건너뜀
        if not aligned_depth_frame or not color_frame:
            continue

        # 깊이 프레임과 컬러 프레임을 numpy 배열로 변환
        depth_image = np.asanyarray(aligned_depth_frame.get_data())
        img = np.asanyarray(color_frame.get_data())

        # 깊이와 컬러 이미지를 상하좌우로 반전
        depth_image = cv2.flip(depth_image, -1)
        img = cv2.flip(img, -1)

        # 관심 영역(ROI) 좌표를 사용해 이미지를 크롭
        x1, y1 = roi_pts[0]
        x2, y2 = roi_pts[1]
        cropped_image = img[y1:y2, x1:x2]  # 컬러 이미지에서 ROI 추출
        cropped_depth_image = depth_image[y1:y2, x1:x2]  # 깊이 이미지에서 ROI 추출

        # 원본 이미지에 ROI 영역 표시
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)  # 빨간색 사각형으로 표시
        cv2.imshow('Color Image with ROI', img)  # ROI가 표시된 컬러 이미지 표시
        cv2.setMouseCallback("Color Image with ROI", mouse_callback)  # 마우스 콜백 함수 등록

        # YOLO 모델을 사용하여 객체 탐지 수행
        results = model(cropped_image)

        # 탐지된 객체 시각화를 위한 초기화
        blended_image = cropped_image.copy()  # 크롭된 이미지를 복사
        all_colored_mask = np.zeros_like(cropped_image)  # 빈 마스크 생성

        for result in results:
            if result.masks is not None:  # 탐지된 객체에 마스크가 있는 경우
                masks = result.masks.data.cpu().numpy()  # 마스크 데이터 가져오기
                classes = result.boxes.cls.cpu().numpy()  # 클래스 ID 가져오기
                class_names = model.names  # 클래스 이름 매핑

                for i, mask in enumerate(masks):
                    # 마스크를 크롭된 이미지 크기에 맞게 조정
                    resized_mask = cv2.resize(mask, (cropped_image.shape[1], cropped_image.shape[0]))
                    color_mask = (resized_mask > 0.5).astype(np.uint8)  # 바이너리 마스크 생성

                    # 클래스 이름과 색상 얻기
                    key = class_names[int(classes[i])]
                    object_name, color = cls_name_color.get(key, ("Unknown", (255, 255, 255)))

                    # 마스크 영역에 색상 적용
                    all_colored_mask[color_mask == 1] = color

                    if save_depth is None:  # save_depth가 None인 경우 처리
                        print("기준 깊이 데이터가 없습니다. 마우스 클릭으로 기준 깊이를 설정하세요.")
                        continue

                    # 마스크 영역의 좌표와 깊이 데이터 계산
                    mask_indices = np.where(color_mask > 0)
                    y_indices, x_indices = mask_indices
                    original_y_indices = y_indices + y1  # 원본 이미지 좌표로 변환
                    original_x_indices = x_indices + x1

                    # 뎁스 프레임의 내부 파라미터 얻기
                    depth_intrin = aligned_depth_frame.profile.as_video_stream_profile().intrinsics
                    
                    # 3차원 좌표를 저장할 NumPy 배열 초기화
                    points = np.zeros((len(original_x_indices), 3), dtype=np.float64) # float64 타입 중요!
                    
                    for i, (pixel_y, pixel_x) in enumerate(zip(original_y_indices, original_x_indices)):
                        depth_value = depth_image[pixel_y, pixel_x]
                        point_3d = rs.rs2_deproject_pixel_to_point(depth_intrin, [pixel_x, pixel_y], depth_value)
                        save_depth_value = save_depth[pixel_y - y1, pixel_x - x1]
                        points[i] = [point_3d[0], point_3d[1], save_depth_value - point_3d[2]]



                    # try:
                    #     hull = ConvexHull(points)
                    #     volumn_mm3 = calculate_volume(hull)
                    #     volumn_cm3 = volumn_mm3 / 1000  # mm³ to cm³ 변환
                    #     print(f"부피 = {volumn_cm3:.2f}cm^3")

                    try:
                        # Open3D PointCloud 생성
                        pcd = o3d.geometry.PointCloud()
                        pcd.points = o3d.utility.Vector3dVector(points)

                        # 노말 추정 (필수)
                        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))

                        # Poisson surface reconstruction
                        mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=8, width=0, scale=1.1, linear_fit=False)

                        # 메쉬가 watertight한지 확인
                        if mesh.is_watertight():
                            volume = mesh.get_volume()
                            print(f"Volume: {volume}")
                        else:
                            print("Mesh is not watertight!")
                            volume = 0 # 부피 계산 실패시 0으로 설정
                                            

                    except Exception as e : #scipy 오류 예외처리
                        print("ConvexHull 오류:",e)

                    # 객체 이름과 부피, 높이를 이미지에 표시
                    min_x, max_x = np.min(x_indices), np.max(x_indices)
                    min_y, max_y = np.min(y_indices), np.max(y_indices)
                    label_x, label_y = min_x, min_y - 10  # 라벨 위치 설정
                    cv2.putText(all_colored_mask, f"{object_name}:V:{volume:.2f}cm_3",
                                (label_x, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # 마스크와 원본 이미지를 혼합하여 시각화
        blended_image = cv2.addWeighted(cropped_image, 0.7, all_colored_mask, 0.3, 0)

        # 결과 이미지 출력
        cv2.imshow('Segmented Mask with Heights', blended_image)

        # ESC 키 입력 시 루프 종료
        if cv2.waitKey(1) == 27:
            break

# 종료 시 리소스 정리
finally:
    pipeline.stop()  # 파이프라인 정지
    cv2.destroyAllWindows()  # 모든 OpenCV 창 닫기