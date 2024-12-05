import os
import random
import cv2
import numpy as np

def scale_and_save_images_with_coordinates(input_dir, output_dir, coordinates=None, class_folders=None):
    """
    폴더명을 기반으로 클래스를 판단하여 이미지를 축소 후 지정된 좌표에 배치.
    이미 640x640 크기로 처리.

    Args:
        input_dir (str): 클래스별 이미지가 저장된 폴더 경로 (e.g., 'dataset/classname').
        output_dir (str): 축소된 이미지를 저장할 폴더 경로.
        coordinates (dict): 클래스별 이미지를 배치할 좌표와 크기 {'rice': (x, y, w, h), 'soup': (x, y, w, h), 'side': [(x, y, w, h), ...]}.
        class_folders (dict): 클래스별 폴더 이름 {'rice': ['rice_folder1', ...], 'soup': [...], 'side': [...]}.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if coordinates is None:
        raise ValueError("Coordinates must be provided for each class (e.g., rice, soup, side).")

    if class_folders is None:
        raise ValueError("Class folders must be specified as a dictionary with keys 'rice', 'soup', 'side'.")

    # 클래스별 폴더 순회
    for class_folder in os.listdir(input_dir):
        class_path = os.path.join(input_dir, class_folder)

        # 폴더 내의 이미지 파일 확인
        if os.path.isdir(class_path):
            images = [f for f in os.listdir(class_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

            # 클래스 판단
            if class_folder in class_folders['rice']:
                class_label = 'rice'
            elif class_folder in class_folders['soup']:
                class_label = 'soup'
            elif class_folder in class_folders['side']:
                class_label = 'side'
            else:
                print(f"Skipping unknown class folder: {class_folder}")
                continue

            # 입력 폴더명을 그대로 사용하여 출력 폴더 생성
            class_output_dir = os.path.join(output_dir, class_folder)
            os.makedirs(class_output_dir, exist_ok=True)

            # 이미지 파일의 50%만 무작위로 선택
            selected_images = random.sample(images, len(images) // 2)

            # 선택된 이미지 파일만 처리
            for image_file in selected_images:
                image_path = os.path.join(class_path, image_file)
                original_image = cv2.imread(image_path)

                if original_image is None:
                    print(f"Failed to read image: {image_path}")
                    continue

                # 좌표와 크기 설정
                if class_label == 'rice':
                    x, y, w, h = coordinates['rice']
                elif class_label == 'soup':
                    x, y, w, h = coordinates['soup']
                elif class_label == 'side':
                    x, y, w, h = random.choice(coordinates['side'])  # 반찬은 랜덤하게 좌표 선택

                # 이미지 크기 조정
                resized_image = cv2.resize(original_image, (w, h), interpolation=cv2.INTER_AREA)

                # 640x640 캔버스 생성 (흰 배경)
                canvas = np.ones((640, 640, 3), dtype=np.uint8) * 255

                # 이미지를 캔버스 위에 배치
                canvas[y:y + h, x:x + w] = resized_image

                # 파일명을 수정하여 저장
                new_filename = os.path.splitext(image_file)[0] + os.path.splitext(image_file)[1]
                save_path = os.path.join(class_output_dir, new_filename)
                cv2.imwrite(save_path, canvas)
                print(f"Saved: {save_path}")


# 사용 예시(경로 수정해야함)
input_directory = "C:/Users/SBA/Desktop/scale/scale_0612008" # 원본이미지 폴더경로
output_directory = "C:/Users/SBA/Desktop/scale/scale_0612008/output"  # 축소된 이미지 저장 경로

# 좌표 및 크기 설정 (640x640 기준)
coordinates = {
    'rice': (0, 384, 320, 256),  # 하단 왼쪽 (x, y, width, height)
    'soup': (320, 384, 320, 256),  # 하단 오른쪽 (x, y, width, height)
    'side': [
        (0, 0, 160, 256),    # 상단 1구 (x, y, width, height)
        (160, 0, 160, 256),  # 상단 2구 (x, y, width, height)
        (320, 0, 160, 256),  # 상단 3구 (x, y, width, height)
        (480, 0, 160, 256)   # 상단 4구 (x, y, width, height)
    ]
}

# 클래스별 폴더 이름
class_folders = {
    'rice': ['01011001', '01012006', '01012002'],  # 밥이 포함된 폴더 이름
    'soup': ['04011005', '04011007', '04017001', '04011011'],  # 국이 포함된 폴더 이름
    'side': ['06012004', '07014001', '11013007', '12011008', '06012008', '08011003', '10012001', '11013002', '12011003', '07013003', '11013010']  # 반찬이 포함된 폴더 이름
}

# 함수 실행
scale_and_save_images_with_coordinates(
    input_dir=input_directory,
    output_dir=output_directory,
    coordinates=coordinates,
    class_folders=class_folders
)