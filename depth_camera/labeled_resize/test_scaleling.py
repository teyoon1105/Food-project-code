import os
import random
import cv2
import numpy as np
import json

def create_composite_images_with_polygon(input_dir, output_dir, coordinates, class_folders, num_images=10):
    """
    여러 클래스의 이미지를 결합하여 하나의 캔버스에 배치하고, 폴리곤 좌표를 변환하여 JSON 파일 생성.

    Args:
        input_dir (str): 클래스별 이미지가 저장된 폴더 경로.
        output_dir (str): 생성된 이미지를 저장할 폴더 경로.
        coordinates (dict): 각 클래스별 이미지를 배치할 좌표 및 크기 {'rice': (x, y, w, h), ...}.
        class_folders (dict): 클래스별 폴더 이름 {'rice': [...], ...}.
        num_images (int): 생성할 이미지의 총 개수.
    """
    # 출력 폴더 생성
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 캔버스 크기 설정 (640x640, 흰색 배경)
    canvas_size = (640, 640)

    # 지정된 수만큼 이미지를 생성
    for img_idx in range(num_images):
        # 흰색 캔버스 생성
        canvas = np.ones((canvas_size[1], canvas_size[0], 3), dtype=np.uint8) * 255
        shapes = []  # JSON에 저장할 폴리곤 데이터를 담을 리스트

        # 클래스별로 선택된 이미지 저장 (밥, 국, 반찬)
        selected_images = {'rice': None, 'soup': None, 'side': []}
        side_candidates = []  # 반찬 후보 이미지 경로를 저장할 리스트

        # 각 클래스에 대해 이미지 선택
        for class_label, folders in class_folders.items():
            for folder_name in folders:
                folder_path = os.path.join(input_dir, folder_name)
                if not os.path.exists(folder_path):
                    continue

                # 해당 폴더에서 이미지 파일 목록 가져오기
                images = [f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
                if not images:
                    continue

                # 밥과 국은 하나씩 반드시 선택
                if class_label in ['rice', 'soup'] and selected_images[class_label] is None:
                    selected_images[class_label] = os.path.join(folder_path, random.choice(images))

                # 반찬은 각 폴더에서 하나씩 후보에 추가
                if class_label == 'side':
                    side_candidates.append(os.path.join(folder_path, random.choice(images)))

        # 반찬은 총 4개를 무작위로 선택 (중복 제거)
        if len(side_candidates) < 4:
            raise ValueError("Not enough side dish folders to select 4 unique images.")
        selected_images['side'] = random.sample(side_candidates, 4)

        # 선택된 이미지를 캔버스에 배치
        for class_label, image_paths in selected_images.items():
            # 밥과 국은 리스트로 변환하여 반복문 처리
            if class_label in ['rice', 'soup']:
                image_paths = [image_paths]

            # 선택된 이미지들을 순회하며 배치
            for i, image_path in enumerate(image_paths):
                # 해당 이미지와 동일한 이름의 JSON 파일 경로
                json_path = os.path.splitext(image_path)[0] + ".json"
                if not os.path.exists(json_path):
                    print(f"JSON file not found for image: {image_path}")
                    continue

                # 이미지를 읽어오기
                original_image = cv2.imread(image_path)
                if original_image is None:
                    print(f"Failed to read image: {image_path}")
                    continue

                # 원본 이미지의 높이와 너비
                original_h, original_w = original_image.shape[:2]

                # 해당 클래스의 좌표 및 크기 가져오기
                if class_label == 'rice':
                    x, y, w, h = coordinates['rice']
                elif class_label == 'soup':
                    x, y, w, h = coordinates['soup']
                elif class_label == 'side':
                    x, y, w, h = coordinates['side'][i]  # 반찬은 순서대로 배치

                # 이미지를 지정된 크기로 조정
                # resized_image = cv2.resize(original_image, (w, h), interpolation=cv2.INTER_AREA)

                # 조정된 이미지를 캔버스의 지정된 위치에 배치
                canvas[y:y + h, x:x + w] = original_image

                # JSON 파일 읽어오기
                with open(json_path, 'r') as jf:
                    json_data = json.load(jf)

                # 기존 폴리곤 좌표를 새 캔버스에 맞게 변환
                for shape in json_data['shapes']:
                    transformed_points = []
                    for point in shape['points']:
                        old_x, old_y = point
                        # 좌표 변환 공식 적용
                        new_x = x + (old_x / original_w) * w
                        new_y = y + (old_y / original_h) * h
                        transformed_points.append([new_x, new_y])

                    # 변환된 폴리곤 좌표를 저장
                    shapes.append({
                        "label": shape['label'],  # 기존 라벨 유지
                        "points": transformed_points,  # 변환된 좌표
                        "group_id": shape.get('group_id', None),  # 그룹 ID 유지
                        "shape_type": shape['shape_type'],  # 폴리곤 타입 유지
                        "flags": shape['flags']  # 추가 플래그 유지
                    })

        # 결과 이미지 저장
        output_image_name = f"composite_image_{img_idx + 1}.jpg"
        output_image_path = os.path.join(output_dir, output_image_name)
        cv2.imwrite(output_image_path, canvas)

        # 변환된 JSON 파일 저장
        output_json_name = os.path.splitext(output_image_name)[0] + ".json"
        output_json_path = os.path.join(output_dir, output_json_name)
        output_json_data = {
            "version": "0.4.15",
            "flags": {},
            "shapes": shapes,  # 변환된 폴리곤 데이터
            "imagePath": output_image_name,  # 생성된 이미지 파일명
            "imageData": None,
            "imageHeight": canvas_size[1],  # 캔버스 높이
            "imageWidth": canvas_size[0]  # 캔버스 너비
        }

        with open(output_json_path, 'w') as jf:
            json.dump(output_json_data, jf, indent=4)

        print(f"Saved composite image and JSON: {output_image_path}, {output_json_path}")


# Example usage
input_directory = "C:/Users/Sesame/Desktop/6mix"  # 원본 이미지 폴더 경로
output_directory = "C:/Users/Sesame/Desktop/6mix/output"  # 결과 저장 폴더 경로

# 클래스별 배치 좌표와 크기 설정
coordinates = {
    'rice': (0, 384, 320, 256),  # 하단 왼쪽
    'soup': (320, 384, 320, 256),  # 하단 오른쪽
    'side': [
        (0, 0, 160, 256),    # 상단 1
        (160, 0, 160, 256),  # 상단 2
        (320, 0, 160, 256),  # 상단 3
        (480, 0, 160, 256)   # 상단 4
    ]
}

# 클래스별 폴더 설정
class_folders = {
    'rice': ['01011001'],  # 밥
    'soup': ['04017001'],  # 국
    'side': ['06012004', '07014001', '11013007', '12011008']  # 반찬
}

# 함수 실행
create_composite_images_with_polygon(
    input_dir=input_directory,
    output_dir=output_directory,
    coordinates=coordinates,
    class_folders=class_folders,
    num_images=10
)
