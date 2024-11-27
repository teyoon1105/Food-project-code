import json
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt

def create_mask_from_annotation(annotations, image_shape):
    """JSON annotations (your specific format) to binary mask."""
    mask = np.zeros(image_shape[:2], dtype=np.uint8)

    for shape in annotations['shapes']: 
        if shape['shape_type'] == 'polygon':
            points = np.array(shape['points'], dtype=np.int32)
            cv2.fillPoly(mask, [points], 1)

    return mask

def plot_image(image):
    # Matplotlib을 사용하여 이미지 표시
    plt.figure(figsize=(10, 5))

    plt.subplot(1, 3, 1)
    plt.imshow(image)
    plt.title("Original Image")

    plt.subplot(1, 3, 2)
    plt.imshow(mask, cmap='gray') # 회색조로 마스크 표시
    plt.title("Mask")

    mask_rgb = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    mask_rgb *= 255
    combined_image = cv2.addWeighted(image, 0.7, mask_rgb, 0.3, 0)
    plt.subplot(1,3,3)
    plt.imshow(combined_image)
    plt.title("Combined Image")

    plt.tight_layout()
    plt.show()

# --- 한장의 이미지와 JSON 파일 경로 설정 ---
# image_path = "D:/data_set/images/06_062_06012004_160275519380311.jpg"  # 이미지 파일 경로를 여기에 입력하세요
# json_path = "D:/data_set/labels/06_062_06012004_160275519380311.json"  # JSON 파일 경로를 여기에 입력하세요

# mask dir 만들기
data_dir = "D:/data_set"

# 데이터셋 경로 아래에 mask dir만들기
mask_dir = os.path.join(data_dir, 'masks')
os.makedirs(mask_dir, exist_ok=True)

# 디렉토리 순회하면서 모든 이미지와 Json 파일 가져와 처리
image_dir_path = "D:/data_set/images"
json_dir_path = "D:/data_set/labels"

# 디렉토리가 존재하는지 확인
if not os.path.exists(image_dir_path):
    
    print(f"오류: 이미지 디렉토리 '{image_dir_path}'를 찾을 수 없습니다.")
    exit()
if not os.path.exists(json_dir_path):
    print(f"오류: JSON 디렉토리 '{json_dir_path}'를 찾을 수 없습니다.")
    exit()


for image_filename in os.listdir(image_dir_path):
    # image 파일 경로 설정
    # 파일이 jpg, jpeg, png 등 이미지 파일이 아니면 다음 파일로 넘어가기
    if not image_filename.lower().endswith(('.png', '.jpg', '.jpeg')): # 파일 형식 확인하여 오류 방지
        continue

    image_path = os.path.join(image_dir_path, image_filename)
    
    # --- 이미지 로드 ---
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not load image from {image_path}")
        exit()
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # json 파일 경로 설정
    json_filename = os.path.splitext(image_filename)[0] + ".json"  # 확장자 제거 후 파일 이름 추출
    json_path = os.path.join(json_dir_path, json_filename)

    # --- JSON 로드 ---
    try:
        with open(json_path, 'r') as f:
            annotations = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Error loading JSON: {e}")
        exit()


    # --- 마스크 생성 및 출력 ---
    mask = create_mask_from_annotation(annotations, image.shape)

    # 저장할 파일 이름 경로를 통해 생성
    file_name = 'masks_' + image_filename

    mask_img_path = os.path.join(mask_dir, file_name)
    cv2.imwrite(mask_img_path, mask * 255) # 0~1을 0~255로 변환하여 저장_

# TODO
# instance mask를 사용할 예정이기 때문에
# 현재 단일채널(흑백) 마스크 데이터를
# 다중 채널로 확장, 첫번째와 세번째 채널에 주석 정보 매핑