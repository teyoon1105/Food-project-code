import os
import shutil

def filter_labels_by_existing_images(image_dir, label_dir, output_dir):
    """
    이미지 폴더에 있는 파일 이름과 일치하는 JSON 라벨만 필터링하여 새 폴더에 저장.

    Args:
        image_dir (str): 이미지 파일들이 저장된 폴더 경로.
        label_dir (str): JSON 라벨 파일들이 저장된 폴더 경로.
        output_dir (str): 필터링된 라벨 파일을 저장할 폴더 경로.
    """
    # 출력 디렉토리 생성
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    # 이미지 파일 목록 가져오기 (확장자 제거)
    image_files = [
        os.path.splitext(f)[0]  # 확장자 제거한 파일명만
        for f in os.listdir(image_dir)
        if f.lower().endswith(('.png', '.jpg', '.jpeg'))
    ]

    # 라벨 파일 목록 가져오기
    label_files = {os.path.splitext(f)[0]: f for f in os.listdir(label_dir) if f.endswith('.json')}

    # 이미지 이름과 일치하는 라벨 파일 필터링
    matched_labels = [label_files[img_name] for img_name in image_files if img_name in label_files]

    # 라벨 파일 복사
    for label_file in matched_labels:
        src_path = os.path.join(label_dir, label_file)
        dest_path = os.path.join(output_dir, label_file)
        shutil.copy(src_path, dest_path)
        print(f"Copied: {label_file}")

    print(f"Filtered {len(matched_labels)} labels to {output_dir}")


# 사용 예시
image_folder = "C:/Users/SBA/Desktop/scale/scale_0411005/output/04011005"  # 이미지 폴더 경로
label_folder = "C:/Users/SBA/Desktop/scale/scale_0411005/label"  # 라벨 폴더 경로
output_label_folder = "C:/Users/SBA/Desktop/scale/scale_0411005/04011005_json_label"  # 필터링된 라벨 저장 폴더 경로

# 함수 실행
filter_labels_by_existing_images(
    image_dir=image_folder,
    label_dir=label_folder,
    output_dir=output_label_folder
)
