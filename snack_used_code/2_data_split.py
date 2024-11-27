import os
import shutil
import random

def split_dataset(data_dir, output_dir, train_ratio=0.8, valid_ratio=0.1):
    # 데이터 폴더에서 모든 jpg 파일 목록을 가져오기 (txt 파일은 jpg 파일과 짝지어 이동)
    image_files = [f for f in os.listdir(data_dir) if f.endswith('.jpg')]
    random.shuffle(image_files)  # 무작위로 파일 리스트 섞기

    # 데이터셋 분할 인덱스 계산
    total_count = len(image_files)
    train_end = int(total_count * train_ratio)
    valid_end = int(total_count * (train_ratio + valid_ratio))

    # 폴더 경로 생성
    for subset in ['train', 'valid', 'test']:
        os.makedirs(os.path.join(output_dir, subset, 'images'), exist_ok=True)
        os.makedirs(os.path.join(output_dir, subset, 'labels'), exist_ok=True)

    # 파일 복사 함수
    def copy_files(file_list, subset):
        for image_file in file_list:
            # 이미지 파일 복사
            shutil.copy(
                os.path.join(data_dir, image_file),
                os.path.join(output_dir, subset, 'images', image_file)
            )

            # 이미지 파일과 동일한 이름의 텍스트 파일 복사
            txt_file = image_file.replace('.jpg', '.txt')
            txt_path = os.path.join(data_dir, txt_file)
            if os.path.exists(txt_path):  # 텍스트 파일이 존재할 경우에만 복사
                shutil.copy(
                    txt_path,
                    os.path.join(output_dir, subset, 'labels', txt_file)
                )

    # 데이터셋을 train, valid, test로 나누어 복사
    copy_files(image_files[:train_end], 'train')
    copy_files(image_files[train_end:valid_end], 'valid')
    copy_files(image_files[valid_end:], 'test')

    print(f"데이터셋 분할 완료: {output_dir}/train, {output_dir}/valid, {output_dir}/test")

    # 각 폴더의 파일 개수를 계산하는 함수
    def count_files_in_directory(directory):
        return len([f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))])

    # 각 폴더의 파일 개수 출력
    print(f"Train 폴더 이미지 개수: {count_files_in_directory(os.path.join(output_dir, 'train', 'images'))}")
    print(f"Train 폴더 라벨 개수: {count_files_in_directory(os.path.join(output_dir, 'train', 'labels'))}")
    print(f"Valid 폴더 이미지 개수: {count_files_in_directory(os.path.join(output_dir, 'valid', 'images'))}")
    print(f"Valid 폴더 라벨 개수: {count_files_in_directory(os.path.join(output_dir, 'valid', 'labels'))}")
    print(f"Test 폴더 이미지 개수: {count_files_in_directory(os.path.join(output_dir, 'test', 'images'))}")
    print(f"Test 폴더 라벨 개수: {count_files_in_directory(os.path.join(output_dir, 'test', 'labels'))}")

# 메인 함수 실행
data_dir = 'snack_image_data'  # 원본 데이터 폴더 경로
output_dir = 'snack_split_data'  # 데이터를 나눠서 저장할 폴더
split_dataset(data_dir, output_dir)
