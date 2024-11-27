import cv2
import numpy as np

def map_to_multichannel(input_path):
    # 1. 흑백 마스크 로드
    mask = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)  # 단일 채널 이미지

    # 2. 연결된 객체를 분리 (Connected Components)
    num_labels, labels = cv2.connectedComponents(mask)  # labels: 객체별 ID 맵핑

    # 3. 다중 채널 이미지 생성 (H, W, 3)
    # h, w = mask.shape
    # multi_channel_mask = np.zeros((h, w, 3), dtype=np.uint8)

    return num_labels, labels

img_path = "D:/data_set/masks/masks_06_062_06012004_160336019675129_1.jpeg"
num1, num2 = map_to_multichannel(img_path)

# 레이블 배열 출력
# labels 배열 출력을 "labels.txt" 파일에 저장
np.savetxt("labels.txt", num2, fmt="%d", delimiter=" ")