import numpy as np

# numpy library test code
# 인위적으로 만든 넘파이 배열에서 0보다 큰 부분을 인덱싱하는 코드
binary_mask = np.array([
    [0, 1, 0, 0, 1],
    [0, 0, 1, 1, 0],
    [1, 1, 0, 0, 0],
    [0, 0, 0, 1, 0]
])

mask_indices = np.where(binary_mask > 0)

print((mask_indices))
