import numpy as np

binary_mask = np.array([
    [0, 1, 0, 0, 1],
    [0, 0, 1, 1, 0],
    [1, 1, 0, 0, 0],
    [0, 0, 0, 1, 0]
])

mask_indices = np.where(binary_mask > 0)

print((mask_indices))
