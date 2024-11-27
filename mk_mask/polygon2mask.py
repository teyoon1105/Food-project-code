import numpy as np
from ultralytics.data.utils import polygon2mask
import json
import cv2
import os
import matplotlib.pyplot as plt

def create_mask_from_annotation(annotations, image_shape):
    """JSON annotations (your specific format) to binary mask."""
    mask = np.zeros(image_shape[:2], dtype=np.uint8)
    img_shape = (1024, 1024)
    for shape in annotations['shapes']: 
        if shape['shape_type'] == 'polygon':
            points = np.array(shape['points'], dtype=np.int32)
            # mask 만들기
            mask = polygon2mask(
                img_shape,  # tuple
                points,  # input as list
                color=255,  # 8-bit binary
                downsample_ratio=1,
            )
    return mask