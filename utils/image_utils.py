import cv2
from typing import Tuple


def resize_width(frame, width: int):
    h, w = frame.shape[:2]
    if w == width:
        return frame
    scale = width / float(w)
    new_h = int(h * scale)
    return cv2.resize(frame, (width, new_h), interpolation=cv2.INTER_AREA)
