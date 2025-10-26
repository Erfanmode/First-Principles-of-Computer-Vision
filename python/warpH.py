import cv2
import numpy as np

def warpH(im, H, out_size, fill_value=0):
    # Define the transformation matrix
    tform = cv2.warpPerspective(im, H, (out_size[1], out_size[0]), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(fill_value, fill_value, fill_value))

    return tform