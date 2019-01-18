# -*- coding: utf-8 -*-
import cv2
import numpy as np
from scipy.signal import argrelextrema

frame_num = 1278
z_img = np.zeros((640, frame_num, 3), np.uint8)
N = 100
nib_list = np.load("nib_list.npy")

weights = np.hanning(N)
smooth = np.convolve(weights/weights.sum(), nib_list[:, 2])[N-1:-N+1]
smooth = np.insert(smooth, 0, [smooth[0] for i in range(N // 2)])
smooth = np.append(smooth, [smooth[-1] for i in range(N // 2)])
diff = np.diff(smooth)

point = argrelextrema(smooth, np.greater)
print(point)

result = np.zeros((600, 800, 3), np.uint8)
for i in range(frame_num):
    z_img[int(nib_list[i, 2] * 8) % 640, i] = (255, 255, 255)
    z_img[int(smooth[i] * 8) % 640, i] = (0, 255, 0)
    z_img[(int(diff[i] * 32)+200) % 640, i] = (0, 255, 255)
    z_img[200, i] = (128, 128, 128)

    if nib_list[i, 2] > smooth[i]:
        result[int(nib_list[i, 0] * 4) % 600, int(nib_list[i, 1] * 4) % 800] = 255

for n in point[0]:
    z_img[int(smooth[n] * 8) % 640, n + N // 2] = (0, 0, 255)

cv2.imshow("Z", z_img)
cv2.imshow("result", result)
cv2.waitKey(0)