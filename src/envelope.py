# -*- coding: utf-8 -*-
import cv2
import numpy as np
from scipy.signal import argrelextrema

z_img = np.zeros((640, 1278, 3), np.uint8)
frame_num = 1278
N = 20
nib_list = np.load("nib_list.npy")
weights = np.hanning(N)
smooth = np.convolve(weights/weights.sum(), nib_list[:, 2])[N-1:-N+1]
point = argrelextrema(smooth, np.greater)
print(point)
threshold = np.zeros((1278))
threshold[0 : point[0][0] + N // 2] = smooth[point[0][0]]
threshold[point[0][-1] + N // 2 : -1] = smooth[point[0][-1]]
for i in range(point[0].shape[0] - 1):
    delta = (smooth[point[0][i + 1]] - smooth[point[0][i]]) / (point[0][i + 1] - point[0][i])
    for n in range(point[0][i] + 1, point[0][i + 1]):
        threshold[n] = threshold[n - 1] + delta


print(threshold)
for i in range(1278 - N):
    z_img[int(nib_list[i, 2] * 8) % 640, i] = (255, 255, 255)
    z_img[int(smooth[i - N // 2] * 8) % 640, i] = (0, 255, 0)
    z_img[int(threshold[i] * 8) % 640, i] = (255, 0, 0)
for n in point[0]:
    z_img[int(smooth[n] * 8) % 640, n + N // 2] = (0, 0, 255)

cv2.imshow("Z", z_img)
cv2.waitKey(0)