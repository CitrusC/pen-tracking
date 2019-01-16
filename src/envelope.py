# -*- coding: utf-8 -*-
import cv2
import numpy as np
from scipy.signal import argrelextrema

z_img = np.zeros((640, 1278, 3), np.uint8)

N = 20
nib_list = np.load("nib_list.npy")
weights = np.hanning(N)
sam = np.convolve(weights/weights.sum(), nib_list[:, 2])[N-1:-N+1]
point = argrelextrema(sam, np.greater)
print(point)

for i in range(1278 - N):
    z_img[int(nib_list[i, 2] * 8) % 640, i] = (255, 255, 255)
    z_img[int(sam[i - N // 2] * 8) % 640, i] = (0, 255, 0)
for n in point[0]:
    z_img[int(sam[n] * 8) % 640, n + N // 2] = (0, 0, 255)
cv2.imshow("Z", z_img)
cv2.waitKey(0)