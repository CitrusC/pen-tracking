# -*- coding: utf-8 -*-
import cv2
import numpy as np
import matplotlib.pyplot as plt


N = 100
zoom = 16
nib_list = np.load("nib_list.npy")
frame_num = nib_list.shape[0]
z_img = np.zeros((640, frame_num, 3), np.uint8)
outputVideo = False

weights = np.hanning(N)
smooth = np.convolve(weights/weights.sum(), nib_list[:, 2])[N-1:-N+1]
smooth = np.insert(smooth, 0, [smooth[0] for i in range(N // 2)])
smooth = np.append(smooth, [smooth[-1] for i in range(N // 2 - 1)])

H = int((np.max(nib_list[:, 0]) - np.min(nib_list[:, 0])) * zoom + 100)
W = int((np.max(nib_list[:, 1]) - np.min(nib_list[:, 1])) * zoom + 100)
offset = np.array([np.min(nib_list[:, 0]),
                   np.min(nib_list[:, 1])])
print(H, W)
print(offset)

if outputVideo:
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('../video/output_half_dot.avi', fourcc, 20.0, (W, H))
result = np.zeros((H, W, 3), np.uint8)
for i in range(frame_num):
    z_img[int(nib_list[i, 2] * 8) % 640, i] = (255, 255, 255)
    z_img[int(smooth[i] * 8) % 640, i] = (0, 255, 0)

    if nib_list[i, 2] > smooth[i]:
        result[int((nib_list[i, 0] - offset[0]) * zoom),
               int((nib_list[i, 1] - offset[1]) * zoom)] = 255
        cv2.circle(result, (int((nib_list[i, 1] - offset[1]) * zoom),
                            int((nib_list[i, 0] - offset[0]) * zoom)), 3, (255, 255, 255), -1)
    #     cv2.line(result, (int((nib_list[i-1, 1] - offset[1]) * zoom),
    #                       int((nib_list[i-1, 0] - offset[0]) * zoom)),
    #                      (int((nib_list[i, 1] - offset[1]) * zoom),
    #                       int((nib_list[i, 0] - offset[0]) * zoom)), (255, 255, 255), 2)

    if outputVideo:
        out.write(result)

    x = range(0, frame_num)
    plt.subplot(1, 1, 1)
    plt.plot(x, nib_list[:, 2])
    plt.plot(x, smooth)
    plt.axis([0, frame_num, 520, 460])
    plt.show()
# result[int(nib_list[i, 0] * 4) % 600, int(nib_list[i, 1] * 4) % 800] = 255


cv2.imshow("Z", z_img)
cv2.imshow("result", result)
if outputVideo:
    out.release()
cv2.waitKey(0)
