# -*- coding: utf-8 -*-
import cv2
import numpy as np
from math import atan2, cos, sin, sqrt, pi, degrees

def getOrientation(pts, img):
    sz = len(pts[0])
    data_pts = np.empty((sz, 2), dtype=np.float64)
    for i in range(data_pts.shape[0]):
        data_pts[i, 0] = pts[1][i]
        data_pts[i, 1] = pts[0][i]

    # Perform PCA analysis
    mean = np.empty((0))
    mean, eigenvectors, eigenvalues = cv2.PCACompute2(data_pts, mean)
    # Store the center of the object
    cntr = (int(mean[0, 0]), int(mean[0, 1]))


    p1 = (
    int(mean[0, 0] - 1.73 * eigenvectors[0, 0] * sqrt(eigenvalues[0, 0])), int(mean[0, 1] - 1.73 * eigenvectors[0, 1] * sqrt(eigenvalues[0, 0])))
    cv2.circle(img, p1, 3, (0, 0, 255), 2)
    cv2.line(img, cntr, p1, (0, 0, 255))

    angle = atan2(eigenvectors[0, 1], eigenvectors[0, 0])  # orientation in radians

    print(mean)
    print(eigenvectors)

    return angle

# greenLower = (29, 86, 6)
greenLower = (29, 35, 25)
greenUpper = (64, 255, 255)

if __name__ == '__main__':

    # read video
    # cap = cv2.VideoCapture("../video/Test/C0003.MP4")
    # cap = cv2.VideoCapture("../video/Test/IMG_1948.MOV")
    cap = cv2.VideoCapture("../video/Trimmed/Angle90_R.mov")
    ret, frame = cap.read()

    frame_num = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_H = frame.shape[0]
    frame_W = frame.shape[1]

    frame = cv2.resize(frame, (frame_W // 2, frame_H // 2))
    cv2.imshow("Frame", frame)
    frame_I = 0

    while (True):
        key = cv2.waitKey(0)

        if (key == ord('q')):
            exit()
        elif (key == ord('d')):
            frame_I = min(frame_I + 1, frame_num - 1)
        elif (key == ord('s')):
            frame_I = max(frame_I - 1, 0)
        elif (key == ord('f')):
            frame_I = min(frame_I + 10, frame_num - 1)
        elif (key == ord('a')):
            frame_I = max(frame_I - 10, 0)

        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_I)
        ret, frame = cap.read()
        frame = cv2.resize(frame, (frame_W // 2, frame_H // 2))

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        green_mask = cv2.inRange(hsv, greenLower, greenUpper)
        green_mask = cv2.erode(green_mask, None, iterations = 2)
        green_mask = cv2.dilate(green_mask, None, iterations = 2)

        c = np.where(green_mask == 255)
        print(degrees(getOrientation(c, frame)))

        # _, contours, _ = cv2.findContours(green_mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        # for i, c in enumerate(contours):
        #     # Calculate the area of each contour
        #     area = cv2.contourArea(c);
        #     # Ignore contours that are too small or too large
        #     if area < 1e2 or 1e5 < area:
        #         continue
        #
        #     # Draw each contour only for visualisation purposes
        #     cv2.drawContours(frame, contours, i, (0, 0, 255), 1);
        #     # Find the orientation of each shape
        #     print(degrees(getOrientation(c, frame)))


        cv2.imshow("Frame", frame)
        cv2.imshow("Green mask", green_mask)

        print("%d/%d" % (frame_I, frame_num) )
