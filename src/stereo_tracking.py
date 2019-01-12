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
    cntr = (mean[0, 0], mean[0, 1])

    p1 = (
        cntr[0] - 1.73205 * eigenvectors[0, 0] * sqrt(eigenvalues[0, 0]),
        cntr[1] - 1.73205 * eigenvectors[0, 1] * sqrt(eigenvalues[0, 0]))

    p1_int = (int(p1[0]), int(p1[1]))
    cntr_int = (int(cntr[0]), int(cntr[1]))
    cv2.circle(img, p1_int, 3, (0, 0, 255), 3)
    cv2.arrowedLine(img, p1_int, cntr_int, (0, 0, 255), 2)

    vector = (eigenvectors[0, 0], eigenvectors[0, 1])
    return p1, vector


def calculateFrame(cap, frame_I, greenRange):
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_I)
    ret, frame = cap.read()
    frame = cv2.resize(frame, (frame_W, frame_H))
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    green_mask = cv2.inRange(hsv, greenRange[0], greenRange[1])
    green_mask = cv2.erode(green_mask, None, iterations=2)
    green_mask = cv2.dilate(green_mask, None, iterations=2)
    c = np.where(green_mask == 255)
    position, vector = getOrientation(c, frame)
    return frame, position, vector


def reconstruct_90(result, positionL, positionR, vectorL, vectorR, penLength):
    position3D = (-positionR[0], -positionL[0], (positionL[1] + positionR[1]) / 2)
    vector3D = (-vectorR[0], -vectorL[0], (vectorL[1] + vectorR[1]) / 2)
    nib = (position3D[0] + vector3D[0] * penLength,
           position3D[1] + vector3D[1] * penLength,
           position3D[2] + vector3D[2] * penLength)
    return nib


def map(value, istart, istop, ostart, ostop):
    return np.clip(int(ostart + (ostop - ostart) * ((value - istart) / (istop - istart))), ostart, ostop)


greenLower = (29, 35, 25)
greenUpper = (64, 255, 255)
resize = 2
penLength = 830 // resize

if __name__ == '__main__':
    # read video
    capL = cv2.VideoCapture("../video/Trimmed/Angle90_R.mov")
    capR = cv2.VideoCapture("../video/Trimmed/Angle90_L.mov")
    retL, frameL = capL.read()
    retR, frameR = capR.read()
    frame_num = int(capL.get(cv2.CAP_PROP_FRAME_COUNT)) - 1
    frame_H = frameL.shape[0] // resize
    frame_W = frameL.shape[1] // resize
    if (
            frame_num != int(capR.get(cv2.CAP_PROP_FRAME_COUNT)) - 1
            or frame_H != frameR.shape[0]
            or frame_W != frameR.shape[1]):
        print("Warning: Inconsistent of stereo frames")
    frame_I = 0
    greenRange = (greenLower, greenUpper)
    result = np.zeros((frame_H, frame_W, 3), np.uint8)
    run = False
    offset = np.array([0, 0, 0])
    z_threshold = 5

    while True:
        frameL, positionL, vectorL = calculateFrame(capL, frame_I, greenRange)
        frameR, positionR, vectorR = calculateFrame(capR, frame_I, greenRange)
        nib = reconstruct_90(result, positionL, positionR, vectorL, vectorR, penLength)

        if -z_threshold < (nib - offset)[2] < z_threshold:
            result[
                (int((nib[0] - offset[0]) * 8) + 100) % frame_H, (
                        int((nib[1] - offset[1]) * 8) + 100) % frame_W] = map(
                abs((nib - offset)[2]), 0, z_threshold, 255, 0)

        print(
            "{}/{} {} {} {}, {}".format(frame_I, frame_num, nib[0] - offset[0], nib[1] - offset[1], nib[2] - offset[2]))
        cv2.imshow("Frame Left", frameL)
        cv2.imshow("Frame Right", frameR)
        cv2.imshow("Result", result)

        if run and frame_I < frame_num:
            key = cv2.waitKey(1) & 0xFF
            frame_I += 1
        else:
            run = False
            key = cv2.waitKey(0)

        if (key == ord('q')):
            cv2.destroyAllWindows()
            break
        elif (key == ord('r')):
            run = not run
        elif (key == ord('d')):
            frame_I = min(frame_I + 1, frame_num - 1)
        elif (key == ord('s')):
            frame_I = max(frame_I - 1, 0)
        elif (key == ord('f')):
            frame_I = min(frame_I + 100, frame_num - 1)
        elif (key == ord('a')):
            frame_I = max(frame_I - 100, 0)
        elif (key == ord('k')):
            offset = np.array(nib)
        elif (key == ord('c')):
            result = np.zeros((frame_H, frame_W, 3), np.uint8)
