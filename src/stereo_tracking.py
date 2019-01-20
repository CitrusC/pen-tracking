# -*- coding: utf-8 -*-
import cv2
import numpy as np
from math import sqrt
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt



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
    cv2.imshow("Result", green_mask)
    return frame, position, vector


def reconstruct_90(positionL, positionR, vectorL, vectorR, penLength):
    position3D = np.array([-positionR[0], -positionL[0], (positionL[1] + positionR[1]) / 2])
    vector3D = np.array([-vectorR[0], -vectorL[0], (vectorL[1] + vectorR[1]) / 2])
    nib = position3D + vector3D * penLength
    return nib, position3D


def map(value, istart, istop, ostart, ostop):
    return np.clip(int(ostart + (ostop - ostart) * ((value - istart) / (istop - istart))), ostart, ostop)


greenLower = (29, 35, 25)
greenUpper = (64, 255, 255)
resize = 2
penLength = 830 // resize
outputVideo = False

if __name__ == '__main__':
    # read video
    capL = cv2.VideoCapture("../video/Trimmed/Angle90_R.mov")
    capR = cv2.VideoCapture("../video/Trimmed/Angle90_L.mov")
    retL, frameL = capL.read()
    retR, frameR = capR.read()
    frame_num = int(capL.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_H = frameL.shape[0]
    frame_W = frameL.shape[1]
    if (
            frame_num != int(capR.get(cv2.CAP_PROP_FRAME_COUNT))
            or frame_H != frameR.shape[0]
            or frame_W != frameR.shape[1]):
        print("Warning: Inconsistent of stereo frames")
    frame_H = frame_H // resize
    frame_W = frame_W // resize
    frame_I = 0
    greenRange = (greenLower, greenUpper)
    nib_list = np.zeros((frame_num, 3))
    result = np.zeros((frame_H, frame_W, 3), np.uint8)
    z_img = np.zeros((640, frame_num, 3), np.uint8)
    mpl.rcParams['legend.fontsize'] = 10
    run = False
    is_finish = False

    mpl.rcParams['legend.fontsize'] = 10
    fig = plt.figure()
    ax = fig.gca(projection='3d')

    if outputVideo:
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        outL = cv2.VideoWriter('../video/output_orig_L.avi', fourcc, 20.0, (frame_W, frame_H))
        outR = cv2.VideoWriter('../video/output_orig_R.avi', fourcc, 20.0, (frame_W, frame_H))


    while True:
        frameL, positionL, vectorL = calculateFrame(capL, frame_I, greenRange)
        frameR, positionR, vectorR = calculateFrame(capR, frame_I, greenRange)
        nib_list[frame_I], position = reconstruct_90(positionL, positionR, vectorL, vectorR, penLength)

        z_img[int(nib_list[frame_I, 2] * 8) % 640, frame_I] = (255, 255, 255)

        print(
            "{}/{} {}".format(frame_I, frame_num, nib_list[frame_I]))

        line = np.array([nib_list[frame_I], position])
        ax.plot(line[:, 0], line[:, 1], line[:, 2], label='nib')
        ax.legend()
        plt.pause(.01)

        cv2.imshow("Frame Left", frameL)
        cv2.imshow("Frame Right", frameR)
        # cv2.imshow("Result", result)
        cv2.imshow("Z", z_img)

        if outputVideo:
            outL.write(frameL)
            outR.write(frameR)

        if run:
            if frame_I < frame_num - 1:
                key = cv2.waitKey(1) & 0xFF
                frame_I += 1
            else:
                run = False
                print("Finished pen tracking")
                np.save("nib_list.npy", nib_list)
                # analytic_signal = hilbert(nib_list[:, 2])
                # for i in range(frame_num):
                #     z_img[int(analytic_signal[i] * 8) % 640, i] = (0, 0, 255)
                # cv2.imshow("Z", z_img)
                key = cv2.waitKey(0)
        else:
            key = cv2.waitKey(0)

        if (key == ord('q')):
            capL.release()
            capR.release()
            if outputVideo:
                outL.release()
                outR.release()
            cv2.destroyAllWindows()
            break
        elif (key == ord('r')):
            run = not run
        elif (key == ord('d')):
            frame_I = min(frame_I + 1, frame_num - 1)
        elif (key == ord('s')):
            frame_I = max(frame_I - 1, 0)
        elif (key == ord('f')):
            frame_I = min(frame_I + 50, frame_num - 1)
        elif (key == ord('a')):
            frame_I = max(frame_I - 50, 0)
