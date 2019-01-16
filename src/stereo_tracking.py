# -*- coding: utf-8 -*-
import cv2
import numpy as np
from math import atan2, cos, sin, sqrt, pi, degrees


class PenTracking:
    def getOrientation(self, pts, img):
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

    def calculateFrame(self, cap, frame_I, greenRange):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_I)
        ret, frame = cap.read()
        frame = cv2.resize(frame, (self.frame_W, self.frame_H))
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        green_mask = cv2.inRange(hsv, greenRange[0], greenRange[1])
        green_mask = cv2.erode(green_mask, None, iterations=2)
        green_mask = cv2.dilate(green_mask, None, iterations=2)
        c = np.where(green_mask == 255)
        position, vector = self.getOrientation(c, frame)
        return frame, position, vector

    def reconstruct_90(self, result, positionL, positionR, vectorL, vectorR, penLength):
        position3D = (-positionR[0], -positionL[0], (positionL[1] + positionR[1]) / 2)
        vector3D = (-vectorR[0], -vectorL[0], (vectorL[1] + vectorR[1]) / 2)
        nib = (position3D[0] + vector3D[0] * penLength,
               position3D[1] + vector3D[1] * penLength,
               position3D[2] + vector3D[2] * penLength)
        return nib

    def map(value, istart, istop, ostart, ostop):
        return np.clip(int(ostart + (ostop - ostart) * ((value - istart) / (istop - istart))), ostart, ostop)

    # mouse callback function
    def mouse_event(self, event, x, y, flags, param):
        if event == cv2.EVENT_MOUSEMOVE:
            if flags == cv2.EVENT_FLAG_LBUTTON:
                self.threshold[x] = y
                self.canvas = [[0 for i in range(3)] for j in range(self.frame_num)]

        if event == cv2.EVENT_LBUTTONUP:
            self.draw()

    def refresh(self):
        cv2.imshow("Frame Left", self.frameL)
        cv2.imshow("Frame Right", self.frameR)
        cv2.imshow("Result", self.result)
        cv2.imshow("Z", self.z_img)

    def draw(self):
        self.frameL, self.positionL, self.vectorL = self.calculateFrame(self.capL, self.frame_I, self.greenRange)
        self.frameR, self.positionR, self.vectorR = self.calculateFrame(self.capR, self.frame_I, self.greenRange)
        self.nib = self.reconstruct_90(self.result, self.positionL, self.positionR, self.vectorL, self.vectorR,
                                  self.penLength)

        self.z_value[self.frame_I] = int((self.nib - self.offset)[2] * 5 + 150) % 300
        self.y_value[self.frame_I] = (int((self.nib[0] - self.offset[0]) * 8) + 100) % self.frame_H
        self.x_value[self.frame_I] = (int((self.nib[1] - self.offset[1]) * 8) + 100) % self.frame_W
        # if -z_threshold < (nib - offset)[2] < z_threshold:
        self.result = np.zeros((self.frame_H, self.frame_W, 3), np.uint8)
        for i in range(self.frame_num):
            if self.z_value[i] > self.threshold[i]:
                y = self.y_value[i]
                x = self.x_value[i]
                # result[y, x] = map(abs((nib - offset)[2]), 0, z_threshold, 255, 0)
                self.canvas[i] = [x, y, 255]
            x, y, value = self.canvas[i]
            self.result[y, x] = value
            self.z_img = np.zeros((300, self.frame_num, 3), np.uint8)
        for x, y in enumerate(self.threshold):
            self.z_img[self.z_value[x], x] = 255
            self.z_img[self.threshold[x], x] = (0, 0, 255)
        # print(
        #     "{}/{} {} {} {}".format(frame_I, frame_num, nib[0] - offset[0], nib[1] - offset[1], nib[2] - offset[2]))
        self.refresh()

    greenLower = (29, 35, 25)
    greenUpper = (64, 255, 255)
    resize = 2
    penLength = 830 // resize

    def main(self):
        # read video
        self.capL = cv2.VideoCapture("../video/Trimmed/Angle90_R.mov")
        self.capR = cv2.VideoCapture("../video/Trimmed/Angle90_L.mov")
        self.retL, self.frameL = self.capL.read()
        self.retR, self.frameR = self.capR.read()
        self.frame_num = int(self.capL.get(cv2.CAP_PROP_FRAME_COUNT)) - 1
        self.frame_H = self.frameL.shape[0] // self.resize
        self.frame_W = self.frameL.shape[1] // self.resize
        if (
                self.frame_num != int(self.capR.get(cv2.CAP_PROP_FRAME_COUNT)) - 1
                or self.frame_H != self.frameR.shape[0]
                or self.frame_W != self.frameR.shape[1]):
            print("Warning: Inconsistent of stereo frames")
        self.frame_I = 0
        self.greenRange = (self.greenLower, self.greenUpper)
        self.result = np.zeros((self.frame_H, self.frame_W, 3), np.uint8)
        self.z_img = np.zeros((300, self.frame_num, 3), np.uint8)
        run = False
        self.offset = np.array([0, 0, 0])
        self.z_threshold = 5
        self.z_value = [0] * self.frame_num
        self.y_value = [0] * self.frame_num
        self.x_value = [0] * self.frame_num
        self.threshold = [0] * self.frame_num
        cv2.namedWindow("Z", cv2.WINDOW_NORMAL)
        cv2.setMouseCallback("Z", self.mouse_event)
        self.canvas = [[0 for i in range(3)] for j in range(self.frame_num)]
        while True:
            self.draw()

            if run and self.frame_I < self.frame_num:
                key = cv2.waitKey(1) & 0xFF
                self.frame_I += 1
            else:
                run = False
                key = cv2.waitKey(0)

            if (key == ord('q')):
                cv2.destroyAllWindows()
                break
            elif (key == ord('r')):
                run = not run
            elif (key == ord('d')):
                self.frame_I = min(self.frame_I + 1, self.frame_num - 1)
            elif (key == ord('s')):
                self.frame_I = max(self.frame_I - 1, 0)
            elif (key == ord('f')):
                self.frame_I = min(self.frame_I + 100, self.frame_num - 1)
            elif (key == ord('a')):
                self.frame_I = max(self.frame_I - 100, 0)
            elif (key == ord('k')):
                self.offset = np.array(self.nib)
                self.z_value = [0] * self.frame_num
            elif (key == ord('z')):
                self.z_value = [0] * self.frame_num
            elif (key == ord('c')):
                self.canvas = [[0 for i in range(3)] for j in range(self.frame_num)]


if __name__ == '__main__':
    pt = PenTracking()
    pt.main()
