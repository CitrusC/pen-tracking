# -*- coding: utf-8 -*-
# ex2.py
import numpy as np
import cv2

greenLower = (29, 86, 6)
greenUpper = (64, 255, 255)

#load image
frame = cv2.imread( "../img/001.png" )
frame = cv2.resize(frame, (1280, 720))

hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
green_mask = cv2.inRange(hsv, greenLower, greenUpper)
green_mask = cv2.erode(green_mask, None, iterations=2)
green_mask = cv2.dilate(green_mask, None, iterations=2)
# #draw rect and dots
# cv2.line     (img,(100,100),(300,200), (0,255,255),2)
# cv2.circle   (img,(100,100), 50,       (255,255,0),1)
# cv2.rectangle(img,(100,100),(200,200), (255,0,255),1)

#display img

cv2.imshow("show image", green_mask)
cv2.waitKey()