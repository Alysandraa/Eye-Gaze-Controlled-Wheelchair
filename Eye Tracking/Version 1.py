import os, sys, inspect
import cv2
import numpy as np
import math
import serial
import time


def frame_prep(frame):
    #put frame in black and white
    bw = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #blur the black and white frame
    blur = cv2.GaussianBlur(bw, (9, 9), 0)
    #detect edges on the product of the previous 2 steps
    edges = cv2.Canny(blur, 50, 150)
    cv2.imshow("Edges",edges)
    return edges

cam = cv2.VideoCapture(0)
cam.set(cv2.CAP_PROP_FRAME_WIDTH,320)
cam.set(cv2.CAP_PROP_FRAME_HEIGHT,240)

ser = serial.Serial('/dev/ttyACM1', 9600, timeout=1)

while True:
    ret, frame = cam.read()
    frame = cv2.flip(frame,-1)
    height, width, _ = frame.shape

    key = cv2.waitKey(1)        
    if key != -1:
        break

cam.release()
cv2.destroyAllWindows()
