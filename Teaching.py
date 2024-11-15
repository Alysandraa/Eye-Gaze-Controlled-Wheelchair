import numpy as np
import matplotlib.pyplot as plt
import math
import serial
import time

face_data = ('haarcascade_frontalface_default.xml')
eye_data = ('haarcascade_eye.xml')
face_cascade = cv2.CascadeClassifier(face_data)
eye_cascade = cv2.CascadeClassifier(eye_data)

detector_params = cv2.SimpleBlobDetector_Params()
detector_params.filterByArea = True
detector_params.maxArea = 1500
detector = cv2.SimpleBlobDetector_create(detector_params)

def find_face(cascade, gray, img):
    coords = cascade.detectMultiScale(gray, 1.3, 5)
    if len(coords) > 1:
        biggest = (0, 0, 0, 0)
        for i in coords:
            if i[3] > biggest[3]:
                biggest = i
        biggest = np.array([i], np.int32)
    elif len(coords) == 1:
        biggest = coords
    else:
        return None
    for (x, y, w, h) in biggest:
       cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
       frame = img[y:y + h, x:x + w]
    return frame

def find_eyes(cascade, gray, frame):
    eyes = cascade.detectMultiScale(gray)
    for (x, y, w, h) in eyes:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
    return frame

def find_pupils(img, gray, threshold, detector):
    _, img = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
    img = cv2.erode(img, None, iterations=2)#idea is exactly like soil erosion. erodes away boundaries of foreground object
    img = cv2.dilate(img, None, iterations=4)#commonly used with erosion (for noise removal in images) because erode also shrinks the object so this makes it larger again 
    img = cv2.medianBlur(img, 5)#takes the median values of each pixel and replaces the central element with this value
    keypoints = detector.detect(img)
    return keypoints
    
def nothing(x):
    pass


cv2.namedWindow('Eye Tracking')
cv2.createTrackbar('threshold', 'Eye Tracking', 0, 255, nothing)
while True:
    frame  = cv2.imread("face.jpg") 
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = find_face(face_cascade, gray, frame)
    if faces is not None:
        eyes, left_eye, right_eye, one_eye = find_eyes(eye_cascade, gray, frame)
        for eye in eyes:
            if eye is not None:
                threshold = 52
                keypoints = find_pupils(eyes, gray, threshold, detector)
                if keypoints != ():
                    blank = cv2.imread("face.jpg") 
                    eye = cv2.drawKeypoints(eyes, keypoints, eyes, (0, 0, 255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
                    cv2.imshow('coordinates', blank)
    cv2.imshow('Eye Tracking', frame) 
    if cv2.waitKey(1) & 0xFF == ord('q'):
            break
cv2.destroyAllWindows()
