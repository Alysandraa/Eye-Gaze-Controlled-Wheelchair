#this one was mainly working but it was finding 2 sets of eyes and rerawing the shapes mutliple ties which then messed up the rerun of the program
#also wasn't drawing pupil contours

import cv2
import numpy as np
import matplotlib.pyplot as plt

face_data = ('haarcascade_frontalface_default.xml')
eye_data = ('haarcascade_eye.xml')
face_cascade = cv2.CascadeClassifier(face_data)
eye_cascade = cv2.CascadeClassifier(eye_data)
detector_params = cv2.SimpleBlobDetector_Params()
detector_params.filterByArea = True
detector_params.maxArea = 1500
detector = cv2.SimpleBlobDetector_create(detector_params)

#cap = cv2.VideoCapture(0)
def find_face(gray, img, frame):
    coords = (cv2.CascadeClassifier('haarcascade_frontalface_default.xml')).detectMultiScale(gray, 1.3, 5)
    #looking for the largest rectangle
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
       #this is the one that was drawing the wrong set of rectangles for the eyes
       eyes_frame = find_eyes(eye_cascade, gray, frame)
    return eyes_frame, frame

def find_eyes(cascade, gray, frame):
    eyes = cascade.detectMultiScale(gray)
    for (x, y, w, h) in eyes:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
    return frame, eyes

def find_pupils(frame, threshold, detector):
    _, frame = cv2.threshold(frame, threshold, 255, cv2.THRESH_BINARY)
    frame = cv2.erode(frame, None, iterations=2)#idea is exactly like soil erosion. erodes away boundaries of foreground object
    frame = cv2.dilate(frame, None, iterations=4)#commonly used with erosion (for noise removal in images) because erode also shrinks the object so this makes it larger again 
    frame = cv2.medianBlur(frame, 5)#takes the median values of each pixel and replaces the central element with this value
    keypoints = detector.detect(frame)#blob detection look at example on laptop
    print(keypoints)
    return keypoints

def nothing(x):
    pass

#theory about why it draws it 100 times is that it gets written over again and again and again cause first two functions just returns frame - i was right but it was because the line defiuning it was not also in the while ture loop
frame  = cv2.imread("face.jpg")

while True:
    #ret, frame = cap.read() 
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cv2.namedWindow('image')
    cv2.createTrackbar('threshold', 'image', 0, 255, nothing)
    faces = find_face(gray, frame, face_cascade)
    if faces is not None:
        eyes = eye_cascade.detectMultiScale(gray)
        for eye in eyes:
                eyes = find_eyes(eye_cascade, gray, frame)
                if eye is not None:
                    threshold = cv2.getTrackbarPos('threshold', 'image')
                   # eye = cut_eyebrows(eye)
                    keypoints = find_pupils(frame, threshold, detector)
                    eye = cv2.drawKeypoints(frame, keypoints, eye, (0, 0, 255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    cv2.imshow('Eye Tracking', frame)
    #print("showing image")            
    if cv2.waitKey(1) & 0xFF == ord('q'):
            break

#cap.release()
#cv2.destroyAllWindows()
