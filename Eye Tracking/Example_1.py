import cv2
import numpy as np
import matplotlib.pyplot as plt

face_data = ('haarcascade_frontalface_default.xml')
eye_data = ('haarcascade_eye.xml')
face_cascade = cv2.CascadeClassifier(face_data)
eye_cascade = cv2.CascadeClassifier(eye_data)

#cap = cv2.VideoCapture(0)

while True:
    frame  = cv2.imread("face.jpg")

    #ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray)
        cv2.rectangle(roi_color, (x, y), (x+w, y+h), (255, 0, 0), 2)
        print("found face")
        for (ex, ey, ew, eh) in eyes:
            #print("entering 1st loop")
            cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)
            
            eye_roi_gray = roi_gray[ey:ey+eh, ex:ex+ew]
            eye_roi_color = roi_color[ey:ey+eh, ex:ex+ew]
            _, eye_thresh = cv2.threshold(eye_roi_gray, 50, 255, cv2.THRESH_BINARY)
            contours, _ = cv2.findContours(eye_thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            #cv2.drawContours(frame, contours, -1, (0, 255, 0), -1)
            print("found eyes")
            print(contours)
            if len(contours) > 0:
                #print("entering 2nd loop")
                pupil = max(contours, key = cv2.contourArea)
                x1, y1, w1, h1 = cv2.boundingRect(pupil)
                #cv2.rectangle(frame, (x1, y1), (x1+w1, y1+h1), (0, 255, 0), 2)
                center = (int(x1 + w1/2), int(y1 + h1/2))
                cv2.circle(frame, center, 3, (0, 0, 255), -1)
                print("found pupils")
    cv2.imshow('Eye Tracking', frame)
    #print("showing image")            
    key = cv2.waitKey(1)
    if key != -1:
        break

#cap.release()
#cv2.destroyAllWindows()
