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
def find_face(cascade, gray, img):
    coords = cascade.detectMultiScale(gray, 1.3, 5)
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
       #eyes_frame = find_eyes(eye_cascade, gray, frame)
    return frame #, eye_frame

def find_eyes(cascade, gray, frame):
    eyes = cascade.detectMultiScale(gray)
    eye_info = []
    for (x, y, w, h) in eyes:
        x2 = x+w
        y2 = y+h
        cv2.rectangle(frame, (x, y), (x2, y2), (0, 255, 0), 2)
        eye_info = (((x+x2)/2), ((y+y2)/2), x, x2)
    return frame, eye_info

def find_pupils(img, gray, threshold, detector):
    _, img = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
    img = cv2.erode(img, None, iterations=2)#idea is exactly like soil erosion. erodes away boundaries of foreground object
    img = cv2.dilate(img, None, iterations=4)#commonly used with erosion (for noise removal in images) because erode also shrinks the object so this makes it larger again 
    img = cv2.medianBlur(img, 5)#takes the median values of each pixel and replaces the central element with this value
    keypoints = detector.detect(img)#blob detection look at example on laptop
    if len(keypoints) == 2: #should this be more than/equal to?
      ptsA = []
      ptsA.extend(np.asarray(keypoints[0]))
      print(ptsA)
      ptsB = []
      ptsB.extend(np.asarray(keypoints[1]))
      print(ptsB)
      if ((int(ptsB[0])) - (int(ptsA[0]))) <= 0:
        left = ptsA
        right = ptsB
        one = 0
      else:
        left = ptsB
        right = ptsA
        one = 0
    elif len(keypoints) == 1:
      left = 0
      right = 0
      one = ptsA
    else:
      left = 0
      right = 0
      one = 0
      
    print(keypoints)
    return keypoints, left, right, one

def nothing(x):
    pass

def postion(left, right, one, eye_info):
    if sum(left, right, one) != 0:
        #do i use one or both eyes?
        if one == 0:
            
            #check the left pupil is inside left eye and right pupil is inside right eye
            #then find the difference of center coordinates to get osition
            #then average???? the differences for each eye to get a more reliable result?
        else:
            if((one_eye[2]) <= (one[0]) <= (one_eye[3])):
                #then blah
             if((left_eye[2]) <= (one[0]) <= (left_eye[3])):
                 #then blah
             if((right_eye[2]) <= (one[0]) <= (right_eye[3])): # or do i need to write as ((one[0] >= (right_eye[2])) && (one[0] <= (right_eye[3]))):
                 #then blah
    else:
        position = 0
        print ("no keypoints")
    return position
  
cv2.namedWindow('Eye Tracking')
cv2.createTrackbar('threshold', 'Eye Tracking', 0, 255, nothing)
while True:
    #ret, frame = cap.read() 
    frame  = cv2.imread("face.jpg") 
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = find_face(face_cascade, gray, frame)
    if faces is not None:
        eyes, left_eye, right_eye, one_eye = find_eyes(eye_cascade, gray, frame)
        for eye in eyes:
            if eye is not None:
                threshold = cv2.getTrackbarPos('threshold', 'Eye Tracking')
                keypoints, left, right, one = find_pupils(eyes, gray, threshold, detector)
                eye = cv2.drawKeypoints(eyes, keypoints, eyes, (0, 0, 255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
                if keypoints != ():
                    print("entering loop")
                    blank = cv2.imread("face.jpg") 
                    pupil = cv2.drawKeypoints(blank, keypoints, blank, (255, 255, 255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
                    cv2.imshow('keypoints', blank)
    cv2.imshow('Eye Tracking', frame)      
    if cv2.waitKey(1) & 0xFF == ord('q'):
            break
#cap.release()
cv2.destroyAllWindows()
