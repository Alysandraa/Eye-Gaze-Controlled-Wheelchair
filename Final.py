
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
    return frame

def find_eyes(cascade, gray, frame):
    #print("entering loop")
    eyes = cascade.detectMultiScale(gray)
    eye_centre = []
    #eye_centre.clear()
    for (x, y, w, h) in eyes:
        #print("drawing rectangles")
        x2 = x + w
        y2 = y + h
        cx = (x+x2)/2
        cy = (y+y2)/2
        cv2.rectangle(frame, (x, y), (x2, y2), (0, 255, 0), 2)
        eye_pts = [cx, cy, x, x2]
        #print(eye_pts)
        #eye_centre = []
        eye_centre.extend(eye_pts)
        #print("eye info:", eye_centre)

    if len(eye_centre) >= 8: #should this be more than/equal to?
        eyeA = [eye_centre[0], eye_centre[1], eye_centre[2], eye_centre[3]] #[centre x, centre y, start x, end x]
        eyeB = [eye_centre[4], eye_centre[5], eye_centre[6], eye_centre[7]]
        #eyeA[1] = 256 - eyeA[1]
        #eyeB[1] = 256 - eyeB[1]
        #print("A:", eyeA)
        #print("B:", eyeB)
        if ((int(eyeB[0]))-(int(eyeA[0]))) >= 0:
            right_eye = eyeA
            left_eye = eyeB
            one_eye = 0
        else:
            right_eye = eyeB
            left_eye = eyeA
            one_eye = 0
        #print("right eye:", right_eye, "left eye:", left_eye)
        #eyeA.clear()
        #eyeB.clear()
        #eye_pts.clear()
    elif len(eye_centre) == 4:
        eyeX_pts = [eye_centre[0], eye_centre[1], eye_centre[2], eye_centre[3]]
        #eyeX_pts[1] = 256 - eyeX_pts[1]
        eyeX = []
        eyeX.append(eyeX_pts)
        left_eye = 0
        right_eye = 0
        one_eye = eyeX
        print("one eye:", eyeX)
        #eyeX_pts.clear()
        #eyeX.clear()
        #eye_pts.clear()
    else:
        left_eye = 0
        right_eye = 0
        one_eye = 0
    #eye_centre.clear()
    print("right eye:", right_eye, "left eye:", left_eye)
    return frame, left_eye, right_eye, one_eye

def find_pupils(img, gray, threshold, detector):
    _, img = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
    img = cv2.erode(img, None, iterations=2)#idea is exactly like soil erosion. erodes away boundaries of foreground object
    img = cv2.dilate(img, None, iterations=4)#commonly used with erosion (for noise removal in images) because erode also shrinks the object so this makes it larger again 
    img = cv2.medianBlur(img, 5)#takes the median values of each pixel and replaces the central element with this value
    keypoints = detector.detect(img)
    
    if len(keypoints) == 2:
       ptsA = []
       ptsA.extend(np.asarray(keypoints[0].pt))
       #print(ptsA)
       #ptsA[1] = 256 - ptsA[1]
       ptsB = []
       ptsB.extend(np.asarray(keypoints[1].pt))
       #ptsB[1] = 256 - ptsB[1]
       #print(ptsB)
       if ((int(ptsB[0]))-(int(ptsA[0]))) >= 0:
            rightP = ptsA
            leftP = ptsB
            oneP = 0 
       else:
            rightP = ptsB
            leftP = ptsA
    elif len(keypoints) == 1:
       x = []
       x.extend(np.asarray(keypoints[0].pt))
       #x[1] = 256 - (int(x[1]))
       print("one pupil:", x)
       leftP = 0
       rightP = 0
       oneP = x
    else:
       oneP = 0
       leftP = 0
       rightP = 0
    #print("left pupil:", leftP)
    #print("right pupil:", rightP)
    return keypoints, leftP, rightP, oneP
    
def nothing(x):
    pass

def position(leftP, rightP, oneP, left_eye, right_eye, one_eye, blank):
    rAngle = 0
    lAngle = 0
    angle = 0
    #print("left pupil:", leftP)
    #print("right pupil:", rightP)
    #print("right eye:", right_eye, "left eye:", left_eye)
    ser = serial.Serial('/dev/ttyACM1', 9600, timeout=1)
    pX_coords = [leftP[0], rightP[0], oneP[0]]
    i = 0
    #print("new list:", pX_coords)
    x = (int(pX_coords[0])), (int(pX_coords[1])), (int(pX_coords[2]))
    #angles = []
    #print("sum:", (sum(x)))
    if sum(x) != 0:
        #if oneP == 0:
            #check if left pupil is in left eye and the same for right
            #do calculations for position
            #would it be better to put the x values for each set of pupil coordinates into a sperate list here and then just run through a loop that checks them because then it can check for one eye at the same time
            #and the 0 wouldnt be a problem because of the way the axis works on the image so 0 will never be ebtween both points unless both points are 0f
            #if (left_eye[2] < leftP[0]) and (leftP[0] < left_eye[3]):
            #    left = [left_eye[0], left_eye[1], leftP[0], leftP[1]]
            #    print("left:", left)
            #elif (right_eye[2] < leftP[0]) and (leftP[0] < right_eye[3]):
        while i < len(pX_coords):
            if (left_eye[2] < x[i]) and (x[i] < left_eye[3]):
                if i == 2:
                    left = [one_eye[0], one_eye[1], oneP[0], oneP[1]]
                else:
                    left = [left_eye[0], left_eye[1], leftP[0], leftP[1]]
                #print("left:", left)
                (lex, ley) = left[0], left[1]
                cv2.circle(blank, (int(lex), int(ley)), 1, (0, 255, 0), 1)
                (lpx, lpy) = left[2], left[3]
                cv2.circle(blank, (int(lpx), int(lpy)), 1, (0, 0, 255), 1)
                lpy_flipped = 256 - lpy
                ley_flipped = 256 - ley
                lDiff = ((ley_flipped - lpy_flipped), (lex - lpx))
                #print(lDiff)
                lRadian = math.atan2(lDiff[0], lDiff[1])
                lAngle = lRadian*180/math.pi
                if lAngle < 0:
                    lAngle = int(lAngle) + 360
            elif (right_eye[2] < x[i]) and (x[i] < right_eye[3]):
                if i == 2:
                    right = [one_eye[0], one_eye[1], oneP[0], oneP[1]]
                else:
                    right = [right_eye[0], right_eye[1], rightP[0], rightP[1]]
                    #need to set left to 0??
                #print("right:", right)
                (rex, rey) = right[0], right[1]
                cv2.circle(blank, (int(rex), int(rey)), 1, (0, 255, 0), 1)
                (rpx, rpy) = right[2], right[3]
                cv2.circle(blank, (int(rpx), int(rpy)), 1, (0, 0, 255), 1)
                rpy_flipped = 256 - rpy
                rey_flipped = 256 - rey
                rDiff = ((rey_flipped - rpy_flipped), (rex - rpx))
                #print(rDiff)
                rRadian = math.atan2(rDiff[0], rDiff[1])
                rAngle = rRadian*180/math.pi
                if rAngle < 0:
                    rAngle = int(rAngle) + 360
            if i > 2:
                i = 0
            else:
                i = i + 1
            #angles.extend(str(lAngle))
            #angles.extend(str(rAngle))
            #print(diff)
            #radian = math.atan2(diff[0], diff[1])
            #angle = radian*180/math.pi
            #if angle < 0:
            #    angle = int(angle) + 360
            print(rAngle)
            print(lAngle)
            #if rAngle != 0 and lAngle != 0:
             #   print("entering loop")
              #  angle = int((rAngle + lAngle)/2)
               # print("angle1:", angle)
            #need to make sure this is the right space to put threshold for i
            angles = (rAngle, lAngle)
            print(angles)
            if ((lAngle *  rAngle) == 0):
                angle = sum(angles)
            else:
                angle = sum(angles)/len(angles)
            ser.write(int(angle))
            print("angle:", angle)
            line = ser.readline().decode().rstrip()
            print(line)
    else:
        left = 0
        right = 0
        angle = 0
        print("no pupils found")
    ser.close()
    return angle


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
                threshold = 52
                keypoints, leftP, rightP, oneP = find_pupils(eyes, gray, threshold, detector)#had to change first arguement here and second and third arguements below to eyes instead of eye
                eye = cv2.drawKeypoints(eyes, keypoints, eyes, (0, 0, 255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
                if keypoints != ():
                    #print("entering loop")
                    blank = cv2.imread("face.jpg") 
                    angle = position(leftP, rightP, oneP, left_eye, right_eye, one_eye, blank)
                  #  pupil = cv2.drawKeypoints(blank, keypoints, blank, (255, 255, 255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
                    cv2.imshow('coordinates', blank)
    cv2.imshow('Eye Tracking', frame)
    #print("showing image")            
    if cv2.waitKey(1) & 0xFF == ord('q'):
            break
#cap.release()
cv2.destroyAllWindows()
