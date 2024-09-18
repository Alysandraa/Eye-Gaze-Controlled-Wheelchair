import cv2
import numpy as np
import matplotlib.pyplot as plt

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
detector_params = cv2.SimpleBlobDetector_Params()
detector_params.filterByArea = True
detector_params.maxArea = 1500
detector = cv2.SimpleBlobDetector_create(detector_params)

def find_face (img, cascade):
    gray_frame = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    coords = cascade.detectMultiScale(gray_frame, 1.3, 5)
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

def find_eyes(img, cascade, frame):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray)
        #cv2.rectangle(roi_color, (x, y), (x+w, y+h), (255, 0, 0), 2)
        print("found face")
        for (ex, ey, ew, eh) in eyes:
            #print("entering 1st loop")
            cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)
    eyes_img = img[y:y + h, x:x + w]
    return frame, eyes_img


def blob_process(img, threshold, detector):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, img = cv2.threshold(img, threshold, 255, cv2.THRESH_BINARY)
    img = cv2.erode(img, None, iterations=2)#idea is exactly like soil erosion. erodes away boundaries of foreground object
    img = cv2.dilate(img, None, iterations=4)#commonly used with erosion (for noise removal in images) because erode also shrinks the object so this makes it larger again 
    img = cv2.medianBlur(img, 5)#takes the median values of each pixel and replaces the central element with this value
    keypoints = detector.detect(img)#blob detection look at example on laptop
    print(keypoints)
    return keypoints

def nothing(x):#whats the point in doing this?
    pass

def main():
    #cap = cv2.VideoCapture(0)
    cv2.namedWindow('image')
    cv2.createTrackbar('threshold', 'image', 0, 255, nothing)
    while True:
        #_, frame = cap.read()
        frame  = cv2.imread("face.jpg")
        face_frame = find_face(frame, face_cascade)
        if face_frame is not None:
            eyes = find_eyes(face_frame, eye_cascade, frame)
            for eye in eyes:
                if eye is not None:
                    threshold = cv2.getTrackbarPos('threshold', 'image')
                    keypoints = blob_process(eye, threshold, detector)
                    eye = cv2.drawKeypoints(eye, keypoints, eye, (0, 0, 255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        cv2.imshow('image', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    #cap.release()
    #cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
