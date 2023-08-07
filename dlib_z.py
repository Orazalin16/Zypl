import dlib
import cv2
import time
import matplotlib.pyplot as plt

hog_face_detector = dlib.get_frontal_face_detector()

cap = cv2.VideoCapture("/Users/macbook/Documents/Zypl_short.mp4")

#img = cv2.imread("/Users/macbook/Documents/people/Jeff Bezos/two.jpeg")

while True:
    _,frame = cap.read()

    
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    results = hog_face_detector(gray,1)

    for bbox in results:

        x1 = bbox.left()
        y1 = bbox.top()
        x2 = bbox.right()
        y2 = bbox.bottom()

        cv2.rectangle(frame, pt1=(x1, y1), pt2=(x2, y2), color=(0, 255, 0), thickness=2)  

    cv2.imshow("frame",frame)
    k = cv2.waitKey(1)
    if k == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()