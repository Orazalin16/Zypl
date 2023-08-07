import mediapipe as mp
import cv2 as cv
import numpy as np


mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils
mp_face = mp.solutions.face_detection.FaceDetection(
    model_selection=1, 
    min_detection_confidence=0.7 # confidence threshold
)
video_file_3min = "/Users/macbook/Documents/Zypl_short.mp4"
cap = cv.VideoCapture(video_file_3min)
counter =0 
f=1
while True:
        sucess,frame = cap.read();

        frame = cv.cvtColor(frame,cv.COLOR_RGB2BGR)
        results = mp_face.process(frame)
       
        
        frame = cv.cvtColor(frame,cv.COLOR_RGB2BGR)
        
        if results.detections:
            for detection in results.detections:
                mp_drawing.draw_detection(frame,detection)
                counter +=1
        cv.imshow("Test",frame)
        k = cv.waitKey(1)
        if k%256==27:
            break
print("number of detections: {} ".format(counter))
cap.release()
cv.destroyAllWindows()