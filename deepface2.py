from deepface import DeepFace
import cv2 
import numpy as np


models = ["opencv","ssd","dlib","mtcnn","retinaface","mediapipe"]
video_file_3min = "/Users/macbook/Documents/Zypl_short.mp4"
cap = cv2.VideoCapture(video_file_3min)
checker = 1

counter =0 
while True:
        sucess,frame = cap.read();

        frame = cv2.cvtColor(frame,cv2.COLOR_RGB2BGR)
        cv2.imwrite("frame.jpg",frame)
        try:
            #face = DeepFace.extract_faces("frame.jpg",detector_backend=models[0])
            
            face = DeepFace.stream("face-db/",enable_face_analysis=False,detector_backend=models[0],source="/Users/macbook/Documents/Zypl_short.mp4")
            print(face)
        except:
            print("no faces")
            checker =0;
        if checker == 1:

            if face:
                for detection in face:
        
                    x,y,w,h = detection['facial_area'].values()

                    cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),3)
                
                    
                    counter +=1
        cv2.imshow("Test",frame)
        k = cv2.waitKey(1)
        if k%256==27:
            break
print("number of detections: {} ".format(counter))
cap.release()
cv2.destroyAllWindows()