import dlib
import cv2
import time
import matplotlib.pyplot as plt
from deepface import DeepFace 
import numpy
from PIL import Image
import os
min  = 0.4
person = "person (1).jpg"
hog_face_detector = dlib.get_frontal_face_detector()

cap = cv2.VideoCapture("Zypl_new3.mp4")
number =1
success = True
#img = cv2.imread("/Users/macbook/Documents/people/Jeff Bezos/two.jpeg")
temp = "temporary.jpg"
while True:
    _,frame = cap.read()
    cv2.imwrite(temp,frame)
    imag = Image.open(temp)
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    print("THIS IS FOR EACH FRAME")
    results = hog_face_detector(gray,1)
    for bbox in results:
        
        print("THIS IS FOR EACH BOX\n")
        x1 = bbox.left()
        y1 = bbox.top()
        x2 = bbox.right()
        y2 = bbox.bottom()
        rect = (x1,y1,x2,y2)
        imag.crop(rect)
        imag.save(temp)
        #for filename in os.listdir("faces"):
        while True:
                filename="faces/person ("+str(number)+").jpg"
                result = DeepFace.verify(filename,temp,"Facenet")
                print(result)
                if result["verified"]==True:
                    person = str(number)
                    success=True
                    break
                    # if result["distance"]<min:
                    #     min = result["distance"]
                    #     person = filename

                number +=1
                if number==33:
                     success=False
                     break
                
        if success==False:
            person = "UNRECOGNIZED"
        cv2.rectangle(frame, pt1=(x1, y1), pt2=(x2, y2), color=(0, 255, 0), thickness=2)  
        cv2.putText(frame,person,(x1,y1-20),cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)
        cv2.imshow("image",frame)
        number = 1
    k = cv2.waitKey(1)
    if k%256==27:
        break
cap.release()
cv2.destroyAllWindows()