import os
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import tensorflow as tf
import cv2
import mediapipe as mp
from keras.models import load_model
import numpy as np
import time
from PIL import Image
import pandas as pd


np.set_printoptions(suppress=True)
mphands = mp.solutions.hands
hands = mphands.Hands()
mp_drawing = mp.solutions.drawing_utils


model = load_model("keras_Model.h5", compile=False)
rect = (0,0,0,0)
mphands = mp.solutions.hands
hands = mphands.Hands()
mp_drawing = mp.solutions.drawing_utils
cap = cv2.VideoCapture(0)

_, frame = cap.read()

h, w, c = frame.shape

x_max = 0
y_max = 0
x_min = w
y_min = h
img_counter = 0
analysisframe = ''
letterpred = open("labels.txt", "r").readlines()

while True:

    _, frame2 = cap.read()
    img = Image.fromarray(frame2)
    analysisframe = frame
    showframe = analysisframe
    
    k = cv2.waitKey(1)
    if k%256 == 27:
        # ESC pressed
        print("Escape hit, closing...")
        break

    elif np.shape(img)!=(0,0):#k%256 == 32:
        frame = cv2.resize(frame2, (224, 224), interpolation=cv2.INTER_AREA)
            
    # Make the image a numpy array and reshape it to the models input shape.
        frame = np.asarray(frame, dtype=np.float32).reshape(1, 224, 224, 3)

    # Normalize the image array
        frame = (frame / 127.5) - 1

        
        #cv2.imshow("Frame", frame2)
        framergbanalysis = cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB)
        resultanalysis = hands.process(framergbanalysis)
        hand_landmarksanalysis = resultanalysis.multi_hand_landmarks
        if hand_landmarksanalysis:
            for handLMsanalysis in hand_landmarksanalysis:
                
                x_max = 0
                y_max = 0
                x_min = w
                y_min = h
                for lmanalysis in handLMsanalysis.landmark:
                    x, y = int(lmanalysis.x * w), int(lmanalysis.y * h)
                    if x > x_max:
                        x_max = x
                    if x < x_min:
                        x_min = x
                    if y > y_max:
                        y_max = y
                    if y < y_min:
                        y_min = y
                y_min -= 20
                y_max += 20
                x_min -= 20
                x_max += 20 
                rect = (x_min-150, y_min-150, x_max+150, y_max+150)

        img=img.crop(rect)  

        if int(np.size(img,1))!=0 and int(np.size(img,0))!=0:
                img = cv2.resize(np.asarray(img),(224,224),3)
                
                img = np.asarray(img, dtype=np.float32).reshape(1, 224, 224, 3)
                img = (img / 127.5) - 1  

                

                prediction = model.predict(img)
                index = np.argmax(prediction)
                class_name = letterpred[index]
                confidence_score = prediction[0][index]

                print("Class:", class_name[2:], end="")
                print("Confidence Score:", str(np.round(confidence_score * 100))[:-2], "%")
                
        else:
            pass

        time.sleep(0)

    
    framergb = cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB)
    result = hands.process(framergb)
    hand_landmarks = result.multi_hand_landmarks
    if hand_landmarks:
        
        for handLMs in hand_landmarks:
            x_max = 0
            y_max = 0
            x_min = w
            y_min = h
            for lm in handLMs.landmark:
                x, y = int(lm.x * w), int(lm.y * h)
                if x > x_max:
                    x_max = x
                if x < x_min:
                    x_min = x
                if y > y_max:
                    y_max = y
                if y < y_min:
                    y_min = y
            y_min -= 20
            y_max += 20
            x_min -= 20
            x_max += 20
            cv2.rectangle(frame2, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
            
    cv2.imshow("Frame", frame2)

cap.release()
cv2.destroyAllWindows()


