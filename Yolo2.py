# TechVidvan Vehicle counting and Classification

import cv2
import csv
import collections
import numpy as np
from tracker import *
import os
import re
from PIL import Image
import shutil
# Initialize Tracker
tracker = EuclideanDistTracker()

# Initialize the videocapture object
cap = cv2.VideoCapture('/Users/macbook/Downloads/video2.mp4')
input_size = 320
k:int = 0
# Detection confidence threshold
confThreshold =0.2
nmsThreshold= 0.2

font_color = (0, 0, 255)
font_size = 0.5
font_thickness = 2

# Middle cross line position
middle_line_position = 225   
up_line_position = middle_line_position - 15  
down_line_position = middle_line_position + 15

# Store Coco Names in a list
classesFile = '/Users/macbook/Downloads/vechile_detection/coco.names'
classNames = open(classesFile).read().strip().split('\n')
print(classNames)
#print(len(classNames))

# class index for our required detection classes
required_class_index = [7,8,9]

detected_classNames = []


src_dir_2 = '/Users/macbook/Downloads/Trucks/train_2/Full/'
src_dir = '/Users/macbook/Downloads/Trucks/train/Full/'

## Model Files
modelConfiguration = '/Users/macbook/Downloads/vechile_detection/yolov3-320.cfg'
modelWeigheights = '/Users/macbook/Downloads/vechile_detection/yolov3-320.weights'

# configure the network model
net = cv2.dnn.readNetFromDarknet(modelConfiguration, modelWeigheights)

# Configure the network backend

#net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA) - CUDA is for parallel computing thus for GPU, but I use CPU here
#net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

# Define random colour for each class
np.random.seed(42)
colors = np.random.randint(0, 255, size=(len(classNames), 3), dtype='uint8')

# Function for finding the center of a rectangle
def find_center(x, y, w, h):
    x1=int(w/2)
    y1=int(h/2)
    cx = x+x1
    cy=y+y1
    return cx, cy
    
# List for store vehicle count information
temp_up_list = []
temp_down_list = []
up_list = [0, 0, 0, 0]
down_list = [0, 0, 0, 0]

# Function for count vehicle
# def count_vehicle(box_id, img):

#     x, y, w, h, id, index = box_id

#     # Find the center of the rectangle for detection
#     center = find_center(x, y, w, h)
#     ix, iy = center
    
#     # Find the current position of the vehicle
#     if (iy > up_line_position) and (iy < middle_line_position):

#         if id not in temp_up_list:
#             temp_up_list.append(id)

#     elif iy < down_line_position and iy > middle_line_position:
#         if id not in temp_down_list:
#             temp_down_list.append(id)
            
#     elif iy < up_line_position:
#         if id in temp_down_list:
#             temp_down_list.remove(id)
#             up_list[index] = up_list[index]+1

#     elif iy > down_line_position:
#         if id in temp_up_list:
#             temp_up_list.remove(id)
#             down_list[index] = down_list[index] + 1

#     # Draw circle in the middle of the rectangle
#     cv2.circle(img, center, 2, (0, 0, 255), -1)  # end here
#     # print(up_list, down_list)

# Function for finding the detected objects from the network output
def postProcess(outputs,photo,img,i):
    
    global detected_classNames 
    height, width = img.shape[:2]
    boxes = []
    classIds = []
    confidence_scores = []
    detection = []
    for output in outputs:
        for det in output:
            scores = det[5:]
            classId = np.argmax(scores)
            confidence = scores[classId]
            if classId in required_class_index:
                if confidence > confThreshold:
                    # print(classId)
                    w,h = int(det[2]*width) , int(det[3]*height)
                    x,y = int((det[0]*width)-w/2) , int((det[1]*height)-h/2)
                    boxes.append([x,y,w,h])
                    classIds.append(classId)
                    confidence_scores.append(float(confidence))

    # Apply Non-Max Suppression
    indices = cv2.dnn.NMSBoxes(boxes, confidence_scores, confThreshold, nmsThreshold)
    # print(classIds)
    if len(indices) > 0:
        for i in indices.flatten():
            x, y, w, h = boxes[i][0], boxes[i][1], boxes[i][2], boxes[i][3]
            # print(x,y,w,h)

            color = [int(c) for c in colors[classIds[i]]]
            name = classNames[classIds[i]]
            detected_classNames.append(name)
            # Draw classname and confidence score 
            cv2.putText(img,f'{name.upper()} {int(confidence_scores[i]*100)}%',
                    (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

            # Draw bounding rectangle
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 1)
            detection.append([x, y, w, h, required_class_index.index(classIds[i])])
            if name.upper() == "TRUCK":
                    
                rect = (x, y, x + w, y + h)
                img2=photo.crop(rect)
                file_local = "default.jpg"
                img2.save(file_local)
                file_remote = "default_"+str(i)+".jpg"
                #print(file_remote)
                dst = src_dir_2 + file_remote
                shutil.copyfile(file_local,dst)     

                print("process")
            elif name.upper() != "TRUCK":
                print(f'no detection on number {i}')
            
                #img2.show()
    # Update the tracker for each object
    #boxes_ids = tracker.update(detection)
    
    #cv2.imshow("image",img)
    
    
    #for box_id in boxes_ids:
        #count_vehicle(box_id, img) -- related to counting vehciles function,thus if we remove the function we should rmeove this part as well


def realTime():
    
    while True:
        success, img = cap.read()
        img = cv2.resize(img,(0,0),None,0.5,0.5)
        ih, iw, channels = img.shape
        blob = cv2.dnn.blobFromImage(img, 1 / 255, (input_size, input_size), [0, 0, 0], 1, crop=False)

        # Set the input of the network
        net.setInput(blob)
        layersNames = net.getLayerNames()
  
        outputNames = [(layersNames[i - 1]) for i in net.getUnconnectedOutLayers()]
        
        # Feed data to the network
        outputs = net.forward(outputNames)
        
        # Find the objects from the network output
        postProcess(outputs,img)

        # Draw the crossing lines

        # cv2.line(img, (0, middle_line_position), (iw, middle_line_position), (255, 0, 255), 2)
        # cv2.line(img, (0, up_line_position), (iw, up_line_position), (0, 0, 255), 2)
        # cv2.line(img, (0, down_line_position), (iw, down_line_position), (0, 0, 255), 2)

        # Draw counting texts in the frame
        # cv2.putText(img, "Up", (110, 20), cv2.FONT_HERSHEY_SIMPLEX, font_size, font_color, font_thickness)
        # cv2.putText(img, "Down", (160, 20), cv2.FONT_HERSHEY_SIMPLEX, font_size, font_color, font_thickness)
        # cv2.putText(img, "Car:        "+str(up_list[0])+"     "+ str(down_list[0]), (20, 40), cv2.FONT_HERSHEY_SIMPLEX, font_size, font_color, font_thickness)
        # cv2.putText(img, "Motorbike:  "+str(up_list[1])+"     "+ str(down_list[1]), (20, 60), cv2.FONT_HERSHEY_SIMPLEX, font_size, font_color, font_thickness)
        # cv2.putText(img, "Bus:        "+str(up_list[2])+"     "+ str(down_list[2]), (20, 80), cv2.FONT_HERSHEY_SIMPLEX, font_size, font_color, font_thickness)
        # cv2.putText(img, "Truck:      "+str(up_list[3])+"     "+ str(down_list[3]), (20, 100), cv2.FONT_HERSHEY_SIMPLEX, font_size, font_color, font_thickness)

        # Show the frames
        cv2.imshow('Output', img)

        if cv2.waitKey(1) == ord('q'):
            break

    # Write the vehicle counting information in a file and save it

    with open("data.csv", 'w') as f1:
        cwriter = csv.writer(f1)
        cwriter.writerow(['Direction', 'car', 'motorbike', 'bus', 'truck'])
        up_list.insert(0, "Up")
        down_list.insert(0, "Down")
        cwriter.writerow(up_list)
        cwriter.writerow(down_list)
    f1.close()
    # print("Data saved at 'data.csv'")
    # Finally realese the capture object and destroy all active windows
    cap.release()
    cv2.destroyAllWindows()


#image_file = '/Users/macbook/Downloads/Trucks/train/Empty/Empty_38.jpg'
def from_static_image(image,k):
    imag = Image.open(image)
    img = cv2.imread(image)
   
    if img is None:
        print("error")
    else:
        blob = cv2.dnn.blobFromImage(img, 1 / 255, (input_size, input_size), [0, 0, 0], 1, crop=False)

    # Set the input of the network
        net.setInput(blob)
        layersNames = net.getLayerNames()
        outputNames = [(layersNames[i - 1]) for i in net.getUnconnectedOutLayers()]
    # Feed data to the network
        outputs = net.forward(outputNames)
    #for file in os.listdir(src):
    # Find the objects from the network output
        postProcess(outputs,imag,img,k)

    # count the frequency of detected classes
    #frequency = collections.Counter(detected_classNames)
    #print(frequency)
    # Draw counting texts in the frame
    #cv2.putText(img, "Car:        "+str(frequency['car']), (20, 40), cv2.FONT_HERSHEY_SIMPLEX, font_size, font_color, font_thickness)
    #cv2.putText(img, "Motorbike:  "+str(frequency['motorbike']), (20, 60), cv2.FONT_HERSHEY_SIMPLEX, font_size, font_color, font_thickness)
    #cv2.putText(img, "Bus:        "+str(frequency['bus']), (20, 80), cv2.FONT_HERSHEY_SIMPLEX, font_size, font_color, font_thickness)
    #cv2.putText(img, "Truck:      "+str(frequency['truck']), (20, 100), cv2.FONT_HERSHEY_SIMPLEX, font_size, font_color, font_thickness)


       

        #cv2.waitKey(0)

    # save the data to a csv file
    # with open("static-data.csv", 'a') as f1:
    #     cwriter = csv.writer(f1)
    #     cwriter.writerow([image, frequency['car'], frequency['motorbike'], frequency['bus'], frequency['truck']])
    # f1.close()

#image_file = '/Users/macbook/Downloads/Trucks/train/Empty/Empty_52.jpg'
if __name__ == '__main__':
    #realTime()
    #from_static_image(image_file,k)

    for file in os.listdir(src_dir):
        #image_file = '/Users/macbook/Downloads/Trucks/train/Empty/Empty_38.jpg'
        image_file = src_dir+file
        print(file)
        l = re.findall('\d+',file) # finds a number in a file name
        print(l)
        try:
            k = int(l[0])
            
            from_static_image(image_file,k)
        except:
            print("error")
            continue
        
        


