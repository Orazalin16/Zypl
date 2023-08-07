import face_recognition
import cv2



video_file_3min = "Zypl_short.mp4"
cap = cv2.VideoCapture(video_file_3min)
counter =0 

while True:
    _,frame=cap.read()
    frame = cv2.cvtColor(frame,cv2.COLOR_RGB2BGR)
    face = face_recognition.face_locations(frame)
    frame = cv2.cvtColor(frame,cv2.COLOR_RGB2BGR)

    for i in face:
        cv2.rectangle(frame,(i[3],i[0]),(i[1],i[2]),(255,0,255),2)


    cv2.imshow("frame",frame)
    k=cv2.waitKey(1)
    if k%256==27:
        break
cap.release()
cv2.destroyAllWindows()
# while True:
#         sucess,frame = cap.read();

#         frame = cv2.cvtColor(frame,cv2.COLOR_RGB2BGR)
#         face = face_recognition.face_locations(frame)
       
        
#         frame = cv2.cvtColor(frame,cv2.COLOR_RGB2BGR)
        
        
#                 counter +=1
#         cv2.imshow("Test",frame)
#         k = cv2.waitKey(1)
#         if k%256==27:
#             break
# print("number of detections: {} ".format(counter))
# cap.release()
# cv2.destroyAllWindows()