import cv2
import dlib
import os 
cap = cv2.VideoCapture("/Users/macbook/Documents/Zypl_short.mp4")

pwd = os.path.dirname(__file__)
dlib_facelandmark = dlib.shape_predictor(pwd+"/shape_predictor_68_face_landmarks.dat")
hog_face_detector = dlib.get_frontal_face_detector()



while True:
    _, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = hog_face_detector(gray)
    for face in faces:

        face_landmarks = dlib_facelandmark(gray, face)

        for n in range(0, 16):
            x = face_landmarks.part(n).x
            y = face_landmarks.part(n).y
            cv2.circle(frame, (x, y), 1, (0, 255, 255), 1)


    cv2.imshow("Face Landmarks", frame)

    key = cv2.waitKey(1)
    if key == 27:
        break
cap.release()
cv2.destroyAllWindows()