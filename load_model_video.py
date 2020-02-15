import numpy as np
import cv2
import os

import faceRecognition as fr
print (fr)

face_recognizer=cv2.face.LBPHFaceRecognizer_create()
face_recognizer.read(r'C:/Users/balajiam/Documents/ML Data Analysis/Face-Recognition-master/trainingData.yml')

cap=cv2.VideoCapture(0)

name={0:"Balaji", 1:"Srini"}
while True:
    ret,test_img=cap.read()
    faces_detected,gray_img=fr.faceDetection(test_img)
    print("face Detected: ",faces_detected)
    for (x,y,w,h) in faces_detected:
        cv2.rectangle(test_img,(x,y),(x+w,y+h),(0,255,0),thickness=5)

    for face in faces_detected:
        (x,y,w,h)=face
        roi_gray=gray_img[y:y+h,x:x+h]
        label,confidence=face_recognizer.predict(roi_gray)
        print ("Confidence :",confidence)
        print("label :",label)
        fr.draw_rect(test_img,face)
        predicted_name=name[label]
        fr.put_text(test_img,predicted_name,x,y)

    resized_img=cv2.resize(test_img,(1000,700))

    cv2.imshow("face detection ", resized_img)
    if cv2.waitKey(10)==ord('q'):
        break
