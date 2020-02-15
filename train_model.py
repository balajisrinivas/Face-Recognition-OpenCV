import numpy as np
import cv2
import os

import faceRecognition as fr
print (fr)

test_img = cv2.imread(r'C:/Users/balajiam/Documents/ML Data Analysis/Face-Recognition-master/1.jpg')

faces_detected,gray_img = fr.faceDetection(test_img)
print("face Detected: ",faces_detected)

#Training will begin from here

faces,faceID=fr.labels_for_training_data(r'C:/Users/balajiam/Documents/ML Data Analysis/Face-Recognition-master/train-images') #Give path to the train-images folder which has both labeled folder as 0 and 1
face_recognizer=fr.train_classifier(faces,faceID)
face_recognizer.save(r'C:/Users/balajiam/Documents/ML Data Analysis/Face-Recognition-master/trainingData.yml') #It will save the trained model. Just give path to where you want to save

name={0:"Balaji", 1:"Srini"}


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
cv2.waitKey(0)
cv2.destroyAllWindows
