import cv2
import numpy as np

face_cascade=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
cap=cv2.VideoCapture(0)
img_name="photo_0.png"
while True:
    ret,frame=cap.read()
    cv2.imshow('image',frame)
    if cv2.waitKey(1)==13:
        img_name="photo_0.png"
        cv2.imwrite(img_name,frame)
        break
while True:
    img = cv2.imread('photo_0.png')
    faces=face_cascade.detectMultiScale(img,scaleFactor=1.1,minNeighbors= 9)
    for x,y,w,h in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),5)
        cv2.imshow('FACE REC',img)
    if cv2.waitKey(1)==13:
        break

cap.release()
cv2.destroyAllWindows()
