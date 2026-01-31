import cv2
import numpy as np
from deepface import DeepFace
import os
img=cv2.imread("p.jpg")
embedding=None
faces=[]
facial_data=[]
embedding = DeepFace.represent(img, model_name='Facenet',detector_backend='ssd')
if embedding:
    for  i in range(len(embedding)):
        x, y, w, h = embedding[i]['facial_area']['x'], embedding[i]['facial_area']['y'], embedding[i]['facial_area']['w'], embedding[i]['facial_area']['h']
        x1, y1, x2, y2 = x, y, x+w, y+h
        faces.append((x1, y1, x2, y2 ))
        facial_data.append(embedding[i]['embedding'])

# print(faces,facial_data)
# cv2.imshow("img",img)
# cv2.waitKey(0)
if len(facial_data)>=1:
    print("Successfully downloaded FaceNet and SSD.")
