import cv2
import numpy as np
from deepface import DeepFace
import math
import time
def deep_data_extract(img):
    embedding=None
    faces=[]
    facial_data=[]
    try:
        embedding = DeepFace.represent(img, model_name='Facenet',detector_backend='ssd')
        if embedding:
          for  i in range(len(embedding)):
            x, y, w, h = embedding[i]['facial_area']['x'], embedding[i]['facial_area']['y'], embedding[i]['facial_area']['w'], embedding[i]['facial_area']['h']
            x1, y1, x2, y2 = x, y, x+w, y+h
            faces.append((x1, y1, x2, y2 ))
            facial_data.append(embedding[i]['embedding'])
    except:
        pass
    return faces,facial_data


def rgb_to_bgr(rgb_color):
    bgr_color = (rgb_color[2], rgb_color[1], rgb_color[0])
    return bgr_color
    
def drawBox(img, x1, y1, x2, y2, l=30, t=5, rt=1, text="Unknown", id=None,display_id=False,draw_rect=False,color=(2, 240, 228),text_color=(255,255,255)):
    # Define the sci-fi style font
    font = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = 0.7
    thickness = 2
    # color = (255, 255, 255)
    color=rgb_to_bgr(color)
    text_color=rgb_to_bgr(text_color)
    # Draw the ID of the detected person on top of the bounding box
    ((id_width, id_height), _) = cv2.getTextSize(str(id), font, fontScale=fontScale, thickness=thickness)
    id_offset_x = x1 + int((x2 - x1 - id_width) / 2)
    id_offset_y = y1 - 35
    if display_id:
        cv2.putText(img, str(id), (id_offset_x, id_offset_y+25), font, fontScale, text_color, thickness)
        # Draw the name of the detected person inside the bounding box
        ((text_width, text_height), _) = cv2.getTextSize(text, font, fontScale=fontScale, thickness=thickness)
        text_offset_x = x1 + int((x2 - x1 - text_width) / 2)
        text_offset_y = y2 + 25
        cv2.putText(img, text, (text_offset_x, text_offset_y), font, fontScale, text_color, thickness)
        # Draw box around face
    if draw_rect:
        cv2.rectangle(img, (x1, y1), (x2, y2), color,thickness=rt)
    t=t-3
    face_width = x2 - x1
    face_height = y2 - y1
    # l = int(l * min(face_width, face_height) / 100)-20
    
    # Draw top-left corner
    cv2.line(img, (x1, y1), (x1 + l, y1), color, thickness=t)
    cv2.line(img, (x1, y1), (x1, y1 + l), color, thickness=t)
    # Draw top-right corner
    cv2.line(img, (x2, y1), (x2 - l, y1), color, thickness=t)
    cv2.line(img, (x2, y1), (x2, y1 + l), color, thickness=t)
    # Draw bottom-left corner
    cv2.line(img, (x1, y2), (x1 + l, y2), color, thickness=t)
    cv2.line(img, (x1, y2), (x1, y2 - l), color, thickness=t)
    # Draw bottom-right corner
    cv2.line(img, (x2, y2), (x2 - l, y2), color, thickness=t)
    cv2.line(img, (x2, y2), (x2, y2 - l), color, thickness=t)
    return img

def white_overlay(img):
    white_img = np.ones_like(img) * 255
    alpha = 0.5
    result = cv2.addWeighted(img, alpha, white_img, 1-alpha, 0)
    x1 = 60
    y1 = 60
    x2 = img.shape[1] - 60
    y2 = img.shape[0] - 60
    mid_x = (img.shape[1]) // 2
    roi = img[y1:y2, x1:x2]
    result[y1:y2, x1:x2] = roi
    return result



def fps_display(img,pTime,mid_x):
    fps = 0
    cTime = time.time()
    if cTime - pTime > 0:
        fps = 1 / (cTime - pTime)
    pTime = cTime
    # text = f'FPS: {int(fps)}'
    text=str(int(fps))
    font = cv2.FONT_HERSHEY_PLAIN
    font_scale = 3
    thickness = 3
    text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
    x = img.shape[1] - text_size[0] - 20
    color=rgb_to_bgr((240, 0, 148))
    cv2.putText(img, text, (mid_x, 45), font, font_scale,color, thickness)
    return img,pTime


   
cap = cv2.VideoCapture(0)
pTime = 0
while True:
    ret, img = cap.read()
    img=cv2.flip(img,1)
    faces,facial_data=deep_data_extract(img)
    show_img=white_overlay(img)
    mid_x = (img.shape[1]) // 2
    
    if len(faces)!=0 and len(facial_data)!=0:
        if len(faces)==len(facial_data):
            print(faces[0],facial_data[0])
            x1,y1,x2,y2=faces[0]
            crop_img = img[y1:y2, x1:x2]
            l = int(0.1 * math.sqrt((x2-x1)**2 + (y2-y1)**2))
            draw_img=drawBox(img, x1, y1, x2, y2, l=l, t=5, rt=1, text="Unknown", id=None,display_id=True,draw_rect=True,color=(2, 240, 228),text_color=(255,255,255))
            overlay = white_overlay(draw_img)
            show_img=overlay     
    show_img,pTime=fps_display(show_img,pTime,mid_x)         
    cv2.imshow('Video ', show_img)

    # Exit on key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release video capture device and close windows
cap.release()
cv2.destroyAllWindows()
