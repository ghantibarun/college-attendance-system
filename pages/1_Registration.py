import cv2
import numpy as np
from deepface import DeepFace
import math
import time
import sqlite3
import json
import streamlit as st

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


def overlay_icon(img):
    mid_x = (img.shape[1]) // 2
    mid_x=mid_x-20
    logo = cv2.imread('./fps.png', cv2.IMREAD_UNCHANGED)
    logo = cv2.resize(logo, (50, 50))

    # Extract the alpha channel and convert to 8-bit unsigned integer
    alpha_channel = logo[:, :, 3]
    alpha_channel = cv2.convertScaleAbs(alpha_channel)

    # Remove the alpha channel from the logo and convert to BGR format
    logo = logo[:, :, :3]
    logo = cv2.cvtColor(logo, cv2.COLOR_BGRA2BGR)

    # Create a mask from the alpha channel and resize it
    mask = cv2.threshold(alpha_channel, 0, 255, cv2.THRESH_BINARY)[1]
    mask = cv2.resize(mask, (logo.shape[1], logo.shape[0]))

    # Overlay the logo on the image
    x = mid_x - 40
    y = 5
    overlay = img.copy()
    roi = overlay[y:y+logo.shape[0], x:x+logo.shape[1]]
    roi_bg = cv2.bitwise_and(roi, roi, mask=cv2.bitwise_not(mask))
    roi_fg = cv2.bitwise_and(logo, logo, mask=mask)
    roi_combined = cv2.add(roi_bg, roi_fg)
    overlay[y:y+logo.shape[0], x:x+logo.shape[1]] = roi_combined

    return overlay

def fps_display(img,pTime):
    mid_x = (img.shape[1]) // 2
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
def time_count(img,num_seconds):
    time_remaining = int(30 - num_seconds)
    # Add text overlay to display time remaining
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    text = f'Time remaining: {time_remaining}s'
    text_width, text_height = cv2.getTextSize(text, font, font_scale, thickness=1)[0]
    # Calculate text position at the bottom center of the image
    text_offset_x = (img.shape[1] - text_width) // 2
    text_offset_y = img.shape[0] - text_height - 3
    cv2.putText(img, text, (text_offset_x, text_offset_y), font, font_scale, (0, 0, 0), thickness=1, lineType=cv2.LINE_AA)
    return img

    
import os 
def video_capture(name,id_number,branch_name,designation):
    cap = cv2.VideoCapture(0)
    # cap = cv2.VideoCapture("./videos/2.mp4")
    pTime = 0
    FRAME_WINDOW = st.image([]) 
    t0=time.time()
    while True:
        ret, img = cap.read()
        # img=cv2.flip(img,1)
        faces,facial_data=deep_data_extract(img)
        show_img=white_overlay(img)
        
        
        if len(faces)!=0 and len(facial_data)!=0:
            if len(faces)==len(facial_data):
                # print(faces[0],facial_data[0])
                face_data=facial_data[0]
                face_data=convert_string(face_data) 
                # st.write(face_data, name, id_number, branch_name, designation)

                add_attendance_record(face_data, name, id_number, branch_name, designation)
                x1,y1,x2,y2=faces[0]
                l = int(0.1 * math.sqrt((x2-x1)**2 + (y2-y1)**2))
                draw_img=drawBox(img, x1, y1, x2, y2, l=l, t=5, rt=1, text=name, id=id_number,display_id=True,draw_rect=True,color=(2, 240, 228),text_color=(255,255,255))
                overlay = white_overlay(draw_img)
                show_img=overlay     
        show_img,pTime=fps_display(show_img,pTime)  
        t1 = time.time() 
        num_seconds = t1 - t0 
        show_img=overlay_icon(show_img) 
        show_img=time_count(show_img,num_seconds)   
        if num_seconds > 30:  
            break
        frame = cv2.cvtColor(show_img, cv2.COLOR_BGR2RGB)
        FRAME_WINDOW.image(frame)
    FRAME_WINDOW.image([])
    cap.release()
# st.title("Registration")


st.markdown("<h1 style='text-align: center;'>Registration</h1>", unsafe_allow_html=True)




def create_database():

    conn1 = sqlite3.connect("database.sqlite")
    c1 = conn1.cursor()
    c1.execute('''CREATE TABLE IF NOT EXISTS attendance_records
                (face_data TEXT, name TEXT, id_number INTEGER, branch_name TEXT, designation TEXT)''')
    conn1.commit()
    conn1.close()



    conn2 = sqlite3.connect("name.sqlite")
    c2 = conn2.cursor()
    c2.execute('''CREATE TABLE IF NOT EXISTS id_info
                    (id_number INTEGER PRIMARY KEY, name TEXT, branch_name TEXT, designation TEXT, email TEXT)''')
    conn2.commit()
    conn2.close()

create_database()


def add_attendance_record(face_data, name, id_number, branch_name, designation):
    try:
        conn = sqlite3.connect("database.sqlite")
        c = conn.cursor()
        c.execute("INSERT INTO attendance_records (face_data, name, id_number, branch_name, designation) VALUES (?, ?, ?, ?, ?)",
                  (face_data, name, id_number, branch_name, designation))
        conn.commit()
        # print("Data inserted successfully.")
        conn.close()
    except sqlite3.Error as error:
        print("An error occurred:", error)

def convert_string(face_data):
    string_face_data = json.dumps(face_data)
    string_face_data = "[" + string_face_data[1:-1] + "]"
    return string_face_data



def get_id(id_number):
    try:
        conn = sqlite3.connect("name.sqlite")
        c = conn.cursor()
        c.execute("SELECT * FROM id_info WHERE id_number = ?", (id_number,))
        result = c.fetchone()
        conn.close()
        if result is not None:
            id_number, name, branch_name, designation, email = result[0], result[1], result[2], result[3], result[4]
            # print(f"Welcome, {name} ({designation}, {branch_name}, {email})!")
            # print("data exist in database")
            return True
        else:
            # print("ID number not found.")
            return False
    except sqlite3.Error as error:
        # print("An error occurred:", error)
        return False



def insert_data(id_number, name, branch_name, designation, email):
    try:
        conn = sqlite3.connect("name.sqlite")
        c = conn.cursor()
        c.execute('''INSERT OR IGNORE INTO id_info (id_number, name, branch_name, designation, email) 
                     VALUES (?, ?, ?, ?, ?)''',
                  (id_number, name, branch_name, designation, email))
        conn.commit()
        print("Data inserted successfully.")
        conn.close()
    except sqlite3.Error as error:
        print("An error occurred:", error)




create_database()
import re

def is_valid_email(email):
    # regex pattern for email validation
    pattern = r'^([a-zA-Z0-9_.+-]+)@([a-zA-Z0-9-]+\.)+([a-zA-Z0-9]{2,})$'
    # match the pattern with the email
    match = re.match(pattern, email)
    # if match is found, email is valid
    if match:
        return True
    else:
        return False
import re

# def is_valid_email(email):
#     # regex pattern for email validation
#     pattern = r'^([a-zA-Z0-9_.+-]+)@([a-zA-Z0-9-]+\.)+([a-zA-Z0-9]{2,})$'
#     # match the pattern with the email
#     match = re.match(pattern, email)
#     # if match is found, email is valid
#     if match:
#         return True
#     else:
#         return False
def check_email(email):
    try:
        conn = sqlite3.connect("name.sqlite")
        c = conn.cursor()
        c.execute("SELECT * FROM id_info WHERE email = ?", (email,))
        result = c.fetchone()
        conn.close()
        if result is not None:
            return True
        else:
            return False

    except sqlite3.Error as error:
        print("An error occurred:", error)
        return False
    
form = st.form(key='my-form')
name = form.text_input('Enter your name')
id_number = form.number_input("Enter ID number", value=1, step=1)
branch_name = form.text_input('Enter Branch Name')
designation = form.selectbox("Designation", ("Student", "Teacher"))
email = form.text_input('Enter your email')
submit = form.form_submit_button('Submit')

st.caption('You have only :blue[30 seconds] to scan yourself')


if submit:
    name=name.title()
    branch_name=branch_name.replace(".","").upper()
    if is_valid_email(email)==True:
        pass
    else:
        st.error(f'{designation} email: {email} is not valid')
        st.stop()
    if  get_id(id_number)==False: 
        if check_email(email)==False:
            # print(get_id,check_email)
            insert_data(id_number, name, branch_name, designation, email)
            st.write(f'Name: {name}')
            st.write(f'Student Id: {id_number}')
            st.write(f'Branch Name: {branch_name}')
            st.write(f'Designation: {designation}')
            st.write(f'Designation: {email}')
            st.markdown('<p style="color:green">Please wait for a few seconds while the camera is opening...</p>', unsafe_allow_html=True)
            video_capture(name,id_number,branch_name,designation)
            st.success("Data saved successfully")
        else:
            st.error(f'{designation} email: {email} already exists')
            
    else:
        if get_id(id_number)==True:         
            st.error(f'Student Id: {id_number} already exists')
        if check_email(email)==True: 
            st.error(f'{designation} email: {email} already exists')
        
        
        


hide_st_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            header {visibility: hidden;}
            </style>
            """
st.markdown(hide_st_style, unsafe_allow_html=True)
