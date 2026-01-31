import cv2
import numpy as np
from deepface import DeepFace
import math
import time
import sqlite3
import json
import streamlit as st

import threading
import datetime
import smtplib
from dotenv import load_dotenv
import os 
load_dotenv()

st.markdown("<h1 style='text-align: center;'>Real-time Attendance with Email Confirmation</h1>", unsafe_allow_html=True)

HOST = "smtp-mail.outlook.com"
PORT = 587
FROM_EMAIL = os.getenv("email")
PASSWORD = os.getenv("password")
send_mail= os.getenv("send_mail")
def send_attendance_email(id_number, name, branch_name, designation, email, subject):
    now = datetime.datetime.now()
    date = now.strftime('%d-%m-%Y')
    time = now.strftime('%I:%M:%S %p')


    message = f"Subject: Attendance Confirmation\n\n"
    message += f"Dear {name},\n\n"
    message += f"Your attendance for the following class has been recorded:\n\n"
    message += f"Subject: {subject}\n"
    message += f"ID Number: {id_number}\n"
    message += f"Name: {name}\n"
    message += f"Branch Name: {branch_name}\n"
    message += f"Designation: {designation}\n"
    message += f"Date: {date}\n"
    message += f"Time: {time}\n\n"
    message += f"If you believe this is in error, please contact us immediately.\n\n"
    message += f"Thank you,\nThe Attendance Team"

    try:
        with smtplib.SMTP(HOST, PORT) as smtp:
            smtp.ehlo()
            smtp.starttls()
            smtp.login(FROM_EMAIL, PASSWORD)
            smtp.sendmail(FROM_EMAIL, email, message)
        print("Email sent successfully.")
    except Exception as e:
        print(f"An error occurred while sending the email: {e}")

def run_attendance_email(id_number, name, branch_name, designation, email, subject):
    t = threading.Thread(target=send_attendance_email, args=(id_number, name, branch_name, designation, email, subject))
    t.start()
    print("Attendance email processing in the background...")
    
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


    
# with open("./svm_model.pkl", "rb") as f:
#     svm_model = pickle.load(f)

def video_capture(subject):
    global svm_model
    cap = cv2.VideoCapture(0)
    pTime = 0
    FRAME_WINDOW = st.image([]) 
    t0=time.time()
    while True:
        ret, img = cap.read()
        img=cv2.flip(img,1)
        faces,facial_data=deep_data_extract(img)
        show_img=white_overlay(img)
        
        
        if len(faces)!=0 and len(facial_data)!=0:
            if len(faces)==len(facial_data):
                # print(faces[0],facial_data[0])
                face_data=facial_data[0]
                x1,y1,x2,y2=faces[0]
                l = int(0.1 * math.sqrt((x2-x1)**2 + (y2-y1)**2))
                person_id = svm_model.predict([face_data])
                info = get_id_info(int(person_id))
                if info is not None:
                    name, branch_name, designation, email=info
                    att=mark_attendance(int(person_id), name, branch_name, designation, subject)
                    if att:
                        # run_attendance_email(id_number=int(person_id), name=name, branch_name=branch_name, designation=designation, email=send_mail, subject=subject)
                        run_attendance_email(id_number=int(person_id), name=name, branch_name=branch_name, designation=designation, email=email, subject=subject)

                    # print(f"Name: {name}, Branch: {branch_name}, Designation: {designation}")
                    draw_img=drawBox(img, x1, y1, x2, y2, l=l, t=5, rt=1, text=name, id=str(person_id),display_id=True,draw_rect=True,color=(2, 240, 228),text_color=(255,255,255))
                else:
                    # print("ID number not found in database.")
                    draw_img=drawBox(img, x1, y1, x2, y2, l=l, t=5, rt=1, text="unknown", id=None,display_id=True,draw_rect=True,color=(255, 0, 0),text_color=(255,255,255))
                overlay = white_overlay(draw_img)
                show_img=overlay     
        show_img,pTime=fps_display(show_img,pTime)  
        show_img=overlay_icon(show_img) 
        frame = cv2.cvtColor(show_img, cv2.COLOR_BGR2RGB)
        FRAME_WINDOW.image(frame)
    FRAME_WINDOW.image([])
    cap.release()
    cv2.destroyAllWindows()




def create_database():

    conn2 = sqlite3.connect("name.sqlite")
    c2 = conn2.cursor()
    c2.execute('''CREATE TABLE IF NOT EXISTS id_info
                    (id_number INTEGER PRIMARY KEY, name TEXT, branch_name TEXT, designation TEXT, email TEXT)''')
    conn2.commit()
    conn2.close()
    
    conn3 = sqlite3.connect('attendance.sqlite')
    c3 = conn3.cursor()
    c3.execute('''CREATE TABLE IF NOT EXISTS attendance
                (date TEXT, subject TEXT, id_number INTEGER, time TEXT, name TEXT, branch_name TEXT, designation TEXT,
                PRIMARY KEY (date, subject, id_number))''')
    conn3.commit()
    conn3.close()
        
create_database()



import sqlite3

def get_id_info(id_num):
    conn = sqlite3.connect("name.sqlite")
    c = conn.cursor()
    c.execute("SELECT name, branch_name, designation, email FROM id_info WHERE id_number=?", (id_num,))
    result = c.fetchone()
    conn.close()
    
    if result is not None:
        name, branch_name, designation, email = result
        return name, branch_name, designation, email
    else:
        return None



def convert_string(face_data):
    string_face_data = json.dumps(face_data)
    string_face_data = "[" + string_face_data[1:-1] + "]"
    return string_face_data









def mark_attendance(id_number, name, branch_name, designation, subject):
    # Get the current date and time
    now = datetime.datetime.now()
    current_time = now.strftime('%I:%M:%S %p')
    today = datetime.date.today().strftime('%d-%m-%Y')
    
    # Connect to the database or create it if it doesn't exist
    conn = sqlite3.connect('attendance.sqlite')
    
    # Create a cursor object
    c = conn.cursor()
    
    # Check if the date is already in the database
    c.execute("SELECT date FROM attendance WHERE date = ?", (today,))
    date_exists = c.fetchone() is not None
    
    # If the date doesn't exist, insert it into the database
    if not date_exists:
        c.execute("INSERT INTO attendance(date, subject, id_number, time, name, branch_name, designation) VALUES (?, ?, ?, ?, ?, ?, ?)",
                  (today, subject, id_number, current_time, name, branch_name, designation))
        conn.commit()
        c.close()
        return True
    
    # Check if the subject is already in the database for today's date
    c.execute("SELECT subject FROM attendance WHERE date = ? AND subject = ?", (today, subject))
    subject_exists = c.fetchone() is not None
    
    # If the subject doesn't exist, insert it into the database
    if not subject_exists:
        c.execute("INSERT INTO attendance(date, subject, id_number, time, name, branch_name, designation) VALUES (?, ?, ?, ?, ?, ?, ?)",
                  (today, subject, id_number, current_time, name, branch_name, designation))
        conn.commit()
        c.close()
        return True
    
    # Check if the id_number exists for the subject and date
    c.execute("SELECT id_number FROM attendance WHERE date = ? AND subject = ? AND id_number = ?", (today, subject, id_number))
    id_exists = c.fetchone() is not None
    
    # If the id_number doesn't exist, insert it into the database
    if not id_exists:
        c.execute("INSERT INTO attendance(date, subject, id_number, time, name, branch_name, designation) VALUES (?, ?, ?, ?, ?, ?, ?)",
                  (today, subject, id_number, current_time, name, branch_name, designation))
        conn.commit()
        c.close()
        return True
    
    # If the id_number already exists, don't update the database
    c.close()
    return False

import pickle
load_model = True
try:
    with open("./svm_model.pkl", "rb") as f:
        svm_model = pickle.load(f)
except:
    load_model = False
submit_disabled = not load_model
# Inform the user if the model is not loaded

create_database()
form = st.form(key='my-form')
subject = form.text_input('Enter your subject name')


if not load_model:
    st.markdown('<p style="color:red">Model not found! Please train the model before proceeding.</p>', unsafe_allow_html=True)

submit = form.form_submit_button('Submit', disabled=submit_disabled)

# If the submit button is clicked and the model is loaded
if submit and load_model:
    st.markdown('<p style="color:green">Please wait for a few seconds while the camera is opening...</p>', unsafe_allow_html=True)
    subject = subject.title()
    video_capture(subject)  # Call the video capture function




hide_st_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            header {visibility: hidden;}
            </style>
            """
st.markdown(hide_st_style, unsafe_allow_html=True)
