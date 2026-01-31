import threading
import smtplib
from dotenv import load_dotenv
import os 
from email.mime.multipart import MIMEMultipart
from email.mime.application import MIMEApplication
import streamlit as st 
load_dotenv()

HOST = "smtp-mail.outlook.com"
PORT = 587
FROM_EMAIL = os.getenv("email")
PASSWORD = os.getenv("password")
ADMIN_EMAIL= os.getenv("admin_mail")

def send_attendance_email(attendance_file):
    message = MIMEMultipart()
    message["Subject"] = "Attendance File"
    message["From"] = FROM_EMAIL
    message["To"] = ADMIN_EMAIL

    with open(attendance_file, 'rb') as file:
        attachment = MIMEApplication(file.read(), _subtype='xlsx')
        attachment.add_header('Content-Disposition', 'attachment', filename=attendance_file)
        message.attach(attachment)

    try:
        with smtplib.SMTP(HOST, PORT) as smtp:
            smtp.ehlo()
            smtp.starttls()
            smtp.login(FROM_EMAIL, PASSWORD)
            smtp.sendmail(FROM_EMAIL, ADMIN_EMAIL, message.as_string()) # send to admin email
        print("Email sent successfully.")
    except Exception as e:
        print(f"An error occurred while sending the email: {e}")

def run_attendance_email(attendance_file):
    t = threading.Thread(target=send_attendance_email, args=(attendance_file,))
    t.start()
    print("Attendance email processing in the background...")

import os

filename = 'attendance_report.xlsx'
if os.path.isfile(filename):
    pass
else:
    st.write('Attendance Report Not Found')
    st.stop()

# st.markdown("<h1 style='text-align: center;'>Send attendance record excel file to a teacher via email</h1>", unsafe_allow_html=True)
st.title('Send attendance record excel file to a teacher via email')
if st.button('Send Attendance Report'):
    run_attendance_email("attendance_report.xlsx")
    st.markdown("""
        <div style='background-color: #F63366; color: white; padding: 10px; border-radius: 5px; text-align: center;'>
            Attendance Report Sent Successfully
        </div>
    """, unsafe_allow_html=True)

hide_st_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            header {visibility: hidden;}
            </style>
            """
st.markdown(hide_st_style, unsafe_allow_html=True)