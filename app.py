import streamlit as st
import sqlite3
import os 
import base64
st.set_page_config(page_title="This is a Multipage WebApp")
# st.title("This is the Home Page college attendance system.")
# st.sidebar.success("Select Any Page from here")

# st.sidebar.title("Navigation")



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
# check tmp folder exists or not
if not os.path.exists("temp"):
    os.mkdir("temp")




def get_img_as_base64(file):
    with open(file, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()


img = get_img_as_base64("banner.png")

page_bg_img = f"""
<style>
[data-testid="stAppViewContainer"] > .main {{
background-image: url("data:image/png;base64,{img}");
background-size: 100%;
background-position: top left;
background-repeat: no-repeat;
background-attachment: local;
}}

</style>
"""

st.markdown(page_bg_img, unsafe_allow_html=True)


hide_st_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            header {visibility: hidden;}
            </style>
            """
st.markdown(hide_st_style, unsafe_allow_html=True)