import sqlite3
import pandas as pd
import streamlit as st

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

    conn3 = sqlite3.connect('attendance.sqlite')
    c3 = conn3.cursor()
    c3.execute('''CREATE TABLE IF NOT EXISTS attendance
                (date TEXT, subject TEXT, id_number INTEGER, time TEXT, name TEXT, branch_name TEXT, designation TEXT,
                PRIMARY KEY (date, subject, id_number))''')
    conn3.commit()
    conn3.close()

create_database()

# Check if data exists in name.sqlite file
conn2 = sqlite3.connect("name.sqlite")
c2 = conn2.cursor()
c2.execute("SELECT COUNT(*) FROM id_info")
result = c2.fetchone()[0]
conn2.close()

if result == 0:
    st.error("Please add data in the name.sqlite file")
else:
    # Convert name.sqlite data to a pandas dataframe
    conn2 = sqlite3.connect("name.sqlite")
    df = pd.read_sql_query("SELECT * FROM id_info", conn2)
    conn2.close()

    # Display dataframe in a streamlit table
    st.table(df)

    # Visualize unique id numbers and corresponding name and email
    unique_ids = df['id_number'].unique()
    st.write("Unique ID Numbers:")
    st.write(unique_ids)
    st.write("Corresponding Names and Emails:")
    st.write(df[df['id_number'].isin(unique_ids)][['id_number', 'name', 'email']])

    # Take input from user (comma-separated id numbers to delete)
    delete_ids_input = st.text_input("Enter ID number(s) to delete (comma-separated)", "")

    # Delete id numbers and corresponding records from name.sqlite and attendance.sqlite
    if delete_ids_input:
        delete_ids = [int(id.strip()) for id in delete_ids_input.split(",")]

        # Function to delete the records
        def delete_records():
            conn2 = sqlite3.connect("name.sqlite")
            c2 = conn2.cursor()
            c2.execute("DELETE FROM id_info WHERE id_number IN ({})".format(",".join("?" * len(delete_ids))), delete_ids)
            conn2.commit()
            conn2.close()

            conn1 = sqlite3.connect("database.sqlite")
            c1 = conn1.cursor()
            c1.execute("DELETE FROM attendance_records WHERE id_number IN ({})".format(",".join("?" * len(delete_ids))), delete_ids)
            conn1.commit()
            conn1.close()

            conn3 = sqlite3.connect("attendance.sqlite")
            c3 = conn3.cursor()
            c3.execute("DELETE FROM attendance WHERE id_number IN ({})".format(",".join("?" * len(delete_ids))), delete_ids)
            conn3.commit()
            conn3.close()

            st.success("Deleted ID number(s) and corresponding records successfully.")

        # Button to trigger the deletion process
        if st.button("Delete Records"):
            delete_records()
