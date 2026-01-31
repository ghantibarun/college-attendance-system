import sqlite3
import pandas as pd
import plotly.express as px
import streamlit as st
st.markdown("<h1 style='text-align: center;'>Analyze Attendance Records</h1>", unsafe_allow_html=True)


conn3 = sqlite3.connect('attendance.sqlite')
c3 = conn3.cursor()
c3.execute('''CREATE TABLE IF NOT EXISTS attendance
            (date TEXT, subject TEXT, id_number INTEGER, time TEXT, name TEXT, branch_name TEXT, designation TEXT,
            PRIMARY KEY (date, subject, id_number))''')
conn3.commit()
conn3.close()
# Connect to the SQLite database
conn = sqlite3.connect("attendance.sqlite")

# Load attendance data from the database into a DataFrame
df = pd.read_sql_query("SELECT * from attendance", conn)

# Close the database connection
conn.close()
df.to_csv("attendance_report.csv", index=False)
df.to_excel("attendance_report.xlsx", index=False)


# Show the attendance data in the Streamlit app
st.write("Attendance Data")
# st.write(df.head())
st.write(df)
# Get the total number of students in the attendance data
num_students = len(df['name'].unique())

# Show the total number of students in the attendance data
st.write(f"Total number of students: {num_students}")

# Get the total number of subjects in the attendance data
num_subjects = len(df['subject'].unique())

# Show the total number of subjects in the attendance data
st.write(f"Total number of subjects: {num_subjects}")

# Show the number of attendances per subject
st.write(f"The number of attendances per subject")
attendances_per_subject = df.groupby('subject')['id_number'].count().reset_index(name='Count')
st.write(attendances_per_subject)

st.write(f"The number of attendances per student")
# Show the number of attendances per student
attendances_per_student = df.groupby('name')['id_number'].count().reset_index(name='Count')
st.write(attendances_per_student)

# Show the number of attendances per branch
attendances_per_branch = df.groupby('branch_name')['id_number'].count().reset_index(name='Count')
st.write(attendances_per_branch)

# Create bar chart of attendance data by date and subject using Plotly
group_df = df.groupby(['date', 'subject']).size().reset_index(name='Count')
fig1 = px.bar(group_df, x='date', y='Count', color='subject', barmode='group', title='Attendance Count by Date and subject')
st.plotly_chart(fig1)

# Create stacked bar chart of attendance data by student and subject using Plotly
group_df2 = df.groupby(['name', 'subject']).size().reset_index(name='Count')
fig2 = px.bar(group_df2, x='name', y='Count', color='subject', title='Attendance Count by Student and subject', barmode='stack')
st.plotly_chart(fig2)

# Create line chart of attendance data over time using Plotly
group_df3 = df.groupby(['date']).size().reset_index(name='Count')
fig3 = px.line(group_df3, x='date', y='Count', title='Attendance Count over Time')
st.plotly_chart(fig3)

# Create histogram of attendance data by time of day using Plotly
df['Hour'] = pd.to_datetime(df['time']).dt.hour
fig4 = px.histogram(df, x='Hour', nbins=24, title='Attendance by Time of Day')
st.plotly_chart(fig4)

# Create bar chart of attendance data by branch using Plotly
group_df5 = df.groupby(['branch_name']).size().reset_index(name='Count')
fig5 = px.bar(group_df5, x='branch_name', y='Count', title='Attendance by Branch')
st.plotly_chart(fig5)

hide_st_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            header {visibility: hidden;}
            </style>
            """
st.markdown(hide_st_style, unsafe_allow_html=True)
