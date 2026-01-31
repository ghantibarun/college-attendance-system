import sqlite3
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import os
import csv
import numpy as np
try:
    os.remove("database.csv")
except:
    pass
st.markdown("<h1 style='text-align: center;'>Train Model</h1>", unsafe_allow_html=True)

if not os.path.exists("database.sqlite"):
    st.error("First Add data to database")

def create_database():
    conn1 = sqlite3.connect("database.sqlite")
    c1 = conn1.cursor()
    c1.execute('''CREATE TABLE IF NOT EXISTS attendance_records
                (face_data TEXT, name TEXT, id_number INTEGER, branch_name TEXT, designation TEXT)''')
    conn1.commit()
    conn1.close()
create_database()

st.session_state.setdefault('visibility', True)
label_visibility = st.session_state.visibility

def generate_csv():
    conn = sqlite3.connect("database.sqlite")
    df = pd.read_sql_query("SELECT * from attendance_records", conn)
    conn.close()
    df.to_csv("database.csv", index=False)


    with open("database.csv", mode='a', newline='') as file1:
        writer = csv.writer(file1)
        with open('unknown.csv', mode='r') as file2:
            reader = csv.reader(file2)
            next(reader)
            for row in reader:
                writer.writerow(row)

    print('Data appended successfully!')
generate_csv()
df=pd.read_csv("database.csv")
store_df=df
df["face_data"] = df["face_data"].apply(eval)
df = df.sample(frac=1).reset_index(drop=True)

# Display the dataset
st.markdown("## This is the dataset of the registered faces")
st.dataframe(df.head())



# Extract the unique values in the 'id_number' column
unique_ids = df['id_number'].unique()

# Create a new DataFrame that counts the number of occurrences of each unique id_number
id_counts = pd.DataFrame({'id_number': unique_ids, 'count': df['id_number'].value_counts()})

# Create a histogram of the id_number counts
fig = go.Figure(data=[go.Bar(x=id_counts['id_number'], y=id_counts['count'])])
fig.update_layout(title='Distribution of Unique ID Numbers', xaxis_title='ID Number', yaxis_title='Count')
st.plotly_chart(fig)


# Extract the unique values in the 'name' column
unique_names = df['name'].unique()

# Create a new DataFrame that counts the number of occurrences of each unique name
name_counts = pd.DataFrame({'name': unique_names, 'count': df['name'].value_counts()})

# Create a histogram of the name counts
fig = go.Figure(data=[go.Bar(x=name_counts['name'], y=name_counts['count'])])
fig.update_layout(title='Distribution of Unique Names', xaxis_title='Name', yaxis_title='Count')
st.plotly_chart(fig)




# Display an overview of the dataset
st.markdown("## Overview of the dataset")
grouped_data = df.groupby(['id_number', 'name']).count()
st.dataframe(grouped_data)


your_choice = st.selectbox("Clean Data", ("No","Yes"))
if your_choice=="Yes":
    id_numbers = df['id_number'].unique()
    selected_ids = st.text_input("Enter your id_number: (use comma to delete multiple id_numbers)")
    for selected_id in selected_ids.split(","):
        selected_id=selected_id.strip()
        if selected_id != "":
            selected_id = np.int64(selected_id.strip())
        if selected_id in id_numbers:
            df = df.drop(df[df['id_number'] == selected_id].index)
            st.success("Rows with id_number {} have been deleted.".format(selected_id))
            grouped_data = df.groupby(['id_number', 'name']).count()   
        else:
            st.warning("Rows with id_number {} not found.".format(selected_id)) 
            grouped_data = df.groupby(['id_number', 'name']).count()   
    st.dataframe(grouped_data)    
else:
    st.warning("No rows have been deleted.")



from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import json
import pickle
from sklearn.metrics import classification_report
def run_model():
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(df["face_data"], df["id_number"], test_size=0.2, random_state=42)

    # Create the SVM model
    # clf = svm.SVC(kernel='linear')
    clf = svm.SVC(kernel='poly', degree=3)
    # clf = svm.SVC(kernel='rbf')
    # clf = svm.SVC(kernel='sigmoid')

    # Train the model on the training set
    clf.fit(list(X_train), list(y_train))

    # Test the model on the testing set
    y_pred = clf.predict(list(X_test))

    # Calculate the accuracy of the model
    acc = accuracy_score(list(y_test), y_pred)
    #display accuracy
    
    report_dict = classification_report(list(y_test), y_pred, output_dict=True)
    report_df = pd.DataFrame(report_dict).transpose()
    st.write("Classification Report:")
    st.dataframe(report_df)
    st.write("Accuracy: ",acc)
    pickle.dump(clf, open('svm_model.pkl', 'wb'))
    st.caption('Model saved as :blue[model.pkl] file')
if st.button('Run Model'):
    unique_ids = df['id_number'].unique()
    if len(unique_ids) < 2:
        st.error("Insufficient unique IDs to run the SVM multiclass model. At least two unique IDs are required.")
    else:
        run_model()


hide_st_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            header {visibility: hidden;}
            </style>
            """
st.markdown(hide_st_style, unsafe_allow_html=True)
