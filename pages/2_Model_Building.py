import sqlite3
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import os
import numpy as np
import pickle

from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler

# --------------------------------------------------
# Page Title
# --------------------------------------------------
st.markdown("<h1 style='text-align: center;'>Train Model</h1>", unsafe_allow_html=True)

# --------------------------------------------------
# Database Setup
# --------------------------------------------------
def create_database():
    conn = sqlite3.connect("database.sqlite")
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS attendance_records
        (face_data TEXT, name TEXT, id_number INTEGER, branch_name TEXT, designation TEXT)
    ''')
    conn.commit()
    conn.close()

create_database()

# --------------------------------------------------
# Load Data
# --------------------------------------------------
def load_data():
    try:
        conn = sqlite3.connect("database.sqlite")
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM attendance_records")
        count = cursor.fetchone()[0]
        conn.close()

        if count == 0:
            st.error("❌ No face data found. Please register faces first.")
            return None

        conn = sqlite3.connect("database.sqlite")
        df = pd.read_sql_query("SELECT * FROM attendance_records", conn)
        conn.close()

        if df.empty:
            st.error("❌ Failed to load face data.")
            return None

        st.success(f"✅ Loaded {len(df)} face records")
        return df

    except Exception as e:
        st.error(f"❌ Error loading data: {e}")
        return None


df = load_data()
if df is None:
    st.stop()

# --------------------------------------------------
# Process face data
# --------------------------------------------------
try:
    df["face_data"] = df["face_data"].apply(eval)
    df = df.sample(frac=1).reset_index(drop=True)
except Exception as e:
    st.error(f"❌ Error processing face data: {e}")
    st.stop()

# --------------------------------------------------
# Display Dataset
# --------------------------------------------------
st.markdown("## Registered Face Dataset")
st.dataframe(df.head(20))

# --------------------------------------------------
# Data Distribution Plots
# --------------------------------------------------
id_counts = df['id_number'].value_counts().reset_index()
id_counts.columns = ['id_number', 'count']

fig = go.Figure([go.Bar(x=id_counts['id_number'], y=id_counts['count'])])
fig.update_layout(title='Distribution of ID Numbers', xaxis_title='ID', yaxis_title='Count')
st.plotly_chart(fig)

name_counts = df['name'].value_counts().reset_index()
name_counts.columns = ['name', 'count']

fig = go.Figure([go.Bar(x=name_counts['name'], y=name_counts['count'])])
fig.update_layout(title='Distribution of Names', xaxis_title='Name', yaxis_title='Count')
st.plotly_chart(fig)

# --------------------------------------------------
# Overview
# --------------------------------------------------
st.markdown("## Dataset Overview")
st.dataframe(df.groupby(['id_number', 'name']).count())

# --------------------------------------------------
# Optional Data Cleaning
# --------------------------------------------------
st.markdown("## Clean Data")
choice = st.selectbox("Delete records by ID?", ("No", "Yes"))

if choice == "Yes":
    selected_ids = st.text_input("Enter ID numbers (comma separated)")
    for sid in selected_ids.split(","):
        sid = sid.strip()
        if sid.isdigit():
            sid = int(sid)
            if sid in df['id_number'].values:
                df = df[df['id_number'] != sid]
                st.success(f"Deleted records for ID {sid}")
            else:
                st.warning(f"ID {sid} not found")
    st.dataframe(df.groupby(['id_number', 'name']).count())
else:
    st.info("No rows deleted")

# --------------------------------------------------
# Train Model
# --------------------------------------------------
def run_model():
    # Remove classes with < 2 samples
    counts = df['id_number'].value_counts()
    valid_ids = counts[counts >= 2].index
    df_filtered = df[df['id_number'].isin(valid_ids)]

    if df_filtered['id_number'].nunique() < 2:
        st.error("❌ Need at least 2 unique IDs with 2+ samples each.")
        return

    # Prepare data
    X = np.stack(df_filtered["face_data"].values)
    y = df_filtered["id_number"].values

    # Flatten
    X = X.reshape(X.shape[0], -1)

    # Train-test split (STRATIFIED)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # Double safety check
    if len(np.unique(y_train)) < 2:
        st.error("❌ Training data still has only one class.")
        return

    # Scale
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Train SVM
    clf = svm.SVC(kernel='poly', degree=3)
    clf.fit(X_train, y_train)

    # Predict
    y_pred = clf.predict(X_test)

    # Metrics
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)

    st.markdown("## Model Performance")
    st.dataframe(pd.DataFrame(report).transpose())
    st.write("🎯 Accuracy:", acc)

    # Save model
    pickle.dump((clf, scaler), open("svm_model.pkl", "wb"))
    st.success("✅ Model trained and saved successfully")


# --------------------------------------------------
# Run Button
# --------------------------------------------------
if st.button("Run Model"):
    if df['id_number'].nunique() < 2:
        st.error("❌ At least two unique IDs are required to train the model.")
    else:
        run_model()

# --------------------------------------------------
# Hide Streamlit UI
# --------------------------------------------------
st.markdown("""
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)
