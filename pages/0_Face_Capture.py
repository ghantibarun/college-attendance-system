import cv2
import numpy as np
from deepface import DeepFace
import sqlite3
import json
import streamlit as st
from PIL import Image
import os

def deep_data_extract(img):
    """Extract face embeddings from image using DeepFace"""
    embedding = None
    faces = []
    facial_data = []
    
    print(f"\n[DEBUG] ===== FACE DETECTION START =====")
    print(f"[DEBUG] Image shape: {img.shape}")
    print(f"[DEBUG] Image dtype: {img.dtype}")
    print(f"[DEBUG] Image min/max values: {img.min()}/{img.max()}")
    
    try:
        # Try with different backends for more robust detection
        backends = ['opencv', 'ssd', 'mtcnn', 'retinaface']
        
        for backend in backends:
            try:
                print(f"\n[DEBUG] Attempting backend: {backend}")
                embedding = DeepFace.represent(img, model_name='Facenet', detector_backend=backend, enforce_detection=False)
                
                if embedding and len(embedding) > 0:
                    print(f"[DEBUG] ✅ SUCCESS with {backend}!")
                    print(f"[DEBUG] Found {len(embedding)} face(s)")
                    break
                else:
                    print(f"[DEBUG] Backend {backend} returned empty result")
            except Exception as backend_error:
                error_msg = str(backend_error)[:150]
                print(f"[DEBUG] ❌ Backend {backend} failed: {error_msg}")
                continue
        
        if embedding and len(embedding) > 0:
            print(f"\n[DEBUG] Processing {len(embedding)} detected face(s)...")
            for i in range(len(embedding)):
                try:
                    x = embedding[i]['facial_area']['x']
                    y = embedding[i]['facial_area']['y']
                    w = embedding[i]['facial_area']['w']
                    h = embedding[i]['facial_area']['h']
                    x1, y1, x2, y2 = x, y, x+w, y+h
                    faces.append((x1, y1, x2, y2))
                    
                    emb = embedding[i]['embedding']
                    facial_data.append(emb)
                    print(f"[DEBUG] Face {i}: coords=({x1},{y1},{x2},{y2}), embedding_length={len(emb)}")
                except Exception as e:
                    print(f"[DEBUG] ❌ Error processing face {i}: {e}")
        else:
            print(f"\n[DEBUG] ❌ No embedding returned from any backend")
            
    except Exception as e:
        print(f"\n[DEBUG] ❌ CRITICAL ERROR in face detection: {e}")
        import traceback
        traceback.print_exc()
    
    print(f"[DEBUG] ===== FACE DETECTION END =====")
    print(f"[DEBUG] FINAL RESULT: {len(faces)} faces, {len(facial_data)} embeddings\n")
    
    return faces, facial_data


def get_registered_users():
    """Get list of all registered users from name.sqlite"""
    try:
        conn = sqlite3.connect('name.sqlite')
        c = conn.cursor()
        c.execute('''SELECT id_number, name FROM id_info ORDER BY name''')
        users = c.fetchall()
        conn.close()
        return users
    except Exception as e:
        st.error(f"Error fetching users: {e}")
        return []


def update_face_data(id_number, face_data_json):
    """Insert or update face data for a user"""
    try:
        conn = sqlite3.connect('database.sqlite')
        c = conn.cursor()
        
        # Check if user already has face data
        c.execute('SELECT * FROM attendance_records WHERE id_number = ?', (id_number,))
        existing = c.fetchone()
        
        if existing:
            # Update existing record
            c.execute('''UPDATE attendance_records SET face_data = ? WHERE id_number = ?''',
                     (face_data_json, id_number))
            conn.commit()
            conn.close()
            return True, "Face data updated successfully!"
        else:
            # Insert new record
            c.execute('''INSERT INTO attendance_records (id_number, face_data) VALUES (?, ?)''',
                     (id_number, face_data_json))
            conn.commit()
            conn.close()
            return True, "Face data saved successfully!"
            
    except Exception as e:
        return False, f"Error saving face data: {e}"


def convert_string(face_data):
    """Convert face embeddings to JSON string"""
    try:
        if isinstance(face_data, list) and len(face_data) > 0:
            # Convert numpy arrays to lists for JSON serialization
            face_data_list = []
            for item in face_data:
                if isinstance(item, np.ndarray):
                    face_data_list.append(item.tolist())
                else:
                    face_data_list.append(item)
            return json.dumps(face_data_list)
        return None
    except Exception as e:
        print(f"Error converting face data: {e}")
        return None


def main():
    st.set_page_config(page_title="Face Capture", layout="wide")
    st.title("📸 Face Data Capture/Re-capture")
    
    st.markdown("""
    **Use this page to:**
    - Capture face data for registered users who don't have it yet
    - Re-capture/update face data for existing users
    
    **Instructions:**
    1. Select a registered user from the dropdown
    2. Click the camera button to take a photo
    3. Make sure your face is clearly visible and centered
    4. The system will extract and save your face data
    """)
    
    # Get registered users
    users = get_registered_users()
    
    if not users:
        st.error("❌ No registered users found! Please register users first in the Registration page.")
        return
    
    # Create user selection dropdown
    user_options = [f"{user[1]} (ID: {user[0]})" for user in users]
    selected_user = st.selectbox("Select a registered user:", user_options)
    
    # Extract ID number from selection
    selected_id = selected_user.split("ID: ")[1].rstrip(")")
    
    st.divider()
    st.subheader(f"Capturing face data for: {selected_user}")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("### 📷 Take Photo")
        st.markdown("Click below to capture a photo using your camera")
        picture = st.camera_input("Take a picture", key=f"camera_{selected_id}")
        
        if picture is not None:
            # Convert to PIL Image then to OpenCV format
            img_pil = Image.open(picture)
            img_rgb = np.array(img_pil)
            img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
            
            # Save to temp folder
            os.makedirs('temp', exist_ok=True)
            temp_filename = f'temp/capture_{selected_id}_{int(np.random.random() * 10000)}.jpg'
            cv2.imwrite(temp_filename, img_bgr)
            
            st.success(f"✅ Photo captured and saved")
    
    with col2:
        st.markdown("### 🎯 Photo Preview")
        if picture is not None:
            st.image(picture, caption="Your captured photo", use_column_width=True)
    
    # Process captured image
    if picture is not None:
        st.divider()
        st.subheader("Processing Face Data...")
        
        with st.spinner("Extracting face embeddings..."):
            print(f"\n{'='*60}")
            print(f"Processing face capture for user ID: {selected_id}")
            print(f"{'='*60}")
            
            # Extract faces
            faces, facial_data = deep_data_extract(img_bgr)
            
            if len(faces) > 0 and len(facial_data) > 0:
                st.success(f"✅ Successfully detected {len(faces)} face(s)")
                print(f"✅ Face detection successful: {len(faces)} face(s) detected")
                
                # Convert to JSON and save
                face_data_json = convert_string(facial_data)
                
                if face_data_json:
                    success, message = update_face_data(selected_id, face_data_json)
                    
                    if success:
                        st.success(f"✅ {message}")
                        print(f"✅ {message}")
                        
                        # Show statistics
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Faces Detected", len(faces))
                        with col2:
                            st.metric("Embeddings Extracted", len(facial_data))
                        with col3:
                            st.metric("Embedding Dimension", len(facial_data[0]) if facial_data else 0)
                    else:
                        st.error(f"❌ {message}")
                        print(f"❌ {message}")
                else:
                    st.error("❌ Failed to convert face data to JSON format")
                    print("❌ Failed to convert face data")
            else:
                st.error("❌ No faces detected in the photo")
                st.warning("Please try again with:")
                st.markdown("""
                - **Good lighting** - Face should be well-lit
                - **Face centered** - Your face should be in the center of the frame
                - **Straight angle** - Look directly at the camera
                - **Clear view** - No obstructions or filters
                """)
                print(f"❌ Face detection failed: {len(faces)} faces detected")
    
    # Show status of all users
    st.divider()
    st.subheader("📊 User Face Data Status")
    
    try:
        conn = sqlite3.connect('database.sqlite')
        c = conn.cursor()
        c.execute('SELECT DISTINCT id_number FROM attendance_records WHERE face_data IS NOT NULL')
        users_with_faces = set([row[0] for row in c.fetchall()])
        conn.close()
        
        status_data = []
        for user_id, user_name in users:
            has_face = "✅ YES" if user_id in users_with_faces else "❌ NO"
            status_data.append({"User ID": user_id, "Name": user_name, "Face Data": has_face})
        
        st.dataframe(status_data, use_container_width=True)
        
        # Summary
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Users", len(users))
        with col2:
            st.metric("Users with Face Data", len(users_with_faces))
        with col3:
            st.metric("Users Pending Face Capture", len(users) - len(users_with_faces))
            
    except Exception as e:
        st.error(f"Error fetching status: {e}")


if __name__ == "__main__":
    main()
