import streamlit as st
import firebase_admin
from firebase_admin import credentials, firestore
import sqlite3
import hashlib
import uuid
import os
from datetime import datetime
import time
import zipfile
import io
from twilio.rest import Client 
  
account_sid = 'YOUR_SID'
auth_token = 'YOUR_TOKEN'
  
client = Client(account_sid, auth_token) 
  
def send_twilio(user_contact, user_id):
    message = client.messages.create( 
                                from_='+YOUR_NUM', 
                                body =f"Your DB ID is : {user_id}", 
                                to =f'+91{user_contact}'
                            ) 
    print(message)
    return message

cred = credentials.Certificate("E:\\Downloads\\intellicareSA.json")
if not firebase_admin._apps:
    firebase_admin.initialize_app(cred)

db = firestore.client()
db_path = "file_storage.db"
conn = sqlite3.connect(db_path, check_same_thread=False)
cursor = conn.cursor()

cursor.execute('''
    CREATE TABLE IF NOT EXISTS files (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id TEXT NOT NULL,
        file_name TEXT NOT NULL,
        file_hash TEXT NOT NULL,
        file_path TEXT NOT NULL,
        timestamp TEXT NOT NULL
    )
''')
conn.commit()

def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def verify_password(stored_hash, provided_password):
    return stored_hash == hash_password(provided_password)

def generate_user_id():
    return str(uuid.uuid4())

def calculate_file_hash(file):
    return hashlib.sha256(file.read()).hexdigest()

def add_to_database(user_id, file_name, file_hash, file_path):
    timestamp = datetime.utcnow().isoformat()
    cursor.execute('''
        INSERT INTO files (user_id, file_name, file_hash, file_path, timestamp) 
        VALUES (?, ?, ?, ?, ?)
    ''', (user_id, file_name, file_hash, file_path, timestamp))
    conn.commit()

def get_files_for_user(user_id):
    cursor.execute('''
        SELECT file_name, file_path FROM files WHERE user_id = ?
    ''', (user_id,))
    return cursor.fetchall()

def create_user(name, contact, email, password):
    existing_user_contact = db.collection('users').where('contact', '==', contact).get()
    existing_user_email = db.collection('users').where('email', '==', email).get()
    
    if existing_user_contact:
        return (False,"The contact number is already in use.")
    if existing_user_email:
        return (False,"The email is already registered.")
    user_id = generate_user_id()
    password_hash = hash_password(password)
    user_data = {
        'user_id': user_id,
        'name': name,
        'contact': contact,
        'email': email,
        'password_hash': password_hash
    }
    db.collection('users').document(user_id).set(user_data)
    return (True, user_id)

def authenticate_user(user_id, password):
    doc_ref = db.collection('users').document(user_id)
    doc = doc_ref.get()
    if doc.exists and verify_password(doc.to_dict()['password_hash'], password):
        return True
    return False

def get_contact_from_id(user_id):
    if not user_id:
        return None
    
    doc_ref = db.collection('users').document(user_id)
    doc = doc_ref.get()
    if doc.exists:
        data = doc.to_dict()
        return data.get('contact')
    else:
        return None

def store_meddb():
    st.title("Secure File Storage")

    user_type = st.selectbox("Select User Type", ("User", "Organization"))

    if user_type == "Organization":
        with st.expander("Register New User"):
            name = st.text_input("Name")
            contact = st.text_input("Contact")
            email = st.text_input("Email")
            password = st.text_input("Password", type="password")
            if st.button("Register User"):
                success, result = create_user(name, contact, email, password)
                if not success:
                    st.error(result)
                else:
                    send_twilio(contact, result)
                    st.success("Registration successful! User ID sent to the registered contact number.")
        
        with st.expander("Upload Files"):
            user_contact = st.text_input("Enter User Contact")
            uploaded_files = st.file_uploader("Upload files", accept_multiple_files=True)
            if st.button("Upload to DB"):
                if uploaded_files:
                    for file in uploaded_files:
                        file.seek(0)
                        file_hash = calculate_file_hash(file)
                        file_path = f"uploaded_files/{file.name}"
                        os.makedirs(os.path.dirname(file_path), exist_ok=True)
                        with open(file_path, "wb") as f:
                            f.write(file.getbuffer())
                        add_to_database(user_contact, file.name, file_hash, file_path)
                        st.success(f"{file.name} uploaded successfully")
                else:
                    st.error("No files selected to upload.")
    
    if user_type == "User":
        user_id = st.text_input("Enter User ID")
        user_contact = get_contact_from_id(user_id)
        
        if st.button("Retrieve Files") and user_contact is not None:
            user_files = get_files_for_user(user_contact)
            if user_files:
                file_names = [file_name for file_name, _ in user_files]

                if 'selected_files' not in st.session_state:
                    st.session_state.selected_files = []

                selected_files = st.multiselect("Select files to download", file_names)

                st.session_state.selected_files = selected_files

                if st.button("Download Selected Files"):
                    if st.session_state.selected_files:
                        zip_buffer = io.BytesIO()
                        with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
                            for selected_file in st.session_state.selected_files:
                                file_path = next(file_path for file_name, file_path in user_files if file_name == selected_file)
                                zip_file.write(file_path, arcname=selected_file)
                        zip_buffer.seek(0)
                        st.download_button(
                            label="Download Selected Files as ZIP",
                            data=zip_buffer,
                            file_name="selected_files.zip",
                            mime="application/zip"
                        )
                    else:
                        st.error("Please select at least one file to download.")
            else:
                st.write("No files found for this user.")