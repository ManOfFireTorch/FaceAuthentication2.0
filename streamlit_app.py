import streamlit as st
from pymongo import MongoClient
from facenet_pytorch import InceptionResnetV1
import torch
from torchvision import transforms
from PIL import Image
import numpy as np
import base64
import io
import cv2
from datetime import datetime

MONGO_URI = "mongodb+srv://danielqiqiliu:bAYZV5B4njE6ehBT@faceauth.rq2fk.mongodb.net/?retryWrites=true&w=majority&appName=FaceAuth"
client = MongoClient(MONGO_URI)
db = client.FaceRecognitionApp
users_collection = db.Users

face_detector = cv2.dnn.readNetFromCaffe(
    "deploy.prototxt",
    "res10_300x300_ssd_iter_140000.caffemodel"
)

def detect_face(image):
    np_image = np.array(image.convert("RGB"))
    h, w = np_image.shape[:2]
    blob = cv2.dnn.blobFromImage(np_image, scalefactor=1.0, size=(300, 300), mean=(104.0, 177.0, 123.0))
    face_detector.setInput(blob)
    detections = face_detector.forward()
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            x1, y1, x2, y2 = box.astype("int")
            aligned_face = np_image[y1:y2, x1:x2]
            return aligned_face
    return None

def extract_features(image, model):
    face = detect_face(image)
    if face is None:
        return None
    target_size = (160, 160)
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(target_size),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])
    face_tensor = transform(face).unsqueeze(0)
    with torch.no_grad():
        embedding = model(face_tensor)
    return embedding

def compare_faces(embedding1, embedding2, threshold=0.6):
    similarity = torch.nn.functional.cosine_similarity(embedding1, embedding2)
    return similarity.item() > threshold

def save_user(username, image):
    buffered = io.BytesIO()
    image.save(buffered, format="JPEG")
    encoded_image = base64.b64encode(buffered.getvalue()).decode("utf-8")
    users_collection.insert_one({
        "username": username,
        "image": encoded_image,
        "created_at": datetime.utcnow()
    })

def load_user(username):
    user = users_collection.find_one({"username": username})
    if user:
        image_data = base64.b64decode(user["image"])
        return Image.open(io.BytesIO(image_data))
    return None

option = st.radio("Are you a new or existing user?", ("New User", "Existing User"))

if option == "New User":
    username = st.text_input("Create a Username:")
    if username:
        if users_collection.find_one({"username": username}):
            st.warning("Username already exists. Please choose a different one.")
        else:
            captured_image = st.camera_input("Take your photo")
            if captured_image:
                image = Image.open(captured_image)
                save_user(username, image)
                st.success(f"Account created for {username}!")

elif option == "Existing User":
    username = st.text_input("Enter your Username:")
    if username:
        stored_image = load_user(username)
        if not stored_image:
            st.warning("Username not found. Please check your input.")
        else:
            captured_image = st.camera_input("Please take a photo when ready")
            if captured_image:
                new_image = Image.open(captured_image)
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                recognition_model = InceptionResnetV1(pretrained='vggface2').to(device).eval()
                stored_embedding = extract_features(stored_image, recognition_model)
                new_embedding = extract_features(new_image, recognition_model)
                if stored_embedding is not None and new_embedding is not None and compare_faces(stored_embedding, new_embedding):
                    st.success("Authentication successful!")
                else:
                    st.error("Authentication failed. Please try again.")
