# Step => 1 Importing required libraries
import streamlit as st
import cv2
import requests
import numpy as np
from PIL import Image

# Step => 2 Function to send image to backend
def analyze_emotion(frame, employee_id):
    """Sends the image frame to backend API for emotion detection."""
    _, img_encoded = cv2.imencode('.jpg', frame)
    response = requests.post("http://127.0.0.1:5000/detect", 
                             files={"image": img_encoded.tobytes()},
                             data={"employee_id": employee_id})
    if response.status_code == 200:
        return response.json()
    else:
        return None

# Step => 3 Streamlit UI Setup
st.title("AI POWERED TASK OPTIMIZATION")
st.write("Click 'Start' to begin emotion detection from your webcam.")

# Step => 4 User input for Employee ID
employee_id = st.text_input("Enter Employee ID", "EMP001")

# Step => 5 Creating buttons to start/stop the camera
start_cam = st.button("Start Camera")
stop_cam = st.button("Stop Camera")
if start_cam:
    cap = cv2.VideoCapture(0)
    frame_display = st.empty()
    while cap.isOpened():
        ret, video_frame = cap.read()
        if not ret:
            st.error("Failed to capture frame")
            break
        result = analyze_emotion(video_frame, employee_id)
        if result:
            for emotion_data in result:
                x, y, w, h = emotion_data["coordinates"]
                dominant_emotion = emotion_data["emotion"]
                task_suggestion = emotion_data.get("task_suggestion", "No suggestion available.")
                cv2.rectangle(video_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(video_frame, f"{dominant_emotion}", (x, y - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                st.write(f"Detected Emotion: {dominant_emotion}")
                st.write(f"Task Suggestion: {task_suggestion}")
        video_frame = cv2.cvtColor(video_frame, cv2.COLOR_BGR2RGB)
        frame_display.image(video_frame, channels="RGB")
        if stop_cam:
            break    
    cap.release()
    st.write("Camera stopped.")
