# Step => 1 Importing required libraries
import cv2
import numpy as np
from fer import FER
import smtplib
from email.mime.text import MIMEText
from flask import Flask, request, jsonify
import pandas as pd
import os
from datetime import datetime

# Step => 2 Load dataset
DATASET_PATH = "./dataset/dataset.csv"
if not os.path.exists(DATASET_PATH):
    print("Dataset file not found. Please generate dataset.csv first.")
    exit()
df = pd.read_csv(DATASET_PATH)
task_suggestions = {
    "happy": "Continue current work or collaborate on new ideas.",
    "sad": "Take a short break or engage in light tasks.",
    "angry": "Pause and do a relaxation exercise.",
    "neutral": "Maintain workflow or take a short check-in.",
    "excited": "Take on a challenging task or brainstorm ideas.",
    "stressed": "Prioritize tasks and seek assistance if needed."
}

# Step => 3 Initializing Flask app and emotion detector
app = Flask(__name__)
emotion_analyzer = FER(mtcnn=True)

# Step => 4 Defining function to send email notifications
def notify_manager(employee_id, stress_emotion):
    """Send email alert if stress is detected."""
    sender_email = "sender@example.com"
    receiver_email = "receiver@example.com"
    subject = "Employee Stress Alert"
    body = f"Employee {employee_id} is experiencing {stress_emotion}. Please consider reducing workload."
    msg = MIMEText(body)
    msg['Subject'] = subject
    msg['From'] = sender_email
    msg['To'] = receiver_email
    try:
        server = smtplib.SMTP('smtp.example.com', 587)
        server.starttls()
        server.login("your_email@example.com", "your_password")
        server.sendmail(sender_email, receiver_email, msg.as_string())
        server.quit()
        print("Notification sent successfully!")
    except Exception as e:
        print(f"Failed to send notification: {e}")

# Step => 5 Function to update dataset
def update_dataset(employee_id, emotion):
    """Update dataset with new emotion record."""
    global df
    timestamp = datetime.now()
    task_suggestion = task_suggestions.get(emotion, "No task suggestion available.")
    stress_alert = "Yes" if emotion in ["stressed", "angry", "sad"] else "No"
    new_entry = pd.DataFrame([[employee_id, timestamp, emotion, task_suggestion, stress_alert]],
                              columns=["Employee_ID", "Timestamp", "Emotion", "Task_Suggestion", "Stress_Alert"])
    df = pd.concat([df, new_entry], ignore_index=True)
    df.to_csv(DATASET_PATH, index=False)

# Step => 6 Defining API route to detect emotions
@app.route('/detect', methods=['POST'])
def detect_emotion():
    """API Endpoint to detect emotions from image."""
    file = request.files.get('image')
    employee_id = request.form.get('employee_id', "Unknown")
    if not file:
        return jsonify({"error": "No file uploaded"}), 400
    file_bytes = np.frombuffer(file.read(), np.uint8)
    frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    detected_emotions = emotion_analyzer.detect_emotions(frame_rgb)
    response_data = []
    for emotion in detected_emotions:
        x, y, w, h = emotion['box']
        dominant_feeling = max(emotion['emotions'].items(), key=lambda x: x[1])[0]
        task_suggestion = task_suggestions.get(dominant_feeling, "No task suggestion available.")
        if dominant_feeling in ["angry", "sad", "stressed"]:
            notify_manager(employee_id, dominant_feeling)
        update_dataset(employee_id, dominant_feeling)
        response_data.append({
            "emotion": dominant_feeling, 
            "coordinates": [x, y, w, h], 
            "task_suggestion": task_suggestion
        })
    return jsonify(response_data)

# Step => 7 Running the API server
if __name__ == "__main__":
    print("Starting Emotion Detection API Server...")
    app.run(host="0.0.0.0", port=5000, debug=True)
