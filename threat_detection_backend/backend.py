# FastAPI Threat Detection Backend
# File: main.py

from fastapi import FastAPI, UploadFile, File, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from starlette.responses import JSONResponse
import os, shutil, cv2, uuid, io
from datetime import datetime
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import List
from threat_detection_model_backend import greedy_caption
import albumentations as A
import json
from PIL import Image
import base64

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = "./threat_detection_system/uploads"
SCREENSHOT_DIR = "./threat_detection_system/screenshots"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(SCREENSHOT_DIR, exist_ok=True)

THREAT_WORDS    = {"shoot", "gun","knife","fight","fire","accident","kidnap","kill"}

augment = A.Compose([
    A.Resize(224, 224),
])

# === Dummy threat detection ===
def detect_threats(frame):
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = augment(image=frame)['image']
    frame = frame / 255.0
    caption = greedy_caption(frame)
    threat = ''
    for word in THREAT_WORDS:
        if word in caption.lower():
            print("🚨 ALERT:", word)
            threat = word
    print(caption)
    return threat, caption

# === Email Notification (Gmail SMTP) ===
def send_email_alert(subject: str, body: str):
    sender = "your_email@gmail.com"
    password = "your_app_password"
    receiver = "recipient_email@gmail.com"

    msg = MIMEMultipart()
    msg["From"] = sender
    msg["To"] = receiver
    msg["Subject"] = subject
    msg.attach(MIMEText(body, "plain"))

    with smtplib.SMTP("smtp.gmail.com", 587) as server:
        server.starttls()
        server.login(sender, password)
        server.sendmail(sender, receiver, msg.as_string())

clients: List[WebSocket] = []

async def broadcast_detection(alert):
    for client in clients:
        try:
            await client.send_json(alert)
        except WebSocketDisconnect:
            clients.remove(client)

async def upload_video(video_path):
    detections = []
    cap = cv2.VideoCapture(video_path)
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % 30 == 0:  # Process 1 frame/sec (assuming ~30fps)
            threat, caption  = detect_threats(frame)
            if threat:
                timestamp = datetime.now().isoformat()
                image_name = f"{uuid.uuid4()}.jpg"
                image_path = os.path.join(SCREENSHOT_DIR, image_name)
                cv2.putText(frame, caption, (10,30), cv2.FONT_HERSHEY_SIMPLEX, 
                    0.7, (0,255,0), 2, cv2.LINE_AA)
                cv2.imshow("Threat Captioning", frame)
                cv2.imwrite(image_path, frame)
                screenshot_url = f"http://localhost:8000/screenshots/{image_name}"

                img = Image.open(image_path)

                # Convert image to bytes
                img_byte_arr = io.BytesIO()
                img.save(img_byte_arr, format='JPEG') # or 'JPEG'
                img_bytes = img_byte_arr.getvalue()

                base64_encoded_image = base64.b64encode(img_bytes).decode('utf-8')

                alert = {
                    "done": False,
                    "threat": f"{threat.title()} detected",
                    "timestamp": timestamp,
                    "screenshot_url": base64_encoded_image
                }

                detections.append(alert)
                await broadcast_detection(alert)
                # Send email
                # send_email_alert(alert["threat"], f"Time: {timestamp}\nScreenshot: {screenshot_url}")

        frame_count += 1

    cap.release()
    alert = {
        "done": True
    }
    await broadcast_detection(alert)
    # return JSONResponse(content={"detections": detections})

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    clients.append(websocket)
    try:
        received_data = b""
        while True:
            data = await websocket.receive_bytes()
            received_data += data

            filename = f"{uuid.uuid4()}.mp4"
            video_path = os.path.join(UPLOAD_DIR, filename)
            with open(video_path, 'wb') as f:
                f.write(received_data)
                
            await upload_video(video_path)
    except WebSocketDisconnect:
        clients.remove(websocket)

# === API Endpoint ===
# @app.post("/upload")


# Serve screenshots statically
from fastapi.staticfiles import StaticFiles
app.mount("/screenshots", StaticFiles(directory=SCREENSHOT_DIR), name="screenshots")
