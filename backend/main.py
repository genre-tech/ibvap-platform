import cv2
import asyncio
import time
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from engine import SurveillanceEngine

app = FastAPI(title="IBVAP API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# For demo purposes, using a public sample RTSP or the user's local one
CAMERA_URL = "rtsp://admin:dewas@321@192.168.29.104:554/Streaming/Channels/101"
engine = SurveillanceEngine(CAMERA_URL)

@app.on_event("startup")
async def startup_event():
    engine.start()

@app.on_event("shutdown")
async def shutdown_event():
    engine.stop()

def generate_frames():
    while True:
        frame = engine.latest_annotated_frame
        if frame is None:
            time.sleep(0.1)
            continue
            
        # Encode to JPEG
        ret, buffer = cv2.imencode('.jpg', frame)
        if not ret:
            continue
            
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        
        # Adjust frame rate of streaming
        time.sleep(0.05)


@app.get("/api/video_feed")
async def video_feed():
    """Video streaming route. Put this in the src attribute of an img tag."""
    return StreamingResponse(generate_frames(), media_type="multipart/x-mixed-replace; boundary=frame")

@app.websocket("/ws/alerts")
async def websocket_alerts(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            # Wait for new alerts from the engine
            alert = await engine.alert_queue.get()
            await websocket.send_json(alert)
    except WebSocketDisconnect:
        print("Client disconnected from alerts WebSocket.")
