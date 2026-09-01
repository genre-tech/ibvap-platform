# IBVAP – Intelligent Border Video Analytics Platform

> Transform standard CCTV cameras into an intelligent AI-powered surveillance network — no dedicated smart hardware required.

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688?logo=fastapi)
![React](https://img.shields.io/badge/React-18+-61DAFB?logo=react)
![YOLO](https://img.shields.io/badge/YOLOv11-Ultralytics-FF6F00)
![OpenVINO](https://img.shields.io/badge/OpenVINO-Optimized-0071C5?logo=intel)
![License](https://img.shields.io/badge/License-AGPL--3.0-red)

---

## 🎯 What is IBVAP?

IBVAP is a real-time video surveillance system that processes live RTSP camera feeds using AI to perform:

- **🚶 Human Intrusion Detection** — Detects and tracks humans with bounding boxes and confidence scores
- **🚗 Vehicle Detection & Tracking** — Identifies cars, trucks, and motorcycles using ByteTrack
- **🔢 Automatic Number Plate Recognition (ANPR)** — Localizes license plates on vehicles and reads text via OCR
- **⚡ Real-Time Alerts** — Pushes detection events to a web dashboard via WebSocket in real-time
- **🌐 Cloud-Accessible Dashboard** — React-based Command Center accessible from anywhere via Tailscale VPN

---


---

## 📊 Data Flow

```
Camera (RTSP/TCP) → Frame Grabber Thread (5ms loop) → AI Engine
    → YOLO11n (detect humans + vehicles) → Draw bounding boxes
    → ANPR Model (find plates on full frame) → EasyOCR (read text)
    → FastAPI → MJPEG stream + WebSocket alerts → Browser Dashboard
```

---

## 📁 Project Structure

```
sih-survillance/
├── backend/
│   ├── main.py              # FastAPI server — video streaming, WebSocket alerts, static file serving
│   ├── engine.py            # Core AI pipeline — LiveRTSPStream, SurveillanceEngine, ANPR logic
│   └── venv/                # Python virtual environment (server-side)
│
├── frontend/
│   ├── src/
│   │   ├── App.jsx          # React dashboard — live feed viewer + real-time alert sidebar
│   │   ├── App.css          # Dashboard styling (dark theme)
│   │   ├── index.css        # Global styles
│   │   └── main.jsx         # React entry point
│   ├── dist/                # Pre-built production bundle (served by FastAPI)
│   ├── package.json         # Node.js dependencies
│   └── vite.config.js       # Vite build configuration
│
├── yolo11n.pt               # YOLOv11 Nano — general object detection weights
├── yolo11n_openvino_model/   # OpenVINO-optimized YOLOv11 (for fast CPU inference)
├── best.pt                  # Custom-trained ANPR model weights
├── anpr_best_openvino_model/ # OpenVINO-optimized ANPR model
│   ├── anpr_best.bin        # Model binary weights
│   ├── anpr_best.xml        # Model graph definition
│   └── metadata.yaml        # Model metadata (classes: plate)
│
├── camerasurvillance.py     # Original standalone surveillance script (reference)
├── tata.py                  # Basic YOLO detection script (reference)
├── anpr_surveillance.py     # Advanced ANPR pipeline script (reference)
├── deploy.sh                # Deployment helper script
├── AGENTS.md                # AI agent instructions
└── README.md                # This file
```

---

## 🚀 Quick Start

### Prerequisites

- **Python 3.10+**
- **Node.js 18+** (for building frontend)
- **An RTSP camera** on your local network
- **Tailscale** (if deploying to a remote cloud server)

### 1. Clone the Repository

```bash
git clone https://github.com/genre-tech/ibvap-platform.git
cd ibvap-platform
```

### 2. Set Up the Backend

```bash
cd backend
python -m venv venv
source venv/bin/activate        # Linux/Mac
# venv\Scripts\activate         # Windows

pip install fastapi uvicorn websockets opencv-python ultralytics easyocr aiofiles openvino
```

### 3. Configure Camera URL

Edit `backend/main.py` and update the RTSP URL:

```python
CAMERA_URL = "rtsp://username:password@CAMERA_IP:554/Streaming/Channels/101"
```

### 4. Build the Frontend (Optional — pre-built in `dist/`)

```bash
cd frontend
npm install
npm run build
```

### 5. Run the Server

```bash
cd backend
source venv/bin/activate
uvicorn main:app --host 0.0.0.0 --port 8000
```

### 6. Access the Dashboard

Open your browser and navigate to:

```
http://localhost:8000
```

---

## ☁️ Cloud Deployment (via Tailscale)

To access your local camera from a remote cloud server:

```
📹 Camera (192.168.29.104) → 💻 Windows PC (Tailscale Subnet Router) → ☁️ Cloud Server (Tailscale Client) → 🌍 Browser
```

1. **Install Tailscale** on both your local PC and cloud server
2. **Enable Subnet Routing** on the local PC:
   ```bash
   tailscale up --advertise-routes=192.168.29.0/24
   ```
3. **Approve the route** in the [Tailscale Admin Console](https://login.tailscale.com/admin/machines)
4. **SSH into cloud server** and start the backend:
   ```bash
   cd ibvap-platform/backend
   source venv/bin/activate
   nohup uvicorn main:app --host 0.0.0.0 --port 8000 > backend.log 2>&1 &
   ```
5. **Access**: `http://<tailscale-ip>:8000`

### Managing the Server

```bash
# Start
cd ibvap-platform/backend && source venv/bin/activate
nohup uvicorn main:app --host 0.0.0.0 --port 8000 > backend.log 2>&1 &

# Stop
pkill -f uvicorn

# View logs
tail -f ibvap-platform/backend/backend.log

# Check status
ps aux | grep uvicorn
```

---

## 🧠 AI Models

| Model | Purpose | Format | Size | Classes |
|-------|---------|--------|------|---------|
| `yolo11n` | General object detection | OpenVINO / .pt | ~5.6 MB | Person, Car, Motorcycle, Truck |
| `best` (ANPR) | License plate localization | OpenVINO / .pt | ~6.2 MB | Plate |
| EasyOCR | Optical character recognition | Neural Network | Runtime | Alphanumeric text |

### Detection Pipeline

```
Frame → YOLO11n (detect humans + vehicles) → Draw boxes
                                           ↓
                              ANPR Model (find plates on full frame)
                                           ↓
                              EasyOCR (read plate text from crop)
                                           ↓
                              Emit alert via WebSocket
```

---

## 🔧 Key Technical Decisions

### Why a Dedicated Frame Grabber Thread?

RTSP cameras using H.265 (HEVC) encode video using keyframes (I-frames) and delta frames (P-frames). If OpenCV's internal buffer fills up because the AI inference loop is too slow, the decoder starts dropping I-frames. When a P-frame arrives without its reference keyframe, the video **smears and tears**.

The `LiveRTSPStream` class solves this by running `cap.read()` in a tight background loop (every 5ms), constantly discarding old frames. The AI engine always reads the most recent frame with zero buffer lag.

### Why OpenVINO?

The cloud server runs on ARM64 CPU (no GPU). OpenVINO provides:
- **2-3x faster inference** compared to raw PyTorch on CPU
- Optimized memory layout and kernel scheduling for Intel/ARM architectures
- Automatic model graph optimization (operator fusion, quantization support)

### Why Tailscale?

The camera sits behind a home router with no public IP. Tailscale creates a secure WireGuard-based mesh VPN that:
- Lets the cloud server access local network IPs (via subnet routing)
- Requires zero port forwarding or firewall changes
- Encrypts all traffic end-to-end

---

## 📡 API Reference

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Serves the React dashboard (static files) |
| `/api/video_feed` | GET | Live MJPEG video stream with AI annotations |
| `/ws/alerts` | WebSocket | Real-time JSON alerts for detections |

### WebSocket Alert Format

```json
{
  "type": "human_detected",
  "message": "Human intrusion detected.",
  "timestamp": 1725199200.123
}
```

```json
{
  "type": "plate_detected",
  "message": "License Plate: MP09CD1234",
  "plate_number": "MP09CD1234",
  "timestamp": 1725199205.456
}
```

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| **AI / CV** | Ultralytics YOLO11, OpenVINO, EasyOCR, OpenCV |
| **Backend** | Python, FastAPI, Uvicorn, WebSockets |
| **Frontend** | React 18, Vite, CSS3 |
| **Networking** | Tailscale (WireGuard VPN), RTSP/TCP |
| **Deployment** | Oracle Cloud (ARM64), Ubuntu 22.04, nohup |

---

## 📜 License

This project uses YOLO models under the [AGPL-3.0 License](https://www.gnu.org/licenses/agpl-3.0.html).

---

<p align="center">
  Built for <strong>Smart India Hackathon (SIH)</strong> 🇮🇳
</p>
