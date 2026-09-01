## Project Overview: Surveillance and Object Detection System

This project implements a real-time video surveillance system capable of performing object detection (using YOLO) and Automatic Number Plate Recognition (ANPR). It is designed to process live video streams, typically sourced from RTSP cameras, and provide alerts or visualizations based on detected objects or license plates.

### Tech Stack
*   **Language:** Python
*   **Core Libraries:** OpenCV (`cv2`) for video handling, `ultralytics` for YOLO models, and `easyocr` (implied) for OCR/ANPR.
*   **Models:** Pre-trained weights are used, including `yolov8n.pt` and specialized models like `anpr_best.pt`.

### How to Build / Run / Test
**1. Setup:**
First, ensure all dependencies are installed.
```bash
# Assuming a requirements.txt exists or listing key packages
pip install opencv-python ultralytics easyocr
```

**2. Running the System:**
*   **Basic YOLO Detection (`tata.py`):** This script provides a straightforward implementation for running YOLO detection on a specified RTSP stream.
    ```bash
    python tata.py
    ```
*   **Advanced Streaming with ANPR (`camerasurvillance.py`):** This script implements a more robust, threaded streaming pipeline specifically designed for low-latency capture and includes ANPR logic.
    ```bash
    python camerasurvillance.py
    ```

**3. Testing:**
The system is tested by pointing the scripts to live RTSP feeds (e.g., `rtsp://user:pass@ip:port/stream`).

### Directory Layout
*   **Root Directory:** Contains the main executable scripts (`camerasurvillance.py`, `tata.py`) and the pre-trained model weights (`.pt` files).
*   **`runs/`:** Directory intended for storing output results, logs, or processed frames.

### Important Conventions & Gotchas
*   **RTSP Streams:** The system is heavily dependent on stable RTSP connections.
*   **Latency Management:**
    *   `camerasurvillance.py` sets `OPENCV_FFMPEG_CAPTURE_OPTIONS` to `"rtsp_transport;tcp"` to mitigate packet loss issues common with HEVC/H.264 streams.
    *   `tata.py` explicitly sets `cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)` to reduce frame lag.
*   **Model Loading:** The scripts assume the necessary model weights (`.pt` files) are available in the root directory.
*   **ANPR Logic:** The ANPR functionality is implemented within `camerasurvillance.py` and relies on OCR processing of detected plates.
"