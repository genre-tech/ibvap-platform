# ANPR Pipeline Optimization Design

## Goal
Optimize the real-time video surveillance engine (`backend/engine.py`) to process RTSP streams at maximum framerate without blocking or stuttering. The new architecture is tailored for high-stakes border security environments, prioritizing zero-latency video processing and accurate, decoupled OCR.

## Architecture
Transition from a single-threaded synchronous loop to a Producer-Consumer asynchronous architecture using three independent workers (threads) communicating via thread-safe queues.

1. **Ingestion Worker (Producer):** 
   - Continuously pulls frames from the RTSP stream.
   - Maintains a fixed-size `Frame Queue` (e.g., `maxsize=2`). This ensures the system processes the absolute present moment. If the queue is full, it drops the oldest frame (or skips pushing the new one) to prevent the pipeline from lagging behind real-time.

2. **Inference Worker (Processor):** 
   - Pulls the latest frames from the `Frame Queue`.
   - Runs the Base YOLO model with ByteTrack to detect and track humans and vehicles.
   - Runs the Plate YOLO model to detect license plates.
   - If a plate is detected, it crops the plate region and pushes it into the `OCR Queue`.
   - Updates the `Display Buffer` with the latest annotated frame, ensuring the video feed remains perfectly smooth regardless of OCR workload.

3. **OCR Worker (Consumer):**
   - Dedicated background thread that pulls cropped plates from the `OCR Queue`.
   - Runs the EasyOCR engine. (The decoupled nature allows for an easy swap to PaddleOCR in the future if greater accuracy is needed).
   - Upon successful text extraction, emits asynchronous alerts via the existing WebSocket queue.

## Components & Data Flow
- **Queues (Thread-Safe):**
  - `frame_queue`: `queue.Queue(maxsize=2)`.
  - `ocr_queue`: `queue.Queue(maxsize=50)`. Allows OCR to buffer a burst of plates (e.g., a convoy passing through) without dropping plate images.
- **Models:**
  - Base YOLO: OpenVINO / PyTorch fallback.
  - Plate YOLO: OpenVINO / PyTorch fallback.
  - OCR: EasyOCR.
- **Error Handling & Resilience:**
  - Threads wrap operations in `try-except` blocks to prevent a single inference failure from crashing the entire pipeline.
  - If the RTSP stream drops, the Ingestion Worker automatically attempts to reconnect without requiring an app restart.

## Testing Strategy
- Queue tests: Verify that the frame queue successfully drops stale frames under heavy load.
- Pipeline tests: Run a simulated RTSP stream (using a local video file) to confirm the video continues playing smoothly even when OCR is artificially slowed down.
