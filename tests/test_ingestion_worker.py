import sys
import os
import time
import threading
import tempfile
import cv2
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), '../backend'))
from engine import SurveillanceEngine

def create_dummy_video(filepath, num_frames=60, width=320, height=240):
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(filepath, fourcc, 30.0, (width, height))
    for i in range(num_frames):
        color_val = int((i + 1) * 3) % 255
        frame = np.full((height, width, 3), color_val, dtype=np.uint8)
        cv2.putText(frame, f"Frame {i}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        out.write(frame)
    out.release()

def test_ingestion():
    # Use webcam 0 if available, else dummy video for headless test environments
    test_video_path = None
    cap_check = cv2.VideoCapture(0)
    if cap_check.isOpened():
        source = 0
        cap_check.release()
    else:
        cap_check.release()
        tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
        test_video_path = tmp.name
        tmp.close()
        create_dummy_video(test_video_path, num_frames=60)
        source = test_video_path

    try:
        engine = SurveillanceEngine(source)
        engine.running = True

        t = threading.Thread(target=engine._ingestion_worker, daemon=True)
        t.start()

        time.sleep(2)
        assert not engine.frame_queue.empty(), "Ingestion worker did not populate frame_queue"
        
        # Verify queue size constraint (maxsize=2)
        assert engine.frame_queue.qsize() <= engine.frame_queue.maxsize, "Queue size exceeded maxsize"

        frame = engine.frame_queue.get()
        assert frame is not None, "Retrieved frame is None"
        assert len(frame.shape) == 3, f"Unexpected frame shape: {frame.shape}"

        # If synthetic video was used, verify drop-oldest behavior:
        # frame 0 had color ~3, late frames have color > 50
        if test_video_path:
            assert frame[10, 10, 0] > 50, f"Expected older frame to be dropped, got pixel value {frame[10, 10, 0]}"

        engine.running = False
        t.join(timeout=2.0)
        assert not t.is_alive(), "Ingestion worker thread did not stop cleanly"
        print("Ingestion test passed.")
    finally:
        if test_video_path and os.path.exists(test_video_path):
            os.remove(test_video_path)

def test_ingestion_unreachable_source_shutdown():
    # Verify worker survives unconnectable stream and stops promptly on running=False
    engine = SurveillanceEngine("rtsp://127.0.0.1:9999/nonexistent")
    engine.running = True

    t = threading.Thread(target=engine._ingestion_worker, daemon=True)
    t.start()

    time.sleep(1.2)
    assert t.is_alive(), "Worker should still be running and retrying connection"

    engine.running = False
    t.join(timeout=2.5)
    assert not t.is_alive(), "Worker did not stop cleanly when shutting down from unreachable source"
    print("Unreachable source shutdown test passed.")

if __name__ == "__main__":
    test_ingestion()
    test_ingestion_unreachable_source_shutdown()
