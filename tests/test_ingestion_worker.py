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
        create_dummy_video(test_video_path, num_frames=200)
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

def test_ingestion_midstream_reconnect():
    # Simulate mid-stream network drop where isOpened() stays True but read() fails
    class MockVideoCapture:
        instances = []
        def __init__(self, url):
            self.url = url
            self.frame_num = 0
            self._opened = True
            self.released = False
            MockVideoCapture.instances.append(self)

        def set(self, prop, val):
            pass

        def isOpened(self):
            # In OpenCV, isOpened stays True even after network disconnect
            return self._opened

        def read(self):
            self.frame_num += 1
            # Produce 3 valid frames, then simulate mid-stream disconnect
            if self.frame_num <= 3:
                return True, np.full((240, 320, 3), 100, dtype=np.uint8)
            return False, None

        def release(self):
            self._opened = False
            self.released = True

    orig_vc = cv2.VideoCapture
    cv2.VideoCapture = MockVideoCapture
    MockVideoCapture.instances.clear()

    try:
        engine = SurveillanceEngine("mock_rtsp_stream")
        engine.max_read_failures = 5  # Quick failure threshold for test
        engine.running = True

        t = threading.Thread(target=engine._ingestion_worker, daemon=True)
        t.start()

        # Allow worker to read 3 frames, encounter 5 failures, and reconnect
        time.sleep(1.5)

        assert len(MockVideoCapture.instances) >= 2, "Ingestion worker did not reconnect after mid-stream failures"
        assert MockVideoCapture.instances[0].released, "Stale VideoCapture was not released before reconnecting"

        engine.running = False
        t.join(timeout=2.0)
        assert not t.is_alive(), "Ingestion worker thread did not stop cleanly after reconnect"
        print("Mid-stream reconnection test passed.")
    finally:
        cv2.VideoCapture = orig_vc

if __name__ == "__main__":
    test_ingestion()
    test_ingestion_unreachable_source_shutdown()
    test_ingestion_midstream_reconnect()
