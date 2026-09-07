import sys
import os
import time
import threading
import numpy as np
sys.path.append(os.path.join(os.path.dirname(__file__), '../backend'))
from engine import SurveillanceEngine

def test_inference():
    engine = SurveillanceEngine("dummy")
    engine.running = True
    
    # Push a dummy frame
    dummy_frame = np.ones((640, 640, 3), dtype=np.uint8) * 128
    engine.frame_queue.put(dummy_frame)
    
    t = threading.Thread(target=engine._inference_worker, daemon=True)
    t.start()
    
    try:
        # Poll with timeout for queue consumption/frame update
        timeout = 10.0
        start_time = time.time()
        while time.time() - start_time < timeout:
            if engine.latest_annotated_frame is not None:
                break
            time.sleep(0.1)
            
        assert engine.latest_annotated_frame is not None, "Timeout waiting for latest_annotated_frame"
        print("Inference test passed.")
    finally:
        engine.running = False
        t.join(timeout=2.0)
        assert not t.is_alive(), "Inference worker thread did not stop cleanly"

if __name__ == "__main__":
    test_inference()
