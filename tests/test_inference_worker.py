import sys
import os
import time
import numpy as np
sys.path.append(os.path.join(os.path.dirname(__file__), '../backend'))
from engine import SurveillanceEngine
import queue

def test_inference():
    engine = SurveillanceEngine("dummy")
    engine.running = True
    
    # Push a dummy frame
    dummy_frame = np.ones((640, 640, 3), dtype=np.uint8) * 128
    engine.frame_queue.put(dummy_frame)
    
    import threading
    t = threading.Thread(target=engine._inference_worker, daemon=True)
    t.start()
    
    # Poll with timeout for queue consumption/frame update
    timeout = 10.0
    start_time = time.time()
    while time.time() - start_time < timeout:
        if engine.latest_annotated_frame is not None:
            break
        time.sleep(0.1)
        
    assert engine.latest_annotated_frame is not None, "Timeout waiting for latest_annotated_frame"
    engine.running = False
    print("Inference test passed.")

if __name__ == "__main__":
    test_inference()
