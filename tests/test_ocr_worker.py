import sys
import os
import time
import asyncio
import cv2
import numpy as np
sys.path.append(os.path.join(os.path.dirname(__file__), '../backend'))
from engine import SurveillanceEngine

async def test_ocr():
    engine = SurveillanceEngine("dummy")
    engine.running = True
    
    # Start only OCR thread
    import threading
    t = threading.Thread(target=engine._ocr_worker, daemon=True)
    t.start()
    
    # 1. Push a dummy white image (EasyOCR will likely return nothing, but won't crash)
    dummy_plate = np.ones((50, 150, 3), dtype=np.uint8) * 255
    engine.ocr_queue.put((0, 0, 150, 50, dummy_plate))
    
    # Wait for queue to be consumed
    for _ in range(30):
        if engine.ocr_queue.empty():
            break
        time.sleep(0.1)
    assert engine.ocr_queue.empty(), "OCR worker did not consume item from ocr_queue"

    # 2. Push corrupted plate crop (e.g., 1D array) that causes cv2.cvtColor to fail
    # Worker must catch exception, log error, and remain alive
    corrupted_plate = np.array([1, 2, 3], dtype=np.uint8)
    engine.ocr_queue.put((0, 0, 10, 10, corrupted_plate))
    for _ in range(30):
        if engine.ocr_queue.empty():
            break
        time.sleep(0.1)
    assert engine.ocr_queue.empty(), "OCR worker did not consume corrupted plate"
    assert t.is_alive(), "OCR worker thread crashed on corrupted plate"

    # 3. Push an image with readable license plate text to verify alert emission after error
    plate_img = np.ones((100, 300, 3), dtype=np.uint8) * 255
    cv2.putText(plate_img, "MH12DE1433", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 2)
    engine.ocr_queue.put((0, 0, 300, 100, plate_img))

    # Wait for alert to be emitted
    for _ in range(50):
        if not engine.alert_queue.empty():
            break
        time.sleep(0.1)

    assert not engine.alert_queue.empty(), "Alert was not emitted for detected plate"
    alert = engine.alert_queue.get_nowait()
    assert alert["type"] == "plate_detected"
    assert "plate_number" in alert
    assert len(alert["plate_number"]) >= 4

    engine.running = False
    t.join(timeout=2.0)
    assert not t.is_alive(), "OCR worker thread did not stop cleanly"
    print("OCR worker processed item without crashing.")

if __name__ == "__main__":
    asyncio.run(test_ocr())
