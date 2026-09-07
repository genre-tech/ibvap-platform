import sys
import os
import time
import cv2
import numpy as np
sys.path.append(os.path.join(os.path.dirname(__file__), '../backend'))
from engine import SurveillanceEngine

from unittest.mock import MagicMock
import threading

def test_ocr_passes_allowlist():
    engine = SurveillanceEngine("dummy")
    engine.running = True
    
    mock_reader = MagicMock()
    mock_reader.readtext.return_value = [([], "MH12DE1433", 0.95)]
    engine.reader = mock_reader
    
    t = threading.Thread(target=engine._ocr_worker, daemon=True)
    t.start()
    
    try:
        dummy_crop = np.zeros((50, 150, 3), dtype=np.uint8)
        engine.ocr_queue.put((0, 0, 150, 50, dummy_crop))
        
        for _ in range(30):
            if not engine.alert_queue.empty():
                break
            time.sleep(0.1)
            
        assert not engine.alert_queue.empty(), "Alert was not emitted"
        alert = engine.alert_queue.get_nowait()
        assert alert["plate_number"] == "MH12DE1433"
        
        # Verify readtext was called with the exact allowlist
        mock_reader.readtext.assert_called_once()
        _, kwargs = mock_reader.readtext.call_args
        assert "allowlist" in kwargs, "allowlist parameter was not passed to readtext"
        assert kwargs["allowlist"] == "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
    finally:
        engine.running = False
        t.join(timeout=2.0)

def test_ocr_no_positional_heuristic():
    engine = SurveillanceEngine("dummy")
    engine.running = True
    
    # Under the old heuristic:
    # Pos 1-2 ('12') -> 'IZ' (number_to_letter)
    # Pos 3-4 ('AB') -> '48' (letter_to_number)
    # The string '12AB123456' (length 10) was corrupted to 'IZ48123456'.
    # With heuristic removed, it must remain '12AB123456'.
    raw_ocr = "12AB123456"
    mock_reader = MagicMock()
    mock_reader.readtext.return_value = [([], raw_ocr, 0.95)]
    engine.reader = mock_reader
    
    t = threading.Thread(target=engine._ocr_worker, daemon=True)
    t.start()
    
    try:
        dummy_crop = np.zeros((50, 150, 3), dtype=np.uint8)
        engine.ocr_queue.put((0, 0, 150, 50, dummy_crop))
        
        for _ in range(30):
            if not engine.alert_queue.empty():
                break
            time.sleep(0.1)
            
        assert not engine.alert_queue.empty(), "Alert was not emitted"
        alert = engine.alert_queue.get_nowait()
        assert alert["plate_number"] == "12AB123456", f"Expected '12AB123456', got '{alert['plate_number']}'"
    finally:
        engine.running = False
        t.join(timeout=2.0)

def test_ocr_rejects_non_numeric():
    engine = SurveillanceEngine("dummy")
    engine.running = True
    
    mock_reader = MagicMock()
    mock_reader.readtext.return_value = [([], "WATERMARK", 0.95)]
    engine.reader = mock_reader
    
    t = threading.Thread(target=engine._ocr_worker, daemon=True)
    t.start()
    
    try:
        dummy_crop = np.zeros((50, 150, 3), dtype=np.uint8)
        engine.ocr_queue.put((0, 0, 150, 50, dummy_crop))
        
        # Wait for queue to be processed
        for _ in range(20):
            if engine.ocr_queue.empty():
                break
            time.sleep(0.1)
            
        time.sleep(0.3)
        assert engine.alert_queue.empty(), "Alert should not be emitted for text with no digits"
    finally:
        engine.running = False
        t.join(timeout=2.0)

def test_ocr():
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
    test_ocr_passes_allowlist()
    test_ocr_no_positional_heuristic()
    test_ocr_rejects_non_numeric()
    test_ocr()
