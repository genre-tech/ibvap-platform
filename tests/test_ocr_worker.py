import sys
import os
import time
import cv2
import numpy as np
sys.path.append(os.path.join(os.path.dirname(__file__), '../backend'))
from engine import SurveillanceEngine

from unittest.mock import MagicMock
import threading

def test_ocr_paddleocr_integration():
    """Test that PaddleOCR predict results are parsed correctly and alerts are emitted."""
    engine = SurveillanceEngine("dummy")
    engine.running = True
    
    mock_reader = MagicMock()
    # PaddleOCR predict returns a generator of dicts with 'rec_texts' and 'rec_scores'
    mock_reader.predict.return_value = iter([
        {'rec_texts': ['MH12DE1433'], 'rec_scores': [0.95]}
    ])
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
        
        mock_reader.predict.assert_called_once()
    finally:
        engine.running = False
        t.join(timeout=2.0)

def test_ocr_rejects_non_numeric():
    """Watermark text with no digits should be rejected."""
    engine = SurveillanceEngine("dummy")
    engine.running = True
    
    mock_reader = MagicMock()
    mock_reader.predict.return_value = iter([
        {'rec_texts': ['WATERMARK'], 'rec_scores': [0.95]}
    ])
    engine.reader = mock_reader
    
    t = threading.Thread(target=engine._ocr_worker, daemon=True)
    t.start()
    
    try:
        dummy_crop = np.zeros((50, 150, 3), dtype=np.uint8)
        engine.ocr_queue.put((0, 0, 150, 50, dummy_crop))
        
        for _ in range(20):
            if engine.ocr_queue.empty():
                break
            time.sleep(0.1)
            
        time.sleep(0.3)
        assert engine.alert_queue.empty(), "Alert should not be emitted for text with no digits"
    finally:
        engine.running = False
        t.join(timeout=2.0)

def test_ocr_handles_empty_results():
    """PaddleOCR returning empty results should not crash."""
    engine = SurveillanceEngine("dummy")
    engine.running = True
    
    mock_reader = MagicMock()
    mock_reader.predict.return_value = iter([
        {'rec_texts': [], 'rec_scores': []}
    ])
    engine.reader = mock_reader
    
    t = threading.Thread(target=engine._ocr_worker, daemon=True)
    t.start()
    
    try:
        dummy_crop = np.zeros((50, 150, 3), dtype=np.uint8)
        engine.ocr_queue.put((0, 0, 150, 50, dummy_crop))
        
        for _ in range(20):
            if engine.ocr_queue.empty():
                break
            time.sleep(0.1)
            
        time.sleep(0.3)
        assert engine.alert_queue.empty(), "No alert should be emitted for empty OCR results"
        assert t.is_alive(), "OCR worker should not crash on empty results"
    finally:
        engine.running = False
        t.join(timeout=2.0)

def test_ocr_real_image():
    """Test with a real rendered plate image to verify end-to-end PaddleOCR."""
    engine = SurveillanceEngine("dummy")
    engine.running = True
    
    t = threading.Thread(target=engine._ocr_worker, daemon=True)
    t.start()
    
    try:
        # Create a white image with text
        plate_img = np.ones((100, 300, 3), dtype=np.uint8) * 255
        cv2.putText(plate_img, "MH12DE1433", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 2)
        engine.ocr_queue.put((0, 0, 300, 100, plate_img))

        for _ in range(50):
            if not engine.alert_queue.empty():
                break
            time.sleep(0.1)

        assert not engine.alert_queue.empty(), "Alert was not emitted for detected plate"
        alert = engine.alert_queue.get_nowait()
        assert alert["type"] == "plate_detected"
        assert "plate_number" in alert
        assert len(alert["plate_number"]) >= 4
    finally:
        engine.running = False
        t.join(timeout=2.0)
        assert not t.is_alive(), "OCR worker thread did not stop cleanly"
    print("PaddleOCR worker processed items correctly.")

if __name__ == "__main__":
    test_ocr_paddleocr_integration()
    test_ocr_rejects_non_numeric()
    test_ocr_handles_empty_results()
    test_ocr_real_image()
