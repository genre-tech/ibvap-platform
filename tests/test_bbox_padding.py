import sys
import os
import queue
import numpy as np
import torch
from unittest.mock import MagicMock

sys.path.append(os.path.join(os.path.dirname(__file__), '../backend'))
from engine import SurveillanceEngine

class DummyBox:
    def __init__(self, xyxy):
        self.xyxy = torch.tensor([xyxy])

class DummyResult:
    def __init__(self, boxes):
        self.boxes = boxes

def test_bbox_padding_normal():
    engine = SurveillanceEngine("dummy")
    engine.running = True
    
    # Mock models to avoid heavy inference
    engine.base_model = MagicMock()
    engine.base_model.track.return_value = []
    
    # 480x640 frame (height=480, width=640)
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    # Plate box: w=100, h=50. 10% pad: pad_x=10, pad_y=5
    box = DummyBox([100.0, 100.0, 200.0, 150.0])
    engine.plate_model = MagicMock(return_value=[DummyResult([box])])
    
    engine.frame_queue.put(frame)
    
    import threading
    t = threading.Thread(target=engine._inference_worker, daemon=True)
    t.start()
    
    try:
        item = engine.ocr_queue.get(timeout=3.0)
        px1, py1, px2, py2, plate_crop = item
        # Expected padded coordinates:
        # pad_x = int(100 * 0.10) = 10 -> px1 = 90, px2 = 210
        # pad_y = int(50 * 0.10) = 5   -> py1 = 95, py2 = 155
        assert px1 == 90, f"Expected px1=90, got {px1}"
        assert py1 == 95, f"Expected py1=95, got {py1}"
        assert px2 == 210, f"Expected px2=210, got {px2}"
        assert py2 == 155, f"Expected py2=155, got {py2}"
        assert plate_crop.shape == (60, 120, 3), f"Expected shape (60, 120, 3), got {plate_crop.shape}"
    finally:
        engine.running = False
        t.join(timeout=2.0)

def test_bbox_padding_edge_clamp_top_left():
    engine = SurveillanceEngine("dummy")
    engine.running = True
    engine.base_model = MagicMock()
    engine.base_model.track.return_value = []
    
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    # Box near top-left: px1=5, py1=4, px2=55, py2=34 -> w=50, h=30
    # pad_x = 5 -> px1 = max(0, 5 - 5) = 0, px2 = 55 + 5 = 60
    # pad_y = 3 -> py1 = max(0, 4 - 3) = 1, py2 = 34 + 3 = 37
    box = DummyBox([5.0, 4.0, 55.0, 34.0])
    engine.plate_model = MagicMock(return_value=[DummyResult([box])])
    
    engine.frame_queue.put(frame)
    import threading
    t = threading.Thread(target=engine._inference_worker, daemon=True)
    t.start()
    
    try:
        item = engine.ocr_queue.get(timeout=3.0)
        px1, py1, px2, py2, plate_crop = item
        assert px1 == 0, f"Expected px1=0, got {px1}"
        assert py1 == 1, f"Expected py1=1, got {py1}"
        assert px2 == 60, f"Expected px2=60, got {px2}"
        assert py2 == 37, f"Expected py2=37, got {py2}"
        assert plate_crop.shape == (36, 60, 3), f"Expected shape (36, 60, 3), got {plate_crop.shape}"
    finally:
        engine.running = False
        t.join(timeout=2.0)

def test_bbox_padding_edge_clamp_bottom_right():
    engine = SurveillanceEngine("dummy")
    engine.running = True
    engine.base_model = MagicMock()
    engine.base_model.track.return_value = []
    
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    # Box near bottom-right on 480x640: px1=590, py1=440, px2=638, py2=478 -> w=48, h=38
    # pad_x = 4 -> px1 = 586, px2 = min(640, 638 + 4) = 640
    # pad_y = 3 -> py1 = 437, py2 = min(480, 478 + 3) = 480
    box = DummyBox([590.0, 440.0, 638.0, 478.0])
    engine.plate_model = MagicMock(return_value=[DummyResult([box])])
    
    engine.frame_queue.put(frame)
    import threading
    t = threading.Thread(target=engine._inference_worker, daemon=True)
    t.start()
    
    try:
        item = engine.ocr_queue.get(timeout=3.0)
        px1, py1, px2, py2, plate_crop = item
        assert px1 == 586, f"Expected px1=586, got {px1}"
        assert py1 == 437, f"Expected py1=437, got {py1}"
        assert px2 == 640, f"Expected px2=640, got {px2}"
        assert py2 == 480, f"Expected py2=480, got {py2}"
        assert plate_crop.shape == (43, 54, 3), f"Expected shape (43, 54, 3), got {plate_crop.shape}"
    finally:
        engine.running = False
        t.join(timeout=2.0)

if __name__ == "__main__":
    test_bbox_padding_normal()
    test_bbox_padding_edge_clamp_top_left()
    test_bbox_padding_edge_clamp_bottom_right()
    print("All bbox padding tests passed!")
