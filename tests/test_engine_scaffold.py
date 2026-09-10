import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../backend'))
from engine import SurveillanceEngine

def test_engine_scaffold():
    engine = SurveillanceEngine("dummy_url")
    assert hasattr(engine, 'frame_queue')
    assert hasattr(engine, 'ocr_queue')
    engine.start()
    assert len(engine.threads) == 3
    engine.stop()
    print("Scaffold test passed")

if __name__ == "__main__":
    test_engine_scaffold()
