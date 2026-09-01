import os
import threading
import time
import cv2
import easyocr
import asyncio
from ultralytics import YOLO

os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"

class LiveRTSPStream:
    def __init__(self, rtsp_url):
        self.cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.frame = None
        self.ret = False
        self.running = True

        self.thread = threading.Thread(target=self._update, daemon=True)
        self.thread.start()

    def _update(self):
        while self.running:
            if self.cap.isOpened():
                ret, img = self.cap.read()
                if ret:
                    self.ret = ret
                    self.frame = img
                else:
                    self.ret = False
            time.sleep(0.005)

    def read(self):
        return self.ret, self.frame

    def stop(self):
        self.running = False
        self.cap.release()

class SurveillanceEngine:
    def __init__(self, rtsp_url):
        self.rtsp_url = rtsp_url
        self.running = False
        self.latest_frame = None
        self.latest_annotated_frame = None
        self.alert_queue = asyncio.Queue()  # For websockets
        self.stream = None
        
        # Load models
        print("[INFO] Loading OpenVINO-optimized Models...")
        try:
            self.base_model = YOLO("../yolo11n_openvino_model/")
        except Exception:
            self.base_model = YOLO("../yolo11n.pt")
            
        try:
            self.plate_model = YOLO("../anpr_best_openvino_model/")
        except Exception:
            self.plate_model = YOLO("../best.pt") # fallback
            
        print("[INFO] Loading EasyOCR Engine...")
        self.reader = easyocr.Reader(["en"], gpu=False)
        
        self.SURVEILLANCE_CLASSES = [0, 2, 3, 7] # Person, Car, Motorcycle, Truck
        self.VEHICLE_CLASSES = [2, 3, 7]
        self.last_alert_time = 0

    def start(self):
        self.running = True
        self.thread = threading.Thread(target=self._run_pipeline, daemon=True)
        self.thread.start()

    def stop(self):
        self.running = False

    def _run_pipeline(self):
        self.stream = LiveRTSPStream(self.rtsp_url)
        time.sleep(2)
        
        while self.running:
            ret, frame = self.stream.read()
            if not ret or frame is None:
                time.sleep(0.01)
                continue

            self.latest_frame = frame.copy()
            annotated_frame = frame.copy()
            orig_h, orig_w, _ = frame.shape
            
            inference_frame = cv2.resize(frame, (640, 640))
            scale_x = orig_w / 640.0
            scale_y = orig_h / 640.0
            
            # 1. Base Tracking
            base_results = self.base_model.track(
                inference_frame,
                persist=True,
                verbose=False,
                conf=0.4,
                classes=self.SURVEILLANCE_CLASSES,
                tracker="bytetrack.yaml",
            )
            
            current_time = time.time()
            human_detected = False
            
            # Draw Base Results
            for result in base_results:
                if result.boxes is None:
                    continue
                for box in result.boxes:
                    xyxy = box.xyxy.cpu().numpy().astype(int)[0]
                    conf = float(box.conf[0])
                    cls_id = int(box.cls[0])
                    
                    bx1 = int(xyxy[0] * scale_x)
                    by1 = int(xyxy[1] * scale_y)
                    bx2 = int(xyxy[2] * scale_x)
                    by2 = int(xyxy[3] * scale_y)
                    
                    if cls_id == 0:
                        human_detected = True
                        cv2.rectangle(annotated_frame, (bx1, by1), (bx2, by2), (0, 0, 255), 2)
                        cv2.putText(annotated_frame, f"Human {conf:.2f}", (bx1, max(20, by1 - 10)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                        
                    elif cls_id in self.VEHICLE_CLASSES:
                        cv2.rectangle(annotated_frame, (bx1, by1), (bx2, by2), (255, 0, 0), 2)
                        cv2.putText(annotated_frame, f"Vehicle", (bx1, max(20, by1 - 10)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
                        
                        # Process Plates on Vehicle Crop
                        v_y1, v_y2 = max(0, by1), min(orig_h, by2)
                        v_x1, v_x2 = max(0, bx1), min(orig_w, bx2)
                        vehicle_crop = frame[v_y1:v_y2, v_x1:v_x2]
                        if vehicle_crop.size == 0:
                            continue
                        
                        plate_results = self.plate_model(vehicle_crop, verbose=False, conf=0.4)
                        for p_result in plate_results:
                            if p_result.boxes is None: continue
                            for p_box in p_result.boxes:
                                px1, py1, px2, py2 = p_box.xyxy.cpu().numpy().astype(int)[0]
                                plate_crop = vehicle_crop[py1:py2, px1:px2]
                                
                                if plate_crop.size > 0:
                                    gray_plate = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2GRAY)
                                    ocr_results = self.reader.readtext(gray_plate)
                                    for res in ocr_results:
                                        clean_txt = res[1].upper().replace(" ", "").replace("-", "")
                                        if len(clean_txt) >= 6:
                                            abs_px1 = v_x1 + px1
                                            abs_py1 = v_y1 + py1
                                            abs_px2 = v_x1 + px2
                                            abs_py2 = v_y1 + py2
                                            cv2.rectangle(annotated_frame, (abs_px1, abs_py1), (abs_px2, abs_py2), (0, 255, 0), 2)
                                            cv2.putText(annotated_frame, f"Plate: {clean_txt}", (abs_px1, max(20, abs_py1 - 10)),
                                                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                                            self._emit_alert({
                                                "type": "plate_detected",
                                                "message": f"License Plate: {clean_txt}",
                                                "plate_number": clean_txt,
                                                "timestamp": time.time()
                                            })
                                            break
            
            # Emit human alert max once every 5 seconds to avoid spam
            if human_detected and (current_time - self.last_alert_time) > 5:
                self._emit_alert({
                    "type": "human_detected",
                    "message": f"Human intrusion detected.",
                    "timestamp": current_time
                })
                self.last_alert_time = current_time

            self.latest_annotated_frame = annotated_frame
            time.sleep(0.01) # Yield
            
        if self.stream:
            self.stream.stop()
        
    def _emit_alert(self, alert_data):
        try:
            self.alert_queue.put_nowait(alert_data)
        except asyncio.QueueFull:
            pass
