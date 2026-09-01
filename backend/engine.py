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
        print("[INFO] Loading Base YOLO Model...")
        try:
            self.base_model = YOLO("../yolo11n_openvino_model/")
            print("[INFO] Base model loaded (OpenVINO)")
        except Exception as e:
            print(f"[WARN] OpenVINO base model failed: {e}, falling back to .pt")
            self.base_model = YOLO("../yolo11n.pt")
            
        print("[INFO] Loading ANPR Plate Model...")
        try:
            self.plate_model = YOLO("../anpr_best_openvino_model/")
            print("[INFO] ANPR model loaded (OpenVINO)")
        except Exception as e:
            print(f"[WARN] OpenVINO ANPR model failed: {e}, falling back to best.pt")
            self.plate_model = YOLO("../best.pt")
            
        print("[INFO] Loading EasyOCR Engine...")
        self.reader = easyocr.Reader(["en"], gpu=False)
        print("[INFO] All models loaded successfully!")
        
        self.SURVEILLANCE_CLASSES = [0, 2, 3, 7] # Person, Car, Motorcycle, Truck
        self.VEHICLE_CLASSES = [2, 3, 7]
        self.last_alert_time = 0
        self.frame_count = 0

    def start(self):
        self.running = True
        self.thread = threading.Thread(target=self._run_pipeline, daemon=True)
        self.thread.start()

    def stop(self):
        self.running = False

    def _run_pipeline(self):
        self.stream = LiveRTSPStream(self.rtsp_url)
        time.sleep(2)
        print("[INFO] Pipeline started, processing frames...")
        
        while self.running:
            ret, frame = self.stream.read()
            if not ret or frame is None:
                time.sleep(0.01)
                continue

            self.frame_count += 1
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
            
            # 2. ANPR on full frame (independent of vehicle detection)
            plate_results = self.plate_model(frame, verbose=False, conf=0.35)
            
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
            
            # 3. Process Plates & OCR on full-frame plate detections
            for p_result in plate_results:
                if p_result.boxes is None:
                    continue
                for p_box in p_result.boxes:
                    px1, py1, px2, py2 = p_box.xyxy.cpu().numpy().astype(int)[0]
                    
                    plate_crop = frame[max(0, py1):min(orig_h, py2), max(0, px1):min(orig_w, px2)]
                    
                    if plate_crop.size > 0:
                        gray_plate = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2GRAY)
                        ocr_results = self.reader.readtext(gray_plate)
                        
                        plate_text = ""
                        for res in ocr_results:
                            clean_txt = res[1].upper().replace(" ", "").replace("-", "")
                            if len(clean_txt) >= 4:
                                plate_text = clean_txt
                                break
                        
                        # Draw green box around plate always
                        cv2.rectangle(annotated_frame, (px1, py1), (px2, py2), (0, 255, 0), 2)
                        label = f"Plate: {plate_text}" if plate_text else "Plate"
                        cv2.putText(annotated_frame, label, (px1, max(20, py1 - 10)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                        
                        if plate_text:
                            print(f"[ANPR] Plate detected: {plate_text}")
                            self._emit_alert({
                                "type": "plate_detected",
                                "message": f"License Plate: {plate_text}",
                                "plate_number": plate_text,
                                "timestamp": time.time()
                            })
            
            # Emit human alert max once every 5 seconds to avoid spam
            if human_detected and (current_time - self.last_alert_time) > 5:
                self._emit_alert({
                    "type": "human_detected",
                    "message": "Human intrusion detected.",
                    "timestamp": current_time
                })
                self.last_alert_time = current_time
            
            if self.frame_count % 100 == 0:
                print(f"[DIAGNOSTIC] Processed {self.frame_count} frames. Pipeline alive.")

            self.latest_annotated_frame = annotated_frame
            time.sleep(0.01) # Yield
            
        if self.stream:
            self.stream.stop()
        
    def _emit_alert(self, alert_data):
        try:
            self.alert_queue.put_nowait(alert_data)
        except asyncio.QueueFull:
            pass
