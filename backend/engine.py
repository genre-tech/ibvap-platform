import os
import threading
import time
import cv2
import easyocr
import asyncio
import queue
from ultralytics import YOLO

os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"


class SurveillanceEngine:
    def __init__(self, rtsp_url):
        self.rtsp_url = rtsp_url
        self.running = False
        self.latest_frame = None
        self.latest_annotated_frame = None
        self.alert_queue = asyncio.Queue()  # For websockets
        self.stream = None
        
        # Load models
        base_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.abspath(os.path.join(base_dir, ".."))

        base_model_dir = os.path.join(project_root, "yolo11n_openvino_model")
        if not os.path.exists(base_model_dir):
            base_model_dir = "../yolo11n_openvino_model/"
        base_pt = os.path.join(project_root, "yolo11n.pt")
        if not os.path.exists(base_pt):
            base_pt = "../yolo11n.pt"

        plate_model_dir = os.path.join(project_root, "anpr_best_openvino_model")
        if not os.path.exists(plate_model_dir):
            plate_model_dir = "../anpr_best_openvino_model/"
        plate_pt = os.path.join(project_root, "best.pt")
        if not os.path.exists(plate_pt):
            plate_pt = "../best.pt"

        print("[INFO] Loading Base YOLO Model...")
        try:
            self.base_model = YOLO(base_model_dir)
            print("[INFO] Base model loaded (OpenVINO)")
        except Exception as e:
            print(f"[WARN] OpenVINO base model failed: {e}, falling back to .pt")
            self.base_model = YOLO(base_pt)
            
        print("[INFO] Loading ANPR Plate Model...")
        try:
            self.plate_model = YOLO(plate_model_dir)
            print("[INFO] ANPR model loaded (OpenVINO)")
        except Exception as e:
            print(f"[WARN] OpenVINO ANPR model failed: {e}, falling back to best.pt")
            self.plate_model = YOLO(plate_pt)
            
        print("[INFO] Loading EasyOCR Engine...")
        self.reader = easyocr.Reader(["en"], gpu=False)
        print("[INFO] All models loaded successfully!")
        
        self.SURVEILLANCE_CLASSES = [0, 2, 3, 7] # Person, Car, Motorcycle, Truck
        self.VEHICLE_CLASSES = [2, 3, 7]
        self.last_alert_time = 0
        self.frame_count = 0
        self.frame_queue = queue.Queue(maxsize=2)
        self.ocr_queue = queue.Queue(maxsize=50)
        self.max_read_failures = 50
        self.threads = []

    def start(self):
        self.running = True
        self.threads = [
            threading.Thread(target=self._ingestion_worker, daemon=True),
            threading.Thread(target=self._inference_worker, daemon=True),
            threading.Thread(target=self._ocr_worker, daemon=True)
        ]
        for t in self.threads:
            t.start()

    def stop(self):
        self.running = False
        for t in self.threads:
            if t.is_alive():
                t.join(timeout=1.0)

    def _ingestion_worker(self):
        print(f"[INFO] Ingestion Worker connecting to {self.rtsp_url}...")
        cap = cv2.VideoCapture(self.rtsp_url)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        consecutive_failures = 0
        max_failures = getattr(self, "max_read_failures", 50)
        
        try:
            while self.running:
                if not cap.isOpened() or consecutive_failures >= max_failures:
                    if consecutive_failures >= max_failures:
                        print(f"[WARN] Ingestion Worker lost stream from {self.rtsp_url} (failed {consecutive_failures} consecutive reads). Reconnecting...")
                    time.sleep(1)
                    try:
                        cap.release()
                    except Exception:
                        pass
                    cap = cv2.VideoCapture(self.rtsp_url)
                    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                    consecutive_failures = 0
                    continue
                    
                try:
                    ret, frame = cap.read()
                    if not ret or frame is None:
                        consecutive_failures += 1
                        time.sleep(0.01)
                        continue
                        
                    consecutive_failures = 0
                        
                    # Drop oldest frame if queue is full
                    if self.frame_queue.full():
                        try:
                            self.frame_queue.get_nowait()
                        except queue.Empty:
                            pass
                            
                    self.frame_queue.put(frame)
                    time.sleep(0.005)
                except Exception as e:
                    consecutive_failures += 1
                    print(f"[ERROR] Ingestion Worker error reading frame: {e}")
                    time.sleep(0.01)
        finally:
            cap.release()


    def _inference_worker(self):
        print("[INFO] Inference Worker started...")
        while self.running:
            try:
                frame = self.frame_queue.get(timeout=1.0)
            except queue.Empty:
                continue

            try:
                self.frame_count += 1
                annotated_frame = frame.copy()
                orig_h, orig_w, _ = frame.shape
                
                inference_frame = cv2.resize(frame, (640, 640))
                scale_x = orig_w / 640.0
                scale_y = orig_h / 640.0
                
                base_results = self.base_model.track(
                    inference_frame, persist=True, verbose=False,
                    conf=0.4, classes=self.SURVEILLANCE_CLASSES, tracker="bytetrack.yaml"
                )
                plate_results = self.plate_model(frame, verbose=False, conf=0.35)
                
                current_time = time.time()
                human_detected = False
                
                for result in base_results:
                    if result.boxes is None: continue
                    for box in result.boxes:
                        xyxy = box.xyxy.cpu().numpy().astype(int)[0]
                        conf = float(box.conf[0])
                        cls_id = int(box.cls[0])
                        
                        bx1, by1 = int(xyxy[0] * scale_x), int(xyxy[1] * scale_y)
                        bx2, by2 = int(xyxy[2] * scale_x), int(xyxy[3] * scale_y)
                        
                        if cls_id == 0:
                            human_detected = True
                            cv2.rectangle(annotated_frame, (bx1, by1), (bx2, by2), (0, 0, 255), 2)
                            cv2.putText(annotated_frame, f"Human {conf:.2f}", (bx1, max(20, by1 - 10)),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                        elif cls_id in self.VEHICLE_CLASSES:
                            cv2.rectangle(annotated_frame, (bx1, by1), (bx2, by2), (255, 0, 0), 2)
                            cv2.putText(annotated_frame, "Vehicle", (bx1, max(20, by1 - 10)),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
                
                for p_result in plate_results:
                    if p_result.boxes is None: continue
                    for p_box in p_result.boxes:
                        px1, py1, px2, py2 = p_box.xyxy.cpu().numpy().astype(int)[0]
                        plate_crop = frame[max(0, py1):min(orig_h, py2), max(0, px1):min(orig_w, px2)]
                        
                        if plate_crop.size > 0:
                            # Draw green box immediately
                            cv2.rectangle(annotated_frame, (px1, py1), (px2, py2), (0, 255, 0), 2)
                            cv2.putText(annotated_frame, "Reading Plate...", (px1, max(20, py1 - 10)),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                            
                            # Send to OCR worker non-blocking
                            try:
                                self.ocr_queue.put_nowait((px1, py1, px2, py2, plate_crop))
                            except queue.Full:
                                pass # Drop plate if OCR is backed up entirely
                                
                if human_detected and (current_time - self.last_alert_time) > 5:
                    self._emit_alert({"type": "human_detected", "message": "Human intrusion detected.", "timestamp": current_time})
                    self.last_alert_time = current_time
                
                self.latest_annotated_frame = annotated_frame
                time.sleep(0.005) # Yield
            except Exception as e:
                print(f"[ERROR] Inference Worker error processing frame: {e}")

    def _ocr_worker(self):
        print("[INFO] OCR Worker started.")
        while self.running:
            try:
                # Wait for plate with timeout to allow checking self.running
                px1, py1, px2, py2, plate_crop = self.ocr_queue.get(timeout=1.0)
            except queue.Empty:
                continue

            try:
                if plate_crop is not None and getattr(plate_crop, "size", 0) > 0:
                    gray_plate = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2GRAY)
                    ocr_results = self.reader.readtext(gray_plate)
                    
                    plate_text = ""
                    for res in ocr_results:
                        clean_txt = res[1].upper().replace(" ", "").replace("-", "")
                        if len(clean_txt) >= 4:
                            plate_text = clean_txt
                            break
                    
                    if plate_text:
                        print(f"[ANPR] Plate detected: {plate_text}")
                        self._emit_alert({
                            "type": "plate_detected",
                            "message": f"License Plate: {plate_text}",
                            "plate_number": plate_text,
                            "timestamp": time.time()
                        })
            except Exception as e:
                print(f"[ERROR] OCR Worker error processing plate: {e}")


    def _emit_alert(self, alert_data):
        try:
            self.alert_queue.put_nowait(alert_data)
        except asyncio.QueueFull:
            pass
