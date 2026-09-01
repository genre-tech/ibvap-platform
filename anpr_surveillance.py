import os
import threading
import time
import cv2
import easyocr
from ultralytics import YOLO

# Force TCP transport protocol to eliminate HEVC/H.264 packet drop errors[cite: 2]
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"

# Load & Export models with OpenVINO for high-speed CPU inference[cite: 3]
try:
    print("[INFO] Loading OpenVINO-optimized Base Model...")
    base_model = YOLO("yolo11n_openvino_model/")
except Exception:
    print("[INFO] Exporting base model to OpenVINO for acceleration...")
    temp_model = YOLO("yolo11n.pt")
    temp_model.export(format="openvino")
    base_model = YOLO("yolo11n_openvino_model/")

try:
    print("[INFO] Loading OpenVINO-optimized ANPR Plate Model...")
    plate_model = YOLO("anpr_best_openvino_model/")
except Exception:
    print("[INFO] Exporting plate model to OpenVINO...")
    temp_plate = YOLO("anpr_best.pt")
    temp_plate.export(format="openvino")
    plate_model = YOLO("anpr_best_openvino_model/")

print("[INFO] Loading EasyOCR Engine...")
reader = easyocr.Reader(["en"], gpu=False)


class LiveRTSPStream:
    """Threaded RTSP frame grabber to bypass OpenCV buffer delay and keep latency at 0ms[cite: 2]."""

    def __init__(self, rtsp_url):
        self.cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Drop old frames instantly[cite: 2]
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
            time.sleep(0.001)

    def read(self):
        return self.ret, self.frame

    def stop(self):
        self.running = False
        self.cap.release()


# Configuration & Initialization
RTSP_URL = "rtsp://admin:dewas@321@192.168.29.104:554/Streaming/Channels/101"
print("[INFO] Connecting to RTSP Stream...")
cam = LiveRTSPStream(RTSP_URL)
time.sleep(2.0)  # Warm-up

SURVEILLANCE_CLASSES = [0, 2, 3, 7]  # 0: Person, 2: Car, 3: Motorcycle, 7: Truck
VEHICLE_CLASSES = [2, 3, 7]

print("[INFO] System Active. Processing Live Feed...")

try:
    while True:
        success, frame = cam.read()
        if not success or frame is None:
            time.sleep(0.01)
            continue

        orig_h, orig_w, _ = frame.shape

        # Resize frame for rapid inference[cite: 3]
        inference_frame = cv2.resize(frame, (640, 640))
        scale_x = orig_w / 640.0
        scale_y = orig_h / 640.0

        # Stage 1: General Human & Vehicle Tracking
        base_results = base_model.track(
            inference_frame,
            persist=True,
            verbose=False,
            conf=0.4,
            classes=SURVEILLANCE_CLASSES,
            tracker="bytetrack.yaml",
        )

        annotated_frame = frame.copy()

        # ALWAYS run the ANPR model on the frame (or a region near humans) to catch handheld plates
        # Stage 2: License Plate Localization across the whole frame or vehicle crops
        plate_results = plate_model(inference_frame, verbose=False, conf=0.35)

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
                    cv2.rectangle(
                        annotated_frame, (bx1, by1), (bx2, by2), (0, 0, 255), 2
                    )
                    cv2.putText(
                        annotated_frame,
                        f"Human {conf:.2f}",
                        (bx1, max(20, by1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 0, 255),
                        2,
                    )
                elif cls_id in VEHICLE_CLASSES:
                    cv2.rectangle(
                        annotated_frame, (bx1, by1), (bx2, by2), (255, 0, 0), 2
                    )
                    cv2.putText(
                        annotated_frame,
                        f"Vehicle",
                        (bx1, max(20, by1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (255, 0, 0),
                        2,
                    )

        # Stage 3: Draw and OCR License Plates detected anywhere in view (handles handheld plates & cars)
        for p_result in plate_results:
            if p_result.boxes is None:
                continue
            for p_box in p_result.boxes:
                px1 = int(p_box.xyxy[0][0].item() * scale_x)
                py1 = int(p_box.xyxy[0][1].item() * scale_y)
                px2 = int(p_box.xyxy[0][2].item() * scale_x)
                py2 = int(p_box.xyxy[0][3].item() * scale_y)
                p_conf = float(p_box.conf[0])

                plate_crop = frame[
                    max(0, py1) : min(orig_h, py2), max(0, px1) : min(orig_w, px2)
                ]

                if plate_crop.size > 0:
                    gray_plate = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2GRAY)
                    ocr_results = reader.readtext(gray_plate)
                    
                    for res in ocr_results:
                        clean_txt = (
                            res[1].upper().replace(" ", "").replace("-", "")
                        )
                        if len(clean_txt) >= 4:
                            cv2.rectangle(
                                annotated_frame,
                                (px1, py1),
                                (px2, py2),
                                (0, 255, 0),
                                2,
                            )
                            cv2.putText(
                                annotated_frame,
                                f"Plate: {clean_txt}",
                                (px1, max(20, py1 - 10)),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.7,
                                (0, 255, 0),
                                2,
                            )
                            print(f"[ANPR ALERT] Plate Found: {clean_txt}")
                            break

        # Display Real-Time Low-Latency Feed
        cv2.imshow("Real-Time ANPR & Human Detection", annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

except KeyboardInterrupt:
    print("[INFO] Stopping system...")
finally:
    cam.stop()
    cv2.destroyAllWindows()