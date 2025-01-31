import cv2
import math
from ultralytics import YOLO

from config.settings import DETECTION_CONFIDENCE
from utils.logger import get_logger

logger = get_logger()

def detect_brands_in_video(model_path: str, video_path: str):
    """
    Loads YOLO model and runs detection on each frame of video.
    Returns a list of (brand_name, time_in_seconds).
    """
    logger.info(f"Loading model from: {model_path}")
    model = YOLO(model_path)

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    detections_log = []

    frame_id = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Optionally skip frames if speed is an issue
        # e.g. if frame_id % 5 != 0: frame_id += 1; continue

        # Inference with Ultralytics
        results = model.predict(source=frame, conf=DETECTION_CONFIDENCE, verbose=False)
        # YOLOv8 returns a list of Results objects
        if len(results) > 0:
            for box in results[0].boxes:
                cls_id = int(box.cls[0].item())  # class index
                conf = float(box.conf[0].item())
                if conf >= DETECTION_CONFIDENCE:
                    brand_name = results[0].names[cls_id]
                    # Convert frame_id -> time in seconds
                    time_in_seconds = frame_id / fps
                    detections_log.append((brand_name, time_in_seconds))

        frame_id += 1

    cap.release()
    logger.info(f"Total detections: {len(detections_log)}")
    return detections_log