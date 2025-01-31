import os
from ultralytics import YOLO
from config.settings import DATA_DIR, MODEL_DIR
from utils.logger import get_logger

logger = get_logger()

def train_model():
    """
    Trains a YOLOv8 model on the LogoDet-3K data (in YOLO format).
    Expects 'data.yaml' in DATA_DIR.
    """
    data_yaml_path = os.path.join(DATA_DIR, "data.yaml")
    model = YOLO("yolov8n.pt")  # start from a pretrained YOLOv8n model
    logger.info("Starting training on data: " + data_yaml_path)

    results = model.train(
        data=data_yaml_path,
        epochs=50,
        imgsz=640,
        project=MODEL_DIR,
        name="logodet3k_yolov8",
        pretrained=True
    )
    logger.info("Training complete. Best model path: " + str(results.best))

if __name__ == "__main__":
    os.makedirs(MODEL_DIR, exist_ok=True)
    train_model()