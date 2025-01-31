import os
import argparse

from data.download_dataset import (
    download_logodet_3k,
    download_logodet_3k_kaggle
)
from models.inference import detect_brands_in_video
from utils.video_utils import group_detections_by_time, cut_video_segments
from config.settings import MODEL_DIR, DATA_DIR, RESULTS_DIR
from utils.logger import get_logger
from models.train import train_model

logger = get_logger()

def main():
    parser = argparse.ArgumentParser(
        description="Detect brand appearances in a video and cut segments around them."
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="",
        help="Path to a trained YOLO model (.pt). If not provided, will try to train or use the default path."
    )
    parser.add_argument(
        "--no-train",
        action="store_true",
        help="Skip training even if no model is found. (Will fail if model-path isn't valid.)"
    )
    parser.add_argument(
        "--video",
        type=str,
        default="input_video.mp4",
        help="Path to the input video you want to process."
    )
    # Existing RoboFlow download
    parser.add_argument(
        "--download-dataset",
        action="store_true",
        help="Force download the LogoDet-3K dataset from RoboFlow."
    )
    # New Kaggle download
    parser.add_argument(
        "--download-dataset-kaggle",
        action="store_true",
        help="Force download the LogoDet-3K dataset (lyly99/logodet3k) using kagglehub."
    )

    args = parser.parse_args()

    # 1. (Optional) Download dataset from either RoboFlow or Kaggle
    #    They can also be combined if needed, but typically you'd choose one.
    if args.download_dataset:
        if not os.path.exists(DATA_DIR):
            download_logodet_3k()
        else:
            logger.info("DATA_DIR already exists, skipping RoboFlow download.")

    if args.download_dataset_kaggle:
        # If the user specifically wants the Kaggle version:
        if not os.path.exists(DATA_DIR):
            download_logodet_3k_kaggle()
        else:
            logger.info("DATA_DIR already exists, skipping Kaggle download.")

    # 2. Determine the model to use
    if args.model_path:
        # Use user-provided path
        best_model_path = args.model_path
        logger.info(f"Using user-provided model path: {best_model_path}")
    else:
        # Default to whatever was trained
        best_model_path = os.path.join(MODEL_DIR, "logodet3k_yolov8", "weights", "best.pt")

    # If user didn't ask us to skip training, check if best_model_path exists. If not, train.
    if not args.no_train:
        if not os.path.exists(best_model_path):
            logger.info(f"No model found at '{best_model_path}'. Starting training...")
            train_model()
            # After training, best_model_path should exist
        else:
            logger.info(f"Found trained model at '{best_model_path}'. Skipping training.")
    else:
        # --no-train was specified
        if not os.path.exists(best_model_path):
            raise FileNotFoundError(
                f"Model path '{best_model_path}' not found, and training is disabled."
            )

    # 3. Detect brand appearances in the video
    sample_video = args.video
    if not os.path.exists(sample_video):
        raise FileNotFoundError(f"Cannot find video: {sample_video}")

    detections = detect_brands_in_video(best_model_path, sample_video)
    logger.info(f"Number of brand detections: {len(detections)}")

    # 4. Group detections
    grouped_segments = group_detections_by_time(detections, max_gap=1.0)
    logger.info(f"Grouped segments: {grouped_segments}")

    # 5. Cut video segments
    output_clips_dir = os.path.join(RESULTS_DIR, "clips")
    cut_video_segments(sample_video, grouped_segments, output_clips_dir)

if __name__ == "__main__":
    main()