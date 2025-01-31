import os
import shutil
import glob
import random
import xml.etree.ElementTree as ET

import roboflow
import kagglehub

from config.settings import (
    ROBOFLOW_API_KEY,
    ROBOFLOW_WORKSPACE,
    ROBOFLOW_PROJECT,
    DATA_DIR
)
from utils.logger import get_logger

logger = get_logger()

##########################################################
# 1) ROBOFLOW DOWNLOAD (unchanged)
##########################################################
def download_logodet_3k():
    """
    Uses the roboflow package to download the LogoDet-3K dataset
    in YOLOv8 format (Version 1).
    """
    logger.info("Downloading LogoDet-3K dataset from RoboFlow...")
    rf = roboflow.Roboflow(api_key=ROBOFLOW_API_KEY)
    project = rf.workspace(ROBOFLOW_WORKSPACE).project(ROBOFLOW_PROJECT)
    project.version(1).download("yolov8", location=DATA_DIR)
    logger.info(f"Dataset downloaded to: {DATA_DIR}")

##########################################################
# 2) KAGGLE DOWNLOAD + CONVERSION
##########################################################
def download_logodet_3k_kaggle():
    """
    1) Downloads the Kaggle LogoDet-3K dataset (lyly99/logodet3k).
    2) Converts it from PASCAL VOC .xml -> YOLO format.
    3) Writes out data.yaml and train/valid folders under DATA_DIR.
    """
    logger.info("Downloading LogoDet-3K dataset from Kaggle (lyly99/logodet3k)...")
    path = kagglehub.dataset_download("lyly99/logodet3k")  # downloads latest version
    logger.info(f"Kaggle raw dataset path: {path}")

    # Convert it automatically
    logger.info("Converting Kaggle dataset (VOC .xml) -> YOLOv8 format...")
    convert_kaggle_logodet_to_yolo(
        kaggle_dir=path,
        output_dir=DATA_DIR,
        train_ratio=0.8  # 80% train, 20% valid
    )
    logger.info("Kaggle dataset conversion complete!")

##########################################################
# 3) CONVERSION FUNCTION (VOC -> YOLO)
##########################################################
def convert_kaggle_logodet_to_yolo(kaggle_dir, output_dir, train_ratio=0.8):
    """
    Convert the Kaggle LogoDet-3K dataset from PASCAL VOC (xml) to YOLOv8 format.
    - kaggle_dir: path to the raw Kaggle data (where .xml & .jpg are located).
    - output_dir: final YOLO dataset folder (will contain data.yaml, train/, valid/).
    - train_ratio: fraction of data in train (rest in valid).
    """
    # 1. Gather all .xml files (VOC annotations)
    xml_files = glob.glob(os.path.join(kaggle_dir, "**", "*.xml"), recursive=True)
    if not xml_files:
        raise FileNotFoundError(f"No .xml files found under {kaggle_dir}")

    # 2. Prepare output structure
    train_img_dir = os.path.join(output_dir, "train", "images")
    train_lbl_dir = os.path.join(output_dir, "train", "labels")
    valid_img_dir = os.path.join(output_dir, "valid", "images")
    valid_lbl_dir = os.path.join(output_dir, "valid", "labels")

    for d in [train_img_dir, train_lbl_dir, valid_img_dir, valid_lbl_dir]:
        os.makedirs(d, exist_ok=True)

    brand2id = {}
    current_id = 0

    # 3. Shuffle the dataset
    random.shuffle(xml_files)

    # 4. Parse each XML file
    for idx, xml_path in enumerate(xml_files):
        tree = ET.parse(xml_path)
        root = tree.getroot()

        # Extract image filename
        filename_node = root.find("filename")
        if filename_node is None:
            continue
        img_filename = filename_node.text.strip()

        # Actual image is typically in the same folder as the XML
        img_folder = os.path.dirname(xml_path)
        img_path = os.path.join(img_folder, img_filename)

        if not os.path.exists(img_path):
            # Possibly check for alternative extensions
            continue

        objects = root.findall("object")
        if not objects:
            continue

        size_node = root.find("size")
        if size_node is None:
            continue
        img_w = float(size_node.find("width").text)
        img_h = float(size_node.find("height").text)

        # We'll accumulate YOLO lines
        yolo_annotations = []

        for obj in objects:
            brand_name = obj.find("name").text.strip()

            # Brand -> ID
            if brand_name not in brand2id:
                brand2id[brand_name] = current_id
                current_id += 1
            cls_id = brand2id[brand_name]

            bndbox = obj.find("bndbox")
            if bndbox is None:
                continue
            xmin = float(bndbox.find("xmin").text)
            ymin = float(bndbox.find("ymin").text)
            xmax = float(bndbox.find("xmax").text)
            ymax = float(bndbox.find("ymax").text)

            # Convert VOC -> YOLO (normalized)
            x_center = (xmin + xmax) / 2.0 / img_w
            y_center = (ymin + ymax) / 2.0 / img_h
            w = (xmax - xmin) / img_w
            h = (ymax - ymin) / img_h

            yolo_line = f"{cls_id} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}"
            yolo_annotations.append(yolo_line)

        # Train/Valid split
        is_train = (idx < int(len(xml_files) * train_ratio))

        if is_train:
            dst_img_path = os.path.join(train_img_dir, img_filename)
            dst_lbl_path = os.path.join(train_lbl_dir, img_filename.replace(".jpg", ".txt"))
        else:
            dst_img_path = os.path.join(valid_img_dir, img_filename)
            dst_lbl_path = os.path.join(valid_lbl_dir, img_filename.replace(".jpg", ".txt"))

        # Copy image
        shutil.copy2(img_path, dst_img_path)

        # Write label txt
        with open(dst_lbl_path, "w") as f:
            for line in yolo_annotations:
                f.write(line + "\n")

    # 5. Create data.yaml with brand names
    id2brand = [b for b, i in sorted(brand2id.items(), key=lambda x: x[1])]
    num_classes = len(id2brand)

    data_yaml = f"""# YOLOv8 data file for LogoDet-3K (converted from Kaggle)
train: {os.path.join(output_dir, 'train')}
val: {os.path.join(output_dir, 'valid')}

nc: {num_classes}
names: [{', '.join([f"'{b}'" for b in id2brand])}]
"""

    with open(os.path.join(output_dir, "data.yaml"), "w") as f:
        f.write(data_yaml)

    logger.info(f"Conversion complete! YOLO data is in: {output_dir}")
    logger.info(f"Number of classes: {num_classes}")
