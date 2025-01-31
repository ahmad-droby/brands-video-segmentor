import os
import glob
import random
import xml.etree.ElementTree as ET
import shutil

# Adjust your paths here:
KAGGLE_DIR = "/Users/drobya/Documents/BrandsVideoSegmentor/datasets/logodet_3k/LogoDet-3K"
OUTPUT_DIR = "/Users/drobya/Documents/BrandsVideoSegmentor/datasets/logodet_3k"
TRAIN_RATIO = 0.8  # 80% for training, 20% for validation

def convert_kaggle_logodet_to_yolo(kaggle_dir=KAGGLE_DIR, output_dir=OUTPUT_DIR, train_ratio=TRAIN_RATIO):
    """
    Convert the Kaggle LogoDet-3K dataset from PASCAL VOC-style
    annotations to YOLOv8 format.
    """
    # 1. Gather all (xml) annotation files
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

    # We will collect brand names to build a brand -> id map
    brand2id = {}
    current_id = 0

    # 3. Shuffle the dataset
    random.shuffle(xml_files)

    # 4. Process each annotation
    for idx, xml_path in enumerate(xml_files):
        tree = ET.parse(xml_path)
        root = tree.getroot()

        # 4a. Extract image filename from <filename>
        filename_node = root.find("filename")
        if filename_node is None:
            continue
        img_filename = filename_node.text.strip()
        # The actual image (jpg) is typically in the same folder as the XML
        img_folder = os.path.dirname(xml_path)
        img_path = os.path.join(img_folder, img_filename)

        # Make sure image exists
        if not os.path.exists(img_path):
            # Some datasets name differently or have missing images
            # Possibly handle an alternate extension check here
            continue

        # 4b. Identify brand name(s) from <object><name> in the XML
        objects = root.findall("object")
        if not objects:
            continue

        # We will build a YOLO .txt lines array
        yolo_annotations = []

        # 4c. Get image dimensions from <size>
        size_node = root.find("size")
        if size_node is None:
            continue
        img_w = float(size_node.find("width").text)
        img_h = float(size_node.find("height").text)

        for obj in objects:
            brand_name = obj.find("name").text.strip()
            # Map brand to an integer ID
            if brand_name not in brand2id:
                brand2id[brand_name] = current_id
                current_id += 1
            cls_id = brand2id[brand_name]

            # 4d. Parse bounding box from <bndbox>
            bndbox = obj.find("bndbox")
            if bndbox is None:
                continue
            xmin = float(bndbox.find("xmin").text)
            ymin = float(bndbox.find("ymin").text)
            xmax = float(bndbox.find("xmax").text)
            ymax = float(bndbox.find("ymax").text)

            # 4e. Convert VOC to YOLO format: (x_center, y_center, width, height) normalized
            x_center = (xmin + xmax) / 2.0 / img_w
            y_center = (ymin + ymax) / 2.0 / img_h
            w = (xmax - xmin) / img_w
            h = (ymax - ymin) / img_h

            # YOLO annotation line: class_id x_center y_center width height
            yolo_line = f"{cls_id} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}"
            yolo_annotations.append(yolo_line)

        # 4f. Decide if this image goes to train or valid
        is_train = (idx < int(len(xml_files) * train_ratio))
        if is_train:
            dst_img_path = os.path.join(train_img_dir, img_filename)
            dst_lbl_path = os.path.join(train_lbl_dir, img_filename.replace(".jpg", ".txt"))
        else:
            dst_img_path = os.path.join(valid_img_dir, img_filename)
            dst_lbl_path = os.path.join(valid_lbl_dir, img_filename.replace(".jpg", ".txt"))

        # 4g. Copy image to new path
        shutil.copy2(img_path, dst_img_path)

        # 4h. Write the labels to .txt
        with open(dst_lbl_path, "w") as f:
            for line in yolo_annotations:
                f.write(line + "\n")

    # 5. Create data.yaml with brand names
    # Sort brand2id by ID to get consistent ordering
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

    print(f"Conversion complete! YOLO data is in: {output_dir}")
    print(f"Number of classes: {num_classes}")

if __name__ == "__main__":
    convert_kaggle_logodet_to_yolo()