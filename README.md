# BrandsVideoSegmentor

A production-ready pipeline for:
1. Training a YOLOv8 model on RoboFlow's LogoDet-3K dataset.
2. Detecting brand logos in videos.
3. Cutting short clips around each brand appearance.

## Setup

1. **Clone** this repo.
2. **Install** dependencies:

   ```bash
   pip install -r requirements.txt
   ```

   ```bash
   python src/main.py --download-dataset-kaggle --video /Users/drobya/Downloads/index0-2.ts
   ```