import os

# Optionally load from environment variables if you want
ROBOFLOW_API_KEY = os.getenv("ROBOFLOW_API_KEY", "lk13Pj96Kas32pWaQyfS")
ROBOFLOW_WORKSPACE = "drobya"
ROBOFLOW_PROJECT = "LogoDet-3K"

# Local directories
DATA_DIR = os.path.join(os.getcwd(), "datasets", "logodet_3k")
MODEL_DIR = os.path.join(os.getcwd(), "trained_models")
RESULTS_DIR = os.path.join(os.getcwd(), "results")

# In seconds, how much buffer to add before/after brand detection
SEGMENT_PADDING = 2.0

# Confidence threshold for detection
DETECTION_CONFIDENCE = 0.5