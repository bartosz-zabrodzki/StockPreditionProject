import os

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
R_DIR = os.path.join(SRC_DIR, "r")
DATA_DIR = os.path.join(SRC_DIR, "data")
CACHE_DIR = os.path.join(DATA_DIR, "data_cache")
PROCESSED_DIR = os.path.join(DATA_DIR, "data_processed")
FEATURES_DIR = os.path.join(PROCESSED_DIR, "features")
FORECASTS_DIR = os.path.join(PROCESSED_DIR, "forecasts")
OUTPUTS_DIR = os.path.join(SRC_DIR, "outputs")

os.makedirs(OUTPUTS_DIR, exist_ok=True)
