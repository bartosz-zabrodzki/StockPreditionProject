import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]  # repo root
DATA_DIR   = Path(os.getenv("DATA_ROOT",  str(PROJECT_ROOT / "data")))
OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", str(PROJECT_ROOT / "output")))
MODEL_DIR  = Path(os.getenv("MODEL_DIR",  str(PROJECT_ROOT / "models")))
LOG_DIR    = Path(os.getenv("LOG_DIR",    str(PROJECT_ROOT / "logs")))
CACHE_DIR = DATA_DIR / "data_cache"
PROCESSED_DIR = DATA_DIR / "data_processed"
FEATURES_DIR = PROCESSED_DIR / "features"
FORECASTS_DIR = PROCESSED_DIR / "forecasts"
SCALERS_DIR = MODEL_DIR / "scalers"



def cache_file(ticker: str, interval: str) -> Path:
    return CACHE_DIR / f"{ticker}_{interval}.csv"

def ensure_dirs():
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    FEATURES_DIR.mkdir(parents=True, exist_ok=True)
    FORECASTS_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    SCALERS_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)



