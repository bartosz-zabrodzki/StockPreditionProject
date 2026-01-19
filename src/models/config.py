# ============================================================
# src/models/config.py — dynamic per-ticker model configuration
# ============================================================

import os
from pathlib import Path
from src.config.paths import FEATURES_DIR, MODEL_DIR, SCALERS_DIR, OUTPUT_DIR

# === LSTM model configuration ===
LSTM_CONFIG = {
    "input_seq_len": 120,     # number of timesteps per sample
    "hidden_size": 256,
    "num_layers": 4,
    "dropout": 0.3,
    "learning_rate": 0.0003,
    "weight_decay": 1e-4,
    "epochs": 150,
    "patience": 12,
    "batch_size": 32,
}

# === XGBoost configuration ===
XGB_CONFIG = {
    "n_estimators": 500,
    "learning_rate": 0.05,
    "max_depth": 6,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": 42,
}

# === Forecast settings ===
FORECAST_DAYS = 30


# ============================================================
# Dynamic paths (computed at runtime)
# ============================================================
def get_model_paths():
    """Compute file paths dynamically based on environment (ticker, etc.)."""
    ticker = os.getenv("STOCK_TICKER", "AAPL")
    interval = os.getenv("STOCK_INTERVAL", "1d")
    period = os.getenv("STOCK_PERIOD", "max")

    feature_file = FEATURES_DIR / f"{ticker}_features.csv"

    paths = {
        "TICKER": ticker,
        "INTERVAL": interval,
        "PERIOD": period,
        "FEATURE_FILE": feature_file,
        "LSTM_MODEL_PATH": MODEL_DIR / f"lstm_model_{ticker}.pth",
        "XGB_MODEL_PATH": MODEL_DIR / f"xgboost_model_{ticker}.pkl",
        "SCALER_X_PATH": SCALERS_DIR / f"scaler_x_{ticker}.pkl",
        "SCALER_Y_PATH": SCALERS_DIR / f"scaler_y_{ticker}.pkl",
        "TRAINING_HISTORY_PATH": MODEL_DIR / f"lstm_training_history_{ticker}.csv",
        "PREDICTIONS_CSV": OUTPUT_DIR / f"{ticker}_predictions.csv",
        "PREDICTIONS_PNG": OUTPUT_DIR / f"{ticker}_predictions_plot.png",
        "FORECAST_CSV": OUTPUT_DIR / f"{ticker}_forecast.csv",
    }
    return paths
    # ============================================================
    # Overwrite / Cleanup utilities
    # ============================================================

def clean_old_outputs(ticker: str):

    from src.config.paths import OUTPUT_DIR, MODEL_DIR
    import logging

    log = logging.getLogger(__name__)
    deleted = 0

        # Remove old outputs (CSV, PNG, logs)
    for old in Path(OUTPUT_DIR).glob(f"{ticker}_*"):
        try:
            old.unlink()
            deleted += 1
            log.info(f"[ConfigCleanup] Deleted old output → {old.name}")
        except Exception as e:
            log.warning(f"[ConfigCleanup] Could not delete {old.name}: {e}")

        # Optional: clear obsolete model or scaler files if retraining
    for old in Path(MODEL_DIR).glob(f"*{ticker}*.tmp"):
        try:
            old.unlink()
            deleted += 1
            log.info(f"[ConfigCleanup] Deleted temp model → {old.name}")
        except Exception as e:
            log.warning(f"[ConfigCleanup] Could not delete {old.name}: {e}")

    if deleted == 0:
        log.info(f"[ConfigCleanup] No stale outputs found for {ticker}.")
    else:
        log.info(f"[ConfigCleanup] {deleted} stale artifacts removed for {ticker}.")



