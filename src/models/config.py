import os

# === Base directories ===
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_DIR = os.path.join(BASE_DIR, "data")
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, "data_processed", "features")
MODEL_DIR = os.path.join(BASE_DIR, "models")
OUTPUT_DIR = os.path.join(DATA_DIR, "data_output")
SCALER_DIR = os.path.join(MODEL_DIR, "scalers")

# Ensure directories exist
for path in [MODEL_DIR, OUTPUT_DIR, SCALER_DIR]:
    os.makedirs(path, exist_ok=True)

# === File paths ===
FEATURE_FILE = os.path.join(PROCESSED_DATA_DIR, "AAPL_features.csv")
LSTM_MODEL_PATH = os.path.join(MODEL_DIR, "lstm_model.pth")
XGB_MODEL_PATH = os.path.join(MODEL_DIR, "xgboost_model.pkl")
TRAINING_HISTORY_PATH = os.path.join(MODEL_DIR, "lstm_training_history.csv")
SCALER_X_PATH = os.path.join(SCALER_DIR, "scaler_x.pkl")
SCALER_Y_PATH = os.path.join(SCALER_DIR, "scaler_y.pkl")

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
    "batch_size": 32
}

# === XGBoost configuration ===
XGB_CONFIG = {
    "n_estimators": 500,
    "learning_rate": 0.05,
    "max_depth": 6,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": 42
}

# === Forecast settings ===
FORECAST_DAYS = 30

