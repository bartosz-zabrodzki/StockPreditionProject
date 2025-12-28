import sys, os
print("PYTHON:", sys.executable)
print("VERSION:", sys.version)
print("PATH0:", os.environ.get("PATH", "")[:200])

import os
import joblib
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import timedelta
from src.models.config import (
    FEATURE_FILE,
    LSTM_MODEL_PATH,
    XGB_MODEL_PATH,
    SCALER_X_PATH,
    SCALER_Y_PATH,
    LSTM_CONFIG,
    FORECAST_DAYS,
    OUTPUT_DIR,
)



class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out

def load_data():
    if not os.path.exists(FEATURE_FILE):
        raise FileNotFoundError(f"[ERROR] Missing feature file: {FEATURE_FILE}")
    df = pd.read_csv(FEATURE_FILE)
    df = df.dropna().reset_index(drop=True)
    print(f"[Data] Loaded {df.shape[0]} rows × {df.shape[1]} columns")
    return df

def load_models(input_size):
    lstm = LSTMModel(
        input_size=input_size,
        hidden_size=LSTM_CONFIG["hidden_size"],
        num_layers=LSTM_CONFIG["num_layers"],
        dropout=LSTM_CONFIG["dropout"]
    )
    state = torch.load(LSTM_MODEL_PATH, map_location="cpu", weights_only=True)
    lstm.load_state_dict(state)
    lstm.eval()

    xgb = joblib.load(XGB_MODEL_PATH)
    scaler_x = joblib.load(SCALER_X_PATH)
    scaler_y = joblib.load(SCALER_Y_PATH)

    print("[Models] LSTM, XGBoost, and scalers loaded successfully.")
    return lstm, xgb, scaler_x, scaler_y

def make_predictions(df):
    target = "Close"
    features = ["High", "Low", "Open", "Volume"]
    missing = [c for c in features if c not in df.columns]
    if missing:
        raise ValueError(f"[ERROR] Missing required feature columns: {missing}")

    x = df[features].values
    y = df[target].values.reshape(-1, 1)

    lstm, xgb, scaler_x, scaler_y = load_models(input_size=x.shape[1])
    x_scaled = scaler_x.transform(x)
    y_scaled = scaler_y.transform(y)

    SEQ_LEN = LSTM_CONFIG["input_seq_len"]
    x_seq = []
    for i in range(len(x_scaled) - SEQ_LEN):
        x_seq.append(x_scaled[i:(i + SEQ_LEN)])
    x_seq = np.array(x_seq)

    x_t = torch.tensor(x_seq, dtype=torch.float32)

    with torch.no_grad():
        lstm_pred_scaled = lstm(x_t).numpy()
    xgb_pred_scaled = xgb.predict(x_scaled[SEQ_LEN:]).reshape(-1, 1)

    lstm_pred = scaler_y.inverse_transform(lstm_pred_scaled)
    xgb_pred = scaler_y.inverse_transform(xgb_pred_scaled)

    df = df.iloc[SEQ_LEN:].copy()
    df["LSTM_Prediction"] = lstm_pred
    df["XGB_Prediction"] = xgb_pred
    print("[Prediction] Predictions completed successfully.")
    return df, features, lstm, xgb, scaler_x, scaler_y

def forecast_future(df, lstm, xgb, scaler_x, scaler_y, features, forecast_days=FORECAST_DAYS):
    print(f"[Forecast] Generating {forecast_days}-day future forecast...")
    last_date = pd.to_datetime(df["Date"].iloc[-1])
    x = df[features].values.copy()

    SEQ_LEN = LSTM_CONFIG["input_seq_len"]
    future_lstm, future_xgb = [], []

    for _ in range(forecast_days):
        x_seq = x[-SEQ_LEN:].reshape(1, SEQ_LEN, -1)
        x_scaled = scaler_x.transform(x[-1:].reshape(1, -1))
        x_t = torch.tensor(x_scaled.reshape(-1, 1, x_scaled.shape[1]), dtype=torch.float32)

        with torch.no_grad():
            lstm_pred_scaled = lstm(x_t).numpy()
        xgb_pred_scaled = xgb.predict(scaler_x.transform(x[-1:].reshape(1, -1))).reshape(-1, 1)

        lstm_value = scaler_y.inverse_transform(lstm_pred_scaled)[0, 0]
        xgb_value = scaler_y.inverse_transform(xgb_pred_scaled)[0, 0]

        future_lstm.append(lstm_value)
        future_xgb.append(xgb_value)

        new_row = x[-1].copy()
        if "Close" in features:
                new_row[features.index("Close")] = lstm_value
        x= np.vstack([x, new_row])


    future_dates = [last_date + timedelta(days=i + 1) for i in range(forecast_days)]
    forecast_df = pd.DataFrame({
        "Date": future_dates,
        "LSTM_Forecast": future_lstm,
        "XGB_Forecast": future_xgb
    })
    print("[Forecast] Forecast generated successfully.")
    return forecast_df



def visualize_predictions(df):
    plt.figure(figsize=(28, 14))
    plt.style.use("seaborn-v0_8-darkgrid")

    plt.plot(df["Date"], df["Close"], label="Actual", color="steelblue", linewidth=2)
    plt.plot(df["Date"], df["LSTM_Prediction"], label="LSTM Prediction", linestyle="--", color="darkorange",
             linewidth=2)
    plt.plot(df["Date"], df["XGB_Prediction"], label="XGBoost Prediction", linestyle=":", color="green", linewidth=2)

    plt.title("AAPL Stock Price Prediction (LSTM vs XGBoost)", fontsize=16, fontweight="bold")
    plt.xlabel("Date", fontsize=12)
    plt.ylabel("Stock Price (USD)", fontsize=12)
    plt.legend(frameon=True, fontsize=10)
    plt.grid(alpha=0.3)
    plt.tight_layout()

    output_path = os.path.join(OUTPUT_DIR, "AAPL_predictions_plot.png")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"[Plot] Saved high-quality chart → {output_path}")

    plt.show()


if __name__ == "__main__":
    df = load_data()
    df_pred, features, lstm, xgb, scaler_x, scaler_y = make_predictions(df)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_path = os.path.join(OUTPUT_DIR, "AAPL_predictions.csv")

    df_pred.to_csv(output_path, index=False, encoding="utf-8")
    print(f"[Save] Predictions saved → {output_path}")

    visualize_predictions(df_pred)

    forecast_df = forecast_future(df_pred, lstm, xgb, scaler_x, scaler_y, features)
    forecast_path = os.path.join(OUTPUT_DIR, "AAPL_forecast.csv")
    forecast_df.to_csv(forecast_path, index=False, encoding="utf-8")
    print(f"[Save] Forecast saved → {forecast_path}")










