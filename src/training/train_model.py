from __future__ import annotations
import os, sys, joblib, json
import numpy as np, pandas as pd, torch
import torch.nn as nn
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score
from xgboost import XGBRegressor
from torch.utils.data import DataLoader, TensorDataset
from src.prediction.predict import run as run_predict, PredictConfig
from src.config.paths import MODEL_DIR, FEATURES_DIR, OUTPUT_DIR
from src.models.config import get_model_paths, LSTM_CONFIG, XGB_CONFIG


# ============================================================
# UTILS
# ============================================================

def save_feature_schema(ticker: str, feature_cols: list[str]) -> Path:
    p = Path(MODEL_DIR) / f"feature_schema_{ticker}.json"
    payload = {"ticker": ticker, "feature_cols": feature_cols}
    p.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"[Schema] Saved feature schema → {p}")
    return p


def save_overwrite(path, data, is_torch=False):
    force = os.environ.get("FORCE", "1") == "1"
    if os.path.exists(path) and not force:
        raise FileExistsError(f"File exists: {path}. Set FORCE=1 to overwrite.")
    if os.path.exists(path):
        os.remove(path)
    if is_torch:
        torch.save(data, path)
    else:
        joblib.dump(data, path)
    print(f"[Save] Successfully saved → {path}")


# ============================================================
# MODELS
# ============================================================

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])


# ============================================================
# TRAINING
# ============================================================

def train_lstm(x_train, y_train, x_val, y_val, lstm_path, history_path):
    input_size = x_train.shape[2]
    model = LSTMModel(
        input_size,
        hidden_size=LSTM_CONFIG["hidden_size"],
        num_layers=LSTM_CONFIG["num_layers"],
        dropout=LSTM_CONFIG["dropout"],
    )

    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LSTM_CONFIG["learning_rate"],
        weight_decay=LSTM_CONFIG["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
    )

    train_loader = DataLoader(TensorDataset(x_train, y_train), batch_size=32, shuffle=True)
    val_loader = DataLoader(TensorDataset(x_val, y_val), batch_size=32, shuffle=False)

    best_loss = float("inf")
    patience_counter = 0
    train_losses, val_losses = [], []
    best_state = None

    for epoch in range(1, LSTM_CONFIG["epochs"] + 1):
        model.train()
        last_loss = None

        for xb, yb in train_loader:
            optimizer.zero_grad()
            out = model(xb)
            loss = criterion(out, yb)
            loss.backward()
            optimizer.step()
            last_loss = loss

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                preds = model(xb)
                val_loss += criterion(preds, yb).item()
        val_loss /= max(len(val_loader), 1)

        scheduler.step(val_loss)
        train_losses.append(float(last_loss.item() if last_loss is not None else 0.0))
        val_losses.append(float(val_loss))

        print(f"[LSTM] Epoch {epoch}/{LSTM_CONFIG['epochs']} | Train Loss={train_losses[-1]:.6f} | Val Loss={val_losses[-1]:.6f}")

        if val_loss < best_loss:
            best_loss = val_loss
            patience_counter = 0
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1

        if patience_counter >= LSTM_CONFIG["patience"]:
            print(f"[LSTM] Early stopping triggered at epoch {epoch}.")
            break

    if best_state is None:
        raise RuntimeError("LSTM training failed: best_state is None")

    save_overwrite(lstm_path, best_state, is_torch=True)
    model.load_state_dict(best_state)
    model.eval()

    with torch.no_grad():
        preds = model(x_val).numpy()

    rmse = float(np.sqrt(mean_squared_error(y_val.numpy(), preds)))
    r2 = float(r2_score(y_val.numpy(), preds))
    print(f"[LSTM] Final RMSE={rmse:.4f}, R²={r2:.4f}")

    history = pd.DataFrame({"epoch": range(1, len(train_losses) + 1), "train_loss": train_losses, "val_loss": val_losses})
    history.to_csv(history_path, index=False)
    print(f"[LSTM] Training history saved → {history_path}")

    return rmse, r2


def train_xgboost(x_train, y_train, x_test, y_test, scaler_X, scaler_Y, xgb_path):
    print("[XGBoost] Training model...")
    model = XGBRegressor(**XGB_CONFIG)
    y_train_1d = np.asarray(y_train).reshape(-1)
    y_test_1d = np.asarray(y_test).reshape(-1)

    model.fit(x_train, y_train_1d)
    preds = model.predict(x_test)
    rmse = float(np.sqrt(mean_squared_error(y_test_1d, preds)))
    r2 = float(r2_score(y_test_1d, preds))

    save_overwrite(xgb_path, model)
    print(f"[XGBoost] Saved → {xgb_path}")
    print(f"[XGBoost] RMSE={rmse:.4f}, R²={r2:.4f}")
    return rmse, r2


# ============================================================
# MAIN PIPELINE
# ============================================================

def main() -> None:
    from src.config.paths import ensure_dirs
    ensure_dirs()

    paths = get_model_paths()
    TICKER = paths["TICKER"]
    FEATURE_FILE = paths["FEATURE_FILE"]
    LSTM_MODEL_PATH = paths["LSTM_MODEL_PATH"]
    XGB_MODEL_PATH = paths["XGB_MODEL_PATH"]
    SCALER_X_PATH = paths["SCALER_X_PATH"]
    SCALER_Y_PATH = paths["SCALER_Y_PATH"]
    TRAINING_HISTORY_PATH = paths["TRAINING_HISTORY_PATH"]

    print(f"\n[Init] Training started for {TICKER}")
    print(f"[Init] Using feature file: {FEATURE_FILE}")

    if not Path(FEATURE_FILE).exists():
        raise FileNotFoundError(f"Missing feature file: {FEATURE_FILE}")

    df = pd.read_csv(FEATURE_FILE).dropna().reset_index(drop=True)
    target = "Close"

    numeric_cols = [
        c for c in df.columns
        if c not in {"Date", target} and pd.api.types.is_numeric_dtype(df[c])
    ]
    save_feature_schema(TICKER, numeric_cols)

    X = df[numeric_cols].values
    y = df[[target]].values
    scaler_X, scaler_Y = MinMaxScaler(), MinMaxScaler()
    x_scaled = scaler_X.fit_transform(X)
    y_scaled = scaler_Y.fit_transform(y)

    seq_len = int(LSTM_CONFIG["input_seq_len"])
    xs, ys = [], []
    for i in range(len(x_scaled) - seq_len):
        xs.append(x_scaled[i:i+seq_len])
        ys.append(y_scaled[i+seq_len])
    x_seq, y_seq = np.array(xs), np.array(ys)

    x_train, x_test, y_train, y_test = train_test_split(x_seq, y_seq, test_size=0.2, shuffle=False)
    x_train_flat, x_test_flat = x_train[:, -1, :], x_test[:, -1, :]

    x_train_t = torch.tensor(x_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.float32)
    x_test_t = torch.tensor(x_test, dtype=torch.float32)
    y_test_t = torch.tensor(y_test, dtype=torch.float32)

    # Save scalers
    SCALER_X_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(scaler_X, SCALER_X_PATH)
    joblib.dump(scaler_Y, SCALER_Y_PATH)
    print(f"[Scaler] Saved → {SCALER_X_PATH}, {SCALER_Y_PATH}")

    lstm_rmse, lstm_r2 = train_lstm(x_train_t, y_train_t, x_test_t, y_test_t, LSTM_MODEL_PATH, TRAINING_HISTORY_PATH)
    xgb_rmse, xgb_r2 = train_xgboost(x_train_flat, y_train, x_test_flat, y_test, scaler_X, scaler_Y, XGB_MODEL_PATH)

    print(f"\n[SUMMARY] {TICKER}: LSTM RMSE={lstm_rmse:.4f}, R²={lstm_r2:.4f} | XGB RMSE={xgb_rmse:.4f}, R²={xgb_r2:.4f}")


# ============================================================
# ENTRYPOINT
# ============================================================



def run(ticker: str | None = None):
    if ticker:
        os.environ["STOCK_TICKER"] = ticker
    active_ticker = os.getenv("STOCK_TICKER", "AAPL")

    print(f"[Train] Starting full training pipeline for {active_ticker}")
    main()
    print(f"[Train] Completed training for {active_ticker}")

    config = PredictConfig(
        feature_file=FEATURES_DIR / f"{active_ticker}_features.csv",
        output_dir=OUTPUT_DIR,
        ticker=active_ticker,
        forecast_days=30,
    )
    run_predict(config)



if __name__ == "__main__":
    print("[CLI] Manual training run starting...")
    main()
