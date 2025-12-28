import torch, sys
print("torch file:", torch.__file__)
print("python:", sys.executable)
import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score
from xgboost import XGBRegressor
import joblib
from torch.utils.data import DataLoader, TensorDataset
from src.models.config import (
    FEATURE_FILE,
    MODEL_DIR,
    SCALER_X_PATH,
    SCALER_Y_PATH,
    TRAINING_HISTORY_PATH,
    LSTM_CONFIG,
    XGB_CONFIG
)



def save_overwrite(path, data, is_torch=False):
    if os.path.exists(path):
        print(f"[Warning] Overwriting existing file: {path}")
        os.remove(path)
    if is_torch:
        torch.save(data, path)
    else:
        joblib.dump(data, path)
    print(f"[Save] Successfully saved → {path}")



print(f"[Init] Using data file: {FEATURE_FILE}")
os.makedirs(MODEL_DIR, exist_ok=True)

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2, dropout=0.2):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out

def load_features(DATA_PATH):
    print(f"[Data] Loading processed features from: {DATA_PATH}")

    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(
            f"[ERROR] Processed data not found: {DATA_PATH}\n"
            f"Please run R analysis first to generate 'AAPL_features.csv'."
        )

    df = pd.read_csv(DATA_PATH)
    df = df.dropna().reset_index(drop=True)
    for col in df.columns:
        try:
            df[col] = pd.to_numeric(df[col])
        except Exception:
            pass

    print(f"[Data] Loaded shape: {df.shape}")
    return df


def train_lstm(x_train, y_train, x_val, y_val):
    input_size = x_train.shape[2]
    model = LSTMModel(input_size,  hidden_size=LSTM_CONFIG["hidden_size"], num_layers=LSTM_CONFIG["num_layers"], dropout= LSTM_CONFIG["dropout"])
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LSTM_CONFIG["learning_rate"],
        weight_decay=LSTM_CONFIG["weight_decay"]
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
    )

    train_dataset = TensorDataset(x_train, y_train)
    val_dataset = TensorDataset(x_val, y_val)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    best_loss = float("inf")
    patience_counter = 0
    train_losses, val_losses = [], []

    for epoch in range(1, LSTM_CONFIG["epochs"] + 1):
        model.train()
        optimizer.zero_grad()
        for xb, yb in train_loader:
            optimizer.zero_grad()
            output = model(xb)
            loss = criterion(output, yb)
            loss.backward()
            optimizer.step()


        val_loss = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                preds = model(xb)
                val_loss += criterion(preds, yb).item()
        val_loss /= len(val_loader)

        model.eval()
        with torch.no_grad():
            val_output = model(x_val)
            val_loss = criterion(val_output, y_val)

        scheduler.step(val_loss)
        train_losses.append(loss.item())
        val_losses.append(val_loss.item())

        print(f"[LSTM] Epoch {epoch}/{LSTM_CONFIG['epochs']} | Train Loss: {loss.item():.6f} | Val Loss: {val_loss.item():.6f}")

        if val_loss < best_loss:
            best_loss = val_loss
            patience_counter = 0
            best_state = model.state_dict()
        else:
            patience_counter += 1

        if patience_counter >= LSTM_CONFIG["patience"]:
            print(f"[LSTM] Early stopping triggered at epoch {epoch}.")
            break
    save_overwrite(os.path.join(MODEL_DIR, "lstm_model.pth"), best_state, is_torch=True)

    model.load_state_dict(best_state)
    torch.save(model.state_dict(), os.path.join(MODEL_DIR, "lstm_model.pth"))
    print(f"[LSTM] Best model saved → {os.path.join(MODEL_DIR, 'lstm_model.pth')}")

    model.eval()
    with torch.no_grad():
        preds = model(x_val).numpy()
    rmse = np.sqrt(mean_squared_error(y_val.numpy(), preds))
    r2 = r2_score(y_val.numpy(), preds)
    print(f"[LSTM] Final RMSE: {rmse:.4f}, R²: {r2:.4f}")

    history = pd.DataFrame(
        {"epoch": range(1, len(train_losses) + 1), "train_loss": train_losses, "val_loss": val_losses})
    history.to_csv(TRAINING_HISTORY_PATH, index=False, encoding="utf-8")
    print(f"[LSTM] Training history saved → {TRAINING_HISTORY_PATH}")

    return rmse, r2

def train_xgboost(x_train, y_train, x_test, y_test):
    print("[XGBoost] Training model...")
    model = XGBRegressor(**XGB_CONFIG)
    model.fit(x_train, y_train)
    preds = model.predict(x_test)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2 = r2_score(y_test, preds)

    save_overwrite(os.path.join(MODEL_DIR, "xgboost_model.pkl"), model)
    save_overwrite(SCALER_X_PATH, scaler_X)
    save_overwrite(SCALER_Y_PATH, scaler_Y)

    print(f"[XGBoost] RMSE: {rmse:.4f}, R²: {r2:.4f}")
    joblib.dump(model, os.path.join(MODEL_DIR, "xgboost_model.pkl"))
    print(f"[XGBoost] Model saved to {os.path.join(MODEL_DIR, 'xgboost_model.pkl')}")
    return rmse, r2
if __name__ == "__main__":
    df = load_features(FEATURE_FILE)
    target = "Close"
    features = [col for col in df.columns if col != target and df[col].dtype != "object"]
    X, y = df[features].values, df[target].values.reshape(-1, 1)


    def create_sequences(x, y, seq_length):
        xs, ys = [], []
        for i in range(len(x) - seq_length):
            xs.append(x[i:(i + seq_length)])
            ys.append(y[i + seq_length])
        return np.array(xs), np.array(ys)

    scaler_X, scaler_Y = MinMaxScaler(), MinMaxScaler()
    x_scaled = scaler_X.fit_transform(X)
    y_scaled = scaler_Y.fit_transform(y)
    joblib.dump(scaler_X, SCALER_X_PATH)
    joblib.dump(scaler_Y, SCALER_Y_PATH)
    print(f"[Scaler] Saved → {SCALER_X_PATH}, {SCALER_Y_PATH}")

    seq_len = LSTM_CONFIG["input_seq_len"]
    x_seq, y_seq = create_sequences(x_scaled, y_scaled, seq_len)
    x_train, x_test, y_train, y_test = train_test_split(x_seq, y_seq, test_size=0.2, shuffle=False)
    x_train_flat, x_test_flat = x_train[:, -1, :], x_test[:, -1, :]

    x_train_t = torch.tensor(x_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.float32)
    x_test_t = torch.tensor(x_test, dtype=torch.float32)
    y_test_t = torch.tensor(y_test, dtype=torch.float32)

    lstm_rmse, lstm_r2 = train_lstm(x_train_t, y_train_t, x_test_t, y_test_t)
    xgb_rmse, xgb_r2 = train_xgboost(x_train_flat, y_train, x_test_flat, y_test)

    print(f"\nSUMMARY → LSTM RMSE: {lstm_rmse:.4f}, R²: {lstm_r2:.4f} | XGB RMSE: {xgb_rmse:.4f}, R²: {xgb_r2:.4f}")

