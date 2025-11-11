import os
import numpy as np
import pandas as pd
import torch
import logging
import torch.nn as nn
import tensorflow
from pyexpat import features
from sklearn.metrics import r2_score
from sympy.logic.utilities import load_file
from tensorflow.python.distribute.multi_process_runner import manager
from tensorflow.python.keras.backend import dtype
from tensorflow.python.ops.losses.losses_impl import mean_squared_error
from torch.utils.data import DataLoader, TensorDataset
import xgboost as xgb
import logging
from datetime import datetime
from sklearn.model_selection import train_test_split
from torch.xpu import device

from src.data.data import StockDataLoader

class TrainingManager:

    def __init__(self, logs_dir="logs", model_dir ="models", batch_size= 32, epochs=20, lr=0.001):
        self.logs_dir = logs_dir
        self.model_dir = model_dir
        self.batch_size = batch_size
        self.epochs = epochs
        self.lr = lr

        os.makedirs(logs_dir, exist_ok=True)
        os.makedirs(model_dir, exist_ok=True)
        self.logger = self._setup_logger()

    def _setup_logger(self):
        logs_file = os.path.join(self.logs_dir, "training.log")
        logger = logging.getLogger("TrainingManager")

        if not logger.handlers:
            logger.setLevel(logging.INFO)
            fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", "%Y-%m-%d %H:%M:%S")

            fh = logging.FileHandler(logs_file)
            fh.setFormatter(fmt)
            ch = logging.StreamHandler()
            ch.setFormatter(fmt)

            logger.addHandler(fh)
            logger.addHandler(ch)

        return logger

    def _log(self, level, msg):
        getattr(self.logger, level)(msg)


    def prepare_sequences(self, data, seq_len=30):
        close = data["Close"].values
        x, y = [], []

        for i in range(seq_len, len(close)):
            x.append(close[i - seq_len: i])
            y.append(close[i])
        x = np.array(x).reshape(-1, seq_len, 1)
        y = np.array(y).reshape(-1, 1)
        return x, y

class LSTMModel(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=64, num_layers =2, output_dim=1):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True )
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        out = self.fc(out)
        return out

def train_lstm(manager,train_df, test_df):
    manager._log("info", "Starting LSTM training...")
    seq_len = 30
    x_train, y_train = manager.prepare_sequences(train_df, seq_len)
    x_test, y_test = manager.prepare_sequences(test_df, seq_len)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = LSTMModel().to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=manager.lr)

    train_data = TensorDataset(
        torch.tensor(x_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.float32)
    )
    loader = DataLoader(train_data, batch_size=manager.batch_size, shuffle=True, drop_last= True)

    for epoch in range(manager.epochs):
        model.train()
        total_loss = 0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        manager._log("info", f"LSTM Epoch {epoch+1}/{manager.epochs} | Loss: {total_loss/len(loader):.6f}")

    model.eval()
    with torch.no_grad():
        X_test_tensor = torch.tensor(x_test, dtype=torch.float32).to(device)
        preds = model(X_test_tensor).cpu().numpy()

    min_len = min(len(preds), len(y_test))
    preds = preds[:min_len]
    y_true = y_test[:min_len]

    rmse = np.sqrt(mean_squared_error(y_true, preds))
    r2 = r2_score(y_true, preds)
    manager._log("info", f"LSTM Evaluation | RMSE: {rmse:.4f} | R2: {r2:.4f}")

    return rmse, r2

def train_xgboost(manager,train_df, test_df):
    manager._log("info", "Starting XGBoost training...")
    features = ["Open", "High", "Low", "Close", "Volume"]
    target = "Close"
    x_train, y_train = train_df[features], train_df[target]
    x_test, y_test = test_df[features], test_df[target]
    y_test = np.ravel(y_test)

    model = xgb.XGBRegressor(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
    )
    model.fit(x_train, y_train, eval_set=[(x_test, y_test)], verbose=False)

    preds = model.predict(x_test)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2 = r2_score(y_test, preds)

    manager._log("info", f"XGBoost Evaluation | RMSE: {rmse:.4f} | R2: {r2:.4f}")

    model.save_model(os.path.join(manager.model_dir, "xgboost_model.json"))
    manager._log("info", "Saved XGBoost model -> models/xgboost_model.json")


    return rmse, r2

if __name__ == "__main__":
    loader = StockDataLoader(cache_expiry_days=1)
    manager = TrainingManager(logs_dir="logs", model_dir="models", epochs=10, batch_size=32, lr=0.001)


    df = loader.fetch("AAPL", start="2020-01-01", interval="1d")
    df = loader.preprocess(df, normalize=True)
    train_df, test_df = loader.split(df)

    lstm_rmse, lstm_r2 = train_lstm(manager, train_df, test_df)

    xgb_rmse, xgb_r2 = train_xgboost(manager, train_df, test_df)

    manager._log("info", f"SUMMARY -> LSTM RMSE: {lstm_rmse:.4f}, R2: {lstm_r2:.4f} | XGB RMSE: {xgb_rmse:.4f}, R2: {xgb_r2:.4f}")
    print("Training complete. See logs/training.log for full report.")