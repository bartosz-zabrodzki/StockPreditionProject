import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score
import torch
from torch import nn

def load_features(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.dropna()
    return df

def split_data(df, target_col="Close", train_size=0.8):
    n = int(len(df) * train_size)
    train_df = df.iloc[:n]
    test_df = df.iloc[n:]
    return train_df, test_df

def prepare_lstm_sequences(df, target_col="Close", seq_len=30):
    data = df[target_col].values
    X, y = [], []
    for i in range(seq_len, len(data)):
        X.append(data[i-seq_len:i])
        y.append(data[i])
    return np.array(X), np.array(y)

def evaluate(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    return rmse, r2

class LSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out
