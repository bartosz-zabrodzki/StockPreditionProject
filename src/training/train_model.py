import os
import torch
import subprocess
from torch import nn, optim
import xgboost as xgb
from main import ensure_r_environment, run_r_script
from model_utils import load_features, split_data, prepare_lstm_sequences, evaluate, LSTMModel

ensure_r_environment()

run_r_script("src/r/analyze.R")
run_r_script("src/r/diagnostics.R")

DATA_PATH = "src/data/data_processed/AAPL_features.csv"
MODEL_DIR = "src/models"

if not os.path.exists(DATA_PATH):
    print(f"[Warning] Missing {DATA_PATH}, generating via R script...")
    subprocess.run(
        ["C:/Program Files/R/R-4.5.2/bin/Rscript.exe", "--vanilla", "--encoding=UTF-8", "src/r/analyze.R"],
        check=True,
    )
    try:
        print("[R Bridge] Running analytics pipeline...")
        subprocess.run(
            ["C:/Program Files/R/R-4.5.2/bin/Rscript.exe", "--vanilla", "--encoding=UTF-8", "src/r/analyze.R"],
            check=True,
        )
        print("[R Bridge] Analytics completed successfully.")
    except subprocess.CalledProcessError as e:
        print(f"[Error] R script failed. Exit code: {e.returncode}")
        print("Check src/r/analyze.R manually using Rscript to debug.")
        raise

os.makedirs(MODEL_DIR, exist_ok=True)


df = load_features(DATA_PATH)
train_df, test_df = split_data(df)


x_train = train_df.drop(columns=["Close"])
y_train = train_df["Close"]
x_test = test_df.drop(columns=["Close"])
y_test = test_df["Close"]

xgb_model = xgb.XGBRegressor(n_estimators=200, learning_rate=0.05, max_depth=5)
print("X_train shape:", x_train.shape)
print("y_train shape:", y_train.shape)
print("First 5 y_train values:", y_train[:5])

xgb_model.fit(x_train, y_train)
xgb_preds = xgb_model.predict(x_test)

xgb_rmse, xgb_r2 = evaluate(y_test, xgb_preds)
print(f"[XGBoost] RMSE: {xgb_rmse:.4f}, R²: {xgb_r2:.4f}")

xgb_model.save_model(os.path.join(MODEL_DIR, "xgb_model.json"))

seq_len = 30
x, y = prepare_lstm_sequences(df)
split = int(len(x) * 0.8)
x_train, x_test = x[:split], x[split:]
y_train, y_test = y[:split], y[split:]

X_train_tensor = torch.tensor(x_train, dtype=torch.float32).unsqueeze(-1)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(-1)
X_test_tensor = torch.tensor(x_test, dtype=torch.float32).unsqueeze(-1)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32).unsqueeze(-1)

model = LSTMModel()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

for epoch in range(10):
    model.train()
    optimizer.zero_grad()
    output = model(X_train_tensor)
    loss = criterion(output, y_train_tensor)
    loss.backward()
    optimizer.step()
    print(f"Epoch [{epoch+1}/10], Loss: {loss.item():.6f}")

model.eval()
with torch.no_grad():
    preds = model(X_test_tensor).numpy().flatten()
    rmse, r2 = evaluate(y_test, preds)
    print(f"[LSTM] RMSE: {rmse:.4f}, R²: {r2:.4f}")

torch.save(model.state_dict(), os.path.join(MODEL_DIR, "lstm_model.pt"))
print("\nModels saved successfully.")
