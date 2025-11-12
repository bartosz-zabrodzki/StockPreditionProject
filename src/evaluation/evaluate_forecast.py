import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import numpy as np
import os

r_forecast_path = "src/models/arima_forecast.csv"
lstm_path = "src/models/lstm_predictions.csv"
xgb_path = "src/models/xgb_predictions.csv"

r_df = pd.read_csv(r_forecast_path)
lstm_df = pd.read_csv(lstm_path)
xgb_df = pd.read_csv(xgb_path)