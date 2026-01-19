import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import numpy as np

R_FORECAST_PATH = "src/models/arima_forecast.csv"
LSTM_PATH = "src/models/lstm_predictions.csv"
XGB_PATH = "src/models/xgb_predictions.csv"

def main():
    r_df = pd.read_csv(R_FORECAST_PATH)
    lstm_df = pd.read_csv(LSTM_PATH)
    xgb_df = pd.read_csv(XGB_PATH)

    #

if __name__ == "__main__":
    main()

