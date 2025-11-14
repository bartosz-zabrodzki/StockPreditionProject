import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import lightgbm as lgb
from prophet import Prophet
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import yfinance as yf


def load_data(ticker, start="2000-01-01", end="2025-12-31"):
    df = yf.download(ticker, start=start, end=end)
    df = df.reset_index()
    return df


def lstm_forecast(df, period, ticker):
    df = df[['Date', 'Close']].dropna()
    df.set_index('Date', inplace=True)
    data = df[['Close']].values
    if len(data) < 100:
        print(f"Not enough data for LSTM for period {period}")
        return None
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(data)
    X, y = [], []
    lookback = 60
    for i in range(lookback, len(scaled)):
        X.append(scaled[i - lookback:i, 0])
        y.append(scaled[i, 0])
    X, y = np.array(X), np.array(y)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(X.shape[1], 1)))
    model.add(Dropout(0.2))
    model.add(LSTM(50))
    model.add(Dropout(0.2))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    model.fit(X, y, epochs=10, batch_size=64, validation_split=0.1, verbose=0)
    preds = model.predict(X)
    preds_rescaled = scaler.inverse_transform(preds)
    last_sequence = scaled[-lookback:]
    last_sequence = np.reshape(last_sequence, (1, lookback, 1))
    next_pred = model.predict(last_sequence)
    tomorrow_price = scaler.inverse_transform(next_pred)[0][0]
    plt.figure(figsize=(15, 8))
    plt.plot(df.index[lookback:], data[lookback:], label='True')
    plt.plot(df.index[lookback:], preds_rescaled, label='LSTM Forecast')
    plt.legend()
    plt.title(f'LSTM Forecast - {period} | Tomorrow: {tomorrow_price:.2f}')
    plt.savefig(f'{ticker}_LSTM_{period}.png')
    plt.close()
    print(f"LSTM forecast for next day after {period}: {tomorrow_price:.2f}")
    return tomorrow_price


def prophet_forecast(df, period, ticker):
    prophet_df = df[['Date', 'Close']].dropna()
    prophet_df.columns = ['ds', 'y']
    if prophet_df.shape[0] < 2:
        print(f"Not enough data for Prophet for period {period}")
        return None
    model = Prophet(changepoint_prior_scale=0.5)
    model.add_seasonality(name='monthly', period=30.5, fourier_order=5)
    model.fit(prophet_df)
    future = model.make_future_dataframe(periods=1)
    forecast = model.predict(future)
    tomorrow_price = forecast.iloc[-1]['yhat']
    plt.figure(figsize=(15, 8))
    plt.plot(prophet_df['ds'], prophet_df['y'], label='True')
    plt.plot(forecast['ds'], forecast['yhat'], label='Prophet Forecast')
    plt.legend()
    plt.title(f'Prophet Forecast - {period} | Tomorrow: {tomorrow_price:.2f}')
    plt.savefig(f'{ticker}_Prophet_{period}.png')
    plt.close()
    print(f"Prophet forecast for next day after {period}: {tomorrow_price:.2f}")
    return tomorrow_price


def lightgbm_forecast(df, period, ticker):
    df = df[['Date', 'Close']].dropna()
    df['lag1'] = df['Close'].shift(1)
    df.dropna(inplace=True)
    if len(df) < 10:
        print(f"Not enough data for LightGBM for period {period}")
        return None
    X = pd.DataFrame({'lag1': df['lag1'].astype(float).values})
    y = df['Close'].astype(float)
    # Clean up feature names (robust to any index weirdness)
    X.columns = X.columns.astype(str)
    X.columns = X.columns.str.replace(r'[^A-Za-z0-9_]', '', regex=True)
    model = lgb.LGBMRegressor(n_estimators=1000, learning_rate=0.01)
    model.fit(X, y)
    last_close = df['Close'].iloc[-1]
    input_df = pd.DataFrame({'lag1': [float(last_close.iloc[0])]})
    tomorrow_pred = model.predict(input_df)[0]
    plt.figure(figsize=(15, 8))
    plt.plot(df['Date'], y, label='True')
    plt.plot(df['Date'], model.predict(X), label='LightGBM Forecast')
    plt.legend()
    plt.title(f'LightGBM Forecast - {period} | Tomorrow: {tomorrow_pred:.2f}')
    plt.savefig(f'{ticker}_LightGBM_{period}.png')
    plt.close()
    print(f"LightGBM forecast for next day after {period}: {tomorrow_pred:.2f}")
    return tomorrow_pred



def run_all(ticker):
    periods = {
        '2000': ('2000-01-01', '2025-12-31'),
        '2008': ('2008-01-01', '2025-12-31'),
        '2020': ('2020-01-01', '2025-12-31')
    }
    df_full = load_data(ticker)
    summary = []
    for period, (start, end) in periods.items():
        df_period = df_full[(df_full['Date'] >= pd.to_datetime(start)) & (df_full['Date'] <= pd.to_datetime(end))]
        print(f"\n=== Processing {period} ===")
        lstm_pred = lstm_forecast(df_period.copy(), period, ticker)
        prophet_pred = prophet_forecast(df_period.copy(), period, ticker)
        lightgbm_pred = lightgbm_forecast(df_period.copy(), period, ticker)
        summary.append({
            'Period': period,
            'LSTM': lstm_pred,
            'Prophet': prophet_pred,
            'LightGBM': lightgbm_pred
        })

    print("\n=== Summary Forecast Table ===")
    print(f"{'Period':<10}{'LSTM':>15}{'Prophet':>15}{'LightGBM':>15}")
    for row in summary:
        print(f"{row['Period']:<10}{str(round(row['LSTM'],2)) if row['LSTM'] else 'N/A':>15}{str(round(row['Prophet'],2)) if row['Prophet'] else 'N/A':>15}{str(round(row['LightGBM'],2)) if row['LightGBM'] else 'N/A':>15}")


if __name__ == '__main__':
    ticker = input("Enter stock ticker (e.g. AAPL, MSFT, TSLA): ").upper()
    run_all(ticker)
