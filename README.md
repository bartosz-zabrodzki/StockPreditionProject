# StockPreditionProject
A full-featured AI-driven stock prediction suite using LSTM, LightGBM, and Prophet. 
Combines traditional machine learning, deep learning, and time-series forecasting to predict next-day stock prices for multiple historical periods.
**GitHub Repository Description**

---

A full-featured **AI-driven stock prediction suite** using **LSTM**, **LightGBM**, and **Prophet**.
Combines traditional machine learning, deep learning, and time-series forecasting to predict next-day stock prices for multiple historical periods.

---

* **Multi-Model Forecasting:**

  *  `LSTM` — Deep learning model for sequential trend analysis
  *  `Prophet` — Robust trend & seasonality modeling
  *  `LightGBM` — Gradient-boosted regression for fast predictions

* **Historical Period Comparison:**
  Runs forecasts on multiple windows (2000–, 2008–, 2020–) to test resilience across economic cycles.

* **Automatic Data Loading:**
  Fetches clean market data directly using `yfinance`.

* **Visualization:**
  Auto-generates and saves model performance plots (`ticker_LSTM_period.png`, etc.)

* **Unified Summary Table:**
  Presents all forecasted prices in a clean summary view.

---

###Tech Stack

* **Python 3.9**
* `yfinance`, `pandas`, `numpy`
* `lightgbm`
* `prophet`
* `tensorflow / keras`
* `matplotlib`
* `scikit-learn`

---

###Usage

```bash
pip install pandas numpy matplotlib lightgbm prophet tensorflow scikit-learn yfinance
python forecast.py
```

When prompted:

```
Enter stock ticker (e.g. AAPL, MSFT, TSLA): 
```

The script will:

1. Download all historical data for that ticker.
2. Train & evaluate all models.
3. Save forecast plots and print next-day predictions.

---

###Example Output

```
=== Processing 2020 ===
LSTM forecast for next day after 2020: 192.34
Prophet forecast for next day after 2020: 191.87
LightGBM forecast for next day after 2020: 193.15

=== Summary Forecast Table ===
Period            LSTM        Prophet       LightGBM
2000              187.9        186.5          188.2
2008              145.4        146.1          145.7
2020              192.3        191.8          193.1
```

---

###Notes

* LSTM requires ≥100 closing prices for stable predictions.
* Prophet and LightGBM gracefully skip underfilled data windows.
* Designed for experimentation & research — **not financial advice.**

---
