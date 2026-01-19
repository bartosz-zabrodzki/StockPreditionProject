from __future__ import annotations

import os
import requests
import pandas as pd
import streamlit as st

API_BASE = os.getenv("API_BASE", "http://api:8000")                 # dla kontenerów
PUBLIC_API_BASE = os.getenv("PUBLIC_API_BASE", "http://localhost:8000")  # linki do pobrania w przeglądarce



st.set_page_config(page_title="Stock Predictor", layout="wide")

st.title("StockPredictionProject UI")
# --- API healthcheck (fail fast) ---


with st.sidebar:
    ticker = st.text_input("Ticker", value="AAPL").strip().upper()
    interval = st.selectbox("Interval", ["1d", "1h", "15m"], index=0)
    n = st.slider("Points (n)", min_value=50, max_value=2000, value=300, step=50)
    force_train = st.checkbox("Force retraining", value=True)

# --- guard: ticker must be non-empty ---
ticker = ticker.strip().upper()
if not ticker:
    st.warning("Podaj ticker (np. AAPL, META).")
    st.stop()


colA, colB, colC, colD,colE = st.columns(5)


def post(path: str, payload: dict):
    with st.spinner(f"Running {path}..."):
        r = requests.post(f"{API_BASE}{path}", json=payload, timeout=3600)
    if r.status_code >= 400:
        st.error(f"{path} failed: {r.status_code} {r.text}")
        return None
    st.success(f"{path} OK")
    return r.json()

@st.cache_data(ttl=10)
def get_cached(path: str):
    r = requests.get(f"{API_BASE}{path}", timeout=60)
    if r.status_code >= 400:
        return None, f"{path} failed: {r.status_code} {r.text}"
    return r.json(), None

def get(path: str):
    r = requests.get(f"{API_BASE}{path}", timeout=60)
    if r.status_code >= 400:
        st.error(f"{path} failed: {r.status_code} {r.text}")
        return None
    return r.json()

hc, err = get_cached("/paths")
if err:
    st.error(err)
    st.stop()
if not hc:
    st.error("API unreachable")
    st.stop()

payload = {"ticker": ticker, "interval": interval, "force": force_train}

with colA:
    if st.button("Fetch"):
        post("/fetch", payload)
with colB:
    if st.button("Build Features"):
        payload = {"ticker": ticker, "interval": "1d", "force": st.session_state.get("force", False)}

        # Step 1: Fetch data first (if missing)
        fetch_res = requests.post(f"{API_BASE}/fetch", json=payload)
        if fetch_res.status_code != 200:
            st.error(f" Fetch failed: {fetch_res.text}")
        else:
            st.info(" Data fetched successfully. Building features...")

            # Step 2: Then build features
            feat_res = requests.post(f"{API_BASE}/features", json=payload)
            if feat_res.status_code == 200:
                st.success("Features built successfully!")
            else:
                st.error(f" Feature build failed: {feat_res.text}")

with colC:
    if st.button("Train"):
        post("/train", payload)
with colD:
    if st.button("Pipeline"):
        res = post("/pipeline", payload)
        if res is not None:
            st.cache_data.clear()
            st.rerun()
with colE:
    if st.button("Predict"):
        res = post("/predict", payload)
        if res is not None:
            st.cache_data.clear()
            st.rerun()


st.divider()

left, right = st.columns([2, 1])

with left:
    st.subheader("Predictions chart")
    series, err = get_cached(f"/results/series?ticker={ticker}&n={n}")
    if err:
        st.error(err)
    elif series and "points" in series:
        df = pd.DataFrame(series["points"])
        if not df.empty:
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
            df = df.dropna(subset=["date"]).sort_values("date").set_index("date")
            st.line_chart(df[["close", "lstm", "xgb", "hybrid"]])


with right:
    st.subheader("Latest")
    latest, err = get_cached(f"/results/latest?ticker={ticker}")
    if err:
        st.error(err)
    elif latest and "latest" in latest:
        l = latest["latest"]
        st.metric("Date", l["date"])
        st.metric("Close", f'{l["close"]:.4f}')
        st.metric("LSTM", f'{l["lstm"]:.4f}', f'{(l["lstm"] - l["close"]):.4f}')
        st.metric("XGB", f'{l["xgb"]:.4f}', f'{(l["xgb"] - l["close"]):.4f}')

    st.subheader("Artifacts")
    art, err = get_cached(f"/artifacts?ticker={ticker}")
    if err:
        st.error(err)
    elif art:
        st.write("Models:", art.get("models", []))
        st.write("Scalers:", art.get("scalers", []))
        st.write("Outputs:", art.get("outputs", []))

    if latest and "downloads" in latest:
        d = latest["downloads"]
        st.subheader("Downloads")
        if d.get("predictions_csv"):
            st.link_button("predictions.csv", f"{PUBLIC_API_BASE}{d['predictions_csv']}")
        if d.get("forecast_csv"):
            st.link_button("forecast.csv", f"{PUBLIC_API_BASE}{d['forecast_csv']}")
        if d.get("plot_png"):
            st.link_button("plot.png", f"{PUBLIC_API_BASE}{d['plot_png']}")
    # Preview plot image (served by API)
    if latest and latest.get("downloads", {}).get("plot_png"):
        st.subheader("Plot preview")
        st.image(
            f"{PUBLIC_API_BASE}{latest['downloads']['plot_png']}",
            caption=f"{ticker} predictions plot",
            use_container_width=True,
        )


