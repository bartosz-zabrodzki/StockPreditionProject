from __future__ import annotations
import os
from fastapi import HTTPException
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse

from pydantic import BaseModel, Field
from fastapi.middleware.cors import CORSMiddleware
from src.config.paths import (
    ensure_dirs,
    DATA_DIR,
    CACHE_DIR,
    FEATURES_DIR,
    FORECASTS_DIR,
    MODEL_DIR,
    OUTPUT_DIR,
    LOG_DIR,
)


app = FastAPI(title="StockPredictionProject API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # docelowo zawężysz
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/download/output/{filename}")
def download_output(filename: str):
    if "/" in filename or "\\" in filename or ".." in filename:
        raise HTTPException(400, "invalid filename")
    p = Path(OUTPUT_DIR) / filename
    if not p.exists():
        raise HTTPException(404, "file not found")
    return FileResponse(p)

@app.get("/download/model/{filename}")
def download_model(filename: str):
    if "/" in filename or "\\" in filename or ".." in filename:
        raise HTTPException(400, "invalid filename")
    p = Path(MODEL_DIR) / filename
    if not p.exists():
        raise HTTPException(404, "file not found")
    return FileResponse(p)
class TickerReq(BaseModel):
    ticker: str = Field(..., min_length=1, examples=["AAPL"])
    interval: str = Field("1d", examples=["1d"])
    force: bool = False

def _set_env(ticker: str, interval: str, force: bool = False) -> None:
    t = ticker.strip().upper()
    if not t:
        raise HTTPException(status_code=400, detail="ticker is required")
    os.environ["STOCK_TICKER"] = t
    os.environ["STOCK_INTERVAL"] = interval.strip() or "1d"
    os.environ["FORCE"] = "1" if force else "0"


def _safe_call(fn, *args, **kwargs):
    """Safely call a function and handle known recoverable exceptions once."""
    try:
        return fn(*args, **kwargs)

    except FileExistsError as e:
        existing = str(e).split(":")[-1].strip()
        if os.path.exists(existing):
            os.remove(existing)
            print(f"[safe_call] Removed existing file: {existing}")
        # ⚠️ Call only once more
        try:
            return fn(*args, **kwargs)
        except Exception as inner_e:
            raise HTTPException(status_code=500, detail=f"{fn.__name__} failed after retry: {inner_e}")

    except SystemExit as e:
        code = e.code if isinstance(e.code, int) else 0
        if code != 0:
            raise HTTPException(status_code=500, detail=f"{fn.__name__} exited with code {code}")
        return None

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"{fn.__name__} failed: {type(e).__name__}: {e}")

@app.get("/")
def root():
    return {"ok": True, "docs": "/docs"}


@app.get("/paths")
def show_paths():
    # u Ciebie paths.py to stałe, więc zwracamy stałe
    return {
        "DATA_DIR": str(DATA_DIR),
        "CACHE_DIR": str(CACHE_DIR),
        "FEATURES_DIR": str(FEATURES_DIR),
        "FORECASTS_DIR": str(FORECASTS_DIR),
        "MODEL_DIR": str(MODEL_DIR),
        "OUTPUT_DIR": str(OUTPUT_DIR),
        "LOG_DIR": str(LOG_DIR),
    }


@app.get("/artifacts")
def artifacts(ticker: str = "AAPL"):
    t = ticker.strip().upper()

    model_dir = Path(MODEL_DIR)
    out_dir = Path(OUTPUT_DIR)
    scaler_dir = model_dir / "scalers"

    models = sorted([p.name for p in model_dir.glob(f"*{t}*") if p.is_file()])
    outputs = sorted([p.name for p in out_dir.glob(f"*{t}*") if p.is_file()])
    scalers = sorted([p.name for p in scaler_dir.glob(f"*{t}*") if p.is_file()]) if scaler_dir.exists() else []

    return {"ticker": t, "models": models, "scalers": scalers, "outputs": outputs}


@app.post("/fetch")
def fetch(req: TickerReq):
    ensure_dirs()
    _set_env(req.ticker, req.interval, req.force)


    from src.data.data import run as run_fetch

    run_fetch(
        ticker=os.environ["STOCK_TICKER"],
        interval=os.environ["STOCK_INTERVAL"],
        normalize=True,
        split=True,
    )
    return {"status": "fetched", "ticker": os.environ["STOCK_TICKER"], "interval": os.environ["STOCK_INTERVAL"]}



@app.post("/features")
def features(req: TickerReq):
    ensure_dirs()
    _set_env(req.ticker, req.interval, req.force)

    from src.data.data import run as run_fetch
    from src.features.build_features import run as run_features

    ticker = os.environ["STOCK_TICKER"]
    interval = os.environ["STOCK_INTERVAL"]

    # 👇 auto-fetch missing data
    cache_file = Path(f"/data/data_cache/{ticker}_{interval}.csv")
    if not cache_file.exists():
        print(f"[AutoFetch] No cache for {ticker}, fetching...")
        run_fetch(ticker=ticker, interval=interval, normalize=True, split=True)

    out = run_features(ticker=ticker, interval=interval)
    return {"status": "features_built", "file": str(out)}



@app.post("/train")
def train(req: TickerReq):
    ensure_dirs()
    _set_env(req.ticker, req.interval, req.force)


    from src.training.train_model import run as run_train
    _safe_call(run_train, ticker=os.environ["STOCK_TICKER"])

    return {"status": "trained", "ticker": os.environ["STOCK_TICKER"], "force": os.environ["FORCE"]}

@app.post("/predict")
def predict(req: TickerReq):
    ensure_dirs()
    _set_env(req.ticker, req.interval, req.force)

    from src.prediction.predict import run as run_predict

    _safe_call(run_predict, ticker=os.environ["STOCK_TICKER"])
    return {"status": "predicted", "ticker": os.environ["STOCK_TICKER"]}
@app.post("/pipeline")
def pipeline(req: TickerReq):
    """
    Full end-to-end pipeline:
      1. Fetch data
      2. Build features
      3. Train models (LSTM + XGB)
      4. Predict with hybrid R integration
    """
    import traceback
    import logging
    from src.prediction.predict import run as run_predict, PredictConfig

    ensure_dirs()
    _set_env(req.ticker, req.interval, req.force)

    from src.data.data import run as run_fetch
    from src.features.build_features import run as run_features
    from src.training.train_model import run as run_train

    log = logging.getLogger(__name__)

    ticker = os.environ["STOCK_TICKER"]
    interval = os.environ["STOCK_INTERVAL"]
    force = os.environ["FORCE"]

    try:
        # 1 Fetch (skip if cache exists)
        cache_path = Path(CACHE_DIR) / f"{ticker}_{interval}.csv"
        if req.force or not cache_path.exists():
            _safe_call(run_fetch, ticker=ticker, interval=interval, normalize=True, split=True)
        else:
            log.info(f"[pipeline] Skipping fetch — cache already exists at {cache_path}")

        # 2 Features (skip if file exists)
        feature_path = Path(FEATURES_DIR) / f"{ticker}_{interval}_features.csv"
        if not feature_path.exists():
            feature_path = Path(FEATURES_DIR) / f"{ticker}_features.csv"
        if not feature_path.exists():
            raise HTTPException(
                status_code=404,
                detail=f"Feature file not found for {ticker}: {feature_path}"
            )

        # 3 Train models only if needed
        lstm_path = Path(MODEL_DIR) / f"lstm_model_{ticker}.pth"
        xgb_path = Path(MODEL_DIR) / f"xgboost_model_{ticker}.pkl"

        if req.force or not (lstm_path.exists() and xgb_path.exists()):
            msg = f"[pipeline] Training models for {ticker} ... (force={req.force})"
            print(msg)
            log.info(msg)
            _safe_call(run_train, ticker=ticker)
        else:
            msg = f"[pipeline] Skipping retraining — models for {ticker} already exist."
            print(msg)
            log.info(msg)

        # 4 Predict — run once, with concurrency lock
        config = PredictConfig(
            feature_file=feature_path,
            output_dir=Path(OUTPUT_DIR),
            ticker=ticker,
            forecast_days=30,
        )

        lock_path = Path(f"/tmp/predict_{ticker}.lock")
        if lock_path.exists():
            raise HTTPException(429, f"Prediction already running for {ticker}")

        try:
            lock_path.touch()
            _safe_call(run_predict, config=config)
        finally:
            if lock_path.exists():
                lock_path.unlink()

        return {
            "status": "ok",
            "ticker": ticker,
            "interval": interval,
            "force": force,
            "message": (
                "Pipeline completed successfully with hybrid forecast. "
                f"(retrain={'yes' if req.force else 'no'})"
            ),
        }

    except Exception as e:
        err_trace = "".join(traceback.format_exception(type(e), e, e.__traceback__))
        err_log = Path(OUTPUT_DIR) / f"{ticker}_pipeline_error.log"
        err_log.write_text(err_trace, encoding="utf-8")

        raise HTTPException(
            status_code=500,
            detail=f"Pipeline failed: {e.__class__.__name__}: {e}",
        )



@app.get("/debug/env")
def debug_env():
    return {
        "STOCK_TICKER": os.getenv("STOCK_TICKER"),
        "STOCK_INTERVAL": os.getenv("STOCK_INTERVAL"),
        "FORCE": os.getenv("FORCE"),
    }


@app.get("/results/series")
def series(ticker: str = "AAPL", n: int = 200):
    import pandas as pd

    t = ticker.strip().upper()
    pred_csv = Path(OUTPUT_DIR) / f"{t}_predictions.csv"
    if not pred_csv.exists():
        raise HTTPException(404, f"missing predictions: {pred_csv.name}")

    df = pd.read_csv(pred_csv).tail(int(n))

    out = []
    for _, r in df.iterrows():
        out.append({
            "date": str(r["Date"]),
            "close": float(r["Close"]),
            "lstm": float(r["LSTM_Prediction"]),
            "xgb": float(r["XGB_Prediction"]),
        })

    return {"ticker": t, "n": len(out), "points": out}

@app.get("/results/series")
def series(ticker: str = "AAPL", n: int = 200):
    import pandas as pd
    import numpy as np
    import json

    t = ticker.strip().upper()
    pred_csv = Path(OUTPUT_DIR) / f"{t}_predictions.csv"
    if not pred_csv.exists():
        raise HTTPException(404, f"missing predictions: {pred_csv.name}")

    df = pd.read_csv(pred_csv).tail(int(n))

    # Sanitize everything here, only once
    df.replace([np.inf, -np.inf], 0, inplace=True)
    df.fillna(0, inplace=True)

    #  Build output
    out = [
        {
            "date": str(r.get("Date", "")),
            "close": float(r.get("Close", 0) or 0),
            "lstm": float(r.get("LSTM_Prediction", 0) or 0),
            "xgb": float(r.get("XGB_Prediction", 0) or 0),
        }
        for _, r in df.iterrows()
    ]

    #  Guarantee it can be serialized (no NaN/inf)
    return json.loads(json.dumps({"ticker": t, "n": len(out), "points": out}))





