from __future__ import annotations

import os
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from src.models.config import clean_old_outputs
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
    try:
        return fn(*args, **kwargs)


    except FileExistsError as e:

        existing = getattr(e, "filename", None)

        if existing and os.path.exists(existing):
            os.remove(existing)

            print(f"[safe_call] Removed existing file: {existing}")

        return fn(*args, **kwargs)


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
    cache_file = Path(CACHE_DIR) / f"{ticker}_{interval}.csv"
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

    ticker = os.environ["STOCK_TICKER"]
    interval = os.environ["STOCK_INTERVAL"]

    _safe_call(run_train, ticker=ticker, interval=interval)


    return {"status": "trained", "ticker": ticker, "interval": interval, "force": req.force}

@app.post("/predict")
def predict(req: TickerReq):
    ensure_dirs()
    clean_old_outputs(req.ticker)
    _set_env(req.ticker, req.interval, req.force)

    from src.prediction.predict import run as run_predict, PredictConfig

    ticker = os.environ["STOCK_TICKER"]
    interval = os.environ["STOCK_INTERVAL"]

    feature_path = Path(FEATURES_DIR) / f"{ticker}_{interval}_features.csv"
    alt_path = Path(FEATURES_DIR) / f"{ticker}_features.csv"
    if alt_path.exists() and not feature_path.exists():
        feature_path = alt_path

    if not feature_path.exists():
        raise HTTPException(404, f"missing features file: {feature_path.name} (run /features first)")

    config = PredictConfig(
        feature_file=feature_path,
        output_dir=Path(OUTPUT_DIR),
        ticker=ticker,
        forecast_days=30,
    )

    _safe_call(run_predict, config=config)
    return {"status": "predicted", "ticker": ticker, "interval": interval, "feature_file": str(feature_path)}


@app.post("/pipeline")
def pipeline(req: TickerReq):
    import traceback
    import logging
    from src.prediction.predict import run as run_predict, PredictConfig

    ensure_dirs()
    _set_env(req.ticker, req.interval, req.force)

    ticker = os.environ["STOCK_TICKER"]
    interval = os.environ["STOCK_INTERVAL"]

    from src.data.data import run as run_fetch
    from src.features.build_features import run as run_features
    from src.training.train_model import run as run_train

    log = logging.getLogger(__name__)
    try:
        # 1 Fetch (skip if cache exists)
        cache_path = Path(CACHE_DIR) / f"{ticker}_{interval}.csv"
        if req.force:
            if cache_path.exists():
                try:
                    cache_path.unlink()
                    log.info(f"[pipeline] Deleted old cache → {cache_path}")
                except Exception as e:
                    log.warning(f"[pipeline] Could not delete cache {cache_path}: {e}")

        if req.force or not cache_path.exists():
            _safe_call(run_fetch, ticker=ticker, interval=interval, normalize=True, split=True)
        else:
            log.info(f"[pipeline] Skipping fetch — cache already exists at {cache_path}")

        # 2 Build or rebuild features
        feature_path = Path(FEATURES_DIR) / f"{ticker}_{interval}_features.csv"
        alt_path = Path(FEATURES_DIR) / f"{ticker}_features.csv"

        # Ensure the correct output file name (depending on your R script)
        if alt_path.exists() and not feature_path.exists():
            feature_path = alt_path

        # If forcing rebuild, delete the old features first
        if req.force and feature_path.exists():
            try:
                feature_path.unlink()
                log.info(f"[pipeline] Deleted old features → {feature_path}")
            except Exception as e:
                log.warning(f"[pipeline] Could not delete {feature_path}: {e}")

        # Always run features if forced or missing
        if req.force or not feature_path.exists():
            log.info(f"[pipeline] Building features for {ticker} (force={req.force})...")
            feature_path = _safe_call(run_features, ticker=ticker, interval=interval, force=req.force)
        else:
            log.info(f"[pipeline] Skipping feature build — using existing {feature_path}")

        # 3 Train models only if needed
        lstm_path = Path(MODEL_DIR) / f"lstm_model_{ticker}.pth"
        xgb_path = Path(MODEL_DIR) / f"xgboost_model_{ticker}.pkl"

        if req.force or not (lstm_path.exists() and xgb_path.exists()):
            msg = f"[pipeline] Training models for {ticker} ... (force={req.force})"
            print(msg)
            log.info(msg)
            _safe_call(run_train, ticker=ticker, interval=interval)
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

        # ---  Always clean old outputs before predicting ---
        old_outputs = list(Path(OUTPUT_DIR).glob(f"{ticker}_*.csv"))
        if old_outputs:
            print(f"[Pipeline] Cleaning old prediction files for {ticker} ...")
            for old in old_outputs:
                try:
                    old.unlink()
                    print(f"[Pipeline] Deleted old output → {old.name}")
                except Exception as e:
                    print(f"[Pipeline] Warning: Could not delete {old.name}: {e}")

        lock_path = Path(f"/tmp/predict_{ticker}.lock")
        if lock_path.exists():
            raise HTTPException(429, f"Prediction already running for {ticker}")

        try:

            clean_old_outputs(ticker)
            lock_path.touch()
            _safe_call(run_predict, config=config)
        finally:
            if lock_path.exists():
                lock_path.unlink()

        return {
            "status": "ok",
            "ticker": ticker,
            "interval": interval,
            "force": req.force,
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
            "close": float(r.get("Close", 0)),
            "lstm": float(r.get("LSTM_Prediction", r.get("DL_Forecast", 0)) or 0),
            "xgb": float(r.get("XGB_Prediction", r.get("XGB_Forecast", 0)) or 0),
            "hybrid": float(r.get("Hybrid_Prediction", r.get("Hybrid_Forecast", 0)) or 0),
        })
    return {"ticker": t, "n": len(out), "points": out}


@app.get("/results/latest")
def latest(ticker: str = "AAPL"):
    import pandas as pd
    import numpy as np
    import json

    t = ticker.strip().upper()
    pred_csv = Path(str(OUTPUT_DIR)).resolve() / f"{t}_predictions.csv"
    fc_csv = Path(str(OUTPUT_DIR)).resolve() / f"{t}_forecast.csv"
    plot_png = Path(str(OUTPUT_DIR)).resolve() / f"{t}_predictions_plot.png"

    # 🔍 Debug: show path resolution in logs (optional)
    print(f"[DEBUG] latest() checking path: {pred_csv} (exists={pred_csv.exists()})")

    if not pred_csv.exists():
        raise HTTPException(404, f"missing predictions: {pred_csv.name}")

    df = pd.read_csv(pred_csv)

    # 🧹 Sanitize numeric data to prevent JSON float errors
    df.replace([np.inf, -np.inf], 0, inplace=True)
    df.fillna(0, inplace=True)

    if df.empty:
        raise HTTPException(400, f"{t}_predictions.csv is empty")

    # Pick last record safely
    r = df.iloc[-1]

    latest_data = {
        "date": str(r.get("Date", "")),
        "close": float(r.get("Close", 0) or 0),
        "lstm": float(r.get("LSTM_Prediction", 0) or 0),
        "xgb": float(r.get("XGB_Prediction", 0) or 0),
    }

    #  Build JSON-safe response
    result = {
        "ticker": t,
        "latest": latest_data,
        "downloads": {
            "predictions_csv": f"/download/output/{pred_csv.name}",
            "forecast_csv": f"/download/output/{fc_csv.name}" if fc_csv.exists() else None,
            "plot_png": f"/download/output/{plot_png.name}" if plot_png.exists() else None,
        },
    }

    return json.loads(json.dumps(result))





