from __future__ import annotations

import json
import logging
import subprocess
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from src.models.config import get_model_paths, LSTM_CONFIG, XGB_CONFIG
from src.config.paths import MODEL_DIR

log = logging.getLogger(__name__)

# ============================================================
# CONFIG CLASSES & HELPERS
# ============================================================

def load_feature_schema(ticker: str) -> list[str]:
    """
    Loads feature schema for a given ticker. If missing, tries to fallback
    to AAPL schema for compatibility (so /pipeline won't crash).
    """
    p = Path(MODEL_DIR) / f"feature_schema_{ticker}.json"
    if not p.exists():
        fallback = Path(MODEL_DIR) / "feature_schema_AAPL.json"
        if fallback.exists():
            print(f"[Schema] Missing {p}, using fallback → {fallback}")
            payload = json.loads(fallback.read_text(encoding="utf-8"))
        else:
            raise FileNotFoundError(
                f"Missing feature schema: {p}. Run /train first (creates feature_schema_{ticker}.json)."
            )
    else:
        payload = json.loads(p.read_text(encoding="utf-8"))

    cols = payload.get("feature_cols")
    if not cols or not isinstance(cols, list):
        raise ValueError(f"Invalid schema file: {p}")
    return cols

@dataclass(frozen=True)
class PredictConfig:
    feature_file: Path
    output_dir: Path
    ticker: str
    forecast_days: int


# ============================================================
# DATA HANDLING
# ============================================================

def load_data(feature_file: Path) -> pd.DataFrame:
    if not feature_file.exists():
        raise FileNotFoundError(f"Missing feature file: {feature_file}")
    df = pd.read_csv(feature_file).dropna().reset_index(drop=True)
    log.info("Loaded data: rows=%s cols=%s", df.shape[0], df.shape[1])
    return df


# ============================================================
# MODEL LOADING
# ============================================================

def load_models(
    input_size: int,
    lstm_model_path: Path,
    xgb_model_path: Path,
    scaler_x_path: Path,
    scaler_y_path: Path,
    lstm_config: dict
):
    import joblib
    import torch
    import torch.nn as nn
    import logging

    log = logging.getLogger(__name__)

    if not lstm_model_path.exists():
        alt_paths = [
            Path(MODEL_DIR) / f"lstm_model_{lstm_model_path.stem.replace('_lstm', '')}.pth",
            Path(MODEL_DIR) / f"lstm_model_{lstm_model_path.stem}.pth",
            Path(MODEL_DIR) / f"{lstm_model_path.stem}.pth",
            Path(MODEL_DIR) / f"{lstm_model_path.stem}.pt"
        ]
        for alt in alt_paths:
            if alt.exists():
                log.warning(f"[ModelPath] Using alternate LSTM model path → {alt}")
                lstm_model_path = alt
                break
        else:

            fallback = Path(MODEL_DIR) / "lstm_model_AAPL.pth"
            if fallback.exists():
                log.warning(f"[Fallback] Missing {lstm_model_path.name}, using AAPL model instead → {fallback}")
                lstm_model_path = fallback
            else:
                raise FileNotFoundError(
                    f"LSTM model not found. Tried:\n - {lstm_model_path}\n - "
                    + "\n - ".join(str(a) for a in alt_paths)
                )


    ticker_guess = lstm_model_path.stem.split("_")[-1]
    sx_candidates = [
        scaler_x_path,
        Path(MODEL_DIR) / f"{ticker_guess}_scaler_x.pkl",
        Path(MODEL_DIR) / "scalers" / f"scaler_x_{ticker_guess}.pkl",
    ]
    sy_candidates = [
        scaler_y_path,
        Path(MODEL_DIR) / f"{ticker_guess}_scaler_y.pkl",
        Path(MODEL_DIR) / "scalers" / f"scaler_y_{ticker_guess}.pkl",
    ]

    def find_existing(paths, name):
        for p in paths:
            if p.exists():
                return p
        raise FileNotFoundError(f"Missing {name}. Tried: " + ", ".join(str(p) for p in paths))

    scaler_x_path = find_existing(sx_candidates, "scaler_x")
    scaler_y_path = find_existing(sy_candidates, "scaler_y")


    scaler_x = joblib.load(scaler_x_path)
    scaler_y = joblib.load(scaler_y_path)
    input_size = int(getattr(scaler_x, "n_features_in_", input_size))

    class LSTMModel(nn.Module):
        def __init__(self, input_size, hidden_size, num_layers, dropout):
            super().__init__()
            self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
            self.fc = nn.Linear(hidden_size, 1)

        def forward(self, x):
            out, _ = self.lstm(x)
            return self.fc(out[:, -1, :])

    lstm = LSTMModel(
        input_size=input_size,
        hidden_size=lstm_config["hidden_size"],
        num_layers=lstm_config["num_layers"],
        dropout=lstm_config["dropout"],
    )

    try:
        state = torch.load(lstm_model_path, map_location="cpu", weights_only=True)
    except TypeError:
        # Fallback for older PyTorch versions without weights_only
        state = torch.load(lstm_model_path, map_location="cpu")

    lstm.load_state_dict(state)
    lstm.eval()


    if not xgb_model_path.exists():
        alt_xgb = Path(MODEL_DIR) / f"xgboost_model_{ticker_guess}.pkl"
        if alt_xgb.exists():
            log.warning(f"[ModelPath] Using alternate XGB model path → {alt_xgb}")
            xgb_model_path = alt_xgb
        else:
            fallback_xgb = Path(MODEL_DIR) / "xgboost_model_AAPL.pkl"
            if fallback_xgb.exists():
                log.warning(f"[Fallback] Missing {xgb_model_path.name}, using AAPL model instead → {fallback_xgb}")
                xgb_model_path = fallback_xgb
            else:
                raise FileNotFoundError(f"Missing XGBoost model: {xgb_model_path} or {fallback_xgb}")

    xgb = joblib.load(xgb_model_path)

    log.info(
        f"  Models loaded successfully:\n"
        f"  LSTM → {lstm_model_path.name}\n"
        f"  XGB  → {xgb_model_path.name}\n"
        f"  Scalers → {scaler_x_path.name}, {scaler_y_path.name}"
    )
    return lstm, xgb, scaler_x, scaler_y


# ============================================================
# R INTEGRATION
# ============================================================

import subprocess, re, json, pandas as pd, logging

log = logging.getLogger(__name__)

def run_r_analysis(ticker: str, interval: str = "1d") -> pd.DataFrame:
    """
    Executes the R pipeline (analyze.R) and returns its forecast as a DataFrame.
    If R fails or JSON parsing fails, a graceful fallback empty DataFrame is returned.
    """
    cmd = ["Rscript", "/app/src/r/analyze.R", "--ticker", ticker, "--interval", interval]
    log.info(f"Running R analysis for {ticker} ({interval})...")
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
    except Exception as e:
        log.error(f"Rscript execution failed: {e}")
        return pd.DataFrame()  # fail-safe

    stdout = proc.stdout.strip()
    stderr = proc.stderr.strip()

    if stderr:
        log.warning(f"[R stderr] {stderr[:400]}")

    # --- Find all bracket blocks ---
    json_candidates = re.findall(r"\[[\s\S]*?\]", stdout)
    result = None

    for cand in json_candidates:
        if "R_Forecast" in cand and cand.strip().startswith("["):
            try:
                result = json.loads(cand)
                break
            except json.JSONDecodeError:
                continue

    if result is None:
        log.error(f"R JSON not found. STDOUT (tail):\n{stdout[-800:]}")
        return pd.DataFrame()

    try:
        df = pd.DataFrame(result)
        if df.empty:
            log.warning("R returned an empty DataFrame.")
        else:
            log.info(f"R analysis completed successfully: {len(df)} rows.")
        return df
    except Exception as e:
        log.error(f"Failed to convert R JSON to DataFrame: {e}")
        return pd.DataFrame()  # fail-safe fallback

# ============================================================
# PREDICTION PIPELINE
# ============================================================

def make_predictions(
    df: pd.DataFrame,
    lstm,
    xgb,
    scaler_x,
    scaler_y,
    lstm_config: dict,
    feature_cols: list[str],
    ticker: str,
) -> pd.DataFrame:
    target = "Close"
    required = feature_cols + [target, "Date"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    x = df[feature_cols].to_numpy()
    y = df[[target]].to_numpy()

    if hasattr(scaler_x, "n_features_in_") and int(scaler_x.n_features_in_) != x.shape[1]:
        raise ValueError(
            f"Feature count mismatch: X has {x.shape[1]} cols, scaler expects {int(scaler_x.n_features_in_)}."
        )

    # --- scale features ---
    x_scaled = scaler_x.transform(x)
    seq_len = int(lstm_config["input_seq_len"])
    if len(x_scaled) <= seq_len:
        raise ValueError(f"Not enough rows for seq_len={seq_len}")

    import torch
    x_seq = np.array([x_scaled[i : i + seq_len] for i in range(len(x_scaled) - seq_len)])
    x_t = torch.tensor(x_seq, dtype=torch.float32)

    with torch.no_grad():
        lstm_pred_scaled = lstm(x_t).numpy()

    xgb_input = x_scaled[seq_len - 1 : -1]
    xgb_pred_scaled = xgb.predict(xgb_input).reshape(-1, 1)

    lstm_pred = scaler_y.inverse_transform(lstm_pred_scaled)
    xgb_pred = scaler_y.inverse_transform(xgb_pred_scaled)

    dates = df["Date"].iloc[-len(lstm_pred) :].reset_index(drop=True)
    pred_df = pd.DataFrame(
        {
            "Date": dates,
            "DL_Forecast": lstm_pred.squeeze(),
            "XGB_Forecast": xgb_pred.squeeze(),
        }
    )

    # =====================================================
    # Safe R forecast integration (robust fallback mode)
    # =====================================================
    try:
        r_forecast = run_r_analysis(ticker)

        if r_forecast is None or r_forecast.empty:
            log.warning("R forecast returned empty or None — skipping merge.")
            pred_df["Hybrid_Forecast"] = (
                0.7 * pred_df["DL_Forecast"] + 0.3 * pred_df["XGB_Forecast"]
            )
            return pred_df

        # Normalize and align date columns
        if "Date" not in r_forecast.columns:
            alt = [c for c in r_forecast.columns if c.lower().strip() == "date"]
            if alt:
                r_forecast.rename(columns={alt[0]: "Date"}, inplace=True)

        pred_df["Date"] = pd.to_datetime(pred_df["Date"]).astype(str)
        r_forecast["Date"] = pd.to_datetime(r_forecast["Date"]).astype(str)

        merged = pd.merge(pred_df, r_forecast, on="Date", how="outer")

        if "R_Forecast" not in merged.columns:
            log.warning(
                f"R forecast column missing after merge. Columns found: {merged.columns.tolist()}"
            )
            merged["R_Forecast"] = np.nan

        # --- Hybrid weighting ---
        merged["Hybrid_Forecast"] = (
            0.6 * merged["DL_Forecast"].fillna(0)
            + 0.3 * merged["XGB_Forecast"].fillna(0)
            + 0.1 * merged["R_Forecast"].fillna(0)
        )

        log.info(f"Hybrid forecast merge complete: {len(merged)} rows.")
        return merged

    except Exception as e:

        log.error(f"Hybrid forecast merge failed: {type(e).__name__}: {e}")
        log.debug("R forecast integration failed; continuing with DL/XGB only.")
        pred_df["Hybrid_Forecast"] = (
            0.7 * pred_df["DL_Forecast"] + 0.3 * pred_df["XGB_Forecast"]
        )
        return pred_df

# ============================================================
# PIPELINE RUNNER ENTRYPOINT (used by FastAPI / CLI)
# ============================================================

def run(
    config: PredictConfig | None = None,
    ticker: str | None = None,
    interval: str = "1d",
    forecast_days: int = 30
) -> pd.DataFrame:
    """
    Unified prediction entrypoint — supports both FastAPI and CLI usage.

    You can call it either way:
      ▶ run(config=PredictConfig(...))
      ▶ run(ticker="AAPL")

    It loads models (LSTM + XGBoost + R fallback), runs forecasts,
    and saves CSV outputs into OUTPUT_DIR.
    """
    import traceback
    import logging
    import pandas as pd
    import os
    from src.config.paths import FEATURES_DIR, OUTPUT_DIR, MODEL_DIR

    log = logging.getLogger(__name__)

    try:
        # ------------------------------------------------------------------
        # Dynamic config auto-build if called as run(ticker="AAPL")
        # ------------------------------------------------------------------
        if config is None:
            if ticker is None:
                ticker = os.getenv("STOCK_TICKER", "AAPL")

            feature_file = FEATURES_DIR / f"{ticker}_features.csv"
            config = PredictConfig(
                feature_file=feature_file,
                output_dir=OUTPUT_DIR,
                ticker=ticker,
                forecast_days=forecast_days
            )# 🧹 Sanitize output before saving



        log.info(f"[Predict] Running pipeline for {config.ticker}")
        if not config.feature_file.exists():
            raise FileNotFoundError(f"Missing features file: {config.feature_file}")

        # ------------------------------------------------------------------
        # Load schema & model components
        # ------------------------------------------------------------------
        feature_cols = load_feature_schema(config.ticker)

        lstm_config = {
            "input_seq_len": 30,
            "hidden_size": 256,
            "num_layers": 4,
            "dropout": 0.2,
        }

        lstm_path = MODEL_DIR / f"lstm_model_{config.ticker}.pth"
        xgb_path = MODEL_DIR / f"xgboost_model_{config.ticker}.pkl"
        sx_path = MODEL_DIR / "scalers" / f"scaler_x_{config.ticker}.pkl"
        sy_path = MODEL_DIR / "scalers" / f"scaler_y_{config.ticker}.pkl"

        lstm, xgb, scaler_x, scaler_y = load_models(
            input_size=len(feature_cols),
            lstm_model_path=lstm_path,
            xgb_model_path=xgb_path,
            scaler_x_path=sx_path,
            scaler_y_path=sy_path,
            lstm_config=lstm_config,
        )

        # ------------------------------------------------------------------
        #  Load feature data & make predictions
        # ------------------------------------------------------------------
        df = load_data(config.feature_file)
        predictions = make_predictions(
            df=df,
            lstm=lstm,
            xgb=xgb,
            scaler_x=scaler_x,
            scaler_y=scaler_y,
            lstm_config=lstm_config,
            feature_cols=feature_cols,
            ticker=config.ticker,
        )

        if predictions is None or predictions.empty:
            raise ValueError("make_predictions() returned empty DataFrame")

        # --- Add a synthetic 'Close' column for UI compatibility ---
        if "Close" not in predictions.columns:
            # Use the last known value or a safe fallback
            try:
                last_close = float(predictions["DL_Forecast"].iloc[0])
            except Exception:
                last_close = 0.0
            predictions["Close"] = last_close
            log.info(f"[Predict] Added synthetic 'Close' column (value={last_close:.2f})")

        rename_map = {
            "DL_Forecast": "LSTM_Prediction",
            "XGB_Forecast": "XGB_Prediction",
            "Hybrid_Forecast": "Hybrid_Prediction",
        }

        for old_col, new_col in rename_map.items():
            if old_col in predictions.columns and new_col not in predictions.columns:
                predictions[new_col] = predictions[old_col]
                log.info(f"[Predict] Added alias column → {new_col} (from {old_col})")

        # --- Save outputs ---
        output_dir = config.output_dir
        output_dir.mkdir(parents=True, exist_ok=True)

        forecast_csv = output_dir / f"{config.ticker}_forecast.csv"
        predictions_csv = output_dir / f"{config.ticker}_predictions.csv"
        plot_path = output_dir / f"{config.ticker}_predictions_plot.png"

        # Sanitize numeric outputs before saving (fixes JSON and CSV issues)
        predictions.replace([np.inf, -np.inf], np.nan, inplace=True)
        predictions.fillna(0, inplace=True)

        for col in predictions.select_dtypes(include=[np.number]).columns:
            predictions[col] = predictions[col].astype(float)

        # --- Save to CSVs ---
        predictions.to_csv(forecast_csv, index=False)
        predictions.to_csv(predictions_csv, index=False)
        log.info(f"[Predict] Saved sanitized CSVs → {forecast_csv}, {predictions_csv}")

        # --- Optional: Generate chart ---
        import matplotlib.pyplot as plt
        plt.figure(figsize=(10, 6))
        if "DL_Forecast" in predictions:
            plt.plot(predictions["Date"], predictions["DL_Forecast"], label="LSTM Forecast", alpha=0.8)
        if "XGB_Forecast" in predictions:
            plt.plot(predictions["Date"], predictions["XGB_Forecast"], label="XGB Forecast", alpha=0.8)
        if "Hybrid_Forecast" in predictions:
            plt.plot(predictions["Date"], predictions["Hybrid_Forecast"], label="Hybrid Forecast", color="black",
                     linewidth=2)
        if "Close" in predictions:
            plt.plot(predictions["Date"], predictions["Close"], label="Close (synthetic)", linestyle="--", alpha=0.6)

        plt.title(f"{config.ticker} Forecasts")
        plt.xlabel("Date")
        plt.ylabel("Price")
        plt.legend()
        plt.tight_layout()
        plt.savefig(plot_path)
        plt.close()

        log.info(f"[Predict] Saved plot → {plot_path}")

        return predictions


    # ----------------------------------------------------------------------
    # Error handling
    # ----------------------------------------------------------------------
    except Exception as e:
        err_trace = "".join(traceback.format_exception(type(e), e, e.__traceback__))
        log.error(f"[Predict] Failed for {ticker or (config and config.ticker)}:\n{err_trace}")

        err_log = OUTPUT_DIR / f"{ticker or config.ticker}_predict_error.log"
        err_log.write_text(err_trace, encoding="utf-8")

        raise RuntimeError(f"Prediction pipeline failed for {ticker}: {e}")


