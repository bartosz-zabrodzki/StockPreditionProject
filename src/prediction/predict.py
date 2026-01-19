from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd

from src.config.paths import MODEL_DIR, OUTPUT_DIR
import subprocess, re, json, logging

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

        # Detect trained input size reliably from state_dict
    state = torch.load(lstm_model_path, map_location="cpu")

    w_key = "lstm.weight_ih_l0"
    if w_key not in state:
        raise KeyError(f"Missing {w_key} in state_dict: {list(state.keys())[:10]} ...")

    trained_input_size = int(state[w_key].shape[1])
    log.info(f"[Model] Detected trained input size from {w_key}: {trained_input_size}")

    input_size = trained_input_size

    lstm = LSTMModel(
        input_size=input_size,
        hidden_size=lstm_config["hidden_size"],
        num_layers=lstm_config["num_layers"],
        dropout=lstm_config["dropout"],
    )
    lstm.load_state_dict(state, strict=True)
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



def run_r_analysis(ticker: str, interval: str = "1d") -> pd.DataFrame:
    """
    Executes the R pipeline (analyze.R) and returns its forecast as a DataFrame.
    If R fails or JSON parsing fails, a graceful fallback empty DataFrame is returned.
    """
    r_script = (Path(__file__).resolve().parents[1] / "r" / "analyze.R")
    cmd = ["Rscript", str(r_script), "--ticker", ticker, "--interval", interval]

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
    """
    Runs model inference (LSTM + XGB + R hybrid) robustly.
    Automatically detects univariate vs multivariate training.
    """

    import torch
    from sklearn.preprocessing import MinMaxScaler

    target = "Close"
    required = feature_cols + [target, "Date"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # ---------------------------------------------------------
    #  Auto-detect trained input size from loaded LSTM
    # ---------------------------------------------------------
    try:
        trained_input_size = lstm.lstm.input_size
    except AttributeError:
        trained_input_size = getattr(lstm, "input_size", len(feature_cols))

    log.info(f"[Predict] LSTM expects {trained_input_size} input features.")

    # ---------------------------------------------------------
    #  Select correct feature set based on model type
    # ---------------------------------------------------------
    if trained_input_size == 1:
        log.info("[Predict] Using univariate input (Close-only).")
        x = df[[target]].to_numpy()
    else:
        log.info("[Predict] Using multivariate input (feature_cols).")
        x = df[feature_cols].to_numpy()

    y = df[[target]].to_numpy()

    # ---------------------------------------------------------
    #  Ensure scaler and input align
    # ---------------------------------------------------------
    if hasattr(scaler_x, "n_features_in_") and int(scaler_x.n_features_in_) != x.shape[1]:
        log.warning(
            f"[Predict] Feature count mismatch: scaler expects {int(scaler_x.n_features_in_)} cols, got {x.shape[1]} — refitting temporary scaler."
        )
        temp_scaler = MinMaxScaler()
        x_scaled = temp_scaler.fit_transform(x)
    else:
        x_scaled = scaler_x.transform(x)

    seq_len = int(lstm_config["input_seq_len"])
    if len(x_scaled) <= seq_len:
        raise ValueError(f"Not enough rows for seq_len={seq_len}")

    # ---------------------------------------------------------
    #  Prepare LSTM sequence input
    # ---------------------------------------------------------
    x_seq = np.array([x_scaled[i:i + seq_len] for i in range(len(x_scaled) - seq_len)])
    x_t = torch.tensor(x_seq, dtype=torch.float32)

    with torch.no_grad():
        lstm_pred_scaled = lstm(x_t).numpy()

    # ---------------------------------------------------------
    #  XGB predictions (aligned)
    # ---------------------------------------------------------
    xgb_input = x_scaled[seq_len - 1:-1]
    xgb_pred_scaled = xgb.predict(xgb_input).reshape(-1, 1)

    # ---------------------------------------------------------
    #  Safe inverse scaling
    # ---------------------------------------------------------
    # --- Safe inverse scaling ---
    def safe_inverse(y_scaled, scaler):
        # if predictions look already in the same magnitude as Close, skip inverse
        median_val = np.median(y_scaled)
        if median_val > 20:  # heuristically means not scaled (already in raw price range)
            return y_scaled
        try:
            inv = scaler.inverse_transform(y_scaled)
            # sanity check: if result explodes, fall back
            if np.abs(np.median(inv)) > 10_000:
                log.warning("[Predict] Inverse-scaling produced implausible values — reverting to raw output.")
                return y_scaled
            return inv
        except Exception as e:
            log.warning(f"[Predict] Inverse-scaling failed: {e}")
            return y_scaled

    lstm_pred = safe_inverse(lstm_pred_scaled, scaler_y)
    xgb_pred = safe_inverse(xgb_pred_scaled, scaler_y)

    # ---------------------------------------------------------
    # Construct aligned forecast DataFrame
    # ---------------------------------------------------------
    dates = pd.to_datetime(df["Date"].iloc[-len(lstm_pred):]).reset_index(drop=True)
    pred_df = pd.DataFrame({
        "Date": dates,
        "DL_Forecast": pd.to_numeric(pd.Series(lstm_pred.squeeze()), errors="coerce"),
        "XGB_Forecast": pd.to_numeric(pd.Series(xgb_pred.squeeze()), errors="coerce"),
    })

    #pred_df.replace([np.inf, -np.inf], np.nan, inplace=True)
   # pred_df.fillna(0, inplace=True)
    log.info("[Predict] NaNs: DL=%s XGB=%s",
             pred_df["DL_Forecast"].isna().sum(),
             pred_df["XGB_Forecast"].isna().sum())

    nonzero_mask = (pred_df["DL_Forecast"].abs() > 1e-6) | (pred_df["XGB_Forecast"].abs() > 1e-6)
    pred_df = pred_df[nonzero_mask].reset_index(drop=True)

    # ---------------------------------------------------------
    #  Merge with R hybrid forecast (safe)
    # ---------------------------------------------------------
    try:
        r_forecast = run_r_analysis(ticker)
        if not r_forecast.empty:
            if "Date" not in r_forecast.columns:
                r_forecast.rename(columns={r_forecast.columns[0]: "Date"}, inplace=True)

            r_forecast["Date"] = pd.to_datetime(r_forecast["Date"])
            pred_df["Date"] = pd.to_datetime(pred_df["Date"])

            merged = pd.merge(pred_df, r_forecast, on="Date", how="left", suffixes=("", "_r"))
            merged.sort_values("Date", inplace=True)
            merged.reset_index(drop=True, inplace=True)

            for col in ["DL_Forecast", "XGB_Forecast"]:
                if col in merged.columns:
                    merged[col] = pd.to_numeric(merged[col], errors="coerce").ffill()

            merged["R_Forecast"] = pd.to_numeric(merged.get("R_Forecast", 0), errors="coerce").fillna(0)

            merged["Hybrid_Forecast"] = (
                    0.6 * merged["DL_Forecast"] +
                    0.3 * merged["XGB_Forecast"] +
                    0.1 * merged["R_Forecast"]
            )

            log.info(f"[Predict] Hybrid forecast merge complete: {len(merged)} rows.")
            return merged

        else:
            log.warning("[Predict] R returned empty forecast — using DL/XGB hybrid only.")
            pred_df["Hybrid_Forecast"] = 0.7 * pred_df["DL_Forecast"] + 0.3 * pred_df["XGB_Forecast"]
            return pred_df

    except Exception as e:
        log.error(f"[Predict] R merge failed: {e}")
        pred_df["Hybrid_Forecast"] = 0.7 * pred_df["DL_Forecast"] + 0.3 * pred_df["XGB_Forecast"]
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
        # ls: cannot access '/data/output': No such file or directory
        # Dynamic config auto-build if called as run(ticker="AAPL")
        # ------------------------------------------------------------------
        if config is None:
            if ticker is None:
                ticker = os.getenv("STOCK_TICKER", "AAPL")

            interval = interval or os.getenv("STOCK_INTERVAL", "1d")

            p1 = FEATURES_DIR / f"{ticker}_{interval}_features.csv"
            p2 = FEATURES_DIR / f"{ticker}_features.csv"
            feature_file = p1 if p1.exists() else p2

            config = PredictConfig(
                feature_file=feature_file,
                output_dir=OUTPUT_DIR,
                ticker=ticker,
                forecast_days=forecast_days,
            )

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
        # --- Restore real 'Close' column if missing ---
        if "Close" not in predictions.columns:
            try:
                # Attempt to recover from feature file
                raw_df = pd.read_csv(config.feature_file)
                if "Close" in raw_df.columns:
                    # Align number of rows
                    predictions["Close"] = (
                        raw_df["Close"].tail(len(predictions)).reset_index(drop=True)
                    )
                    log.info(f"[Predict] Restored real Close values from features file ({len(predictions)} rows).")
                elif "Close_raw" in raw_df.columns:
                    predictions["Close"] = (
                        raw_df["Close_raw"].tail(len(predictions)).reset_index(drop=True)
                    )
                    log.info(f"[Predict] Restored real Close values from Close_raw column ({len(predictions)} rows).")
                else:
                    raise KeyError("No Close or Close_raw column in features file.")
            except Exception as e:
                # Fallback if features truly missing close data
                last_close = float(predictions["DL_Forecast"].iloc[0])
                predictions["Close"] = last_close
                log.warning(f"[Predict] Synthetic Close used (fallback): {e}")

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
        # --- Clear stale forecast files before saving ---
        for old in output_dir.glob(f"{config.ticker}_*.csv"):
            try:
                old.unlink()
                log.info(f"[Predict] Removed old output → {old.name}")
            except Exception as e:
                log.warning(f"[Predict] Could not delete {old}: {e}")

        forecast_csv = output_dir / f"{config.ticker}_forecast.csv"
        predictions_csv = output_dir / f"{config.ticker}_predictions.csv"
        plot_path = output_dir / f"{config.ticker}_predictions_plot.png"

        # Sanitize numeric outputs before saving (fixes JSON and CSV issues)
        predictions.replace([np.inf, -np.inf], np.nan, inplace=True)

        # tylko to, co ma sens zerować (np. R_Forecast może być 0 jeśli brak)
        for col in ["R_Forecast"]:
            if col in predictions.columns:
                predictions[col] = pd.to_numeric(predictions[col], errors="coerce").fillna(0)

        # DL/XGB/Hybrid zostaw bez sztucznych zer (ew. forward-fill, jeśli chcesz ciągłość)
        for col in ["DL_Forecast", "XGB_Forecast", "Hybrid_Forecast"]:
            if col in predictions.columns:
                predictions[col] = pd.to_numeric(predictions[col], errors="coerce").ffill()

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


