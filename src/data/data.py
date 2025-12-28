import logging
import os
import subprocess
import sys
from datetime import datetime, timedelta
from typing import Optional, Tuple

import pandas as pd
import yfinance as yf

DEFAULT_TICKER = os.getenv("STOCK_TICKER", "AAPL").strip().upper() or "AAPL"


def ensure_latest_yfinance(logger: Optional[logging.Logger] = None) -> None:
    """Optionally refresh the yfinance installation when explicitly requested."""
    log = logger.info if logger else print
    warn = logger.warning if logger else print
    try:
        log("[Update] Checking for yfinance updates...")
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "--upgrade", "yfinance"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        log("[Update] yfinance is up to date.")
    except Exception as exc:
        warn(f"[Warning] Could not verify yfinance version: {exc}")


class StockDataLoader:

    def __init__(
        self,
        cache_dir: str = "data_cache",
        logs_dir: str = "logs",
        cache_expiry_days: int = 1,
        auto_update_yfinance: bool = False,
    ):
        self.cache_dir = cache_dir
        self.cache_expiry_days = cache_expiry_days
        self.logs_dir = logs_dir

        os.makedirs(cache_dir, exist_ok=True)
        os.makedirs(logs_dir, exist_ok=True)

        self.logger = self._setup_logger()
        if auto_update_yfinance:
            ensure_latest_yfinance(self.logger)

    def _setup_logger(self):
        logfile = os.path.join(self.logs_dir, "data_loader.log")
        logger = logging.getLogger("StockDataLoader")
        logger.setLevel(logging.INFO)

        if not logger.handlers:
            file_handler = logging.FileHandler(logfile)
            console_handler = logging.StreamHandler()

            fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", "%Y-%m-%d %H:%M:%S")
            file_handler.setFormatter(fmt)
            console_handler.setFormatter(fmt)

            logger.addHandler(file_handler)
            logger.addHandler(console_handler)

        return logger

    def _get_cache_path(self, ticker: str, interval: str) -> str:
        safe_ticker = ticker.strip().upper().replace("/", "_")
        return os.path.join(self.cache_dir, f"{safe_ticker}_{interval}.csv")

    def _is_cache_fresh(self, cache_path: str) -> bool:
        if not os.path.exists(cache_path):
            return False
        file_time = datetime.fromtimestamp(os.path.getmtime(cache_path))
        return datetime.now()  - file_time < timedelta(days = self.cache_expiry_days)

    def fetch(
            self,
            ticker: str,
            start: str = "2015-01-01",
            end: Optional[str] = None,
            interval: str = "1d",
            force_download: bool = False,
            overwrite_cache: bool = False,
    ) -> pd.DataFrame:

        if end is None:
            end = datetime.today().strftime("%Y-%m-%d")

        cache_path = self._get_cache_path(ticker, interval)
        cache_valid = self._is_cache_fresh(cache_path)

        if cache_valid and not force_download and os.path.exists(cache_path):
            try:
                df = pd.read_csv(cache_path, parse_dates=["Date"])
                if not df.empty and "Close" in df.columns:
                    self.logger.info(f"Loaded cached data for {ticker}")
                    return df
            except Exception as e:
                self.logger.warning(f"Invalid cache {cache_path}: {e}")

        self.logger.info(f"Refreshing data from cache: {ticker}...")

        if overwrite_cache and os.path.exists(cache_path):
            try:
                os.remove(cache_path)
                self.logger.info(f"Old cache removed: {cache_path}")
            except Exception as e:
                self.logger.warning(f"Failed to remove old cache {cache_path}: {e}")

        try:
            print(f"[Download] Fetching {ticker} from Yahoo Finance...")
            df = yf.download(
                ticker.strip().upper(),
                start=start,
                end=end,
                interval=interval,
                progress=False,
                auto_adjust=True,
            )

            if df.empty:
                raise ValueError("No data returned from Yahoo Finance.")

            if isinstance(df.columns, pd.MultiIndex):
                df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]
                self.logger.info("Detected MultiIndex — flattened column names.")

            df.reset_index(inplace=True)

            expected_cols = ["Date", "Open", "High", "Low", "Close", "Volume"]
            df = df[[c for c in df.columns if c in expected_cols]]
            df.rename(columns=lambda x: x.strip().title(), inplace=True)

            if df.index.name == "Date":
                df.reset_index(inplace=True)

            if "Unnamed: 0" in df.columns:
                df.drop(columns=["Unnamed: 0"], inplace=True)

            if list(df.columns).count("Date") > 1:
                df = df.loc[:, ~df.columns.duplicated()]

            df.to_csv(
                cache_path,
                index=False,
                date_format="%Y-%m-%d",
                float_format="%.6f",
                encoding="utf-8"
            )
            self.logger.info(f"Saved clean cache: {cache_path}")

            test_df = pd.read_csv(cache_path, nrows=5)
            if "Date" not in test_df.columns or "Close" not in test_df.columns:
                raise ValueError(f"Corrupted CSV: Missing 'Date' or 'Close' in {cache_path}")

            self.logger.info(f"Validated CSV structure: {list(test_df.columns)}")
            return df



        except Exception as e:
            import traceback
            err = traceback.format_exc()
            self.logger.error(f"Failed to fetch {ticker}: {e}\nTraceback:\n{err}")
            return pd.DataFrame()

    def preprocess(self, df: pd.DataFrame, dropna: bool = True,normalize: bool = False,) -> pd.DataFrame:

        df = df.copy()
        if dropna:
            df.dropna(inplace=True)

        if normalize:
            numeric_cols = df.select_dtypes(include="number").columns
            df[numeric_cols] = (df[numeric_cols] - df[numeric_cols].mean()) / df[numeric_cols].std()
        self.logger.info(f"Preprocessed dataframe (normalize={normalize}) | Rows: {len(df)}")
        return df

    def split(
        self, df: pd.DataFrame, train_ratio: float = 0.8) -> Tuple[pd.DataFrame, pd.DataFrame]:
        split_idx = int(len(df) * train_ratio)
        self.logger.info(f"Split data into train/test ({train_ratio * 100:.0f}%/{(1 - train_ratio) * 100:.0f}%)")
        return df.iloc[:split_idx], df.iloc[split_idx:]


if __name__ == "__main__":
    loader = StockDataLoader(cache_expiry_days=1)

    data = loader.fetch(DEFAULT_TICKER, start="2020-01-01", interval="1d", force_download=True)
    if not data.empty:
        clean = loader.preprocess(data, normalize=True)
        train, test = loader.split(clean)
        print(f"Train: {train.shape}, Test: {test.shape}")
    else:
        print(f"Failed to load {DEFAULT_TICKER} data — see logs for details.")
