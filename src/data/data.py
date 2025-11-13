# src/data/data.py
import os
import pandas as pd
import yfinance as yf
import logging
import subprocess
import sys
from datetime import datetime, timedelta
from typing import Optional, Tuple
from tensorboard.backend.event_processing.data_provider import logger


def ensure_latest_yfinance():
    """
    Check if yfinance is up to date, and upgrade automatically if not.
    Runs only once at program start.
    """
    try:
        print("[Update] Checking for yfinance updates...")
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "--upgrade", "yfinance"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        print("[Update] yfinance is up to date.")
    except Exception as e:
        print(f"[Warning] Could not verify yfinance version: {e}")
ensure_latest_yfinance()



class StockDataLoader:
    """
    Downloads, caches, reloads, and preprocesses stock market data.
    """
    def __init__(self, cache_dir: str = "data_cache", logs_dir: str = "logs" ,cache_expiry_days: int = 1):
        self.cache_dir = cache_dir
        self.cache_expiry_days = cache_expiry_days
        self.logs_dir = logs_dir

        os.makedirs(cache_dir, exist_ok=True)
        os.makedirs(logs_dir, exist_ok=True)

        self.logger = self._setup_logger()

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
        """Generate a consistent filename for cached CSV."""
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
    ) -> pd.DataFrame:

        if end is None:
            end = datetime.today().strftime("%Y-%m-%d")

        cache_path = self._get_cache_path(ticker, interval)
        cache_valid = self._is_cache_fresh(cache_path)

        if cache_valid and not force_download and os.path.exists(cache_path):
            try:
                df = pd.read_csv(cache_path, index_col="Date", parse_dates=True)
                if df.empty or "Close" not in df.columns:
                    self.logger.info(f"Loading data from cache: {ticker}({cache_path})")
                    return df
            except Exception as e:
                self.logger.warning(f"Invalid cache {cache_path}: {e}")
        self.logger.info(f"Refreshing data from cache: {ticker}...")
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
            df.to_csv(cache_path, index=True)
            self.logger.info(f"Saved updated cache: {cache_path}")
            return df
        except Exception as e:
            self.logger.error(f"Failed to fetch {ticker}: {e}")
            return pd.read_csv(cache_path, index_col="Date", parse_dates=True)
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

    data = loader.fetch("AAPL", start="2020-01-01", interval="1d", force_download=True)
    if not data.empty:
        clean = loader.preprocess(data, normalize=True)
        train, test = loader.split(clean)
        print(f"Train: {train.shape}, Test: {test.shape}")
    else:
        print("Failed to load AAPL data — see logs for details.")
