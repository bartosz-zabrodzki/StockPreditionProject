from src.models.config import get_model_paths

import os
import pandas as pd
import yfinance as yf
from pathlib import Path
from src.config.paths import CACHE_DIR

from datetime import datetime, timedelta
from typing import Optional, Tuple
import logging
logger = logging.getLogger(__name__)




DEFAULT_TICKER = os.getenv("STOCK_TICKER", "AAPL").strip().upper() or "AAPL"
paths = get_model_paths()
TICKER = paths["TICKER"]
INTERVAL = paths["INTERVAL"]
PERIOD = paths["PERIOD"]

class StockDataLoader:
    def __init__(self, cache_dir: Path = CACHE_DIR, logs_dir: Path = Path("logs"), cache_expiry_days: int = 1):
        self.cache_dir = Path(cache_dir)
        self.logs_dir = Path(logs_dir)
        self.cache_expiry_days = int(cache_expiry_days)

        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(parents=True, exist_ok=True)

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

    def _get_cache_path(self, ticker: str, interval: str) -> Path:
        safe_ticker = ticker.strip().upper().replace("/", "_")
        return self.cache_dir / f"{safe_ticker}_{interval}.csv"

    def _is_cache_fresh(self, cache_path: Path) -> bool:
        if not cache_path.exists():
            return False
        file_time = datetime.fromtimestamp(cache_path.stat().st_mtime)
        return datetime.now() - file_time < timedelta(days=self.cache_expiry_days)

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

        ticker = ticker.strip().upper()
        interval = interval.strip()

        cache_path = self._get_cache_path(ticker, interval)
        cache_valid = self._is_cache_fresh(cache_path)

        if cache_valid and not force_download and os.path.exists(cache_path):
            df = pd.read_csv(cache_path, parse_dates=["Date"])

        self.logger.info(f"Downloading fresh data: {ticker} (interval={interval})...")

        if overwrite_cache and os.path.exists(cache_path):
            cache_path.unlink()

        try:
            print(f"[Download] Fetching {ticker} from Yahoo Finance...")

            df = yf.download(
                ticker,
                start=start,
                end=end,
                interval=interval,
                auto_adjust=False,
                progress=False,
            )

            if df.empty:
                raise ValueError("No data returned from Yahoo Finance.")

            if isinstance(df.columns, pd.MultiIndex):
                df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]
                self.logger.info("Detected MultiIndex — flattened column names.")

            df.reset_index(inplace=True)

            expected_cols = ["Date", "Open", "High", "Low", "Close", "Volume"]
            df = df[[c for c in expected_cols if c in df.columns]]
            df.rename(columns=lambda x: x.strip().title(), inplace=True)

            if df.index.name == "Date":
                df.reset_index(inplace=True)

            if "Unnamed: 0" in df.columns:
                df.drop(columns=["Unnamed: 0"], inplace=True)

            if list(df.columns).count("Date") > 1:
                df = df.loc[:, ~df.columns.duplicated()]

            df.to_csv(cache_path, index=False, date_format="%Y-%m-%d", float_format="%.6f", encoding="utf-8")

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
            raise  # <-- zamiast: return pd.DataFrame()

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

def run(ticker: str, start: str = "2020-01-01", interval: str = "1d", normalize: bool = True, split: bool = True):
    from src.config.paths import ensure_dirs
    ensure_dirs()

    force = os.getenv("FORCE", "0") == "1"

    loader = StockDataLoader()

    data = loader.fetch(
        ticker,
        start=start,
        interval=interval,
        force_download=force,
        overwrite_cache=force,
    )
    if data.empty:
        raise SystemExit(f"Failed to load {ticker} data — see logs for details.")
    clean = loader.preprocess(data, normalize=normalize)
    if split:
        train, test = loader.split(clean)
        print(f"Train: {train.shape}, Test: {test.shape}")
    return clean


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    default_ticker = (os.getenv("STOCK_TICKER", "AAPL").strip().upper() or "AAPL")
    loader = StockDataLoader(cache_expiry_days=1)
    data = loader.fetch(default_ticker, start="2020-01-01", interval="1d", force_download=True)

    if not data.empty:
        clean = loader.preprocess(data, normalize=True)
        train, test = loader.split(clean)
        print(f"Train: {train.shape}, Test: {test.shape}")
    else:
        print(f"Failed to load {DEFAULT_TICKER} data — see logs for details.")

