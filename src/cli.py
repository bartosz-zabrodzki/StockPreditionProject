import argparse
import os
from pathlib import Path

from src.utils.tickers import load_tickers, pick_interactive


def _validate_ticker_yf(ticker: str) -> None:
    import yfinance as yf
    df = yf.download(ticker, period="5d", interval="1d", progress=False)
    if df is None or df.empty:
        raise SystemExit(f"Ticker nie zwraca danych z yfinance: {ticker}")


def _choose_ticker(args: argparse.Namespace) -> str:
    ticker = (getattr(args, "ticker", None) or "").strip().upper()
    if ticker:
        return ticker

    csv_path = Path(getattr(args, "ticker_list", "configs/tickers.csv"))
    items = load_tickers(csv_path)
    return pick_interactive(items)


def _set_common_env(args: argparse.Namespace, ticker: str) -> None:
    os.environ["STOCK_TICKER"] = ticker

    if getattr(args, "period", None):
        os.environ["STOCK_PERIOD"] = args.period
    if getattr(args, "interval", None):
        os.environ["STOCK_INTERVAL"] = args.interval

    os.environ["NORMALIZE"] = "0" if getattr(args, "no_normalize", False) else "1"
    os.environ["SPLIT"] = "0" if getattr(args, "no_split", False) else "1"


def cmd_fetch(args: argparse.Namespace) -> None:
    ticker = _choose_ticker(args)

    if getattr(args, "validate", False):
        _validate_ticker_yf(ticker)

    _set_common_env(args, ticker)

    # LAZY import -> brak side-effectów przy `-h`
    from src.data.data import run as run_fetch

    run_fetch(
        ticker=ticker,
        interval=os.environ.get("STOCK_INTERVAL", "1d"),
        normalize=(os.environ.get("NORMALIZE", "1") == "1"),
        split=(os.environ.get("SPLIT", "1") == "1"),
    )

def cmd_features(args: argparse.Namespace) -> None:
    ticker = _choose_ticker(args)
    _set_common_env(args, ticker)

    from src.features.build_features import run as run_features
    run_features(ticker=ticker, interval=getattr(args, "interval", None) or "1d")

def cmd_train(args: argparse.Namespace) -> None:
    ticker = _choose_ticker(args)
    _set_common_env(args, ticker)

    if getattr(args, "fast", False):
        os.environ["FAST_DEV_RUN"] = "1"

    from src.training.train_model import run as run_train
    run_train(ticker=ticker)


def cmd_predict(args: argparse.Namespace) -> None:
    ticker = _choose_ticker(args)
    _set_common_env(args, ticker)

    from src.prediction.predict import run as run_predict
    run_predict(ticker=ticker)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="stockpred")
    sub = p.add_subparsers(dest="cmd", required=True)



    # fetch
    f = sub.add_parser("fetch", help="Pobierz dane dla tickera")
    f.add_argument("-t", "--ticker", help="np. AAPL, MSFT (gdy brak, uruchamia wybór interaktywny)")
    f.add_argument("--ticker-list", default="configs/tickers.csv", help="CSV z tickerami")
    f.add_argument("--validate", action="store_true", help="Sprawdź, czy yfinance zwraca dane")
    f.add_argument("--period", default=None, help="np. max, 5y, 1y, 6mo")
    f.add_argument("--interval", default=None, help="np. 1d, 1h, 15m")
    f.add_argument("--no-normalize", action="store_true")
    f.add_argument("--no-split", action="store_true")
    f.set_defaults(func=cmd_fetch)

    fe = sub.add_parser("features", help="Zbuduj plik *_features.csv z cache")
    fe.add_argument("-t", "--ticker", help="np. AAPL, MSFT (gdy brak, uruchamia wybór interaktywny)")
    fe.add_argument("--ticker-list", default="configs/tickers.csv", help="CSV z tickerami")
    fe.add_argument("--interval", default="1d", help="np. 1d, 1h")
    fe.set_defaults(func=cmd_features)

    # train
    t = sub.add_parser("train", help="Trenuj modele dla tickera")
    t.add_argument("-t", "--ticker", help="np. AAPL, MSFT (gdy brak, uruchamia wybór interaktywny)")
    t.add_argument("--ticker-list", default="configs/tickers.csv", help="CSV z tickerami")
    t.add_argument("--fast", action="store_true", help="Szybki tryb (CI/smoke)")
    t.set_defaults(func=cmd_train)

    # predict
    pr = sub.add_parser("predict", help="Wykonaj predykcje i zapisz wyniki")
    pr.add_argument("-t", "--ticker", help="np. AAPL, MSFT (gdy brak, uruchamia wybór interaktywny)")
    pr.add_argument("--ticker-list", default="configs/tickers.csv", help="CSV z tickerami")
    pr.set_defaults(func=cmd_predict)

    return p


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
