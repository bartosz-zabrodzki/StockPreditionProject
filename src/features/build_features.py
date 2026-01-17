from __future__ import annotations

from pathlib import Path
import subprocess

from src.config.paths import CACHE_DIR, FEATURES_DIR, ensure_dirs


def run(ticker: str, interval: str = "1d") -> Path:
    ensure_dirs()

    t = ticker.strip().upper()
    interval = interval.strip() or "1d"

    in_csv = Path(CACHE_DIR) / f"{t}_{interval}.csv"
    if not in_csv.exists():
        raise FileNotFoundError(f"Missing cache file: {in_csv}. Run fetch first.")

    # src/r/analyze.R
    r_script = Path(__file__).resolve().parents[1] / "r" / "analyze.R"
    if not r_script.exists():
        raise FileNotFoundError(f"Missing R script: {r_script}")

    cmd = ["Rscript", str(r_script), "--ticker", t, "--interval", interval]
    p = subprocess.run(cmd, capture_output=True, text=True)

    if p.returncode != 0:
        raise RuntimeError(
            "R pipeline failed\n"
            f"cmd: {' '.join(cmd)}\n"
            f"stdout:\n{p.stdout}\n"
            f"stderr:\n{p.stderr}\n"
        )

    out_csv = Path(FEATURES_DIR) / f"{t}_features.csv"
    if not out_csv.exists():
        raise RuntimeError(f"R finished but missing output: {out_csv}")

    print(f"[Features:R] Saved → {out_csv}")
    return out_csv
