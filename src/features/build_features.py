from __future__ import annotations

from pathlib import Path
import subprocess
import logging
from src.config.paths import CACHE_DIR, FEATURES_DIR, ensure_dirs

log = logging.getLogger(__name__)

def run(ticker: str, interval: str = "1d", force: bool = False) -> Path:
    """
    Builds features for a given ticker via R script.
    If force=True, old feature files are deleted before regeneration.
    """
    ensure_dirs()

    t = ticker.strip().upper()
    interval = interval.strip() or "1d"

    in_csv = Path(CACHE_DIR) / f"{t}_{interval}.csv"
    out_csv = Path(FEATURES_DIR) / f"{t}_{interval}_features.csv"


    if not in_csv.exists():
        raise FileNotFoundError(f"Missing cache file: {in_csv}. Run fetch first.")

    # 🧹 Delete stale file if forcing rebuild
    if out_csv.exists() and force:
        try:
            out_csv.unlink()
            log.info(f"[Features] Deleted old feature file → {out_csv}")
        except Exception as e:
            log.warning(f"[Features] Could not delete {out_csv}: {e}")

    # src/r/analyze.R
    r_script = Path(__file__).resolve().parents[1] / "r" / "analyze.R"
    if not r_script.exists():
        raise FileNotFoundError(f"Missing R script: {r_script}")

    cmd = ["Rscript", str(r_script), "--ticker", t, "--interval", interval]
    log.info(f"[Features] Running R pipeline for {t} ({interval}) ...")
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=180)

    legacy_csv = Path(FEATURES_DIR) / f"{t}_features.csv"
    if legacy_csv.exists() and not out_csv.exists():
        try:
            legacy_csv.replace(out_csv)  # rename
            log.info(f"[Features] Renamed legacy output → {out_csv.name}")
        except Exception as e:
            raise RuntimeError(f"[Features] Could not rename {legacy_csv.name} to {out_csv.name}: {e}")



    if p.returncode != 0:
        raise RuntimeError(
            f"[Features] R pipeline failed (code={p.returncode})\n"
            f"cmd: {' '.join(cmd)}\n"
            f"stdout(tail):\n{p.stdout[-2000:]}\n"
            f"stderr(tail):\n{p.stderr[-2000:]}\n"
        )

    if not out_csv.exists():
        raise RuntimeError(f"[Features] R finished but missing output: {out_csv}")

    log.info(f"[Features] Saved → {out_csv}")
    return out_csv
