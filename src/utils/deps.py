# src/utils/deps.py
from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path


def ensure_latest_yfinance(
    *,
    check_every_hours: int = 24,
    state_file: str = ".cache/yfinance_last_check.txt",
) -> None:
    """
    Auto-update yfinance, ale nie częściej niż raz na N godzin.
    Stan trzymamy w pliku na /app (bind mount), więc działa też między uruchomieniami w Dockerze.
    """
    if os.getenv("YFINANCE_AUTOUPDATE", "1") != "1":
        return

    state_path = Path(state_file)
    state_path.parent.mkdir(parents=True, exist_ok=True)

    now = time.time()
    if state_path.exists():
        try:
            last = float(state_path.read_text().strip() or "0")
            if now - last < check_every_hours * 3600:
                return
        except Exception:
            pass

    print("[Update] Checking for yfinance updates...")
    try:
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "--upgrade", "yfinance"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        print("[Update] yfinance is up to date.")
        state_path.write_text(str(now), encoding="utf-8")
    except Exception as e:
        print(f"[Warning] Could not verify/update yfinance version: {e}")
