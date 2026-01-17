from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable
import subprocess, json
from src.config.paths import ensure_dirs  # jeśli to uruchamiasz w repo (nie w kontenerze)

R_PATH = r"C:\Program Files\R\R-4.5.2\bin\Rscript.exe"
R_LIBS = Path(os.getcwd()) / "venv_R_libs"
REQUIRED_R_PKGS = ["forecast", "tseries", "TTR", "dplyr", "quantmod"]


def ensure_r_environment() -> str:
    """
    Installs required R packages into an isolated library folder and returns that path.
    """
    if not Path(R_PATH).exists():
        raise FileNotFoundError(f"Rscript not found at {R_PATH}")

    R_LIBS.mkdir(parents=True, exist_ok=True)
    r_lib = str(R_LIBS).replace("\\", "/")

    pkg_vec = "c(" + ",".join(f"'{p}'" for p in REQUIRED_R_PKGS) + ")"

    expr = (
        f".libPaths('{r_lib}'); "
        f"req <- {pkg_vec}; "
        f"inst <- rownames(installed.packages(lib.loc='{r_lib}')); "
        f"missing <- req[!(req %in% inst)]; "
        f"if (length(missing) > 0) {{ "
        f"  install.packages(missing, lib='{r_lib}', repos='https://cloud.r-project.org'); "
        f"  cat('Installed missing packages:', paste(missing, collapse=', '), '\\n'); "
        f"}} else {{ "
        f"  cat('All R packages already installed.\\n'); "
        f"}}"
    )

    subprocess.run([R_PATH, "--vanilla", "-e", expr], check=True)

    # ważne: ustawiamy dla procesu Pythona (i potomnych)
    os.environ["R_LIBS_USER"] = r_lib
    return r_lib




def run_r_script(script_path, args):
    cmd = ["Rscript", script_path] + args
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=90)
    if proc.returncode != 0:
        raise RuntimeError(f"R failed: {proc.stderr}")
    return json.loads(proc.stdout)



def run_analyze_r(script_path: str | Path, ticker: str, interval: str = "1d") -> None:
    """
    Calls analyze.R with explicit args (najstabilniej).
    """
    ensure_dirs()

    t = ticker.strip().upper()
    itv = (interval.strip() or "1d")

    # Możesz też trzymać env jako fallback:
    os.environ["STOCK_TICKER"] = t
    os.environ["STOCK_INTERVAL"] = itv

    run_r_script(script_path, args=["--ticker", t, "--interval", itv])
