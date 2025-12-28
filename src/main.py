import os
import subprocess
import sys
import tempfile

R_PATH = r"C:\Program Files\R\R-4.5.2\bin\Rscript.exe"
R_LIBS = os.path.join(os.getcwd(), "venv_R_libs")

def ensure_r_environment():
    if not os.path.exists(R_PATH):
        sys.exit(f"[ERROR] Rscript not found at {R_PATH}")

    os.makedirs(R_LIBS, exist_ok=True)
    r_safe_path = R_LIBS.replace("\\", "/")

    r_script = f"""
    libs <- .libPaths()
    .libPaths('{r_safe_path}')
    req <- c('forecast','tseries','TTR','dplyr','quantmod')
    installed <- rownames(installed.packages(lib.loc='{r_safe_path}'))
    missing <- req[!(req %in% installed)]
    if (length(missing) > 0) {{
        install.packages(missing, lib='{r_safe_path}', repos='https://cloud.r-project.org')
        cat('Installed missing packages:', missing, '\\n')
    }} else {{
        cat('All R packages already installed.\\n')
    }}
    """

    with tempfile.NamedTemporaryFile(mode="w", suffix=".R", delete=False, encoding="utf-8") as tmp_file:
        tmp_file.write(r_script)
        tmp_path = tmp_file.name

    print(f"[Bootstrap] Executing R environment check via script: {tmp_path}")
    try:
        subprocess.run([R_PATH, tmp_path], check=True)
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] R setup failed with exit code {e.returncode}")
        sys.exit(1)
    finally:
        os.remove(tmp_path)

    os.environ["R_LIBS_USER"] = r_safe_path
    os.environ["R_LIBS"] = r_safe_path
    print(f"[Bootstrap] R environment verified and bound to isolated library: {r_safe_path}\n")


def run_r_script(script_path: str, *args):
    """Execute any R script with the virtual library path injected automatically."""
    if not os.path.exists(script_path):
        raise FileNotFoundError(f"R script not found: {script_path}")

    r_safe_path = R_LIBS.replace("\\", "/")
    cmd = [
        R_PATH,
        "--vanilla",
        "-e",
        f".libPaths('{r_safe_path}'); source('{script_path.replace(os.sep, '/')}')"
    ]

    print(f"[R-Run] Executing: {script_path}")
    subprocess.run(cmd, check=True)
    print(f"[R-Run] Completed: {script_path}\n")
