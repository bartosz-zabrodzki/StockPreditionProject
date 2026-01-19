# ============================================================
# Base Image
# ============================================================
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04 AS base

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8

WORKDIR /app

# --- System deps + Python + R ---
RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common ca-certificates curl \
    build-essential gfortran pkg-config \
    libopenblas-dev liblapack-dev \
    libffi-dev \
    libxml2-dev libcurl4-openssl-dev libssl-dev \
    zlib1g-dev libbz2-dev liblzma-dev \
 && add-apt-repository ppa:deadsnakes/ppa -y \
 && apt-get update && apt-get install -y --no-install-recommends \
    python3.11 python3.11-dev python3.11-venv python3.11-distutils \
    r-base r-base-dev \
 && apt-get clean \
 && rm -rf /var/lib/apt/lists/*

RUN Rscript -e "options(repos=c(CRAN='https://cloud.r-project.org')); install.packages(c('forecast','fs','jsonlite','dplyr','ggplot2','lubridate','tseries','TTR','quantmod','readr'), Ncpus=parallel::detectCores())"


# pip for Python 3.11
RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3.11

# Python deps
COPY requirements.docker.txt /app/requirements.docker.txt

RUN python3.11 -m pip install --no-cache-dir -U pip setuptools wheel \
 && python3.11 -m pip install --no-cache-dir \
      --extra-index-url https://download.pytorch.org/whl/cu121 \
      torch torchvision torchaudio \
 && python3.11 -m pip install --no-cache-dir -r /app/requirements.docker.txt

# App code last (better caching)
COPY . /app

# ============================================================
# Stage: API
# ============================================================
FROM base AS api
CMD ["python3.11", "-m", "uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "8000"]

# ============================================================
# Stage: UI
# ============================================================
FROM base AS ui
CMD ["streamlit", "run", "src/ui/app.py", "--server.address=0.0.0.0", "--server.port=8501"]

# ============================================================
# Stage: Stock Predictor
# ============================================================
FROM base AS stock-predictor
CMD ["python3.11", "-m", "src.prediction.predict"]

