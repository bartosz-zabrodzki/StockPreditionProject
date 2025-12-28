FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# System deps + Python 3.11 (deadsnakes) + build deps + R (dla rpy2)
RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common ca-certificates curl \
    build-essential gfortran pkg-config \
    libopenblas-dev liblapack-dev \
    libffi-dev \
    libxml2-dev libcurl4-openssl-dev libssl-dev \
    zlib1g-dev libbz2-dev liblzma-dev \
 && add-apt-repository ppa:deadsnakes/ppa -y \
 && apt-get update && apt-get install -y --no-install-recommends \
    python3.11 python3.11-dev python3.11-venv \
    r-base r-base-dev \
 && rm -rf /var/lib/apt/lists/*

# pip dla Python 3.11
RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3.11

# Torch CUDA 12.1 + reszta zależności
COPY requirements.docker.txt /app/requirements.docker.txt

RUN python3.11 -m pip install --no-cache-dir -U pip \
 && python3.11 -m pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cu121 \
      torch torchvision torchaudio \
 && python3.11 -m pip install --no-cache-dir -r /app/requirements.docker.txt

COPY . /app

CMD ["python3.11", "-m", "src.prediction.predict"]
