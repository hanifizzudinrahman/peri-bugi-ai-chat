# =============================================================================
# Dockerfile — peri-bugi-ai-chat (GPU support)
#
# STRATEGI LAYER CACHING:
# Layer diurutkan dari "paling jarang berubah" ke "paling sering berubah".
# Docker hanya rebuild layer yang berubah dan semua layer di bawahnya.
#
# Layer order:
#   1. Base image (nvidia cuda)       → cache permanent
#   2. System packages                → cache selama apt list tidak berubah
#   3. Install torch + cuda (~2.5 GB) → cache selama versi torch tidak berubah
#   4. Install sentence-transformers  → cache selama versi tidak berubah
#   5. Install requirements-base.txt  → cache selama library tidak berubah
#   6. Copy app code                  → rebuild setiap ada code change (cepat)
#
# KENAPA NVIDIA BASE IMAGE:
# python:3.11-slim tidak punya CUDA runtime — GPU tidak bisa diakses dari
# dalam container. Harus pakai nvidia/cuda base image yang sudah include
# libcuda.so dan cudart.
# =============================================================================

# CUDA 12.1 runtime + cuDNN 8 + Ubuntu 22.04
# Sesuaikan dengan versi CUDA di host kamu (lihat nvidia-smi)
# Host kamu: CUDA 12.3 → bisa pakai image cu121 (backward compatible)
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

# ── Layer 1: System setup ─────────────────────────────────────────────────────
# Set timezone supaya tidak ada interactive prompt saat apt install
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Jakarta

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-dev \
    python3-pip \
    python3.11-distutils \
    gcc \
    g++ \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    # Buat symlink python3 → python3.11 dan pip → pip3
    && ln -sf /usr/bin/python3.11 /usr/bin/python3 \
    && ln -sf /usr/bin/python3.11 /usr/bin/python \
    && pip3 install --upgrade pip

WORKDIR /app

# ── Layer 2: Install torch dengan CUDA (~2.5 GB) ─────────────────────────────
# Layer ini paling berat. Diletakkan PALING AWAL supaya jarang invalide cache.
# Hanya rebuild kalau requirements-gpu.txt berubah (versi torch berubah).
# Kalau kamu ubah app code → layer ini tetap cached, tidak download ulang.
COPY requirements-gpu.txt .
RUN pip install --no-cache-dir \
    --index-url https://download.pytorch.org/whl/cu121 \
    -r requirements-gpu.txt

# ── Layer 3: Install sentence-transformers ───────────────────────────────────
# Dipisah dari base requirements karena butuh torch sudah terinstall dulu.
# Saat pertama load model, akan download ~500MB dari HuggingFace.
# Download ini terjadi saat RUNTIME (pertama kali dipakai), bukan saat build.
COPY requirements-sentence-transformers.txt .
RUN pip install --no-cache-dir -r requirements-sentence-transformers.txt

# ── Layer 4: Install base requirements ───────────────────────────────────────
# Library lain (fastapi, langchain, dll). Layer ini rebuild kalau ada perubahan
# versi library, tapi torch tidak didownload ulang.
COPY requirements-base.txt .
RUN pip install --no-cache-dir -r requirements-base.txt

# ── Layer 5: Copy app code ────────────────────────────────────────────────────
# Paling sering berubah → diletakkan paling akhir.
# Rebuild layer ini cepat (tidak ada download), hanya copy file.
COPY app/ ./app/
COPY scripts/ ./scripts/

# ── Runtime config ────────────────────────────────────────────────────────────
# Set CUDA visible devices — bisa di-override via docker-compose environment
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility

EXPOSE 8003

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8003"]
