# ==============================================================================
# Stage 1: Model Downloader
# Purpose: Download the weights so we don't do it at runtime.
# ==============================================================================
FROM python:3.12-slim as downloader

WORKDIR /downloader
RUN pip install --no-cache-dir faster-whisper

# Download the "small" model (Multilingual, supports Japanese)
# Output path: /models/small
RUN python3 -c "from faster_whisper import download_model; download_model('small', output_dir='/models/small')"

# ==============================================================================
# Stage 2: Builder
# Purpose: Create a clean virtual environment and install dependencies.
# We use Ubuntu 24.04 to match the Runtime OS (ensuring Python binary compatibility).
# ==============================================================================
FROM ubuntu:24.04 as builder

ENV DEBIAN_FRONTEND=noninteractive

# Install Python 3.12 and build tools (needed for compiling any C-extensions)
RUN apt-get update && apt-get install -y \
    python3 \
    python3-venv \
    python3-pip \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

# Create a virtual environment at /opt/venv
RUN python3 -m venv /opt/venv

# Activate the venv for the following commands
ENV PATH="/opt/venv/bin:$PATH"

# Install dependencies into the venv
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# ==============================================================================
# Stage 3: Runtime (Target: NVIDIA T4 / CUDA 12.4)
# Purpose: The final lightweight image.
# ==============================================================================
# We use the cuDNN runtime image which is required for CTranslate2 (Faster-Whisper backend)
FROM nvidia/cuda:12.9.1-cudnn-runtime-ubuntu24.04

WORKDIR /app

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# 1. Install Runtime Dependencies
# - python3: to run the script
# - libsndfile1: required by audio processing libraries
RUN apt-get update && apt-get install -y \
    python3 \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# 2. Copy the Virtual Environment from the Builder stage
COPY --from=builder /opt/venv /opt/venv

# 3. Copy the Pre-downloaded Model from the Downloader stage
COPY --from=downloader /models/small /app/models/small

# 4. Copy Application Code
COPY batch_transcribe.py .
# Optional: Copy default config files if you have them locally
# COPY summarize_prompt.md context.md ./

# 5. Set Environment Variables
# Add the venv to the PATH so 'python' automatically uses the venv version
ENV PATH="/opt/venv/bin:$PATH"
# Point to the internal model path
ENV MODEL_SIZE=/app/models/small
# Default Language
ENV LANGUAGE=ja

# NVIDIA Capabilities
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility

# 6. Run
CMD ["python", "batch_transcribe.py"]