# -------------------------
# Base image: Python 3.9 slim
# -------------------------
FROM python:3.9-slim

# -------------------------
# System dependencies
# -------------------------
# - libsndfile1 : required by soundfile
# - ffmpeg      : required by librosa for audio loading
# -------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
    libsndfile1 \
    ffmpeg \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# -------------------------
# Working directory
# -------------------------
WORKDIR /app

# -------------------------
# Copy project files
# -------------------------
COPY . .

# -------------------------
# Install Python dependencies from pyproject.toml
# -------------------------
RUN pip install --upgrade pip \
 && pip install .

# If you want editable install (recommended during dev):
# RUN pip install -e .

# -------------------------
# Default command
# (can be overridden by docker run)
# -------------------------
CMD ["python", "scripts/diarize.py", "data/short_mulan.wav"]
