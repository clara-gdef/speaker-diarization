FROM python:3.9-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    libsndfile1 \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY . .

# Install deps declared in pyproject.toml
RUN pip install --upgrade pip && pip install .

CMD ["python", "scripts/diarize.py", "data/short_mulan.wav"]
