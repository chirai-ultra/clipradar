FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg fonts-dejavu-core ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY pyproject.toml README.md LICENSE ./
COPY clipradar ./clipradar
RUN pip install --no-cache-dir ".[asr,qr]"

ENV CLIPRADAR_CACHE=/data/cache PYTHONUNBUFFERED=1
VOLUME ["/data"]
EXPOSE 8765
CMD ["clipradar", "serve", "--host", "0.0.0.0", "--port", "8765", "--out", "/data/output"]
