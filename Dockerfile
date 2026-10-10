# ---------- builder: install deps + bake EasyOCR weights into the image ----------
FROM python:3.11-slim AS builder
ENV PIP_NO_CACHE_DIR=1 PYTHONDONTWRITEBYTECODE=1
WORKDIR /app

# torch needs OpenMP at runtime
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --index-url https://download.pytorch.org/whl/cpu \
        torch==2.4.1 torchvision==0.19.1 && \
    pip install -r requirements.txt

# Bake the ~100MB EasyOCR weights at BUILD time so the container never
# re-downloads them on cold start (Heroku's disk is ephemeral).
ENV EASYOCR_MODEL_DIR=/app/models
RUN python -c "import easyocr; easyocr.Reader(['en'], gpu=False, model_storage_directory='/app/models')"

# ---------- runtime: slim image, no build tools ----------
FROM python:3.11-slim AS runtime
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 \
    EASYOCR_MODEL_DIR=/app/models PORT=8080
WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin
COPY --from=builder /app/models /app/models

COPY app.py ./
COPY templates ./templates

EXPOSE 8080
CMD ["sh", "-c", "gunicorn app:app --bind 0.0.0.0:${PORT:-8080} --timeout 120 --workers 1 --threads 4"]
