# Fase de construcción
FROM python:3.10-slim-bullseye as builder

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    libopenblas-base \
    && rm -rf /var/lib/apt/lists/*

ENV PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUSERBASE=/app/.local

WORKDIR /app

COPY requirements.txt .
RUN pip install --user --no-warn-script-location --no-cache-dir -r requirements.txt \
    && pip install --user --no-warn-script-location --no-cache-dir torch==2.2.1 --index-url https://download.pytorch.org/whl/cpu

# Fase final
FROM python:3.10-slim-bullseye

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    libopenblas-base \
    && rm -rf /var/lib/apt/lists/*

# Configura variables críticas
ENV OMP_NUM_THREADS=1 \
    TOKENIZERS_PARALLELISM=true \
    TF_CPP_MIN_LOG_LEVEL=3 \
    PATH=/app/.local/bin:$PATH \
    PYTHONUSERBASE=/app/.local \
    PYTHONPATH=/app

# Crea usuario no-root
RUN groupadd -r appuser && useradd -r -g appuser appuser

# Copia las dependencias instaladas (con permisos correctos)
COPY --chown=appuser:appuser --from=builder /app/.local /app/.local

# Copia la aplicación
WORKDIR /app
COPY --chown=appuser:appuser . .

USER appuser
EXPOSE 8000

CMD ["gunicorn", "app.main:app", "--worker-class", "uvicorn.workers.UvicornWorker", "--bind", "0.0.0.0:8000", "--workers", "2", "--timeout", "120", "--preload"]