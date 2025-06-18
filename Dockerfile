# Fase de construcción
FROM python:3.10-slim-bullseye as builder

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    libopenblas-base \
    && rm -rf /var/lib/apt/lists/*

ENV PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    VIRTUAL_ENV=/app/venv

WORKDIR /app

# Crear entorno virtual
RUN python -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir torch==2.2.1 --index-url https://download.pytorch.org/whl/cpu

# Fase final
FROM python:3.10-slim-bullseye

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    libopenblas-base \
    && rm -rf /var/lib/apt/lists/*

# Configura entorno virtual
ENV OMP_NUM_THREADS=1 \
    TOKENIZERS_PARALLELISM=true \
    TF_CPP_MIN_LOG_LEVEL=3 \
    VIRTUAL_ENV=/app/venv \
    PATH="/app/venv/bin:$PATH" \
    PYTHONPATH=/app

# Crea usuario no-root
RUN groupadd -r appuser && useradd -r -g appuser appuser

# Copia entorno virtual
COPY --chown=appuser:appuser --from=builder /app/venv /app/venv

# Copia la aplicación
WORKDIR /app
COPY --chown=appuser:appuser . .

USER appuser
EXPOSE 8000

CMD ["gunicorn", "app.main:app", "--worker-class", "uvicorn.workers.UvicornWorker", "--bind", "0.0.0.0:8000", "--workers", "2", "--timeout", "120", "--preload"]