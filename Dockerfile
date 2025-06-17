# Usar imagen base más ligera
FROM python:3.10-slim-bullseye as builder

# Instalar solo dependencias esenciales del sistema
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    libopenblas-base \
    && rm -rf /var/lib/apt/lists/*

# Configurar entorno Python
ENV PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

# Instalar dependencias en un directorio temporal
COPY requirements.txt .
RUN pip install --user --no-warn-script-location --no-cache-dir -r requirements.txt \
    && pip install --user --no-warn-script-location --no-cache-dir torch==2.2.1 --index-url https://download.pytorch.org/whl/cpu

# Etapa final de ejecución
FROM python:3.10-slim-bullseye

# Copiar dependencias instaladas
COPY --from=builder /root/.local /root/.local
COPY --from=builder /app /app

# Instalar solo runtime esencial
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    libopenblas-base \
    && rm -rf /var/lib/apt/lists/*

# Variables de entorno para optimización
ENV OMP_NUM_THREADS=1 \
    TOKENIZERS_PARALLELISM=true \
    TF_CPP_MIN_LOG_LEVEL=3 \
    PATH=/root/.local/bin:$PATH

WORKDIR /app

# Crear usuario no privilegiado
RUN groupadd -r appuser && useradd -r -g appuser appuser
USER appuser

# Copiar aplicación (con permisos correctos)
COPY --chown=appuser:appuser . .

EXPOSE 8000

# Comando optimizado
CMD ["gunicorn", "app.main:app", "--worker-class", "uvicorn.workers.UvicornWorker","--bind", "0.0.0.0:8000", "--workers", "2", "--timeout", "120", "--preload"]