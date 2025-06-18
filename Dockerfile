# Fase de construcción - SOLO se ejecuta cuando cambian dependencias
FROM python:3.10-slim-bullseye as builder

# 1. Instalación de librerías del sistema (capa en caché hasta que cambien los paquetes)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    libopenblas-base \
    && rm -rf /var/lib/apt/lists/*

# 2. Variables de entorno para optimización
ENV PIP_NO_CACHE_DIR=off \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    VIRTUAL_ENV=/app/venv

WORKDIR /app

# 3. Crear entorno virtual (capa estable)
RUN python -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# 4. Instalación de dependencias (capa en caché mientras requirements.txt no cambie)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir torch==2.2.1 --index-url https://download.pytorch.org/whl/cpu

# ------------------------------------------------------------
# Fase final - Se reconstruye SOLO cuando cambia el código
# ------------------------------------------------------------
FROM python:3.10-slim-bullseye

# 5. Librerías de sistema necesarias para ejecución
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    libopenblas-base \
    && rm -rf /var/lib/apt/lists/*

# 6. Variables críticas de entorno
ENV OMP_NUM_THREADS=1 \
    TOKENIZERS_PARALLELISM=false \
    TF_CPP_MIN_LOG_LEVEL=3 \
    PYTHONPATH=/app \
    PATH="/app/venv/bin:$PATH"

# 7. Crear usuario no-root con home directory
RUN groupadd -r appuser && \
    useradd -r -g appuser -d /app -s /sbin/nologin appuser

# 8. Copiar entorno virtual (capa estable)
COPY --from=builder /app/venv /app/venv

# 9. Copiar aplicación como último paso (capa que cambia frecuentemente)
WORKDIR /app
COPY --chown=appuser:appuser . .

# 10. Permisos adicionales para seguridad
RUN chmod 755 /app && \
    find /app -type d -exec chmod 755 {} + && \
    find /app -type f -exec chmod 644 {} +

USER appuser
EXPOSE 8000

CMD ["gunicorn", "app.main:app", "--worker-class", "uvicorn.workers.UvicornWorker", \
     "--bind", "0.0.0.0:8000", "--workers", "2", "--timeout", "120", "--preload"]