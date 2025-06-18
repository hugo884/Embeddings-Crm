# Fase de construcción
FROM python:3.10-slim-bullseye as builder

# 1. Instalación de librerías del sistema
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    libopenblas-base \
    && rm -rf /var/lib/apt/lists/*

# 2. Variables de entorno
ENV PIP_NO_CACHE_DIR=off \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    VIRTUAL_ENV=/app/venv

WORKDIR /app

# 3. Crear entorno virtual
RUN python -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# 4. Instalación de dependencias
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir torch==2.2.1 --index-url https://download.pytorch.org/whl/cpu

# ------------------------------------------------------------
# Fase final
# ------------------------------------------------------------
FROM python:3.10-slim-bullseye

# 5. Librerías de sistema
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    libopenblas-base \
    && rm -rf /var/lib/apt/lists/*

# 6. Variables de entorno
ENV OMP_NUM_THREADS=1 \
    TOKENIZERS_PARALLELISM=false \
    TF_CPP_MIN_LOG_LEVEL=3 \
    PYTHONPATH=/app \
    PATH="/app/venv/bin:$PATH" \
    WORKERS_COUNT=2

# 7. Crear usuario no-root
RUN groupadd -r appuser && \
    useradd -r -g appuser -d /app -s /sbin/nologin appuser

# 8. Copiar entorno virtual
COPY --from=builder --chown=appuser:appuser /app/venv /app/venv

# 9. Copiar aplicación y script de entrada
WORKDIR /app
COPY --chown=appuser:appuser . .

# 10. Copiar y configurar entrypoint
COPY --chown=appuser:appuser entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

# 11. Aplicar permisos
RUN chmod 755 /app && \
    find . -path ./venv -prune -o -type d -exec chmod 755 {} + && \
    find . -path ./venv -prune -o -type f -exec chmod 644 {} +

# 12. Configurar usuario y puerto
USER appuser
EXPOSE 8000

# 13. Ejecutar script de entrada
CMD ["/app/entrypoint.sh"]