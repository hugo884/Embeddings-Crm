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

# 5. Librerías de sistema y utilidades (igual)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    libopenblas-base \
    dos2unix \
    && rm -rf /var/lib/apt/lists/*

# 6. Variables de entorno (igual)
ENV OMP_NUM_THREADS=1 \
    TOKENIZERS_PARALLELISM=false \
    TF_CPP_MIN_LOG_LEVEL=3 \
    PYTHONPATH=/app \
    PATH="/app/venv/bin:$PATH"

# 7. Crear usuario no-root (igual)
RUN groupadd -r appuser && \
    useradd -r -g appuser -d /app -s /bin/bash appuser

# 8. Copiar entorno virtual (igual)
COPY --from=builder --chown=appuser:appuser /app/venv /app/venv

# 9. Copiar aplicación (igual)
WORKDIR /app
COPY --chown=appuser:appuser . .

# 10. Convertir formato y ajustar permisos CORREGIDO:
RUN dos2unix entrypoint.sh && \
    # Aplicar permisos globales:
    chmod 755 /app && \
    find . -path ./venv -prune -o -type d -exec chmod 755 {} + && \
    find . -path ./venv -prune -o -type f -exec chmod 644 {} + && \
    # Dar permiso de ejecución SOLO al entrypoint:
    chmod +x entrypoint.sh


# 12. Verificación básica (solo ls)
RUN ls -l entrypoint.sh

# 13. Configurar usuario y puerto
USER appuser
EXPOSE 8000

# 14. Ejecutar script de entrada
ENTRYPOINT ["/app/entrypoint.sh"]