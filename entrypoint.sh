#!/bin/sh

# Calcular número óptimo de workers si no está especificado
if [ -z "$WORKERS_COUNT" ]; then
    # Intentar detectar núcleos disponibles
    if command -v nproc >/dev/null; then
        WORKERS_COUNT=$(nproc)
    else
        # Valor por defecto conservador
        WORKERS_COUNT=2
    fi
fi

# Asegurar que sea un número válido
if ! [ "$WORKERS_COUNT" -eq "$WORKERS_COUNT" ] 2>/dev/null; then
    echo "ERROR: WORKERS_COUNT no es un número válido: $WORKERS_COUNT" >&2
    WORKERS_COUNT=2
fi

# Configurar variables para Gunicorn basadas en memoria disponible
MEM_PER_WORKER=150  # MB estimados por worker
TOTAL_MEM=$(($(awk '/MemTotal/ {print $2}' /proc/meminfo) / 1024))  # Memoria total en MB
MAX_WORKERS_BY_MEM=$((TOTAL_MEM / MEM_PER_WORKER))

# Ajustar workers si excede límite de memoria
if [ "$WORKERS_COUNT" -gt "$MAX_WORKERS_BY_MEM" ]; then
    WORKERS_COUNT=$MAX_WORKERS_BY_MEM
fi

# Mínimo de 2 workers
if [ "$WORKERS_COUNT" -lt 2 ]; then
    WORKERS_COUNT=2
fi

echo "🚀 Iniciando servicio con $WORKERS_COUNT workers"
echo "💻 Memoria disponible: ${TOTAL_MEM}MB (~${MEM_PER_WORKER}MB por worker)"

# Ejecutar Gunicorn con configuración óptima
exec gunicorn app.main:app \
    --worker-class uvicorn.workers.UvicornWorker \
    --bind 0.0.0.0:8000 \
    --workers $WORKERS_COUNT \
    --timeout 120 \
    --preload \
    --access-logfile - \
    --error-logfile - \
    --log-level info