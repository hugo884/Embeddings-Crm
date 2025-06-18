#!/bin/sh

# Calcular número óptimo de workers
if [ -z "$WORKERS_COUNT" ]; then
    if command -v nproc >/dev/null; then
        WORKERS_COUNT=$(nproc)
    else
        WORKERS_COUNT=2
    fi
fi

# Validar que sea número
if ! [ "$WORKERS_COUNT" -eq "$WORKERS_COUNT" ] 2>/dev/null; then
    echo "ERROR: WORKERS_COUNT debe ser número: $WORKERS_COUNT" >&2
    WORKERS_COUNT=2
fi

# Calcular memoria disponible
MEM_PER_WORKER=200
TOTAL_MEM=$(awk '/MemTotal/ {print int($2/1024)}' /proc/meminfo)
MAX_WORKERS_BY_MEM=$((TOTAL_MEM / MEM_PER_WORKER))

# Ajustar workers por memoria
if [ "$WORKERS_COUNT" -gt "$MAX_WORKERS_BY_MEM" ]; then
    WORKERS_COUNT="$MAX_WORKERS_BY_MEM"
fi

# Mínimo de workers
[ "$WORKERS_COUNT" -lt 2 ] && WORKERS_COUNT=2

echo "🚀 Iniciando servicio con $WORKERS_COUNT workers"
echo "💻 Memoria: ${TOTAL_MEM}MB (~${MEM_PER_WORKER}MB/worker)"

# Ejecutar Gunicorn
exec gunicorn app.main:app \
    --worker-class uvicorn.workers.UvicornWorker \
    --bind 0.0.0.0:8000 \
    --workers "$WORKERS_COUNT" \
    --timeout 300 \
    --preload \
    --access-logfile - \
    --error-logfile - \
    --log-level info