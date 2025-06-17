FROM python:3.10-slim-bullseye

RUN apt-get update && apt-get upgrade -y && \
    apt-get install -y --no-install-recommends \
    libgomp1 \
    libopenblas-dev \
    ocl-icd-opencl-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

ENV OMP_NUM_THREADS=1
ENV TOKENIZERS_PARALLELISM=true
ENV TF_CPP_MIN_LOG_LEVEL=3
ENV PIP_NO_CACHE_DIR=1

# Actualizar pip primero para evitar problemas de compatibilidad
RUN pip install --no-cache-dir --upgrade pip

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir torch==2.2.1 --index-url https://download.pytorch.org/whl/cpu

COPY . .

EXPOSE 8000

CMD ["gunicorn", "app.main:app", "--worker-class", "uvicorn.workers.UvicornWorker", "--bind", "0.0.0.0:8000", "--workers", "2", "--timeout", "120"]