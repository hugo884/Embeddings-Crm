FROM python:3.10-slim

# Crear usuario no-root para mayor seguridad
RUN useradd --create-home appuser
WORKDIR /home/appuser/app
RUN chown appuser:appuser /home/appuser/app

# Cambiar a usuario no-root
USER appuser

# Crear entorno virtual
RUN python -m venv /home/appuser/venv
ENV PATH="/home/appuser/venv/bin:$PATH"

# Actualizar pip y configurar entorno
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Copiar e instalar dependencias
COPY --chown=appuser:appuser requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copiar aplicación
COPY --chown=appuser:appuser . .

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]