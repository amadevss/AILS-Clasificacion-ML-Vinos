# Utiliza una imagen base oficial de Python
FROM python:3.10-slim

# Establece variables de entorno
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Instala las dependencias del sistema necesarias
RUN apt-get update && apt-get install -y \
    build-essential \
    libomp-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Crea un usuario no root
RUN useradd -m -u 1000 appuser

# Establece el directorio de trabajo
WORKDIR /app

# Instala libgomp1
RUN apt-get update && apt-get install -y libgomp1

# Copia los archivos de requerimientos e instálalos
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt && \
    rm -rf /root/.cache/pip

# Crea las carpetas necesarias y establece permisos
RUN mkdir -p /app/static /app/templates && \
    chown -R appuser:appuser /app

# Copia primero los archivos estáticos y plantillas
COPY --chown=appuser:appuser static/ /app/static/
COPY --chown=appuser:appuser templates/ /app/templates/

# Copia el resto del código de la aplicación
COPY --chown=appuser:appuser . .

# Cambia al usuario no root
USER appuser

# Expone el puerto de la aplicación
EXPOSE 8000

# Comando por defecto
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"] 