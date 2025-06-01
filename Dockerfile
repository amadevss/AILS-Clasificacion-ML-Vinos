# Utiliza una imagen base oficial de Python
FROM python:3.10-slim

# Establece el directorio de trabajo
WORKDIR /app

# Instala libgomp1
RUN apt-get update && apt-get install -y libgomp1

# Copia los archivos de requerimientos e instálalos
COPY requirements.txt ./
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install mlflow jupyter ucimlrepo lazypredict

# Copia el resto del código de la aplicación
COPY . .

# Expone los puertos necesarios
EXPOSE 8000 5000 8888

# Comando por defecto (puede ser sobreescrito)
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"] 