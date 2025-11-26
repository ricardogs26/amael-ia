# command-executor/Dockerfile
FROM python:3.9-slim

WORKDIR /usr/src/app

# --- PASO 1: Instalar curl y kubectl en una sola capa ---
# Primero actualizamos los paquetes, instalamos curl, luego usamos curl para descargar kubectl.
RUN apt-get update && \
    apt-get install -y curl && \
    curl -LO "https://dl.k8s.io/release/$(curl -L -s https://dl.k8s.io/release/stable.txt)/bin/linux/amd64/kubectl" && \
    install -o root -g root -m 0755 kubectl /usr/local/bin/kubectl && \
    # Limpiamos la caché de apt para mantener la imagen pequeña
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Copia e instala las dependencias de Python
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copia el resto del código de la aplicación
COPY . .

# Expone el puerto en el que corre la app
EXPOSE 8001

# Comando para ejecutar la aplicación
CMD ["uvicorn", "command-executor:app", "--host", "0.0.0.0", "--port", "8001"]
