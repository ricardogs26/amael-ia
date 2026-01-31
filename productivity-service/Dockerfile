# --- Paso 1: Elegir una imagen base ---
# Usamos una versión oficial y ligera de Python. 'slim' es un buen balance entre tamaño y funcionalidad.
FROM python:3.9-slim

# --- Paso 2: Establecer el directorio de trabajo ---
# Todas las acciones siguientes se ejecutarán dentro de este directorio en el contenedor.
WORKDIR /app

# --- Paso 3: Copiar e instalar las dependencias ---
# Primero copiamos solo el archivo de dependencias. Docker cachea esta capa.
# Si requirements.txt no cambia, no reinstalará todo en cada build, acelerando el proceso.
COPY ./requirements.txt .

# Instalamos las dependencias de Python.
# --no-cache-dir evita guardar la caché de pip, manteniendo la imagen más pequeña.
RUN pip install --no-cache-dir --upgrade -r /app/requirements.txt

# --- Paso 4: Copiar el código de la aplicación ---
# Ahora copiamos el resto del código de la aplicación al directorio de trabajo.
# Esta capa se reconstruirá solo si el código fuente cambia.
COPY ./app /app


ENV PYTHONPATH=/app

# --- Paso 5: Exponer el puerto de la aplicación ---
# El servicio FastAPI se ejecutará en el puerto 8001 dentro del contenedor.
# Esto es más para documentación que para funcionalidad, pero es una buena práctica.
EXPOSE 8001

# --- Paso 6: Definir el comando para ejecutar la aplicación ---
# Este es el comando que se ejecutará cuando se inicie el contenedor.
# uvicorn es el servidor ASGI para FastAPI.
# app.main:app -> apunta al objeto 'app' dentro del archivo 'main.py' en la carpeta 'app'.
# --host 0.0.0.0 -> Es CRUCIAL. Hace que el servicio sea accesible desde fuera del contenedor.
# --port 8001 -> El puerto en el que escuchará uvicorn.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8001"]