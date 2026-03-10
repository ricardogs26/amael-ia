# Debian slim (glibc) para que el Chromium bundled de puppeteer-core funcione
FROM node:18-slim

# Instalar dependencias del sistema necesarias para Chromium bundled de puppeteer-core
# No instalar Chromium del sistema - puppeteer-core descargará su propia versión compatible
RUN apt-get update && apt-get install -y \
    wget \
    gnupg \
    ca-certificates \
    chromium \
    libnss3 \
    libatk-bridge2.0-0 \
    libdrm2 \
    libxkbcommon0 \
    libxcomposite1 \
    libxdamage1 \
    libxfixes3 \
    libxrandr2 \
    libgbm1 \
    libasound2 \
    --no-install-recommends \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /usr/src/app

# Copia los archivos de package e instala dependencias
# puppeteer-core descargará su Chromium compatible durante npm install
COPY package*.json ./
RUN npm install

# Copia el resto del código de la aplicación
COPY . .

# Expone el puerto en el que corre la app
EXPOSE 3000

# Eliminar archivos de bloqueo de Chromium antes de arrancar
CMD [ "sh", "-c", "rm -rf /tmp/.org.chromium.Chromium.* /tmp/.com.google.Chrome.* && find .wwebjs_auth -name 'SingletonLock' -exec rm -f {} \\; || true && node index.js" ]