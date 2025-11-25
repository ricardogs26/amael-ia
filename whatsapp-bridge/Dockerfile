# Usa una imagen ligera de Node.js
FROM node:18-alpine

# Establece el directorio de trabajo
WORKDIR /usr/src/app

# --- PASO 1: Instalar Chromium y sus dependencias ---
# Usamos apk, el gestor de paquetes de Alpine Linux
RUN apk add --no-cache \
    chromium \
    nss \
    freetype \
    freetype-dev \
    harfbuzz \
    ca-certificates \
    ttf-freefont

# --- PASO 2: Decirle a Puppeteer dónde encontrar Chromium ---
# Esta es la parte más importante.
ENV PUPPETEER_EXECUTABLE_PATH=/usr/bin/chromium-browser

# Copia los archivos de package
COPY package*.json ./

# Instala las dependencias de Node.js
RUN npm install

# Copia el resto del código de la aplicación
COPY . .

# Expone el puerto en el que corre la app
EXPOSE 3000

# Comando para ejecutar la aplicación
CMD [ "node", "index.js" ]
