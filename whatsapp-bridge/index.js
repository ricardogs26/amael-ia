// index.js
const { Client, LocalAuth } = require('whatsapp-web.js');
const express = require('express');
const axios = require('axios');
const qrcode = require('qrcode-terminal');

const app = express();
app.use(express.json());

// --- CONFIGURACIÓN ---
// URL de tu API de amael-ia DENTRO del clúster de Kubernetes
const AMAEL_API_URL = process.env.AMAEL_API_URL || 'http://backend-service:8000/api/chat';
// Token JWT de un usuario autorizado para que el bot pueda hablar con amael-ia
const AMAEL_JWT_TOKEN = process.env.AMAEL_JWT_TOKEN;

if (!AMAEL_JWT_TOKEN) {
    console.error("ERROR: La variable de entorno AMAEL_JWT_TOKEN no está configurada.");
    process.exit(1);
}

// Almacenará el código QR para mostrarlo en la web
let qrCodeData = null;
let clientStatus = 'initializing';

// Configuración del cliente de WhatsApp para que guarde la sesión
// --- CONFIGURACIÓN ROBUSTA PARA PUPPETEER EN KUBERNETES ---
const puppeteerCore = require('puppeteer-core');

// Usar el Chromium instalado en el sistema
const CHROMIUM_PATH = process.env.PUPPETEER_EXECUTABLE_PATH || '/usr/bin/chromium';
console.log(`Usando Chromium en: ${CHROMIUM_PATH}`);

const client = new Client({
    authStrategy: new LocalAuth(),
    puppeteer: {
        headless: true,
        executablePath: CHROMIUM_PATH,
        protocolTimeout: 600000, // 10 minutes
        args: [
            '--no-sandbox',
            '--disable-setuid-sandbox',
            '--disable-dev-shm-usage',
            '--disable-gpu',
            '--disable-extensions',
            '--disable-software-rasterizer',
            '--disable-background-timer-throttling',
            '--disable-backgrounding-occluded-windows',
            '--disable-renderer-backgrounding',
            '--no-zygote',
            '--single-process',
        ]
    },
});

// --- EVENTOS DE WHATSAPP ---

// Cuando se genera el código QR
client.on('qr', (qr) => {
    console.log('QR Code recibido, escanéalo!');
    qrcode.generate(qr, { small: true }); // Muestra el QR en la consola
    qrCodeData = qr; // Guarda el QR para la web
    clientStatus = 'awaiting_qr';
});

// Cuando el cliente se ha conectado correctamente
client.on('ready', () => {
    console.log('¡Cliente de WhatsApp listo!');
    qrCodeData = 'CLIENTE_LISTO'; // Indica que ya no se necesita el QR
    clientStatus = 'ready';
});

// Cuando hay un fallo de autenticación
client.on('auth_failure', (msg) => {
    console.error('ERROR de autenticación de WhatsApp:', msg);
    clientStatus = 'auth_failure';
    qrCodeData = null;
});

// Cuando el cliente se desconecta
client.on('disconnected', (reason) => {
    console.log('WhatsApp desconectado. Razón:', reason);
    clientStatus = 'disconnected';
    qrCodeData = null;
    // Reintentar inicialización después de 5 segundos
    console.log('Reintentando inicialización en 5 segundos...');
    setTimeout(() => {
        console.log('Reintentando client.initialize()...');
        client.initialize().catch(err => {
            console.error('Error al reintentar initialize():', err);
        });
    }, 5000);
});

// Capturar errores del proceso de Puppeteer
client.on('loading_screen', (percent, message) => {
    console.log(`Cargando WhatsApp Web: ${percent}% - ${message}`);
    clientStatus = `loading:${percent}%`;
});

// Cuando se recibe un mensaje
client.on('message', async message => {
    // Ignorar mensajes de grupos o estados para simplificar
    if (message.from.includes('@g.us') || message.from.includes('status')) {
        return;
    }

    // --- CAMBIO CLAVE: Extraer el número de teléfono del remitente ---
    // El formato de 'message.from' es 'phoneNumber@c.us'
    const phoneNumber = message.from.split('@')[0];

    console.log(`Mensaje recibido de ${message.from} (Número: ${phoneNumber}): ${message.body}`);

    try {
        let payload = {
            prompt: message.body || "",
            user_id: phoneNumber
        };

        // --- SOPORTE MULTIMODAL: Manejo de imágenes ---
        if (message.hasMedia) {
            console.log('Descargando contenido multimedia...');
            const media = await message.downloadMedia();
            if (media && media.mimetype.startsWith('image/')) {
                console.log(`Imagen recibida (${media.mimetype}). Convirtiendo a base64...`);
                payload.image = media.data; // media.data ya es base64 en whatsapp-web.js

                // Si el mensaje no trae texto, le ponemos un prompt por defecto
                if (!payload.prompt) {
                    payload.prompt = "Analiza esta imagen.";
                }
            }
        }

        // Llama a tu API de amael-ia
        const response = await axios.post(AMAEL_API_URL, payload, {
            headers: {
                'Authorization': `Bearer ${AMAEL_JWT_TOKEN}`
            }
        });

        const botResponse = response.data.response;
        console.log(`Respuesta de amael-ia: ${botResponse}`);

        // Envía la respuesta de vuelta a WhatsApp
        await message.reply(botResponse);

    } catch (error) {
        console.error('Error al contactar a la API de amael-ia:', error.message);
        await message.reply('Lo siento, tuve un problema al procesar tu mensaje. Inténtalo de nuevo.');
    }
});

// --- ENDPOINTS WEB ---

// Endpoint para servir la página con el QR (con auto-refresh)
app.get('/qr', (req, res) => {
    if (clientStatus === 'ready') {
        res.send('<h1>✅ El bot ya está conectado y listo.</h1><p>Puedes cerrar esta pestaña.</p>');
    } else if (qrCodeData && clientStatus === 'awaiting_qr') {
        res.send(`
            <!DOCTYPE html>
            <html>
            <head><title>WhatsApp QR</title></head>
            <body style="text-align:center;font-family:sans-serif;padding:40px">
                <h1>📱 Escanea este código QR con WhatsApp</h1>
                <img src="https://api.qrserver.com/v1/create-qr-code/?size=300x300&data=${encodeURIComponent(qrCodeData)}" alt="QR Code">
                <p>Abre WhatsApp → Dispositivos vinculados → Vincular un dispositivo</p>
                <meta http-equiv="refresh" content="30">
            </body>
            </html>
        `);
    } else {
        res.send(`
            <!DOCTYPE html>
            <html>
            <head><title>WhatsApp Bridge</title></head>
            <body style="text-align:center;font-family:sans-serif;padding:40px">
                <h1>⏳ Iniciando WhatsApp Bridge...</h1>
                <p>Estado actual: <strong>${clientStatus}</strong></p>
                <p>Esto puede tomar hasta 30 segundos. La página se recargará automáticamente.</p>
                <meta http-equiv="refresh" content="5">
            </body>
            </html>
        `);
    }
});

// Endpoint de estado para diagnóstico
app.get('/status', (req, res) => {
    res.json({
        status: clientStatus,
        hasQR: !!qrCodeData && qrCodeData !== 'CLIENTE_LISTO',
        timestamp: new Date().toISOString()
    });
});

// Inicializa el cliente de WhatsApp (solo una vez)
console.log("Intentando inicializar el cliente de WhatsApp...");
client.initialize().catch(err => {
    console.error('ERROR FATAL al inicializar el cliente de WhatsApp:', err);
    clientStatus = 'error';
});

const PORT = process.env.PORT || 3000;
app.listen(PORT, () => {
    console.log(`Servidor web corriendo en http://localhost:${PORT}`);
    console.log(`QR disponible en: http://localhost:${PORT}/qr`);
    console.log(`Estado en: http://localhost:${PORT}/status`);
});