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

// Configuración del cliente de WhatsApp para que guarde la sesión
// --- CONFIGURACIÓN ROBUSTA PARA PUPPETEER EN KUBERNETES ---
const client = new Client({
    authStrategy: new LocalAuth(),
    puppeteer: { 
        headless: true,
        // --- RUTA CORRECTA PARA LA IMAGEN DEBIAN/SLIM ---
        executablePath: '/usr/bin/chromium',
        // --- ARGUMENTOS CLAVE PARA CONTENEDORES ---
        args: [
            '--no-sandbox',
            '--disable-setuid-sandbox',
            '--disable-dev-shm-usage',
            '--disable-gpu'
        ]
    }
});

// --- EVENTOS DE WHATSAPP ---

// Cuando se genera el código QR
client.on('qr', (qr) => {
    console.log('QR Code recibido, escanéalo!');
    qrcode.generate(qr, { small: true }); // Muestra el QR en la consola
    qrCodeData = qr; // Guarda el QR para la web
});

// Cuando el cliente se ha conectado correctamente
client.on('ready', () => {
    console.log('¡Cliente de WhatsApp listo!');
    qrCodeData = 'CLIENTE_LISTO'; // Indica que ya no se necesita el QR
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
        // Llama a tu API de amael-ia
        const response = await axios.post(AMAEL_API_URL, {
            prompt: message.body,
            // --- CAMBIO CLAVE: Enviar el phoneNumber como user_id ---
            user_id: phoneNumber 
        }, {
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

// Endpoint para servir la página con el QR
app.get('/qr', (req, res) => {
    if (qrCodeData === 'CLIENTE_LISTO') {
        res.send('<h1>El bot ya está conectado y listo.</h1><p>Puedes cerrar esta pestaña.</p>');
    } else if (qrCodeData) {
        res.send(`
            <h1>Escanea este código QR con WhatsApp</h1>
            <img src="https://api.qrserver.com/v1/create-qr-code/?size=200x200&data=${encodeURIComponent(qrCodeData)}" alt="QR Code">
        `);
    } else {
        res.send('<h1>Esperando el código QR... Por favor, espera.</h1><meta http-equiv="refresh" content="5">');
    }
});

// Inicializa el cliente de WhatsApp (solo una vez)
console.log("Intentando inicializar el cliente de WhatsApp...");
client.initialize();

const PORT = process.env.PORT || 3000;
app.listen(PORT, () => {
    console.log(`Servidor web corriendo en http://localhost:${PORT}`);
});