// index.js
const { Client, LocalAuth, MessageMedia } = require('whatsapp-web.js');
const express = require('express');
const axios = require('axios');
const qrcode = require('qrcode-terminal');
const puppeteer = require('puppeteer-core');

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
        protocolTimeout: 0,
        args: [
            '--no-sandbox',
            '--disable-setuid-sandbox',
            '--disable-dev-shm-usage',
            '--disable-gpu',
            '--disable-extensions',
            '--disable-software-rasterizer',
            '--disable-setuid-sandbox',
            '--no-zygote',
            '--disable-background-networking',
            '--disable-default-apps',
            '--disable-sync',
            '--disable-translate',
            '--hide-scrollbars',
            '--metrics-recording-only',
            '--mute-audio',
            '--safebrowsing-disable-auto-update',
            '--ignore-certificate-errors',
            '--ignore-ssl-errors',
            '--ignore-certificate-errors-spki-list',
            '--font-render-hinting=none',
            '--disable-web-security',
            '--disable-features=IsolateOrigins,site-per-process',
        ]
    },
    authTimeoutMs: 300000, // 5 minutos de timeout para el QR
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
    // El formato de 'message.from' puede ser 'phoneNumber@c.us' o 'phoneNumber:1@c.us' (multi-device)
    const phoneNumber = message.from.split('@')[0].split(':')[0];

    // --- VALIDACIÓN DE WHITELIST INICIAL (Bridge Level) ---
    const allowedNumbersCsv = process.env.ALLOWED_NUMBERS_CSV || "";
    const allowedNumbers = allowedNumbersCsv.split(',').map(n => n.trim()).filter(n => n !== "");
    
    console.log(`[DEBUG] Validando número: "${phoneNumber}" contra whitelist: [${allowedNumbers.join(', ')}]`);

    if (!allowedNumbers.includes(phoneNumber)) {
        console.log(`BLOQUEO BRIDGE: Mensaje ignorado de número no autorizado: ${phoneNumber}`);
        // Solo responder si no es un número de sistema o algo similar
        await message.reply('Lo siento, este número no está autorizado para usar mis servicios.');
        return;
    }

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

        // --- MANEJO DE MEDIA EN LA RESPUESTA ---
        // Si el agente envía un formato [MEDIA:base64_data], lo extraemos y enviamos como imagen
        const mediaRegex = /\[MEDIA:(.+?)\]/;
        const match = botResponse.match(mediaRegex);
        
        if (match) {
            const base64Data = match[1];
            const cleanText = botResponse.replace(mediaRegex, '').trim();
            
            const media = new MessageMedia('image/png', base64Data, 'screenshot.png');
            await client.sendMessage(message.from, media, { caption: cleanText || "Aquí tienes el consumo actual." });
        } else {
            // Envía la respuesta de texto normal
            await message.reply(botResponse);
        }

    } catch (error) {
        console.error('Error al contactar a la API de amael-ia:', error.message);
        await message.reply('Lo siento, tuve un problema al procesar tu mensaje. Inténtalo de nuevo.');
    }
});

// --- NUEVOS ENDPOINTS PARA EL AGENTE ---

// Endpoint para enviar media desde otros servicios (ej. k8s-agent -> backend -> bridge)
app.post('/send-media', async (req, res) => {
    const { phoneNumber, base64, caption, mimetype } = req.body;
    if (!phoneNumber || !base64) {
        return res.status(400).json({ error: 'Faltan parámetros: phoneNumber o base64' });
    }

    try {
        const chatId = phoneNumber.includes('@c.us') ? phoneNumber : `${phoneNumber}@c.us`;
        const media = new MessageMedia(mimetype || 'image/png', base64, 'image.png');
        await client.sendMessage(chatId, media, { caption: caption || '' });
        res.json({ success: true });
    } catch (error) {
        console.error('Error enviando media:', error);
        res.status(500).json({ error: error.message });
    }
});

// Endpoint para tomar capturas de pantalla usando el navegador del Bridge
app.post('/screenshot', async (req, res) => {
    const { url, waitSelector, username, password } = req.body;
    if (!url) return res.status(400).json({ error: 'Falta la URL' });

    console.log(`[SCREENSHOT] Petición recibida para: ${url}`);

    let browser;
    try {
        browser = await puppeteer.launch({
            executablePath: CHROMIUM_PATH,
            args: ['--no-sandbox', '--disable-setuid-sandbox']
        });
        const page = await browser.newPage();
        await page.setViewport({ width: 1280, height: 800 });
        
        // Autenticación proactiva para Grafana si se proporcionan credenciales
        if (url.includes('grafana')) {
            const user = username || 'admin';
            const pass = password || 'admin';
            console.log(`[SCREENSHOT] Aplicando autenticación proactiva para Grafana: ${user}`);
            
            // Inyectar cabecera de autenticación básica proactivamente
            const authHeader = `Basic ${Buffer.from(`${user}:${pass}`).toString('base64')}`;
            await page.setExtraHTTPHeaders({
                'Authorization': authHeader
            });
            
            // También autenticar de la forma tradicional por si acaso (fallback)
            await page.authenticate({ username: user, password: pass });
        }

        console.log(`[SCREENSHOT] Navegando a: ${url}`);
        await page.goto(url, { waitUntil: 'networkidle2', timeout: 60000 });
        
        // Verificar si caímos en la página de login a pesar de la auth
        const isLoginPage = await page.evaluate(() => {
            return document.title.toLowerCase().includes('login') || 
                   !!document.querySelector('input[name="user"]') ||
                   !!document.querySelector('form[name="login"]');
        });

        if (isLoginPage && url.includes('grafana')) {
            const user = username || 'admin';
            const pass = password || 'admin';
            console.log(`[SCREENSHOT] Detectada página de login. Intentando login automático para: ${user}`);
            
            try {
                // Intentar encontrar los selectores de login de Grafana
                await page.waitForSelector('input[name="user"]', { timeout: 10000 });
                await page.type('input[name="user"]', user);
                await page.type('input[name="password"]', pass);
                
                // Hacer clic en el botón de login
                console.log('[SCREENSHOT] Buscando botón de login...');
                await page.waitForSelector('button[type="submit"]', { timeout: 5000 });
                await page.click('button[type="submit"]');
                
                // En Grafana (SPA), la navegación puede no disparar waitForNavigation de forma estándar
                // Esperamos a que el formulario de login desaparezca o aparezca el dashboard
                console.log('[SCREENSHOT] Login enviado, esperando carga del dashboard...');
                await new Promise(r => setTimeout(r, 5000)); // Espera inicial
                
                console.log('[SCREENSHOT] Login parece haber sido enviado.');
            } catch (loginError) {
                console.warn('[SCREENSHOT] Error o timeout durante el login automático:', loginError.message);
            }
        }

        if (waitSelector) {
            console.log(`[SCREENSHOT] Esperando selector: ${waitSelector}`);
            try {
                await page.waitForSelector(waitSelector, { timeout: 15000 });
            } catch (e) {
                console.warn(`[SCREENSHOT] Aviso: No se encontró el selector ${waitSelector}, continuando...`);
            }
        } else {
            // Delay de seguridad más largo para asegurar que los gráficos de Grafana carguen (son dinámicos)
            console.log('[SCREENSHOT] Esperando 7s para carga de gráficos dinámicos...');
            await new Promise(r => setTimeout(r, 7000));
        }

        // Si es Grafana, ocultar elementos innecesarios para la captura
        if (url.includes('grafana')) {
            await page.addStyleTag({ content: '.sidemenu-canvas, .navbar-page-btn, .search-container { display: none !important; }' });
        }

        console.log('[SCREENSHOT] Capturando pantalla...');
        const screenshot = await page.screenshot({ encoding: 'base64' });
        console.log('[SCREENSHOT] Captura completada con éxito.');
        res.json({ base64: screenshot });
    } catch (error) {
        console.error('[SCREENSHOT] Error:', error);
        res.status(500).json({ error: error.message });
    } finally {
        if (browser) {
            await browser.close();
            console.log('[SCREENSHOT] Navegador cerrado.');
        }
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

// --- START SERVER ---
const PORT = process.env.PORT || 3000;
app.listen(PORT, () => {
    console.log(`Servidor web corriendo en http://localhost:${PORT}`);
    console.log(`QR disponible en: http://localhost:${PORT}/qr`);
    console.log(`Estado en: http://localhost:${PORT}/status`);
});

// Inicializa el cliente de WhatsApp con reintentos
const initializeClient = () => {
    console.log("Intentando inicializar el cliente de WhatsApp...");
    clientStatus = 'initializing';
    client.initialize().catch(err => {
        console.error('ERROR al inicializar el cliente de WhatsApp:', err);
        clientStatus = 'error';
        qrCodeData = null;
        console.log('Reintentando inicialización completa en 30 segundos...');
        setTimeout(initializeClient, 30000);
    });
};

initializeClient();