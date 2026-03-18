// index.js — v1.5.2
// P5-E: Bidirectional /sre command routing to k8s-agent
const { Client, LocalAuth, MessageMedia } = require('whatsapp-web.js');
const express = require('express');
const axios = require('axios');
const qrcode = require('qrcode-terminal');
const puppeteer = require('puppeteer-core');

const app = express();
app.use(express.json({ limit: '50mb' }));

// --- CONFIGURACIÓN ---
const AMAEL_BASE_URL        = process.env.AMAEL_BASE_URL        || 'http://backend-service:8000';
const AMAEL_API_URL         = process.env.AMAEL_API_URL         || `${AMAEL_BASE_URL}/api/chat`;
const AMAEL_JWT_TOKEN       = process.env.AMAEL_JWT_TOKEN;
const AMAEL_INTERNAL_SECRET = process.env.AMAEL_INTERNAL_SECRET || '';
const K8S_AGENT_URL         = process.env.K8S_AGENT_URL         || 'http://k8s-agent-service:8002';

if (!AMAEL_JWT_TOKEN) {
    console.error("ERROR: La variable de entorno AMAEL_JWT_TOKEN no está configurada.");
    process.exit(1);
}

const authHeaders     = () => ({ Authorization: `Bearer ${AMAEL_JWT_TOKEN}` });
const internalHeaders = () => ({ Authorization: `Bearer ${AMAEL_INTERNAL_SECRET}` });

// --- ESTADO GLOBAL ---
let qrCodeData   = null;
let clientStatus = 'initializing';

// Mapa en memoria: phoneNumber → { convId, title }
// Persiste mientras el pod esté vivo; si reinicia, crea nueva conv (aceptable en v1)
const convMap = {};

// ── Comandos rápidos: texto que el usuario escribe → prompt expandido al backend ─
const QUICK_COMMANDS = {
    '/estado':   'Dame un reporte completo del estado del cluster kubernetes: pods, namespaces y cualquier alerta activa.',
    '/plan':     'Genera mi plan del día de hoy basado en mi calendario y el estado actual del cluster.',
    '/gastos':   'Muéstrame un resumen de mis gastos recientes guardados en el sistema.',
    '/objetivos':'Muéstrame mis objetivos activos y el progreso de cada uno.',
    '/ayuda':    null, // manejado localmente
};

const AYUDA_MSG = `*Comandos disponibles:*

/estado — Estado del cluster Kubernetes
/plan — Plan del día de hoy
/gastos — Resumen de gastos recientes
/objetivos — Objetivos activos
/sre <cmd> — Agente SRE autónomo (status, incidents, slo, maintenance)
/ayuda — Esta lista de comandos

También puedes escribir cualquier pregunta y te respondo normalmente. 🤖`;

// --- PUPPETEER ---
const CHROMIUM_PATH = process.env.PUPPETEER_EXECUTABLE_PATH || '/usr/bin/chromium';
console.log(`Usando Chromium en: ${CHROMIUM_PATH}`);

const client = new Client({
    authStrategy: new LocalAuth(),
    puppeteer: {
        headless: true,
        executablePath: CHROMIUM_PATH,
        protocolTimeout: 0,
        args: [
            '--no-sandbox', '--disable-setuid-sandbox', '--disable-dev-shm-usage',
            '--disable-gpu', '--disable-extensions', '--disable-software-rasterizer',
            '--no-zygote', '--disable-background-networking', '--disable-default-apps',
            '--disable-sync', '--disable-translate', '--hide-scrollbars',
            '--metrics-recording-only', '--mute-audio', '--safebrowsing-disable-auto-update',
            '--ignore-certificate-errors', '--ignore-ssl-errors',
            '--disable-web-security', '--disable-features=IsolateOrigins,site-per-process',
        ]
    },
    authTimeoutMs: 300000,
});

// ── Helpers ─────────────────────────────────────────────────────────────────────

/** Obtiene o crea una conversación en el backend para un número de WhatsApp. */
async function getOrCreateConv(phoneNumber) {
    if (convMap[phoneNumber]) return convMap[phoneNumber].convId;

    try {
        // Buscar conversaciones existentes del usuario
        const res = await axios.get(`${AMAEL_BASE_URL}/api/conversations`, {
            headers: authHeaders(),
            params: { user_id: phoneNumber },
        });
        const convs = res.data.conversations || [];
        if (convs.length > 0) {
            convMap[phoneNumber] = { convId: convs[0].id, title: convs[0].title };
            console.log(`[CONV] Usando conversación existente ${convs[0].id} para ${phoneNumber}`);
            return convs[0].id;
        }
    } catch (e) {
        console.warn(`[CONV] Error buscando conversaciones: ${e.message}`);
    }

    // Crear nueva conversación
    try {
        const res = await axios.post(`${AMAEL_BASE_URL}/api/conversations`, {
            title: 'WhatsApp',
            user_id: phoneNumber,
        }, { headers: authHeaders() });
        const convId = res.data.id;
        convMap[phoneNumber] = { convId, title: 'WhatsApp' };
        console.log(`[CONV] Nueva conversación ${convId} para ${phoneNumber}`);
        return convId;
    } catch (e) {
        console.error(`[CONV] Error creando conversación: ${e.message}`);
        return null;
    }
}

/** Carga los últimos N mensajes de una conversación como historial. */
async function loadHistory(convId, limit = 10) {
    if (!convId) return [];
    try {
        const res = await axios.get(`${AMAEL_BASE_URL}/api/conversations/${convId}/messages`, {
            headers: authHeaders(),
        });
        const msgs = (res.data.messages || []).slice(-limit);
        return msgs.map(m => ({ role: m.role, content: m.content }));
    } catch (e) {
        console.warn(`[HIST] Error cargando historial: ${e.message}`);
        return [];
    }
}

/** Envía un mensaje de texto a un número de WhatsApp. */
async function sendText(phoneNumber, text) {
    const chatId = phoneNumber.includes('@c.us') ? phoneNumber : `${phoneNumber}@c.us`;
    await client.sendMessage(chatId, text);
}

/** Verifica si un número tiene acceso y devuelve su user_id canónico. */
async function checkUserAccess(phoneNumber) {
    if (!AMAEL_INTERNAL_SECRET) {
        // Fallback: whitelist estática
        const allowed = (process.env.ALLOWED_NUMBERS_CSV || '').split(',').map(n => n.trim()).includes(phoneNumber);
        return { allowed, canonical_user_id: phoneNumber, allow_requests: false };
    }
    try {
        const res = await axios.get(`${AMAEL_BASE_URL}/api/identity/check`, {
            params: { identifier: phoneNumber },
            headers: internalHeaders(),
            timeout: 5000,
        });
        return res.data;
    } catch (e) {
        console.warn(`[ACCESS] Error verificando acceso para ${phoneNumber}: ${e.message}`);
        // Fallback seguro: negar acceso si el backend no responde
        return { allowed: false, canonical_user_id: phoneNumber, allow_requests: false };
    }
}

/** Envía solicitud de acceso al backend. */
async function requestAccess(phoneNumber, name) {
    try {
        await axios.post(`${AMAEL_BASE_URL}/api/auth/access-request`, { phone: phoneNumber, name });
        console.log(`[ACCESS] Solicitud de acceso enviada para ${phoneNumber}`);
    } catch (e) {
        console.warn(`[ACCESS] Error enviando solicitud: ${e.message}`);
    }
}

// --- EVENTOS DE WHATSAPP ────────────────────────────────────────────────────────

client.on('qr', (qr) => {
    console.log('QR Code recibido, escanéalo!');
    qrcode.generate(qr, { small: true });
    qrCodeData   = qr;
    clientStatus = 'awaiting_qr';
});

client.on('ready', () => {
    console.log('¡Cliente de WhatsApp listo!');
    qrCodeData   = 'CLIENTE_LISTO';
    clientStatus = 'ready';
});

client.on('auth_failure', (msg) => {
    console.error('ERROR de autenticación de WhatsApp:', msg);
    clientStatus = 'auth_failure';
    qrCodeData   = null;
});

client.on('disconnected', (reason) => {
    console.log('WhatsApp desconectado. Razón:', reason);
    clientStatus = 'disconnected';
    qrCodeData   = null;
    setTimeout(() => client.initialize().catch(err => console.error(err)), 5000);
});

client.on('loading_screen', (percent, message) => {
    clientStatus = `loading:${percent}%`;
});

// --- MANEJO DE MENSAJES ENTRANTES ───────────────────────────────────────────────

client.on('message', async message => {
    if (message.from.includes('@g.us') || message.from.includes('status')) return;

    const phoneNumber = message.from.split('@')[0].split(':')[0];
    const body = (message.body || '').trim();

    // Verificación dinámica de acceso (backend API)
    const access = await checkUserAccess(phoneNumber);
    if (!access.allowed) {
        console.log(`[BLOQUEO] Número no registrado: ${phoneNumber}`);
        if (body.toLowerCase().startsWith('/solicitar')) {
            const name = body.replace(/^\/solicitar\s*/i, '').trim() || null;
            await requestAccess(phoneNumber, name);
            await message.reply('✅ Tu solicitud fue enviada al administrador. Te avisaremos cuando tengas acceso.');
        } else if (access.allow_requests) {
            await message.reply(
                '⚠️ *No tienes acceso a este asistente.*\n\n' +
                'Puedes solicitar acceso escribiendo:\n*/solicitar <tu nombre>*'
            );
        } else {
            await message.reply('⚠️ Este asistente es de uso privado. Contacta al administrador para obtener acceso.');
        }
        return;
    }

    // Usar el user_id canónico (email) para todas las llamadas al backend
    const canonicalUserId = access.canonical_user_id;
    console.log(`[MSG] De ${phoneNumber} (→${canonicalUserId}): ${body}`);

    // ── Comando /ayuda (respuesta local, sin llamar al backend) ──────────────────
    if (body === '/ayuda') {
        await message.reply(AYUDA_MSG);
        return;
    }

    // ── P5-E: Comando /sre — rutar al k8s-agent ──────────────────────────────────
    if (body.startsWith('/sre')) {
        const sreCmd = body.replace(/^\/sre\s*/i, '').trim() || 'ayuda';
        console.log(`[SRE] Comando SRE de ${phoneNumber}: "${sreCmd}"`);

        let quotedText = null;
        try {
            if (message.hasQuotedMsg) {
                const quotedMsg = await message.getQuotedMessage();
                quotedText = quotedMsg.body;
                console.log(`[SRE] Mensaje citado detectado: "${quotedText.slice(0, 50)}..."`);
            }
        } catch (e) {
            console.warn(`[SRE] Error obteniendo mensaje citado: ${e.message}`);
        }

        try {
            const sreRes = await axios.post(`${K8S_AGENT_URL}/api/sre/command`, {
                command: sreCmd,
                phone:   phoneNumber,
                quoted_text: quotedText,
            }, {
                headers: { Authorization: `Bearer ${AMAEL_INTERNAL_SECRET}` },
                timeout: 30000,
            });
            const reply = sreRes.data.reply || '(sin respuesta)';
            await message.reply(reply);
        } catch (err) {
            console.error(`[SRE] Error llamando k8s-agent: ${err.message}`);
            await message.reply('❌ El agente SRE no está disponible. Intenta más tarde.');
        }
        return;
    }

    // ── Expandir comandos rápidos ────────────────────────────────────────────────
    let prompt = QUICK_COMMANDS[body] || body;
    if (QUICK_COMMANDS[body]) {
        console.log(`[CMD] Comando rápido "${body}" → expandido`);
    }

    try {
        let payload = { prompt, user_id: canonicalUserId, phone: phoneNumber };

        // Multimedia
        if (message.hasMedia) {
            const media = await message.downloadMedia();
            if (media && media.mimetype.startsWith('image/')) {
                payload.image = media.data;
                if (!payload.prompt) payload.prompt = 'Analiza esta imagen.';
            } else if (media && (message.type === 'ptt' || message.type === 'audio')) {
                // Nota de voz o audio → transcribir en el backend
                payload.audio_base64   = media.data;
                payload.audio_mimetype = media.mimetype || 'audio/ogg; codecs=opus';
                if (!payload.prompt) payload.prompt = '[audio]';
                console.log(`[AUDIO] Nota de voz de ${phoneNumber} (${media.mimetype})`);
            }
        }

        // ── Historial de conversación ────────────────────────────────────────────
        const convId  = await getOrCreateConv(canonicalUserId);
        const history = await loadHistory(convId);
        payload.conversation_id = convId;
        payload.history         = history;

        // Llamada al backend
        const response = await axios.post(AMAEL_API_URL, payload, {
            headers: authHeaders(),
            timeout: 180000, // 3 min para agentes lentos
        });

        const botResponse = response.data.response || '';
        console.log(`[RESP] Para ${phoneNumber}: ${botResponse.slice(0, 100)}...`);

        // Enviar respuesta (con imagen si aplica)
        const mediaRegex = /\[MEDIA:(.+?)\]/;
        const match = botResponse.match(mediaRegex);

        if (match) {
            const cleanText = botResponse.replace(mediaRegex, '').trim();
            const media = new MessageMedia('image/png', match[1], 'screenshot.png');
            await client.sendMessage(message.from, media, { caption: cleanText || '' });
        } else {
            await message.reply(botResponse);
        }

    } catch (error) {
        console.error(`[ERROR] Procesando mensaje de ${phoneNumber}:`, error.message);
        await message.reply('Lo siento, tuve un problema al procesar tu mensaje. Inténtalo de nuevo.');
    }
});

// --- ENDPOINTS ──────────────────────────────────────────────────────────────────

/** Envía un mensaje de texto proactivo (usado por backend para notificaciones). */
app.post('/send', async (req, res) => {
    const { phoneNumber, text } = req.body;
    if (!phoneNumber || !text) {
        return res.status(400).json({ error: 'Faltan parámetros: phoneNumber o text' });
    }
    if (clientStatus !== 'ready') {
        return res.status(503).json({ error: `Cliente no listo. Estado: ${clientStatus}` });
    }
    try {
        await sendText(phoneNumber, text);
        res.json({ success: true });
    } catch (error) {
        console.error('[SEND] Error:', error);
        res.status(500).json({ error: error.message });
    }
});

/** Envía media desde el backend. */
app.post('/send-media', async (req, res) => {
    const { phoneNumber, base64, caption, mimetype } = req.body;
    if (!phoneNumber || !base64) {
        return res.status(400).json({ error: 'Faltan parámetros: phoneNumber o base64' });
    }
    try {
        const chatId = phoneNumber.includes('@c.us') ? phoneNumber : `${phoneNumber}@c.us`;
        const media  = new MessageMedia(mimetype || 'image/png', base64, 'image.png');
        await client.sendMessage(chatId, media, { caption: caption || '' });
        res.json({ success: true });
    } catch (error) {
        console.error('[SEND-MEDIA] Error:', error);
        res.status(500).json({ error: error.message });
    }
});

/** Envía nota de voz (PTT) desde el backend — usado por PiperTool. */
app.post('/send-audio', async (req, res) => {
    const { phoneNumber, base64, mimetype, ptt } = req.body;
    if (!phoneNumber || !base64) {
        return res.status(400).json({ error: 'Faltan parámetros: phoneNumber o base64' });
    }
    if (clientStatus !== 'ready') {
        return res.status(503).json({ error: `Cliente no listo. Estado: ${clientStatus}` });
    }
    try {
        const chatId    = phoneNumber.includes('@c.us') ? phoneNumber : `${phoneNumber}@c.us`;
        const audioMime = mimetype || 'audio/ogg; codecs=opus';
        const filename  = audioMime.startsWith('audio/wav') ? 'voice.wav' : 'voice.ogg';
        const media     = new MessageMedia(audioMime, base64, filename);
        // sendAudio: ptt=true → aparece como nota de voz (ícono de micrófono)
        await client.sendMessage(chatId, media, { sendAudioAsVoice: ptt === true });
        res.json({ success: true });
    } catch (error) {
        console.error('[SEND-AUDIO] Error:', error);
        res.status(500).json({ error: error.message });
    }
});

/** Screenshot de Grafana u otras URLs */
app.post('/screenshot', async (req, res) => {
    const { url, waitSelector, username, password } = req.body;
    if (!url) return res.status(400).json({ error: 'Falta la URL' });

    let browser;
    try {
        browser = await puppeteer.launch({
            executablePath: CHROMIUM_PATH,
            args: ['--no-sandbox', '--disable-setuid-sandbox'],
        });
        const page = await browser.newPage();
        await page.setViewport({ width: 1280, height: 800 });

        if (url.includes('grafana')) {
            const user = username || 'admin';
            const pass = password || 'admin';
            const authHeader = `Basic ${Buffer.from(`${user}:${pass}`).toString('base64')}`;
            await page.setExtraHTTPHeaders({ Authorization: authHeader });
            await page.authenticate({ username: user, password: pass });
        }

        await page.goto(url, { waitUntil: 'networkidle2', timeout: 60000 });

        const isLoginPage = await page.evaluate(() =>
            document.title.toLowerCase().includes('login') ||
            !!document.querySelector('input[name="user"]') ||
            !!document.querySelector('form[name="login"]')
        );

        if (isLoginPage && url.includes('grafana')) {
            try {
                await page.waitForSelector('input[name="user"]', { timeout: 10000 });
                await page.type('input[name="user"]', username || 'admin');
                await page.type('input[name="password"]', password || 'admin');
                await page.waitForSelector('button[type="submit"]', { timeout: 5000 });
                await page.click('button[type="submit"]');
                await new Promise(r => setTimeout(r, 5000));
            } catch (e) {
                console.warn('[SCREENSHOT] Error en login automático:', e.message);
            }
        }

        if (waitSelector) {
            await page.waitForSelector(waitSelector, { timeout: 15000 }).catch(() => {});
        } else {
            await new Promise(r => setTimeout(r, 7000));
        }

        if (url.includes('grafana')) {
            await page.addStyleTag({
                content: '.sidemenu-canvas, .navbar-page-btn, .search-container { display: none !important; }',
            });
        }

        const screenshot = await page.screenshot({ encoding: 'base64' });
        res.json({ base64: screenshot });
    } catch (error) {
        console.error('[SCREENSHOT] Error:', error);
        res.status(500).json({ error: error.message });
    } finally {
        if (browser) await browser.close();
    }
});

app.get('/qr', (req, res) => {
    if (clientStatus === 'ready') {
        res.send('<h1>✅ Bot conectado y listo.</h1>');
    } else if (qrCodeData && clientStatus === 'awaiting_qr') {
        res.send(`<!DOCTYPE html><html><head><title>WhatsApp QR</title></head>
        <body style="text-align:center;font-family:sans-serif;padding:40px">
        <h1>📱 Escanea con WhatsApp</h1>
        <img src="https://api.qrserver.com/v1/create-qr-code/?size=300x300&data=${encodeURIComponent(qrCodeData)}" alt="QR">
        <meta http-equiv="refresh" content="30"></body></html>`);
    } else {
        res.send(`<h1>⏳ Estado: ${clientStatus}</h1><meta http-equiv="refresh" content="5">`);
    }
});

app.get('/status', (req, res) => {
    res.json({ status: clientStatus, hasQR: !!qrCodeData && qrCodeData !== 'CLIENTE_LISTO', timestamp: new Date().toISOString() });
});

app.get('/health', (req, res) => {
    if (clientStatus === 'ready') {
        res.status(200).send('OK');
    } else {
        res.status(503).send(clientStatus);
    }
});

// --- START ──────────────────────────────────────────────────────────────────────
const PORT = process.env.PORT || 3000;
app.listen(PORT, () => {
    console.log(`[BRIDGE v1.5.2] Servidor en puerto ${PORT}`);
});

const initializeClient = () => {
    console.log('[BRIDGE] Iniciando proceso de inicialización del cliente...');
    clientStatus = 'initializing';
    client.initialize()
        .then(() => {
            console.log('[BRIDGE] Solicitud de inicialización enviada con éxito.');
        })
        .catch(err => {
            console.error('[BRIDGE] ERROR CRÍTICO al inicializar:', err);
            clientStatus = 'error';
            setTimeout(initializeClient, 30000);
        });
};

console.log('[BRIDGE] Arrancando aplicación...');
initializeClient();
