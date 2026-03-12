'use strict';

const { Client, LocalAuth, MessageMedia } = require('whatsapp-web.js');
const qrcode  = require('qrcode-terminal');
const express = require('express');
const axios   = require('axios');

// ── Configuración ──────────────────────────────────────────────────────────────
const AMAEL_BASE_URL      = process.env.AMAEL_BASE_URL      || 'http://backend-service:8000';
const AMAEL_INTERNAL_SECRET = process.env.AMAEL_INTERNAL_SECRET;
const WHATSAPP_OWNER_JWT  = process.env.WHATSAPP_OWNER_JWT  || '';
const OWNER_USER_ID       = process.env.OWNER_USER_ID       || '';
const CHROMIUM_PATH       = process.env.PUPPETEER_EXECUTABLE_PATH || '/usr/bin/chromium';
const PORT                = parseInt(process.env.PORT || '3001');

if (!AMAEL_INTERNAL_SECRET) {
  console.error('[FATAL] Falta AMAEL_INTERNAL_SECRET');
  process.exit(1);
}

const internalHeaders = () => ({
  'X-Internal-Secret': AMAEL_INTERNAL_SECRET,
  'Content-Type': 'application/json',
});

const ownerAuthHeaders = () => ({
  'Authorization': `Bearer ${WHATSAPP_OWNER_JWT}`,
  'Content-Type': 'application/json',
});

// ── Estado global ──────────────────────────────────────────────────────────────
let qrCodeData   = null;
let clientStatus = 'initializing';   // initializing | awaiting_qr | ready | disconnected | auth_failure
let connectedPhone = null;           // número conectado una vez autenticado

// ── WhatsApp Client ────────────────────────────────────────────────────────────
const client = new Client({
  authStrategy: new LocalAuth({ dataPath: '/usr/src/app/.wwebjs_auth' }),
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
      '--ignore-certificate-errors', '--ignore-ssl-errors', '--disable-web-security',
      '--disable-features=IsolateOrigins,site-per-process',
    ],
  },
  authTimeoutMs: 300000,
});

// ── Helpers ────────────────────────────────────────────────────────────────────

/**
 * Obtiene los settings del dueño desde el backend.
 * Retorna defaults si falla para no romper el flujo.
 */
async function getOwnerSettings() {
  try {
    const res = await axios.get(
      `${AMAEL_BASE_URL}/api/whatsapp-personal/check-settings`,
      { headers: internalHeaders(), timeout: 4000 },
    );
    return res.data;
  } catch (e) {
    console.warn('[SETTINGS] No se pudieron obtener settings, usando defaults:', e.message);
    return {
      auto_reply:   true,
      in_quiet_hours: false,
      reply_scope:  'all',
      ai_assist:    true,
      offline_msg:  null,
    };
  }
}

/**
 * Determina si un mensaje debe procesarse según scope y tipo.
 * scope: 'all' | 'contacts_only' | 'no_groups' | 'custom'
 * allowedContacts: array de strings con números de teléfono (solo para scope=custom)
 */
function shouldProcess(message, contact, scope, allowedContacts = []) {
  const isGroup = message.from.endsWith('@g.us');
  if (scope === 'no_groups' && isGroup) return false;
  if (scope === 'contacts_only' && !contact.isMyContact) return false;
  if (scope === 'custom') {
    // Extraer número limpio del remitente
    const senderNum = message.from.split('@')[0].split(':')[0].replace(/\D/g, '');
    const allowed = allowedContacts.map(n => String(n).replace(/\D/g, ''));
    if (!allowed.includes(senderNum)) return false;
  }
  return true;
}

/**
 * Llama al backend con el mensaje del usuario y obtiene respuesta IA.
 */
async function askAI(userMessage, conversationId) {
  try {
    const payload = {
      prompt: userMessage,
      user_id: OWNER_USER_ID,
      conversation_id: conversationId || undefined,
    };
    const res = await axios.post(
      `${AMAEL_BASE_URL}/api/chat`,
      payload,
      { headers: ownerAuthHeaders(), timeout: 90000 },
    );
    return res.data?.response || null;
  } catch (e) {
    console.error('[AI] Error consultando IA:', e.message);
    return null;
  }
}

// ── Eventos WhatsApp ───────────────────────────────────────────────────────────

client.on('qr', (qr) => {
  qrCodeData   = qr;
  clientStatus = 'awaiting_qr';
  console.log('[QR] Nuevo código QR generado');
  qrcode.generate(qr, { small: true });
});

client.on('ready', async () => {
  clientStatus = 'ready';
  qrCodeData   = null;
  try {
    const info = client.info;
    connectedPhone = info?.wid?.user || null;
    console.log(`[READY] Conectado como: ${connectedPhone}`);
    // Notificar al backend que el servicio está listo
    await axios.post(
      `${AMAEL_BASE_URL}/api/whatsapp-personal/connected`,
      { phone: connectedPhone },
      { headers: internalHeaders(), timeout: 5000 },
    ).catch(() => {});
  } catch (e) {
    console.warn('[READY] No se pudo obtener número:', e.message);
  }
});

client.on('auth_failure', () => {
  clientStatus = 'auth_failure';
  qrCodeData   = null;
  console.error('[AUTH] Fallo de autenticación');
});

client.on('disconnected', (reason) => {
  clientStatus = 'disconnected';
  qrCodeData   = null;
  connectedPhone = null;
  console.warn('[DISCO] Desconectado:', reason);
  // Reconectar automáticamente después de 10s
  setTimeout(() => {
    console.log('[RECONECT] Intentando reconexión...');
    client.initialize().catch(e => console.error('[RECONECT] Error:', e.message));
  }, 10000);
});

client.on('loading_screen', (percent) => {
  clientStatus = `loading:${percent}`;
});

// ── Manejador de mensajes entrantes ───────────────────────────────────────────

client.on('message', async (message) => {
  // Ignorar mensajes propios
  if (message.fromMe) return;
  // Ignorar status/broadcast
  if (message.from === 'status@broadcast') return;

  const settings = await getOwnerSettings();

  // 1. Check global on/off
  if (!settings.auto_reply) {
    console.log(`[MSG] Auto-respuesta desactivada, ignorando mensaje de ${message.from}`);
    return;
  }

  // 2. Check quiet hours
  if (settings.in_quiet_hours) {
    if (settings.offline_msg) {
      await message.reply(settings.offline_msg).catch(() => {});
    }
    console.log(`[MSG] Horario silencioso activo, ${settings.offline_msg ? 'respondido con mensaje offline' : 'ignorado'}`);
    return;
  }

  // 3. Check scope (groups / contacts)
  let contact = null;
  try { contact = await message.getContact(); } catch (e) { contact = { isMyContact: false }; }
  if (!shouldProcess(message, contact, settings.reply_scope, settings.allowed_contacts || [])) {
    console.log(`[MSG] Mensaje de ${message.from} ignorado por regla de scope (${settings.reply_scope})`);
    return;
  }

  // 4. AI assist
  if (!settings.ai_assist) {
    console.log(`[MSG] AI assist desactivado, ignorando mensaje de ${message.from}`);
    return;
  }

  const body = message.body?.trim();
  if (!body) return;

  console.log(`[MSG] Procesando mensaje de ${message.from}: "${body.substring(0, 60)}"`);

  const aiResponse = await askAI(body, null);
  if (aiResponse) {
    await message.reply(aiResponse).catch(e => console.error('[REPLY] Error:', e.message));
  }
});

// ── HTTP API ───────────────────────────────────────────────────────────────────

const app = express();
app.use(express.json());

/** Estado del servicio */
app.get('/status', (req, res) => {
  res.json({
    status:  clientStatus,
    phone:   connectedPhone,
    hasQR:   !!qrCodeData && clientStatus === 'awaiting_qr',
    version: '1.0.0',
    ts:      new Date().toISOString(),
  });
});

/** QR como JSON — el frontend genera la imagen */
app.get('/qr-json', (req, res) => {
  res.json({
    status: clientStatus,
    qr:     (clientStatus === 'awaiting_qr') ? qrCodeData : null,
    phone:  connectedPhone,
  });
});

/** Desconectar sesión */
app.post('/logout', async (req, res) => {
  try {
    await client.logout();
    clientStatus   = 'disconnected';
    connectedPhone = null;
    qrCodeData     = null;
    res.json({ ok: true });
  } catch (e) {
    res.status(500).json({ ok: false, error: e.message });
  }
});

/** Re-inicializar (reconectar/obtener nuevo QR) */
app.post('/reconnect', async (req, res) => {
  try {
    clientStatus = 'initializing';
    await client.initialize();
    res.json({ ok: true });
  } catch (e) {
    res.status(500).json({ ok: false, error: e.message });
  }
});

/** Enviar mensaje proactivo (uso interno) */
app.post('/send', async (req, res) => {
  const { phoneNumber, text } = req.body || {};
  if (!phoneNumber || !text) return res.status(400).json({ error: 'phoneNumber y text requeridos' });
  if (clientStatus !== 'ready') return res.status(503).json({ error: `No conectado: ${clientStatus}` });
  try {
    const chatId = phoneNumber.includes('@') ? phoneNumber : `${phoneNumber}@c.us`;
    await client.sendMessage(chatId, text);
    res.json({ ok: true });
  } catch (e) {
    res.status(500).json({ ok: false, error: e.message });
  }
});

/** Health check */
app.get('/health', (req, res) => res.json({ ok: true, status: clientStatus }));

// ── Arrancar ───────────────────────────────────────────────────────────────────

app.listen(PORT, () => console.log(`[SERVER] whatsapp-personal escuchando en :${PORT}`));

console.log('[INIT] Iniciando cliente WhatsApp personal...');
client.initialize().catch(e => {
  console.error('[INIT] Error al inicializar:', e.message);
});
