import re
from datetime import datetime
import streamlit as st
import streamlit.components.v1 as components
import requests

# --- CONFIGURACIÓN DE PÁGINA ---
st.set_page_config(
    page_title="Amael · Asistente Inteligente",
    page_icon="◆",
    layout="wide",
    initial_sidebar_state="auto",   # P3-A: auto-colapsa en móvil
)

# --- CONFIGURACIÓN ---
BACKEND_URL = "https://amael-ia.richardx.dev/api"

# ---------------------------------------------------------------------------
# SISTEMA DE TOKENS DE DISEÑO
# ---------------------------------------------------------------------------
THEMES = {
    "dark": {
        "bg_base":           "#0a0e1a",
        "bg_surface":        "#111827",
        "bg_elevated":       "#1a2236",
        "primary":           "#6366f1",
        "primary_hover":     "#4f46e5",
        "primary_subtle":    "rgba(99,102,241,0.12)",
        "text_primary":      "#f1f5f9",
        "text_secondary":    "#94a3b8",
        "text_disabled":     "#475569",
        "border":            "#1e2d3d",
        "border_focus":      "#6366f1",
        "user_bubble_bg":    "#1e2d4f",
        "user_bubble_border":"rgba(99,102,241,0.25)",
        "sidebar_bg":        "#0d1117",
        "input_bg":          "#1a2236",
        "scrollbar":         "#1e2d3d",
        "scrollbar_hover":   "#334155",
        "success":           "#22c55e",
        "error":             "#ef4444",
    },
    "light": {
        "bg_base":           "#f8fafc",
        "bg_surface":        "#ffffff",
        "bg_elevated":       "#f1f5f9",
        "primary":           "#6366f1",
        "primary_hover":     "#4f46e5",
        "primary_subtle":    "rgba(99,102,241,0.08)",
        "text_primary":      "#0f172a",
        "text_secondary":    "#64748b",
        "text_disabled":     "#94a3b8",
        "border":            "#e2e8f0",
        "border_focus":      "#6366f1",
        "user_bubble_bg":    "#eef2ff",
        "user_bubble_border":"rgba(99,102,241,0.2)",
        "sidebar_bg":        "#f1f5f9",
        "input_bg":          "#ffffff",
        "scrollbar":         "#e2e8f0",
        "scrollbar_hover":   "#cbd5e1",
        "success":           "#16a34a",
        "error":             "#dc2626",
    },
}

# ---------------------------------------------------------------------------
# ESTADO DE SESIÓN
# ---------------------------------------------------------------------------
def initialize_session_state():
    defaults = {
        "jwt_token":          None,
        "messages":           [],
        "user_name":          None,
        "user_picture":       None,
        "theme":              "dark",
        "last_failed_prompt": None,    # P2-D
        "retry_prompt":       None,    # P2-D
        "conversation_id":    None,    # P3-B
        "conversations":      [],      # P3-B
        "history_loaded":     False,   # P1-B
        "feedback":           {},      # P2-C: {msg_idx: 'positive'|'negative'}
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val

def toggle_theme():
    st.session_state.theme = "light" if st.session_state.theme == "dark" else "dark"

def get_tokens():
    return THEMES[st.session_state.get("theme", "dark")]

def _api_headers():
    return {"Authorization": f"Bearer {st.session_state.jwt_token}"}

# ---------------------------------------------------------------------------
# INYECCIÓN DE CSS
# ---------------------------------------------------------------------------
def inject_base_css():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    html, body, [class*="css"], .stApp { font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif; }
    #MainMenu, footer, .stDeployButton { visibility: hidden; }
    header { background: transparent !important; border-bottom: none !important; }
    [data-testid="stHeader"] { background: transparent !important; }
    ::-webkit-scrollbar { width: 5px; height: 5px; }
    ::-webkit-scrollbar-track { background: transparent; }
    ::-webkit-scrollbar-thumb { background: var(--scrollbar); border-radius: 10px; }
    ::-webkit-scrollbar-thumb:hover { background: var(--scrollbar-hover); }
    </style>
    """, unsafe_allow_html=True)


def inject_theme_vars(t: dict):
    lines = "\n".join(f"    --{k.replace('_','-')}: {v};" for k, v in t.items())
    st.markdown(f"<style>:root {{\n{lines}\n}}</style>", unsafe_allow_html=True)


def inject_login_css():
    st.markdown("""
    <style>
    .stApp { background-color: var(--bg-base) !important; }
    .block-container { padding: 0 !important; max-width: 100% !important; }
    .login-wrap {
        min-height: 100vh; display: flex; align-items: center;
        justify-content: center; padding: 24px;
    }
    .login-card {
        width: 100%; max-width: 400px; background: var(--bg-surface);
        border: 1px solid var(--border); border-radius: 14px;
        padding: 44px 40px 36px; animation: fadeUp 0.35s cubic-bezier(.2,.8,.2,1);
    }
    @keyframes fadeUp {
        from { opacity: 0; transform: translateY(10px); }
        to   { opacity: 1; transform: translateY(0); }
    }
    .login-wordmark { font-size: 24px; font-weight: 700; color: var(--text-primary); letter-spacing: -0.5px; margin-bottom: 4px; }
    .login-tagline  { font-size: 14px; color: var(--text-secondary); margin-bottom: 36px; }
    .login-divider  { height: 1px; background: var(--border); margin: 28px 0; }
    .google-btn {
        display: flex; align-items: center; justify-content: center; gap: 10px;
        width: 100%; padding: 13px 20px; background: #ffffff; color: #1f2937;
        font-size: 15px; font-weight: 500; border-radius: 8px; text-decoration: none;
        border: 1px solid #dadce0; transition: background 0.15s, box-shadow 0.15s;
        cursor: pointer; box-sizing: border-box;
    }
    .google-btn:hover { background: #f6f8fc; box-shadow: 0 1px 6px rgba(60,64,67,0.15); text-decoration: none; }
    .google-btn::before {
        content: "G"; font-size: 16px; font-weight: 700; color: #4285F4;
        font-family: 'Arial', sans-serif; width: 18px; text-align: center; flex-shrink: 0;
    }
    .login-trust { display: flex; align-items: center; gap: 6px; justify-content: center; font-size: 12px; color: var(--text-disabled); margin-top: 18px; }
    .login-trust::before { content: "🔒"; font-size: 11px; }
    </style>
    """, unsafe_allow_html=True)


def inject_chat_css():
    st.markdown("""
    <style>
    /* ---- Layout ---- */
    .stApp { background-color: var(--bg-base) !important; }
    .block-container {
        padding-top: 2rem !important; padding-bottom: 7rem !important;
        max-width: 720px !important; margin: 0 auto !important;
    }

    /* ---- Sidebar ---- */
    [data-testid="stSidebar"] { background-color: var(--sidebar-bg) !important; border-right: 1px solid var(--border) !important; }
    [data-testid="stSidebar"] * { color: var(--text-primary) !important; }
    [data-testid="stSidebar"] .stButton > button {
        width: 100%; background: var(--bg-elevated) !important; border: 1px solid var(--border) !important;
        border-radius: 8px !important; color: var(--text-primary) !important; font-size: 13px !important;
        font-weight: 400 !important; padding: 8px 12px !important; text-align: left !important;
        transition: background 0.15s, border-color 0.15s !important; box-shadow: none !important;
        white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
    }
    [data-testid="stSidebar"] .stButton > button:hover {
        background: var(--primary-subtle) !important; border-color: var(--primary) !important;
        color: var(--primary) !important; transform: none !important;
    }
    /* P3-B: active conversation highlight */
    .conv-active button {
        background: var(--primary-subtle) !important;
        border-color: var(--primary) !important;
        color: var(--primary) !important;
        font-weight: 600 !important;
    }

    /* P3-A: Ocultar sidebar en pantallas pequeñas */
    @media (max-width: 768px) {
        [data-testid="stSidebar"] { display: none !important; }
        [data-testid="collapsedControl"] { display: flex !important; }
    }

    /* ---- File uploader ---- */
    [data-testid="stFileUploader"] {
        background: var(--bg-elevated); border: 1.5px dashed var(--border);
        border-radius: 10px; padding: 16px 12px; transition: border-color 0.15s;
    }
    [data-testid="stFileUploader"]:hover { border-color: var(--primary); }
    [data-testid="stFileUploader"] div,
    [data-testid="stFileUploader"] small,
    [data-testid="stFileUploader"] span { color: var(--text-secondary) !important; }
    [data-testid="stFileUploader"] section > button {
        border-radius: 6px !important; background: var(--bg-surface) !important;
        border: 1px solid var(--border) !important; color: var(--text-primary) !important;
        font-size: 13px !important; padding: 6px 14px !important;
    }

    /* ---- Mensajes base ---- */
    .stChatMessage { background: transparent !important; border: none !important; padding: 0 !important; margin-bottom: 2px !important; }

    /* ---- Burbuja USUARIO ---- */
    [data-testid="chat-message-container-user"] { flex-direction: row-reverse; }
    [data-testid="chat-message-container-user"] [data-testid="stChatMessageAvatarUser"],
    [data-testid="chat-message-container-user"] .stAvatar { display: none !important; }
    [data-testid="chat-message-container-user"] [data-testid="stMarkdownContainer"],
    [data-testid="chat-message-container-user"] .stMarkdown {
        background: var(--user-bubble-bg) !important; border: 1px solid var(--user-bubble-border) !important;
        border-radius: 18px 18px 4px 18px !important; padding: 11px 16px !important;
        max-width: 82% !important; margin-left: auto !important; box-shadow: none !important;
        font-size: 15px !important; line-height: 1.6 !important; color: var(--text-primary) !important;
    }
    [data-testid="chat-message-container-user"] p { color: var(--text-primary) !important; margin: 0 !important; }

    /* ---- Mensajes ASISTENTE (sin burbuja) ---- */
    [data-testid="chat-message-container-assistant"] [data-testid="stMarkdownContainer"],
    [data-testid="chat-message-container-assistant"] .stMarkdown {
        background: transparent !important; border: none !important; padding: 0 4px !important;
        box-shadow: none !important; font-size: 15px !important; line-height: 1.7 !important;
        color: var(--text-primary) !important; max-width: 100% !important;
    }
    [data-testid="chat-message-container-assistant"] p,
    [data-testid="chat-message-container-assistant"] li,
    [data-testid="chat-message-container-assistant"] span,
    [data-testid="chat-message-container-assistant"] td { color: var(--text-primary) !important; }
    [data-testid="chat-message-container-assistant"] p { margin-bottom: 0.6rem !important; }
    [data-testid="chat-message-container-assistant"] p:last-child { margin-bottom: 0 !important; }

    /* ---- P3-D: Tablas ---- */
    [data-testid="chat-message-container-assistant"] table {
        width: 100%; border-collapse: collapse; margin: 12px 0;
        font-size: 14px; border: 1px solid var(--border); border-radius: 8px; overflow: hidden;
    }
    [data-testid="chat-message-container-assistant"] th {
        background: var(--bg-elevated); color: var(--text-secondary); font-weight: 600;
        font-size: 12px; text-transform: uppercase; letter-spacing: 0.04em;
        padding: 9px 12px; border-bottom: 1px solid var(--border); text-align: left;
    }
    [data-testid="chat-message-container-assistant"] td {
        padding: 9px 12px; border-bottom: 1px solid var(--border);
        color: var(--text-primary) !important; vertical-align: top;
    }
    [data-testid="chat-message-container-assistant"] tr:last-child td { border-bottom: none; }
    [data-testid="chat-message-container-assistant"] tr:hover td { background: var(--bg-elevated); }

    /* ---- P3-D: Code blocks ---- */
    [data-testid="chat-message-container-assistant"] code {
        background: var(--bg-elevated) !important; color: var(--primary) !important;
        border-radius: 4px !important; padding: 1px 5px !important; font-size: 13px !important;
    }
    [data-testid="chat-message-container-assistant"] pre {
        background: var(--bg-surface) !important; border: 1px solid var(--border) !important;
        border-radius: 8px !important; padding: 14px 16px !important; overflow-x: auto !important; margin: 10px 0 !important;
    }
    [data-testid="chat-message-container-assistant"] pre code {
        background: transparent !important; color: var(--text-primary) !important;
        padding: 0 !important; font-size: 13px !important;
    }

    /* ---- P3-D: Blockquote ---- */
    [data-testid="chat-message-container-assistant"] blockquote {
        border-left: 3px solid var(--primary); margin: 10px 0;
        padding: 8px 14px; background: var(--primary-subtle);
        border-radius: 0 6px 6px 0; color: var(--text-secondary) !important;
    }

    /* ---- P3-D: Listas ---- */
    [data-testid="chat-message-container-assistant"] ul,
    [data-testid="chat-message-container-assistant"] ol { padding-left: 20px; margin: 6px 0; }
    [data-testid="chat-message-container-assistant"] li { margin-bottom: 4px; line-height: 1.6; }

    /* ---- P3-D: Strong / HR ---- */
    [data-testid="chat-message-container-assistant"] strong { color: var(--text-primary) !important; font-weight: 600; }
    [data-testid="chat-message-container-assistant"] hr {
        border: none; border-top: 1px solid var(--border); margin: 14px 0;
    }

    /* ---- Avatar asistente ---- */
    [data-testid="chat-message-container-assistant"] .stAvatar,
    [data-testid="stChatMessageAvatarAssistant"] {
        width: 28px !important; height: 28px !important; min-width: 28px !important;
        border-radius: 7px !important; background: var(--primary) !important;
        color: #ffffff !important; font-size: 12px !important; font-weight: 700 !important; box-shadow: none !important;
    }

    /* ---- P2-B: Timestamps ---- */
    .msg-ts {
        font-size: 11px; color: var(--text-disabled); margin-top: 4px; margin-bottom: 4px;
    }
    .msg-ts-user  { text-align: right; padding-right: 2px; }
    .msg-ts-assistant { text-align: left; padding-left: 40px; }

    /* ---- P2-C: Feedback buttons ---- */
    .fb-row { display: flex; gap: 4px; padding-left: 40px; margin-bottom: 10px; }
    .fb-given { padding-left: 40px; margin-bottom: 10px; font-size: 13px; color: var(--text-disabled); }

    /* ---- Thinking indicator ---- */
    .thinking-wrap { display: flex; align-items: center; gap: 8px; padding: 4px 0; color: var(--text-secondary); font-size: 13px; }
    .thinking-dots { display: flex; gap: 3px; align-items: center; }
    .thinking-dots span {
        display: block; width: 5px; height: 5px; border-radius: 50%;
        background: var(--primary); animation: dotPulse 1.3s ease-in-out infinite;
    }
    .thinking-dots span:nth-child(2) { animation-delay: 0.18s; }
    .thinking-dots span:nth-child(3) { animation-delay: 0.36s; }
    @keyframes dotPulse {
        0%, 60%, 100% { opacity: 0.25; transform: scale(0.75); }
        30%            { opacity: 1;    transform: scale(1); }
    }

    /* ---- P2-D: Retry bar ---- */
    .retry-bar {
        background: rgba(239,68,68,0.07); border: 1px solid rgba(239,68,68,0.18);
        border-radius: 8px; padding: 10px 14px; font-size: 13px;
        color: var(--text-secondary); margin-bottom: 6px;
    }

    /* ---- Input de chat ---- */
    [data-testid="stChatInput"] {
        background: var(--input-bg) !important; border: 1px solid var(--border) !important;
        border-radius: 14px !important; box-shadow: none !important;
        padding: 4px 12px !important; transition: border-color 0.15s !important;
    }
    [data-testid="stChatInput"]:focus-within {
        border-color: var(--border-focus) !important;
        box-shadow: 0 0 0 3px var(--primary-subtle) !important;
    }
    [data-testid="stChatInput"] textarea {
        color: var(--text-primary) !important; background: transparent !important;
        font-size: 15px !important; caret-color: var(--primary) !important;
    }
    [data-testid="stChatInput"] textarea::placeholder { color: var(--text-disabled) !important; }

    /* ---- Empty state ---- */
    .empty-state {
        display: flex; flex-direction: column; align-items: center; justify-content: center;
        height: 60vh; text-align: center; animation: fadeUp 0.4s ease; padding: 0 24px;
    }
    .empty-icon { width: 56px; height: 56px; background: var(--primary-subtle); border-radius: 14px; display: flex; align-items: center; justify-content: center; margin-bottom: 20px; font-size: 26px; }
    .empty-title { font-size: 22px; font-weight: 600; color: var(--text-primary); margin-bottom: 8px; letter-spacing: -0.3px; }
    .empty-subtitle { font-size: 15px; color: var(--text-secondary); max-width: 340px; line-height: 1.6; margin-bottom: 28px; }
    .empty-chips { display: flex; gap: 8px; flex-wrap: wrap; justify-content: center; }
    .empty-chip { background: var(--bg-elevated); border: 1px solid var(--border); border-radius: 20px; padding: 7px 14px; font-size: 13px; color: var(--text-secondary); }

    /* ---- Sidebar: perfil y estructura ---- */
    .sidebar-profile { display: flex; align-items: center; gap: 10px; padding: 12px 4px; border-bottom: 1px solid var(--border); margin-bottom: 16px; }
    .sidebar-profile-pic { width: 36px; height: 36px; border-radius: 50%; object-fit: cover; border: 1.5px solid var(--border); flex-shrink: 0; }
    .sidebar-profile-pic-placeholder { width: 36px; height: 36px; border-radius: 50%; background: var(--primary-subtle); border: 1.5px solid var(--border); display: flex; align-items: center; justify-content: center; font-size: 15px; font-weight: 600; color: var(--primary); flex-shrink: 0; }
    .sidebar-profile-info { min-width: 0; }
    .sidebar-profile-name { font-size: 14px; font-weight: 600; color: var(--text-primary); white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
    .sidebar-profile-label { font-size: 11px; color: var(--text-secondary); text-transform: uppercase; letter-spacing: 0.05em; }
    .sidebar-section-label { font-size: 11px; font-weight: 600; color: var(--text-disabled); text-transform: uppercase; letter-spacing: 0.06em; margin: 20px 0 8px; padding: 0 2px; }
    .sidebar-wordmark { font-size: 18px; font-weight: 700; color: var(--text-primary); letter-spacing: -0.3px; display: flex; align-items: center; gap: 7px; margin-bottom: 16px; }
    .sidebar-wordmark-dot { width: 8px; height: 8px; border-radius: 2px; background: var(--primary); display: inline-block; }

    .stToastContainer { top: 16px !important; right: 16px !important; }
    @keyframes fadeUp { from { opacity: 0; transform: translateY(8px); } to { opacity: 1; transform: translateY(0); } }
    </style>
    """, unsafe_allow_html=True)


# P2-A: Copy button JS injection via iframe (accesses parent DOM)
def inject_copy_js():
    components.html("""
    <script>
    (function() {
        var p = window.parent.document;

        function injectStyle() {
            if (p.getElementById('amael-copy-style')) return;
            var s = p.createElement('style');
            s.id = 'amael-copy-style';
            s.textContent = [
                '[data-testid="chat-message-container-assistant"] { position: relative; }',
                '.amael-copy-btn { position: absolute; top: 6px; right: 0;',
                '  background: none; border: none; cursor: pointer;',
                '  color: #475569; opacity: 0; transition: opacity 0.15s, color 0.15s, background 0.15s;',
                '  padding: 4px 8px; border-radius: 6px; line-height: 1; font-size: 13px; }',
                '[data-testid="chat-message-container-assistant"]:hover .amael-copy-btn { opacity: 1; }',
                '.amael-copy-btn:hover { color: #6366f1; background: rgba(99,102,241,0.1); }'
            ].join(' ');
            p.head.appendChild(s);
        }

        function addButtons() {
            injectStyle();
            var els = p.querySelectorAll('[data-testid="chat-message-container-assistant"]');
            els.forEach(function(el) {
                if (el.querySelector('.amael-copy-btn')) return;
                var btn = p.createElement('button');
                btn.className = 'amael-copy-btn';
                btn.title = 'Copiar respuesta';
                btn.textContent = '\u29c9';  // ⧉ copy symbol
                btn.addEventListener('click', function() {
                    var md = el.querySelector('[data-testid="stMarkdownContainer"]');
                    var text = md ? md.innerText : '';
                    navigator.clipboard.writeText(text).then(function() {
                        btn.textContent = '\u2713';  // ✓
                        setTimeout(function() { btn.textContent = '\u29c9'; }, 2000);
                    });
                });
                el.appendChild(btn);
            });
        }

        var obs = new MutationObserver(addButtons);
        obs.observe(p.body, { childList: true, subtree: true });
        addButtons();
    })();
    </script>
    """, height=0)


# ---------------------------------------------------------------------------
# AUTENTICACIÓN
# ---------------------------------------------------------------------------
def check_authentication():
    jwt_token = st.query_params.get("token")
    if isinstance(jwt_token, list):
        jwt_token = jwt_token[0]

    error    = st.query_params.get("error")
    user_name = st.query_params.get("name")
    if isinstance(user_name, list):
        user_name = user_name[0]
    user_picture = st.query_params.get("picture")
    if isinstance(user_picture, list):
        user_picture = user_picture[0]

    if jwt_token:
        st.session_state.jwt_token    = jwt_token
        if user_name:    st.session_state.user_name    = user_name
        if user_picture: st.session_state.user_picture = user_picture
        st.query_params.clear()
        st.rerun()

    if error:
        st.error("No tienes permiso para acceder. Contacta al administrador.")
        st.stop()

    return st.session_state.jwt_token is not None


# ---------------------------------------------------------------------------
# LOGIN
# ---------------------------------------------------------------------------
def mostrar_pantalla_login():
    inject_login_css()
    st.markdown(
        f'<div class="login-wrap">'
        f'<div class="login-card">'
        f'<div class="login-wordmark">Amael</div>'
        f'<div class="login-tagline">Tu asistente inteligente personal</div>'
        f'<a href="{BACKEND_URL}/auth/login" class="google-btn" target="_parent">Continuar con Google</a>'
        f'<div class="login-divider"></div>'
        f'<div class="login-trust">OAuth 2.0 &middot; No almacenamos contrase&ntilde;as</div>'
        f'</div></div>',
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# RENDERIZADO DE MENSAJES
# ---------------------------------------------------------------------------
def render_chat_message(content):
    """Renderiza contenido del mensaje soportando QuickChart y [MEDIA:base64]."""
    content = re.sub(r'!\[.*?\]\((https://quickchart\.io/chart.*?)\)', r'\1', content)

    current_text = content
    while "https://quickchart.io/chart" in current_text:
        start_idx = current_text.find("https://quickchart.io/chart")
        if start_idx > 0:
            st.markdown(current_text[:start_idx])

        rest = current_text[start_idx:]
        brace_count, in_braces, end_idx = 0, False, 0
        for i, char in enumerate(rest):
            if char == '{':
                in_braces = True
                brace_count += 1
            elif char == '}':
                brace_count -= 1
                if in_braces and brace_count == 0:
                    end_idx = i + 1
                    break

        if end_idx == 0 or not in_braces:
            match = re.search(r'[\s"\'\n\)]', rest)
            end_idx = match.start() if match else len(rest)

        url = rest[:end_idx]
        try:
            st.image(url.replace(" ", "%20"))
        except Exception:
            st.markdown(f"**Gráfico:** [Ver aquí]({url.replace(' ', '%20')})")

        current_text = rest[end_idx:]
        if current_text.startswith(')'):
            current_text = current_text[1:]

    media_match = re.search(r'\[MEDIA:(.+?)\]', current_text)
    if media_match:
        base64_data  = media_match.group(1)
        current_text = re.sub(r'\[MEDIA:(.+?)\]', '', current_text).strip()
        try:
            st.image(f"data:image/png;base64,{base64_data}")
        except Exception as e:
            st.error(f"Error al renderizar imagen: {e}")

    if current_text.strip():
        st.markdown(current_text)


# P2-B: Timestamp helper
def render_timestamp(ts: str, role: str):
    if not ts:
        return
    css_class = "msg-ts-user" if role == "user" else "msg-ts-assistant"
    st.markdown(
        f'<div class="msg-ts {css_class}">{ts}</div>',
        unsafe_allow_html=True,
    )


# P2-C: Feedback
def send_feedback(msg_idx: int, sentiment: str):
    st.session_state.feedback[msg_idx] = sentiment
    try:
        requests.post(
            f"{BACKEND_URL}/feedback",
            json={
                "conversation_id": st.session_state.conversation_id,
                "message_index": msg_idx,
                "sentiment": sentiment,
            },
            headers=_api_headers(),
            timeout=3,
        )
    except Exception:
        pass
    st.rerun()


# ---------------------------------------------------------------------------
# P1-B / P3-B — GESTIÓN DE CONVERSACIONES
# ---------------------------------------------------------------------------
def load_user_conversations() -> list:
    try:
        r = requests.get(
            f"{BACKEND_URL}/conversations",
            headers=_api_headers(), timeout=5,
        )
        if r.status_code == 200:
            convs = r.json().get("conversations", [])
            st.session_state.conversations = convs
            return convs
    except Exception:
        pass
    return []


def load_conversation_messages(conv_id: int) -> list:
    try:
        r = requests.get(
            f"{BACKEND_URL}/conversations/{conv_id}/messages",
            headers=_api_headers(), timeout=5,
        )
        if r.status_code == 200:
            return r.json().get("messages", [])
    except Exception:
        pass
    return []


def create_new_conversation(switch: bool = True):
    """Creates a new conversation and optionally switches to it."""
    try:
        r = requests.post(
            f"{BACKEND_URL}/conversations",
            json={"title": "Nueva conversación"},
            headers=_api_headers(), timeout=5,
        )
        if r.status_code == 200:
            data = r.json()
            new_conv = {
                "id": data["id"],
                "title": data["title"],
                "last_active_at": datetime.now().isoformat(),
                "created_at": datetime.now().isoformat(),
            }
            st.session_state.conversations.insert(0, new_conv)
            if switch:
                st.session_state.conversation_id = data["id"]
                st.session_state.messages = []
                st.session_state.feedback = {}
                st.session_state.last_failed_prompt = None
                st.session_state.retry_prompt = None
            return data["id"]
    except Exception as e:
        st.error(f"Error al crear conversación: {e}")
    return None


def switch_conversation(conv_id: int):
    st.session_state.conversation_id = conv_id
    st.session_state.messages = load_conversation_messages(conv_id)
    st.session_state.feedback = {}
    st.session_state.last_failed_prompt = None
    st.session_state.retry_prompt = None


def _sync_conv_title(conv_id: int, new_title: str):
    """Update the title in local conversations list after auto-titling."""
    for c in st.session_state.conversations:
        if c["id"] == conv_id and c["title"] == "Nueva conversación":
            c["title"] = new_title
            break


# ---------------------------------------------------------------------------
# LÓGICA DE ENVÍO (DRY)
# ---------------------------------------------------------------------------
_THINKING_HTML = (
    '<div class="thinking-wrap">'
    '<div class="thinking-dots"><span></span><span></span><span></span></div>'
    'Procesando\u2026'
    '</div>'
)


def _call_api(prompt: str):
    """Hace la llamada al backend y muestra la respuesta del asistente."""
    headers = {"Authorization": f"Bearer {st.session_state.jwt_token}"}
    payload = {
        "prompt": prompt,
        "history": st.session_state.messages,
        "conversation_id": st.session_state.conversation_id,
    }

    with st.chat_message("assistant", avatar="assistant"):
        placeholder = st.empty()
        placeholder.markdown(_THINKING_HTML, unsafe_allow_html=True)
        try:
            response = requests.post(f"{BACKEND_URL}/chat", json=payload, headers=headers)
            placeholder.empty()

            if response.status_code == 200:
                full_response = response.json().get("response", "Sin respuesta.")
                resp_ts = datetime.now().strftime("%H:%M")
                render_chat_message(full_response)
                render_timestamp(resp_ts, "assistant")
                st.session_state.messages.append(
                    {"role": "assistant", "content": full_response, "ts": resp_ts}
                )
                st.session_state.last_failed_prompt = None

                # P3-B: auto-update conversation title after first exchange
                if st.session_state.conversation_id and len(st.session_state.messages) == 2:
                    title = prompt[:50].strip() + ("…" if len(prompt) > 50 else "")
                    _sync_conv_title(st.session_state.conversation_id, title)
            else:
                try:
                    detail = response.json().get("detail", f"HTTP {response.status_code}")
                except ValueError:
                    detail = f"HTTP {response.status_code}"
                st.session_state.last_failed_prompt = prompt
                st.error(f"Error del backend: {detail}")

        except requests.exceptions.RequestException as e:
            placeholder.empty()
            st.session_state.last_failed_prompt = prompt
            st.error(f"No se pudo contactar al backend: {e}")


def send_message(prompt: str):
    """Añade el mensaje del usuario al historial y llama a la API."""
    ts = datetime.now().strftime("%H:%M")
    st.session_state.messages.append({"role": "user", "content": prompt, "ts": ts})
    with st.chat_message("user", avatar="user"):
        render_chat_message(prompt)
        render_timestamp(ts, "user")
    _call_api(prompt)


# ---------------------------------------------------------------------------
# PANTALLA PRINCIPAL
# ---------------------------------------------------------------------------
def mostrar_app_principal():
    inject_chat_css()
    inject_copy_js()   # P2-A

    # ---- P1-B: Load history on first render after login ----
    if not st.session_state.history_loaded:
        st.session_state.history_loaded = True
        convs = load_user_conversations()
        if convs:
            most_recent = convs[0]
            st.session_state.conversation_id = most_recent["id"]
            st.session_state.messages = load_conversation_messages(most_recent["id"])
        else:
            create_new_conversation(switch=True)

    # ---- Sidebar ----
    with st.sidebar:
        st.markdown(
            '<div class="sidebar-wordmark"><span class="sidebar-wordmark-dot"></span>Amael</div>',
            unsafe_allow_html=True,
        )

        if st.session_state.user_name:
            first   = st.session_state.user_name.split()[0]
            initial = first[0].upper()
            if st.session_state.user_picture:
                pic_html = f'<img src="{st.session_state.user_picture}" class="sidebar-profile-pic" alt="">'
            else:
                pic_html = f'<div class="sidebar-profile-pic-placeholder">{initial}</div>'
            st.markdown(
                f'<div class="sidebar-profile">{pic_html}'
                f'<div class="sidebar-profile-info">'
                f'<div class="sidebar-profile-name">{st.session_state.user_name}</div>'
                f'<div class="sidebar-profile-label">Cuenta Google</div>'
                f'</div></div>',
                unsafe_allow_html=True,
            )

        # ---- P3-B: Conversation list ----
        st.markdown('<div class="sidebar-section-label">Conversaciones</div>', unsafe_allow_html=True)
        if st.button("＋  Nueva conversación", use_container_width=True, key="btn_new_conv"):
            create_new_conversation(switch=True)
            st.rerun()

        for conv in st.session_state.conversations:
            conv_id    = conv["id"]
            title      = conv["title"]
            is_active  = conv_id == st.session_state.conversation_id
            short      = title[:34] + "…" if len(title) > 34 else title
            label      = f"◆ {short}" if is_active else f"   {short}"
            css_wrap   = "conv-active" if is_active else ""
            st.markdown(f'<div class="{css_wrap}">', unsafe_allow_html=True)
            if st.button(label, key=f"conv_{conv_id}", use_container_width=True):
                if not is_active:
                    switch_conversation(conv_id)
                    st.rerun()
            st.markdown('</div>', unsafe_allow_html=True)

        theme_label = "☀  Modo claro" if st.session_state.theme == "dark" else "🌙  Modo oscuro"
        st.button(theme_label, on_click=toggle_theme, use_container_width=True)

        st.markdown('<div class="sidebar-section-label">Base de conocimiento</div>', unsafe_allow_html=True)
        uploaded_file = st.file_uploader(
            "PDF, TXT o Imagen", type=["pdf", "txt", "png", "jpg", "jpeg"],
            label_visibility="collapsed",
        )
        if uploaded_file is not None:
            is_image    = uploaded_file.name.lower().endswith(('.png', '.jpg', '.jpeg'))
            spinner_msg = "Analizando imagen..." if is_image else "Procesando documento..."
            with st.spinner(spinner_msg):
                files   = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
                headers = {"Authorization": f"Bearer {st.session_state.jwt_token}"}
                try:
                    endpoint = f"{BACKEND_URL}/analyze-image" if is_image else f"{BACKEND_URL}/ingest"
                    resp = requests.post(endpoint, files=files, headers=headers)
                    if resp.status_code == 200:
                        if is_image:
                            data      = resp.json()
                            detalles  = data.get("analisis_detallado", data.get("analysis_result", []))
                            resumen   = data.get("resumen_ia", "Completado sin resumen.")
                            st.success("Imagen analizada")
                            st.json(detalles)
                            ts = datetime.now().strftime("%H:%M")
                            report = (f"He subido una imagen. El modelo detectó: {resumen}. ¿Puedes interpretarlo?")
                            st.session_state.messages.append({"role": "user",      "content": report, "ts": ts})
                            st.session_state.messages.append({"role": "assistant", "content": f"He recibido el análisis: {resumen}. ¿Qué quieres saber?", "ts": ts})
                        else:
                            st.success(resp.json().get("message", "Documento procesado."))
                    else:
                        st.error(f"Error: {resp.json().get('detail', 'Desconocido')}")
                except requests.exceptions.RequestException as e:
                    st.error(f"Error de conexión: {e}")

        st.markdown('<div class="sidebar-section-label">Sesión</div>', unsafe_allow_html=True)
        if st.button("→  Cerrar sesión", use_container_width=True):
            for key in ("jwt_token", "messages", "user_name", "user_picture",
                        "last_failed_prompt", "retry_prompt", "conversation_id",
                        "conversations", "history_loaded", "feedback"):
                if key in ("messages", "conversations"):
                    st.session_state[key] = []
                elif key == "feedback":
                    st.session_state[key] = {}
                elif key == "history_loaded":
                    st.session_state[key] = False
                else:
                    st.session_state[key] = None
            st.rerun()

    # ---- Empty state ----
    if not st.session_state.messages:
        first    = (st.session_state.user_name or "").split()[0] if st.session_state.user_name else ""
        greeting = f"Hola, {first}" if first else "Hola"
        st.markdown(
            f'<div class="empty-state">'
            f'<div class="empty-icon">◆</div>'
            f'<div class="empty-title">{greeting}</div>'
            f'<div class="empty-subtitle">¿En qué puedo ayudarte hoy?</div>'
            f'<div class="empty-chips">'
            f'<span class="empty-chip">🔍 Consultas sobre Kubernetes</span>'
            f'<span class="empty-chip">📅 Organizar mi agenda</span>'
            f'<span class="empty-chip">📄 Analizar documentos</span>'
            f'<span class="empty-chip">📊 Generar gráficos</span>'
            f'</div></div>',
            unsafe_allow_html=True,
        )

    # ---- Historial de mensajes ----
    for i, message in enumerate(st.session_state.messages):
        role = message["role"]
        ts   = message.get("ts", "")
        with st.chat_message(role, avatar="user" if role == "user" else "assistant"):
            render_chat_message(message["content"])
        render_timestamp(ts, role)

        # P2-C: Feedback buttons under assistant messages
        if role == "assistant":
            fb = st.session_state.feedback.get(i)
            if fb is None:
                col_pos, col_neg, _ = st.columns([1, 1, 8])
                with col_pos:
                    if st.button("👍", key=f"fb_pos_{i}", help="Respuesta útil"):
                        send_feedback(i, "positive")
                with col_neg:
                    if st.button("👎", key=f"fb_neg_{i}", help="Respuesta no útil"):
                        send_feedback(i, "negative")
            else:
                icon = "👍" if fb == "positive" else "👎"
                st.markdown(f'<div class="fb-given">{icon} Gracias por tu feedback</div>', unsafe_allow_html=True)

    # ---- P2-D: Retry — si hay un prompt pendiente ----
    if st.session_state.retry_prompt:
        prompt_to_retry               = st.session_state.retry_prompt
        st.session_state.retry_prompt = None
        _call_api(prompt_to_retry)

    # ---- P2-D: Barra de error ----
    if st.session_state.last_failed_prompt:
        st.markdown(
            '<div class="retry-bar">No se pudo obtener respuesta para el último mensaje.</div>',
            unsafe_allow_html=True,
        )
        col_retry, col_discard, _ = st.columns([2, 2, 6])
        with col_retry:
            if st.button("⟳ Reintentar", use_container_width=True, key="btn_retry"):
                st.session_state.retry_prompt       = st.session_state.last_failed_prompt
                st.session_state.last_failed_prompt = None
                st.rerun()
        with col_discard:
            if st.button("✕ Descartar", use_container_width=True, key="btn_discard"):
                if st.session_state.messages and st.session_state.messages[-1]["role"] == "user":
                    st.session_state.messages.pop()
                st.session_state.last_failed_prompt = None
                st.rerun()

    # ---- Chat input ----
    if prompt := st.chat_input("Pregunta algo a Amael..."):
        send_message(prompt)


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
def main():
    initialize_session_state()
    inject_base_css()
    inject_theme_vars(get_tokens())

    if check_authentication():
        mostrar_app_principal()
    else:
        mostrar_pantalla_login()


if __name__ == "__main__":
    main()
