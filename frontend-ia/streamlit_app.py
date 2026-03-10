import re
import streamlit as st
import requests

# --- CONFIGURACIÓN DE PÁGINA (debe ser el primer comando Streamlit) ---
st.set_page_config(
    page_title="Amael · Asistente Inteligente",
    page_icon="◆",
    layout="wide",
    initial_sidebar_state="expanded",
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
        "jwt_token":    None,
        "messages":     [],
        "user_name":    None,
        "user_picture": None,
        "theme":        "dark",
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val

def toggle_theme():
    st.session_state.theme = "light" if st.session_state.theme == "dark" else "dark"

def get_tokens():
    return THEMES[st.session_state.get("theme", "dark")]

# ---------------------------------------------------------------------------
# INYECCIÓN DE CSS
# ---------------------------------------------------------------------------
def inject_base_css():
    """CSS global: fuente, resets, scrollbar, ocultar chrome de Streamlit."""
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"], .stApp {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }

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
    """Inyecta los custom properties del tema activo en :root."""
    lines = "\n".join(f"    --{k.replace('_','-')}: {v};" for k, v in t.items())
    st.markdown(f"<style>:root {{\n{lines}\n}}</style>", unsafe_allow_html=True)


def inject_login_css():
    st.markdown("""
    <style>
    .stApp {
        background-color: var(--bg-base) !important;
    }
    .block-container {
        padding: 0 !important;
        max-width: 100% !important;
    }

    /* ---- Login card ---- */
    .login-wrap {
        min-height: 100vh;
        display: flex;
        align-items: center;
        justify-content: center;
        padding: 24px;
    }
    .login-card {
        width: 100%;
        max-width: 400px;
        background: var(--bg-surface);
        border: 1px solid var(--border);
        border-radius: 14px;
        padding: 44px 40px 36px;
        animation: fadeUp 0.35s cubic-bezier(.2,.8,.2,1);
    }
    @keyframes fadeUp {
        from { opacity: 0; transform: translateY(10px); }
        to   { opacity: 1; transform: translateY(0); }
    }
    .login-wordmark {
        font-size: 24px;
        font-weight: 700;
        color: var(--text-primary);
        letter-spacing: -0.5px;
        margin-bottom: 4px;
    }
    .login-tagline {
        font-size: 14px;
        color: var(--text-secondary);
        margin-bottom: 36px;
    }
    .login-divider {
        height: 1px;
        background: var(--border);
        margin: 28px 0;
    }
    .google-btn {
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 10px;
        width: 100%;
        padding: 13px 20px;
        background: #ffffff;
        color: #1f2937;
        font-size: 15px;
        font-weight: 500;
        border-radius: 8px;
        text-decoration: none;
        border: 1px solid #dadce0;
        transition: background 0.15s, box-shadow 0.15s;
        cursor: pointer;
    }
    .google-btn:hover {
        background: #f6f8fc;
        box-shadow: 0 1px 6px rgba(60,64,67,0.15);
        text-decoration: none;
    }
    .login-trust {
        display: flex;
        align-items: center;
        gap: 6px;
        justify-content: center;
        font-size: 12px;
        color: var(--text-disabled);
        margin-top: 18px;
    }
    </style>
    """, unsafe_allow_html=True)


def inject_chat_css():
    st.markdown("""
    <style>
    /* ---- Layout global ---- */
    .stApp { background-color: var(--bg-base) !important; }

    .block-container {
        padding-top: 2rem !important;
        padding-bottom: 7rem !important;
        max-width: 720px !important;
        margin: 0 auto !important;
    }

    /* ---- Sidebar ---- */
    [data-testid="stSidebar"] {
        background-color: var(--sidebar-bg) !important;
        border-right: 1px solid var(--border) !important;
    }
    [data-testid="stSidebar"] * { color: var(--text-primary) !important; }

    /* ---- Sidebar buttons ---- */
    [data-testid="stSidebar"] .stButton > button {
        width: 100%;
        background: var(--bg-elevated) !important;
        border: 1px solid var(--border) !important;
        border-radius: 8px !important;
        color: var(--text-primary) !important;
        font-size: 14px !important;
        font-weight: 500 !important;
        padding: 10px 16px !important;
        transition: background 0.15s, border-color 0.15s !important;
        box-shadow: none !important;
    }
    [data-testid="stSidebar"] .stButton > button:hover {
        background: var(--primary-subtle) !important;
        border-color: var(--primary) !important;
        color: var(--primary) !important;
        transform: none !important;
    }

    /* ---- File uploader ---- */
    [data-testid="stFileUploader"] {
        background: var(--bg-elevated);
        border: 1.5px dashed var(--border);
        border-radius: 10px;
        padding: 16px 12px;
        transition: border-color 0.15s;
    }
    [data-testid="stFileUploader"]:hover { border-color: var(--primary); }
    [data-testid="stFileUploader"] div,
    [data-testid="stFileUploader"] small,
    [data-testid="stFileUploader"] span { color: var(--text-secondary) !important; }
    [data-testid="stFileUploader"] section > button {
        border-radius: 6px !important;
        background: var(--bg-surface) !important;
        border: 1px solid var(--border) !important;
        color: var(--text-primary) !important;
        font-size: 13px !important;
        padding: 6px 14px !important;
    }

    /* ---- Mensajes: limpiar defaults de Streamlit ---- */
    .stChatMessage {
        background: transparent !important;
        border: none !important;
        padding: 0 !important;
        margin-bottom: 6px !important;
    }

    /* ---- Burbuja del USUARIO (derecha) ---- */
    [data-testid="chat-message-container-user"] {
        flex-direction: row-reverse;
    }
    [data-testid="chat-message-container-user"] [data-testid="stChatMessageAvatarUser"],
    [data-testid="chat-message-container-user"] .stAvatar {
        display: none !important;
    }
    [data-testid="chat-message-container-user"] [data-testid="stMarkdownContainer"],
    [data-testid="chat-message-container-user"] .stMarkdown {
        background: var(--user-bubble-bg) !important;
        border: 1px solid var(--user-bubble-border) !important;
        border-radius: 18px 18px 4px 18px !important;
        padding: 11px 16px !important;
        max-width: 82% !important;
        margin-left: auto !important;
        box-shadow: none !important;
        font-size: 15px !important;
        line-height: 1.6 !important;
        color: var(--text-primary) !important;
    }
    [data-testid="chat-message-container-user"] p {
        color: var(--text-primary) !important;
        margin: 0 !important;
    }

    /* ---- Mensajes del ASISTENTE (izquierda, sin burbuja — estilo Claude) ---- */
    [data-testid="chat-message-container-assistant"] [data-testid="stMarkdownContainer"],
    [data-testid="chat-message-container-assistant"] .stMarkdown {
        background: transparent !important;
        border: none !important;
        padding: 0 4px !important;
        box-shadow: none !important;
        font-size: 15px !important;
        line-height: 1.7 !important;
        color: var(--text-primary) !important;
        max-width: 100% !important;
    }
    [data-testid="chat-message-container-assistant"] p,
    [data-testid="chat-message-container-assistant"] li,
    [data-testid="chat-message-container-assistant"] span,
    [data-testid="chat-message-container-assistant"] td {
        color: var(--text-primary) !important;
    }
    [data-testid="chat-message-container-assistant"] p { margin-bottom: 0.6rem !important; }
    [data-testid="chat-message-container-assistant"] p:last-child { margin-bottom: 0 !important; }

    /* Code blocks */
    [data-testid="chat-message-container-assistant"] code {
        background: var(--bg-elevated) !important;
        color: var(--primary) !important;
        border-radius: 4px !important;
        padding: 1px 5px !important;
        font-size: 13px !important;
    }
    [data-testid="chat-message-container-assistant"] pre {
        background: var(--bg-surface) !important;
        border: 1px solid var(--border) !important;
        border-radius: 8px !important;
        padding: 14px 16px !important;
        overflow-x: auto !important;
    }
    [data-testid="chat-message-container-assistant"] pre code {
        background: transparent !important;
        color: var(--text-primary) !important;
        padding: 0 !important;
        font-size: 13px !important;
    }

    /* Avatar del asistente */
    [data-testid="chat-message-container-assistant"] .stAvatar,
    [data-testid="stChatMessageAvatarAssistant"] {
        width: 28px !important;
        height: 28px !important;
        min-width: 28px !important;
        border-radius: 7px !important;
        background: var(--primary) !important;
        color: #ffffff !important;
        font-size: 12px !important;
        font-weight: 700 !important;
        box-shadow: none !important;
    }

    /* ---- Thinking indicator ---- */
    .thinking-wrap {
        display: flex;
        align-items: center;
        gap: 8px;
        padding: 4px 0;
        color: var(--text-secondary);
        font-size: 13px;
    }
    .thinking-dots {
        display: flex;
        gap: 3px;
        align-items: center;
    }
    .thinking-dots span {
        display: block;
        width: 5px;
        height: 5px;
        border-radius: 50%;
        background: var(--primary);
        animation: dotPulse 1.3s ease-in-out infinite;
    }
    .thinking-dots span:nth-child(2) { animation-delay: 0.18s; }
    .thinking-dots span:nth-child(3) { animation-delay: 0.36s; }
    @keyframes dotPulse {
        0%, 60%, 100% { opacity: 0.25; transform: scale(0.75); }
        30%            { opacity: 1;    transform: scale(1);    }
    }

    /* ---- Input de chat ---- */
    [data-testid="stChatInput"] {
        background: var(--input-bg) !important;
        border: 1px solid var(--border) !important;
        border-radius: 14px !important;
        box-shadow: 0 0 0 0 transparent !important;
        padding: 4px 12px !important;
        transition: border-color 0.15s !important;
    }
    [data-testid="stChatInput"]:focus-within {
        border-color: var(--border-focus) !important;
        box-shadow: 0 0 0 3px var(--primary-subtle) !important;
    }
    [data-testid="stChatInput"] textarea {
        color: var(--text-primary) !important;
        background: transparent !important;
        font-size: 15px !important;
        caret-color: var(--primary) !important;
    }
    [data-testid="stChatInput"] textarea::placeholder {
        color: var(--text-disabled) !important;
    }

    /* ---- Separador de grupos de mensajes ---- */
    .msg-date-label {
        text-align: center;
        font-size: 11px;
        color: var(--text-disabled);
        margin: 20px 0 12px;
        letter-spacing: 0.04em;
        text-transform: uppercase;
    }

    /* ---- Empty state ---- */
    .empty-state {
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        height: 60vh;
        text-align: center;
        animation: fadeUp 0.4s ease;
        padding: 0 24px;
    }
    .empty-icon {
        width: 56px;
        height: 56px;
        background: var(--primary-subtle);
        border-radius: 14px;
        display: flex;
        align-items: center;
        justify-content: center;
        margin-bottom: 20px;
        font-size: 26px;
    }
    .empty-title {
        font-size: 22px;
        font-weight: 600;
        color: var(--text-primary);
        margin-bottom: 8px;
        letter-spacing: -0.3px;
    }
    .empty-subtitle {
        font-size: 15px;
        color: var(--text-secondary);
        max-width: 340px;
        line-height: 1.6;
        margin-bottom: 28px;
    }
    .empty-chips {
        display: flex;
        gap: 8px;
        flex-wrap: wrap;
        justify-content: center;
    }
    .empty-chip {
        background: var(--bg-elevated);
        border: 1px solid var(--border);
        border-radius: 20px;
        padding: 7px 14px;
        font-size: 13px;
        color: var(--text-secondary);
    }

    /* ---- Sidebar: perfil ---- */
    .sidebar-profile {
        display: flex;
        align-items: center;
        gap: 10px;
        padding: 12px 4px;
        border-bottom: 1px solid var(--border);
        margin-bottom: 16px;
    }
    .sidebar-profile-pic {
        width: 36px;
        height: 36px;
        border-radius: 50%;
        object-fit: cover;
        border: 1.5px solid var(--border);
        flex-shrink: 0;
    }
    .sidebar-profile-pic-placeholder {
        width: 36px;
        height: 36px;
        border-radius: 50%;
        background: var(--primary-subtle);
        border: 1.5px solid var(--border);
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 15px;
        font-weight: 600;
        color: var(--primary);
        flex-shrink: 0;
    }
    .sidebar-profile-info { min-width: 0; }
    .sidebar-profile-name {
        font-size: 14px;
        font-weight: 600;
        color: var(--text-primary);
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
    }
    .sidebar-profile-label {
        font-size: 11px;
        color: var(--text-secondary);
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }

    .sidebar-section-label {
        font-size: 11px;
        font-weight: 600;
        color: var(--text-disabled);
        text-transform: uppercase;
        letter-spacing: 0.06em;
        margin: 20px 0 8px;
        padding: 0 2px;
    }
    .sidebar-wordmark {
        font-size: 18px;
        font-weight: 700;
        color: var(--text-primary);
        letter-spacing: -0.3px;
        display: flex;
        align-items: center;
        gap: 7px;
        margin-bottom: 16px;
    }
    .sidebar-wordmark-dot {
        width: 8px;
        height: 8px;
        border-radius: 2px;
        background: var(--primary);
        display: inline-block;
    }

    /* ---- Toast positioning ---- */
    .stToastContainer { top: 16px !important; right: 16px !important; }

    @keyframes fadeUp {
        from { opacity: 0; transform: translateY(8px); }
        to   { opacity: 1; transform: translateY(0); }
    }
    </style>
    """, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# AUTENTICACIÓN
# ---------------------------------------------------------------------------
def check_authentication():
    jwt_token = st.query_params.get("token")
    if isinstance(jwt_token, list):
        jwt_token = jwt_token[0]

    error = st.query_params.get("error")

    user_name = st.query_params.get("name")
    if isinstance(user_name, list):
        user_name = user_name[0]

    user_picture = st.query_params.get("picture")
    if isinstance(user_picture, list):
        user_picture = user_picture[0]

    if jwt_token:
        st.session_state.jwt_token = jwt_token
        if user_name:
            st.session_state.user_name = user_name
        if user_picture:
            st.session_state.user_picture = user_picture
        st.query_params.clear()
        st.rerun()

    if error:
        st.error("No tienes permiso para acceder. Contacta al administrador.")
        st.stop()

    return st.session_state.jwt_token is not None


# ---------------------------------------------------------------------------
# PANTALLA DE LOGIN
# ---------------------------------------------------------------------------
def mostrar_pantalla_login():
    inject_login_css()

    google_svg = """
    <svg width="18" height="18" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
      <path fill="#4285F4" d="M22.56 12.25c0-.78-.07-1.53-.2-2.25H12v4.26h5.92c-.26 1.37-1.04 2.53-2.21 3.31v2.77h3.57c2.08-1.92 3.28-4.74 3.28-8.09z"/>
      <path fill="#34A853" d="M12 23c2.97 0 5.46-.98 7.28-2.66l-3.57-2.77c-.98.66-2.23 1.06-3.71 1.06-2.86 0-5.29-1.93-6.16-4.53H2.18v2.84C3.99 20.53 7.7 23 12 23z"/>
      <path fill="#FBBC05" d="M5.84 14.09c-.22-.66-.35-1.36-.35-2.09s.13-1.43.35-2.09V7.07H2.18C1.43 8.55 1 10.22 1 12s.43 3.45 1.18 4.93l3.66-2.84z"/>
      <path fill="#EA4335" d="M12 5.38c1.62 0 3.06.56 4.21 1.64l3.15-3.15C17.45 2.09 14.97 1 12 1 7.7 1 3.99 3.47 2.18 7.07l3.66 2.84c.87-2.6 3.3-4.53 6.16-4.53z"/>
    </svg>
    """

    lock_svg = """
    <svg width="12" height="12" fill="none" stroke="currentColor" stroke-width="2"
         viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
      <rect x="3" y="11" width="18" height="11" rx="2" ry="2"/>
      <path d="M7 11V7a5 5 0 0 1 10 0v4"/>
    </svg>
    """

    st.markdown(f"""
    <div class="login-wrap">
      <div class="login-card">
        <div class="login-wordmark">Amael</div>
        <div class="login-tagline">Tu asistente inteligente personal</div>

        <a href="{BACKEND_URL}/auth/login" class="google-btn" target="_parent">
          {google_svg}
          Continuar con Google
        </a>

        <div class="login-divider"></div>

        <div class="login-trust">
          {lock_svg}
          OAuth 2.0 · No almacenamos contraseñas
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# RENDERIZADO DE MENSAJES
# ---------------------------------------------------------------------------
def render_chat_message(content):
    """Renderiza el contenido de un mensaje, soportando QuickChart y base64."""
    content = re.sub(
        r'!\[.*?\]\((https://quickchart\.io/chart.*?)\)', r'\1', content
    )

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

    # Soporte para [MEDIA:base64]
    media_match = re.search(r'\[MEDIA:(.+?)\]', current_text)
    if media_match:
        base64_data = media_match.group(1)
        current_text = re.sub(r'\[MEDIA:(.+?)\]', '', current_text).strip()
        try:
            st.image(f"data:image/png;base64,{base64_data}")
        except Exception as e:
            st.error(f"Error al renderizar imagen: {e}")

    if current_text.strip():
        st.markdown(current_text)


# ---------------------------------------------------------------------------
# PANTALLA PRINCIPAL (CHAT)
# ---------------------------------------------------------------------------
def mostrar_app_principal():
    inject_chat_css()

    # -- Sidebar --
    with st.sidebar:
        # Wordmark
        st.markdown("""
        <div class="sidebar-wordmark">
          <span class="sidebar-wordmark-dot"></span>
          Amael
        </div>
        """, unsafe_allow_html=True)

        # Perfil
        if st.session_state.user_name:
            first = st.session_state.user_name.split()[0]
            initial = first[0].upper()
            if st.session_state.user_picture:
                pic_html = f'<img src="{st.session_state.user_picture}" class="sidebar-profile-pic" alt="">'
            else:
                pic_html = f'<div class="sidebar-profile-pic-placeholder">{initial}</div>'
            st.markdown(f"""
            <div class="sidebar-profile">
              {pic_html}
              <div class="sidebar-profile-info">
                <div class="sidebar-profile-name">{st.session_state.user_name}</div>
                <div class="sidebar-profile-label">Cuenta Google</div>
              </div>
            </div>
            """, unsafe_allow_html=True)

        # Controles
        st.markdown('<div class="sidebar-section-label">Conversación</div>', unsafe_allow_html=True)

        if st.button("＋  Nueva conversación", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

        # Tema
        theme_label = "☀  Modo claro" if st.session_state.theme == "dark" else "🌙  Modo oscuro"
        st.button(theme_label, on_click=toggle_theme, use_container_width=True)

        # Archivos
        st.markdown('<div class="sidebar-section-label">Base de conocimiento</div>', unsafe_allow_html=True)
        uploaded_file = st.file_uploader(
            "PDF, TXT o Imagen",
            type=["pdf", "txt", "png", "jpg", "jpeg"],
            label_visibility="collapsed",
        )

        if uploaded_file is not None:
            is_image = uploaded_file.name.lower().endswith(('.png', '.jpg', '.jpeg'))
            spinner_msg = "Analizando imagen..." if is_image else "Procesando documento..."
            with st.spinner(spinner_msg):
                files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
                headers = {"Authorization": f"Bearer {st.session_state.jwt_token}"}
                try:
                    endpoint = f"{BACKEND_URL}/analyze-image" if is_image else f"{BACKEND_URL}/ingest"
                    response = requests.post(endpoint, files=files, headers=headers)
                    if response.status_code == 200:
                        if is_image:
                            data = response.json()
                            detalles = data.get("analisis_detallado", data.get("analysis_result", []))
                            resumen_ia = data.get("resumen_ia", "Completado sin resumen.")
                            st.success("Imagen analizada")
                            st.json(detalles)
                            report = (
                                f"He subido una imagen para análisis. El modelo detectó: {resumen_ia}. "
                                "¿Puedes interpretarlo?"
                            )
                            st.session_state.messages.append({"role": "user", "content": report})
                            st.session_state.messages.append({
                                "role": "assistant",
                                "content": f"He recibido el análisis de la imagen: {resumen_ia}. ¿Qué te gustaría saber?"
                            })
                        else:
                            st.success(response.json().get("message", "Documento procesado."))
                    else:
                        st.error(f"Error: {response.json().get('detail', 'Desconocido')}")
                except requests.exceptions.RequestException as e:
                    st.error(f"Error de conexión: {e}")

        # Logout
        st.markdown('<div class="sidebar-section-label">Sesión</div>', unsafe_allow_html=True)
        if st.button("→  Cerrar sesión", use_container_width=True):
            for key in ("jwt_token", "messages", "user_name", "user_picture"):
                st.session_state[key] = None if key != "messages" else []
            st.rerun()

    # -- Empty state --
    if not st.session_state.messages:
        first = (st.session_state.user_name or "").split()[0] if st.session_state.user_name else ""
        greeting = f"Hola, {first}" if first else "Hola"
        st.markdown(f"""
        <div class="empty-state">
          <div class="empty-icon">◆</div>
          <div class="empty-title">{greeting}</div>
          <div class="empty-subtitle">
            ¿En qué puedo ayudarte hoy?
          </div>
          <div class="empty-chips">
            <span class="empty-chip">🔍 Consultas sobre Kubernetes</span>
            <span class="empty-chip">📅 Organizar mi agenda</span>
            <span class="empty-chip">📄 Analizar documentos</span>
            <span class="empty-chip">📊 Generar gráficos</span>
          </div>
        </div>
        """, unsafe_allow_html=True)

    # -- Historial de mensajes --
    for message in st.session_state.messages:
        if message["role"] == "user":
            with st.chat_message("user", avatar="✦"):
                render_chat_message(message["content"])
        else:
            with st.chat_message("assistant", avatar="◆"):
                render_chat_message(message["content"])

    # -- Input y envío --
    if prompt := st.chat_input("Pregunta algo a Amael..."):
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.chat_message("user", avatar="✦"):
            render_chat_message(prompt)

        headers = {"Authorization": f"Bearer {st.session_state.jwt_token}"}
        payload = {"prompt": prompt, "history": st.session_state.messages}

        with st.chat_message("assistant", avatar="◆"):
            msg_placeholder = st.empty()
            msg_placeholder.markdown("""
            <div class="thinking-wrap">
              <div class="thinking-dots">
                <span></span><span></span><span></span>
              </div>
              Procesando…
            </div>
            """, unsafe_allow_html=True)

            try:
                response = requests.post(
                    f"{BACKEND_URL}/chat", json=payload, headers=headers
                )
                msg_placeholder.empty()

                if response.status_code == 200:
                    full_response = response.json().get("response", "Sin respuesta.")
                    render_chat_message(full_response)
                    st.session_state.messages.append(
                        {"role": "assistant", "content": full_response}
                    )
                else:
                    try:
                        detail = response.json().get("detail", f"HTTP {response.status_code}")
                    except ValueError:
                        detail = f"HTTP {response.status_code}"
                    msg_placeholder.empty()
                    st.error(f"Error del backend: {detail}")

            except requests.exceptions.RequestException as e:
                msg_placeholder.empty()
                st.error(f"No se pudo contactar al backend: {e}")


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
