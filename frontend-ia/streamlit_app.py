import streamlit as st
import requests
from urllib.parse import urlparse, parse_qs
import streamlit.components.v1 as components

# --- CONFIGURACIÓN DE LA PÁGINA DE STREAMLIT ---
st.set_page_config(
    page_title="Amael-IA | Tu Asistente Inteligente",
    page_icon="✨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CONFIGURACIÓN ---
BACKEND_URL = "https://amael-ia.richardx.dev/api"

# Inject Global CSS for fonts and general resets
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    /* Ocultar elementos por defecto de Streamlit */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    /* header {visibility: hidden;} */
    header { background-color: transparent !important; border-bottom: none !important; }
    [data-testid="stHeader"] { background-color: transparent !important; }
    
    .stDeployButton {display:none;}
    
    /* Mejorar el scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }
    ::-webkit-scrollbar-track {
        background: transparent;
    }
    ::-webkit-scrollbar-thumb {
        background: var(--text-secondary, #cbd5e1);
        border-radius: 4px;
    }
    ::-webkit-scrollbar-thumb:hover {
        background: var(--text-primary, #94a3b8);
    }
</style>
""", unsafe_allow_html=True)

# --- ESTADO DE SESIÓN ---
def initialize_session_state():
    if "jwt_token" not in st.session_state:
        st.session_state.jwt_token = None
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "user_name" not in st.session_state:
        st.session_state.user_name = None
    if "user_picture" not in st.session_state:
        st.session_state.user_picture = None

# --- LÓGICA DE AUTENTICACIÓN ---
def check_authentication():
    # En nuevas versiones de Streamlit, st.query_params se comporta diferente
    # Extraemos asegurándonos que sea un string y no una lista
    jwt_token = st.query_params.get("token")
    if isinstance(jwt_token, list): jwt_token = jwt_token[0]
        
    error = st.query_params.get("error")
    
    user_name = st.query_params.get("name")
    if isinstance(user_name, list): user_name = user_name[0]
        
    user_picture = st.query_params.get("picture")
    if isinstance(user_picture, list): user_picture = user_picture[0]

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

def mostrar_pantalla_login():
    st.markdown("""
    <style>
        :root {
            --card-bg: rgba(255, 255, 255, 0.75);
            --text-primary: #1e293b;
            --text-secondary: #64748b;
            --btn-bg: #ffffff;
            --btn-border: #e2e8f0;
            --btn-hover-bg: #f8fafc;
        }
        @media (prefers-color-scheme: dark) {
            :root {
                --card-bg: rgba(30, 41, 59, 0.75);
                --text-primary: #f8fafc;
                --text-secondary: #94a3b8;
                --btn-bg: #0f172a;
                --btn-border: #334155;
                --btn-hover-bg: #1e293b;
            }
        }
    
        .stApp {
            background: linear-gradient(-45deg, #FF9A9E, #FECFEF, #A1C4FD, #C2E9FB);
            background-size: 400% 400%;
            animation: gradientBG 15s ease infinite;
        }
        
        @media (prefers-color-scheme: dark) {
            .stApp {
                background: linear-gradient(-45deg, #0f172a, #1e1b4b, #312e81, #1e293b);
                background-size: 400% 400%;
                animation: gradientBG 15s ease infinite;
            }
        }
        
        @keyframes gradientBG {
            0% {background-position: 0% 50%;}
            50% {background-position: 100% 50%;}
            100% {background-position: 0% 50%;}
        }
        .login-container {
            display: flex;
            justify-content: center;
            align-items: center;
            height: 100vh;
            margin-top: -80px; /* Offset the top padding */
        }
        .login-card {
            background: var(--card-bg);
            backdrop-filter: blur(16px);
            -webkit-backdrop-filter: blur(16px);
            border-radius: 24px;
            padding: 60px 40px;
            box-shadow: 0 10px 40px -10px rgba(0, 0, 0, 0.15);
            border: 1px solid rgba(255, 255, 255, 0.1);
            text-align: center;
            width: 100%;
            max-width: 440px;
            animation: fadeIn 0.8s cubic-bezier(0.2, 0.8, 0.2, 1);
        }
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(30px); }
            to { opacity: 1; transform: translateY(0); }
        }
        .login-card h1 { 
            margin-bottom: 8px; 
            color: var(--text-primary); 
            font-size: 36px; 
            font-weight: 800;
            letter-spacing: -0.02em;
        }
        .login-card p { 
            color: var(--text-secondary); 
            margin-bottom: 48px; 
            font-size: 17px; 
            font-weight: 400;
        }
        .google-btn {
            display: flex;
            align-items: center;
            justify-content: center;
            width: 100%;
            padding: 16px 24px;
            background-color: var(--btn-bg);
            color: var(--text-primary);
            border: 1px solid var(--btn-border);
            border-radius: 16px;
            font-size: 16px;
            font-weight: 600;
            cursor: pointer;
            text-decoration: none;
            transition: all 0.3s ease;
            box-shadow: 0 4px 6px -1px rgba(0,0,0,0.05), 0 2px 4px -1px rgba(0,0,0,0.03);
        }
        .google-btn:hover { 
            background-color: var(--btn-hover-bg); 
            border-color: var(--btn-border);
            box-shadow: 0 10px 15px -3px rgba(0,0,0,0.08), 0 4px 6px -2px rgba(0,0,0,0.04); 
            transform: translateY(-2px);
        }
        .google-btn img { margin-right: 14px; width: 24px; height: 24px; }
        .logo-icon {
            font-size: 56px;
            margin-bottom: 16px;
            display: inline-block;
            background: linear-gradient(135deg, #6366f1 0%, #a855f7 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            filter: drop-shadow(0 4px 6px rgba(99, 102, 241, 0.2));
        }
    </style>
    """, unsafe_allow_html=True)

    login_html = f"""
    <div class="login-container">
        <div class="login-card">
            <div class="logo-icon">✨</div>
            <h1>Amael-IA</h1>
            <p>Tu asistente inteligente personal</p>
            <a href="{BACKEND_URL}/auth/login" class="google-btn" target="_parent">
                <img src="https://www.svgrepo.com/show/475656/google-color.svg" alt="Google Logo">
                Continuar con Google
            </a>
        </div>
    </div>
    """
    st.markdown(login_html, unsafe_allow_html=True)

import re

def render_chat_message(content):
    # Limpiar si el LLM intentó usar markdown clásico de imagen
    content = re.sub(r'!\[.*?\]\((https://quickchart\.io/chart.*?)\)', r'\1', content)
    
    current_text = content
    while "https://quickchart.io/chart" in current_text:
        start_idx = current_text.find("https://quickchart.io/chart")
        
        if start_idx > 0:
            st.markdown(current_text[:start_idx])
            
        rest = current_text[start_idx:]
        
        # Parse for JSON object inside the quickchart URL (counts curly braces)
        brace_count = 0
        in_braces = False
        end_idx = 0
        
        for i, char in enumerate(rest):
            if char == '{':
                in_braces = True
                brace_count += 1
            elif char == '}':
                brace_count -= 1
                if in_braces and brace_count == 0:
                    end_idx = i + 1
                    break
                    
        # Fallback to normal delimiter if no braces
        if end_idx == 0 or not in_braces:
            match = re.search(r'[\s"\'\n\)]', rest)
            end_idx = match.start() if match else len(rest)
            
        url = rest[:end_idx]
        
        try:
            # Streamlit is safer with encoded spaces in URLs
            safe_url = url.replace(" ", "%20")
            st.image(safe_url)
        except Exception as e:
            st.markdown(f"**Gráfico:** [Ver aquí]({safe_url})")
             
        current_text = rest[end_idx:]
        
        # Extra parenthesis cleanup from markdown syntax
        if current_text.startswith(')'):
            current_text = current_text[1:]
            
    if current_text.strip():
        st.markdown(current_text)

def mostrar_app_principal():
    st.markdown("""
    <style>
        :root {
            --app-bg: #f8fafc;
            --glass-bg: rgba(255, 255, 255, 0.9);
            --text-primary: #334155;
            --text-heading: #0f172a;
            --text-secondary: #64748b;
            --border-color: #e2e8f0;
            --assistant-bg: #ffffff;
            --sidebar-bg: #f8fafc;
            --uploader-bg: #ffffff;
            --empty-icon-bg: #e0e7ff;
            --btn-bg: #ffffff;
        }
        @media (prefers-color-scheme: dark) {
            :root {
                --app-bg: #0f172a;
                --glass-bg: rgba(30, 41, 59, 0.9);
                --text-primary: #e2e8f0;
                --text-heading: #f8fafc;
                --text-secondary: #94a3b8;
                --border-color: #334155;
                --assistant-bg: #1e293b;
                --sidebar-bg: #0f172a;
                --uploader-bg: #1e293b;
                --empty-icon-bg: #312e81;
                --btn-bg: #1e293b;
            }
        }
        
        /* Ajuste general del layout principal */
        .block-container {
            padding-top: 2.5rem !important;
            padding-bottom: 6rem !important;
            max-width: 960px !important;
        }
        
        .stApp {
            background-color: var(--app-bg);
        }
        
        /* Contenedor del perfil en la esquina superior */
        .profile-container {
            display: flex;
            align-items: center;
            justify-content: flex-end;
            padding: 8px 16px;
            background: var(--glass-bg);
            backdrop-filter: blur(12px);
            border-radius: 100px;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05), 0 2px 4px -1px rgba(0, 0, 0, 0.03);
            position: fixed;
            top: 16px;
            right: 24px;
            z-index: 999;
            border: 1px solid var(--border-color);
            transition: all 0.3s ease;
        }
        .profile-container:hover {
            box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05);
            transform: translateY(-1px);
        }
        .profile-name {
            font-size: 14px;
            font-weight: 600;
            color: var(--text-primary);
            margin-right: 12px;
        }
        .profile-pic {
            width: 36px;
            height: 36px;
            border-radius: 50%;
            object-fit: cover;
            border: 2px solid var(--border-color);
        }
        
        /* Estilos de los mensajes de chat */
        .stChatMessage {
            background-color: transparent !important;
            border: none !important;
            padding: 0 !important;
            margin-bottom: 1.5rem;
        }
        
        /* Contenedor de cada mensaje */
        [data-testid="chat-message-container-user"] {
            flex-direction: row-reverse;
        }
        [data-testid="chat-message-container-user"] .stMarkdown {
            background: linear-gradient(135deg, #6366f1 0%, #4f46e5 100%);
            color: white;
            padding: 14px 20px;
            border-radius: 24px 24px 4px 24px;
            box-shadow: 0 4px 12px -2px rgba(99, 102, 241, 0.3);
            max-width: 85%;
            font-size: 15.5px;
            line-height: 1.6;
            margin-left: auto;
        }
        /* Ajustar color del texto dentro del burbuja user */
        [data-testid="chat-message-container-user"] .stMarkdown p {
            color: white !important;
            margin: 0;
        }
        /* Ocultar avatar de usuario para un look más moderno tipo iMessage (opcional, aquí solo lo hacemos más sutil) */
        [data-testid="chat-message-container-user"] .stAvatar {
            margin-left: 12px;
            margin-right: 0;
        }
        
        [data-testid="chat-message-container-assistant"] {
        }
        [data-testid="chat-message-container-assistant"] .stMarkdown {
            background-color: var(--assistant-bg);
            color: var(--text-primary);
            padding: 14px 20px;
            border-radius: 24px 24px 24px 4px;
            box-shadow: 0 4px 12px -2px rgba(0, 0, 0, 0.05);
            border: 1px solid var(--border-color);
            max-width: 90%;
            font-size: 15.5px;
            line-height: 1.6;
        }
        [data-testid="chat-message-container-assistant"] .stMarkdown p,
        [data-testid="chat-message-container-assistant"] .stMarkdown li,
        [data-testid="chat-message-container-assistant"] .stMarkdown span {
            color: var(--text-primary) !important;
        }
        [data-testid="chat-message-container-assistant"] .stMarkdown p {
            margin-bottom: 0.5rem;
        }
        [data-testid="chat-message-container-assistant"] .stMarkdown p:last-child {
            margin-bottom: 0;
        }
        
        /* Avatares */
        .stAvatar {
            display: flex;
            align-items: center;
            justify-content: center;
            background-color: var(--assistant-bg);
            box-shadow: 0 2px 4px rgba(0,0,0,0.05);
            border-radius: 50%;
        }
        
        /* Input del chat flotante */
        [data-testid="stChatInput"] {
            background: var(--glass-bg);
            backdrop-filter: blur(12px);
            border: 1px solid var(--border-color);
            border-radius: 30px;
            box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.1), 0 8px 10px -6px rgba(0, 0, 0, 0.1);
            padding: 6px 16px;
            margin-bottom: 20px; /* Separación del borde inferior */
            transition: all 0.3s ease;
        }
        [data-testid="stChatInput"] textarea {
            color: var(--text-primary) !important;
        }
        [data-testid="stChatInput"]:focus-within {
            border-color: #6366f1;
            box-shadow: 0 10px 25px -5px rgba(99, 102, 241, 0.2), 0 0 0 3px rgba(99, 102, 241, 0.1);
            transform: translateY(-2px);
        }
        
        /* Botones del Sidebar estilizados */
        .stButton button {
            width: 100%;
            border-radius: 16px;
            font-weight: 600;
            padding: 12px 20px;
            transition: all 0.3s ease;
            background-color: var(--btn-bg);
            border: 1px solid var(--border-color);
            color: var(--text-primary);
            box-shadow: 0 4px 6px -1px rgba(0,0,0,0.05), 0 2px 4px -1px rgba(0,0,0,0.03);
            display: flex;
            align-items: center;
            justify-content: center;
        }
        .stButton button p {
            font-size: 15px !important;
        }
        .stButton button:hover {
            border-color: #6366f1;
            color: #6366f1;
            background-color: var(--app-bg);
            box-shadow: 0 10px 15px -3px rgba(0,0,0,0.08), 0 4px 6px -2px rgba(0,0,0,0.04);
            transform: translateY(-2px);
        }
        
        /* Headers y texto en sidebar */
        [data-testid="stSidebar"] {
            background-color: var(--sidebar-bg);
            border-right: 1px solid var(--border-color);
        }
        [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2 {
            color: var(--text-heading);
        }
        
        /* Uploader area */
        [data-testid="stFileUploader"] {
            background-color: var(--uploader-bg);
            border-radius: 16px;
            padding: 1.5rem 1rem;
            border: 2px dashed var(--border-color);
            transition: all 0.3s ease;
            text-align: center;
            box-shadow: inset 0 2px 4px 0 rgba(0, 0, 0, 0.02);
        }
        [data-testid="stFileUploader"]:hover {
            border-color: #6366f1;
            background-color: var(--app-bg);
            box-shadow: inset 0 2px 4px 0 rgba(99, 102, 241, 0.05);
        }
        [data-testid="stFileUploader"] div, [data-testid="stFileUploader"] small {
            color: var(--text-primary) !important;
        }
        [data-testid="stFileUploader"] section > button {
            border-radius: 12px !important;
            background-color: var(--btn-bg) !important;
            border: 1px solid var(--border-color) !important;
            color: var(--text-primary) !important;
            font-weight: 500 !important;
            padding: 8px 16px !important;
            margin-top: 12px;
            box-shadow: 0 1px 2px 0 rgba(0, 0, 0, 0.05) !important;
        }
        [data-testid="stFileUploader"] section > button:hover {
            border-color: #6366f1 !important;
            color: #6366f1 !important;
            transform: translateY(-1px) !important;
            box-shadow: 0 4px 6px -1px rgba(0,0,0,0.05) !important;
        }
        
        /* Notificaciones toast top right adjustment */
        .stToastContainer {
            top: 70px !important;
        }
    </style>
    """, unsafe_allow_html=True)

    if st.session_state.user_picture and st.session_state.user_name:
        # Extraer solo el primer nombre para una apariencia más amigable
        first_name = st.session_state.user_name.split()[0]
        st.markdown(f"""
        <div class="profile-container">
            <span class="profile-name">Hola, {first_name}</span>
            <img src="{st.session_state.user_picture}" alt="Profile" class="profile-pic">
        </div>
        """, unsafe_allow_html=True)
    else:
        first_name = "ahí"

    # Empty state wrapper
    if not st.session_state.messages:
        st.markdown(f"""
        <div style="display:flex; flex-direction:column; align-items:center; justify-content:center; height:65vh; text-align:center; animation: fadeIn 0.8s ease;">
            <div style="width: 88px; height: 88px; background: linear-gradient(135deg, var(--empty-icon-bg) 0%, #c7d2fe 100%); border-radius: 50%; display:flex; align-items:center; justify-content:center; margin-bottom: 24px; box-shadow: 0 10px 25px -5px rgba(99,102,241,0.3);">
                <span style="font-size: 44px;">✨</span>
            </div>
            <h1 style="color:var(--text-heading); margin-bottom: 12px; font-weight: 800; font-size: 32px; letter-spacing: -0.02em;">¡Hola, {first_name}!</h1>
            <p style="color:var(--text-secondary); font-size: 18px; max-width: 500px; line-height: 1.6; padding: 0 20px;">
                ¿En qué puedo ayudarte hoy?<br>Pregúntame cualquier cosa o sube datos para que aprenda.
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    with st.sidebar:
        st.markdown("""
        <style>
            .sidebar-title { margin:0; font-weight: 700; color: var(--text-heading); font-size: 20px; }
            .sidebar-subtitle { color: var(--text-secondary); font-size: 13px; margin-top: 4px; }
            .sidebar-hr { margin: 1.5rem 0; border: none; border-top: 1px solid var(--border-color); }
            .sidebar-header-sm { color: var(--text-secondary); font-size: 12px; font-weight: 700; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 1rem; }
        </style>
        <div style="text-align: center; margin-bottom: 1.5rem;">
            <div style="font-size: 36px; margin-bottom: 8px;">✨</div>
            <h2 class="sidebar-title">Amael-IA</h2>
            <p class="sidebar-subtitle">Tu asistente inteligente</p>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("➕ Nueva Conversación", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

        st.markdown("<hr class='sidebar-hr'>", unsafe_allow_html=True)

        st.markdown("<h4 class='sidebar-header-sm'>🧠 Base de Conocimiento y Análisis Visual</h4>", unsafe_allow_html=True)
        uploaded_file = st.file_uploader("Sube PDF, TXT o Imágenes", type=["pdf", "txt", "png", "jpg", "jpeg"], help="Los documentos de texto se procesarán para el asistente. Las imágenes serán analizadas por TensorFlow.")
        
        if uploaded_file is not None:
            # Notificación visual más atractiva usando toast si es posible, o mantenemos success
            is_image = uploaded_file.name.lower().endswith(('.png', '.jpg', '.jpeg'))
            
            with st.spinner("Procesando archivo analíticamente..." if not is_image else "Analizando imagen con TensorFlow..."):
                files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
                headers = {"Authorization": f"Bearer {st.session_state.jwt_token}"}
                try:
                    # Determinar el endpoint basado en el tipo de archivo
                    endpoint = f"{BACKEND_URL}/analyze-image" if is_image else f"{BACKEND_URL}/ingest"
                    
                    response = requests.post(endpoint, files=files, headers=headers)
                    if response.status_code == 200:
                        if is_image:
                            analysis_result = response.json()
                            
                            # Extraer datos procesados por el backend
                            detalles = analysis_result.get("analisis_detallado", analysis_result.get("analysis_result", []))
                            resumen_ia = analysis_result.get("resumen_ia", "Completado sin resumen.")
                            
                            st.success("¡Imagen analizada exitosamente!")
                            st.json(detalles) # Mostrar el top 5 de probabilidades
                            
                            # Pasar el reporte semántico al historial del chat para análisis de la LLM
                            report_text = f"He subido una imagen para su análisis. El modelo de visión detectó lo siguiente: {resumen_ia}. Por favor interpreta estos resultados y dime de qué podría tratarse la imagen o qué elementos destacan."
                            st.session_state.messages.append({"role": "user", "content": report_text})
                            st.session_state.messages.append({"role": "assistant", "content": f"He recibido el análisis de la imagen que muestra: {resumen_ia}. ¿Qué te gustaría saber en detalle sobre estos elementos?"})
                        else:
                            st.success(response.json().get("message", "Documento procesado con éxito."))
                    else:
                        st.error(f"Error: {response.json().get('detail')}")
                except requests.exceptions.RequestException as e:
                    st.error(f"Error de conexión: {e}")

        st.markdown("<hr class='sidebar-hr'>", unsafe_allow_html=True)
        
        if st.button("🚪 Cerrar Sesión", use_container_width=True):
            st.session_state.jwt_token = None
            st.rerun()

    for message in st.session_state.messages:
        # Avatar emoji or keep clean
        avatar = "👤" if message["role"] == "user" else "✨"
        with st.chat_message(message["role"], avatar=avatar):
            render_chat_message(message["content"])

    if prompt := st.chat_input("Escribe tu pregunta o mensaje aquí..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user", avatar="👤"):
            render_chat_message(prompt)

        headers = {"Authorization": f"Bearer {st.session_state.jwt_token}"}
        payload = {
            "prompt": prompt, 
            "history": st.session_state.messages
        }
        
        # Ocupamos un contenedor general (no de chat) para la fase de "Pensando"
        spinner_placeholder = st.empty()
        error_placeholder = st.empty()
        
        try:
            with spinner_placeholder.container():
                with st.spinner("Pensando respuesta..."):
                    response = requests.post(f"{BACKEND_URL}/chat", json=payload, headers=headers)
            
            # Limpiamos el spinner
            spinner_placeholder.empty()

            if response.status_code == 200:
                full_response = response.json().get("response", "No pude generar una respuesta.")
                # Solo pintamos en el chat si tuvimos éxito
                with st.chat_message("assistant", avatar="✨"):
                    render_chat_message(full_response)
                st.session_state.messages.append({"role": "assistant", "content": full_response})
            else:
                try:
                    error_detail = response.json().get("detail", f"Error HTTP {response.status_code}")
                except ValueError:
                    error_detail = f"Error HTTP {response.status_code}: {response.text[:100]}"
                error_placeholder.error(f"Error del backend: {error_detail}")
        except requests.exceptions.RequestException as e:
            spinner_placeholder.empty()
            error_placeholder.error(f"No se pudo contactar al backend: {e}")

def main():
    initialize_session_state()
    
    if check_authentication():
        mostrar_app_principal()
    else:
        mostrar_pantalla_login()

if __name__ == "__main__":
    main()