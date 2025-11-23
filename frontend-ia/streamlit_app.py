import streamlit as st
import requests
from urllib.parse import urlparse, parse_qs

# --- CONFIGURACIÓN DE LA PÁGINA DE STREAMLIT ---
# ESTA LÍNEA DEBE SER LA PRIMERA COMANDO DE STREAMLIT EN EL SCRIPT
st.set_page_config(
    page_title="Amael-IA",
    page_icon="🤖", # Puedes usar un emoji o un archivo de imagen local
    layout="wide"   # Opciones: "centered" o "wide"
)

# --- CONFIGURACIÓN ---
BACKEND_URL = "https://amael-ia.richardx.dev/api"

# --- ESTADO DE SESIÓN ---
# Inicializa el estado de la sesión si no existe
def initialize_session_state():
    if "jwt_token" not in st.session_state:
        st.session_state.jwt_token = None
    if "messages" not in st.session_state:
        st.session_state.messages = []

# --- LÓGICA DE AUTENTICACIÓN ---
def check_authentication():
    """Verifica si el token está en los parámetros de la URL o en el estado de la sesión."""
    jwt_token = st.query_params.get("token")
    error = st.query_params.get("error")

    if jwt_token:
        st.session_state.jwt_token = jwt_token
        st.query_params.clear()
        st.rerun() # Recarga la app para que los cambios de sesión surtan efecto

    if error:
        st.error("No tienes permiso para acceder. Contacta al administrador.")
        st.stop()

    return st.session_state.jwt_token is not None

# --- PANTALLA DE LOGIN ---
def mostrar_pantalla_login():
    """Muestra la pantalla de login con un diseño personalizado."""
    st.markdown("""
    <style>
        .main-header {visibility: hidden;}
        .stDeployButton {display:none;}
        .stAppViewContainer { margin-top: -2.5rem; }
        .login-container {
            display: flex;
            justify-content: center;
            align-items: center;
            height: 80vh;
        }
        .login-card {
            padding: 40px;
            border-radius: 15px;
            box-shadow: 0 10px 30px rgba(0, 0, 0, 0.1);
            text-align: center;
            background-color: #ffffff;
            width: 100%;
            max-width: 400px;
            border: 1px solid #e0e0e0;
        }
        .login-card h1 {
            margin-bottom: 10px;
            color: #1a1a1a;
            font-size: 28px;
        }
        .login-card p {
            color: #666;
            margin-bottom: 30px;
            font-size: 16px;
        }
        .google-btn {
            display: inline-flex;
            align-items: center;
            justify-content: center;
            width: 100%;
            padding: 14px 20px;
            background-color: #4285F4;
            color: white;
            border: none;
            border-radius: 8px;
            font-size: 16px;
            font-weight: 500;
            cursor: pointer;
            text-decoration: none;
            transition: background-color 0.3s, box-shadow 0.3s;
        }
        .google-btn:hover {
            background-color: #357AE8;
            box-shadow: 0 4px 12px rgba(66, 133, 244, 0.4);
        }
        .google-btn svg {
            margin-right: 12px;
        }
    </style>
    """, unsafe_allow_html=True)

    login_html = f"""
    <div class="login-container">
        <div class="login-card">
            <h1>🤖 Bienvenido a Amael-IA</h1>
            <p>Inicia sesión para comenzar a chatear.</p>
            <a href="{BACKEND_URL}/auth/login" class="google-btn">
                <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 48 48" width="24px" height="24px">
                    <path fill="#FFC107" d="M43.611,20.083H42V20H24v8h11.303c-1.649,4.657-6.083,8-11.303,8c-6.627,0-12-5.373-12-12c0-6.627,5.373-12,12-12c3.059,0,5.842,1.154,7.961,3.039l5.657-5.657C34.046,6.053,29.268,4,24,4C12.955,4,4,12.955,4,24c0,11.045,8.955,20,20,20c11.045,0,20-8.955,20-20C44,22.659,43.862,21.35,43.611,20.083z"/>
                    <path fill="#FF3D00" d="M6.306,14.691l6.571,4.819C14.655,15.108,18.961,12,24,12c3.059,0,5.842,1.154,7.961,3.039l5.657-5.657C34.046,6.053,29.268,4,24,4C16.318,4,9.656,8.337,6.306,14.691z"/>
                    <path fill="#4CAF50" d="M24,44c5.166,0,9.86-1.977,13.409-5.192l-6.19-5.238C29.211,35.091,26.715,36,24,36c-5.202,0-9.619-3.317-11.283-7.946l-6.522,5.024C9.505,39.556,16.227,44,24,44z"/>
                    <path fill="#1976D2" d="M43.611,20.083H42V20H24v8h11.303c-0.792,2.237-2.231,4.166-4.087,5.571c0.001-0.001,0.002-0.001,0.003-0.002l6.19,5.238C36.971,39.205,44,34,44,24C44,22.659,43.862,21.35,43.611,20.083z"/>
                </svg>
                Iniciar sesión con Google
            </a>
        </div>
    </div>
    """
    st.markdown(login_html, unsafe_allow_html=True)

# --- APLICACIÓN PRINCIPAL ---
def mostrar_app_principal():
    """Muestra la interfaz principal de la aplicación después del login."""
    # Oculta elementos de Streamlit para una vista más limpia
    hide_st_style = """
                <style>
                #MainMenu {visibility: hidden;}
                footer {visibility: hidden;}
                header {visibility: hidden;}
                .stDeployButton {display:none;}
                </style>
                """
    st.markdown(hide_st_style, unsafe_allow_html=True)

    # CSS para estilizar el chat
    st.markdown("""
    <style>
        .stChatMessage {
            border-radius: 10px;
            padding: 12px;
            margin-bottom: 10px;
        }
        .stChatMessage[data-testid="chat-message-container-user"] {
            background-color: #E3F2FD; /* Azul claro para el usuario */
        }
        .stChatMessage[data-testid="chat-message-container-assistant"] {
            background-color: #F1F3F4; /* Gris claro para el asistente */
        }
    </style>
    """, unsafe_allow_html=True)

    st.title("💬 Amael-IA")

    # --- Barra Lateral ---
    with st.sidebar:
        st.title("⚙️ Configuración")

        if st.button("🔄 Nueva Conversación"):
            st.session_state.messages = []
            st.success("¡Conversación reiniciada!")
            st.rerun()

        st.markdown("---") # Separador

        if st.button("🚪 Cerrar Sesión"):
            st.session_state.jwt_token = None
            st.info("Has cerrado sesión.")
            st.rerun()
        
        st.markdown("---") # Separador

        st.header("📚 Aprender de tus Datos")
        uploaded_file = st.file_uploader("Sube un archivo (PDF o TXT)", type=["pdf", "txt"])
        
        if uploaded_file is not None:
            with st.spinner("Procesando archivo..."):
                files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
                headers = {"Authorization": f"Bearer {st.session_state.jwt_token}"}
                try:
                    response = requests.post(f"{BACKEND_URL}/ingest", files=files, headers=headers)
                    if response.status_code == 200:
                        st.sidebar.success(response.json().get("message"))
                    else:
                        st.sidebar.error(f"Error: {response.json().get('detail')}")
                except requests.exceptions.RequestException as e:
                    st.sidebar.error(f"Error de conexión: {e}")

    # --- Interfaz de Chat ---
    # Mostrar mensajes anteriores
    for message in st.session_state.messages:
        with st.chat_message(message["role"], avatar="🧑‍💻" if message["role"] == "user" else "🤖"):
            st.markdown(message["content"])

    # Input del usuario
    if prompt := st.chat_input("¿En qué puedo ayudarte?"):
        # Añadir mensaje del usuario al estado y mostrarlo
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user", avatar="🧑‍💻"):
            st.markdown(prompt)

        # Obtener respuesta del asistente
        with st.chat_message("assistant", avatar="🤖"):
            message_placeholder = st.empty()
            headers = {"Authorization": f"Bearer {st.session_state.jwt_token}"}
            try:
                payload = {
                    "prompt": prompt, 
                    "history": st.session_state.messages
                }
                response = requests.post(f"{BACKEND_URL}/chat", json=payload, headers=headers)
                
                if response.status_code == 200:
                    full_response = response.json().get("response", "No pude generar una respuesta.")
                    message_placeholder.markdown(full_response)
                    st.session_state.messages.append({"role": "assistant", "content": full_response})
                else:
                    error_detail = response.json().get("detail", "Error desconocido.")
                    message_placeholder.error(f"Error del backend: {error_detail}")
            except requests.exceptions.RequestException as e:
                message_placeholder.error(f"No se pudo contactar al backend: {e}")

# --- FUNCIÓN PRINCIPAL ---
def main():
    """Función principal que orquesta la aplicación."""
    initialize_session_state()
    
    if check_authentication():
        mostrar_app_principal()
    else:
        mostrar_pantalla_login()

# Ejecutar la aplicación
if __name__ == "__main__":
    main()