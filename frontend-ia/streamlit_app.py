import streamlit as st
import requests
import base64
from io import BytesIO

# --- CONFIGURACIÓN ---
BACKEND_URL = "http://amael-ia.richardx.dev/api" # URL pública de tu API
# Para desarrollo local, podrías usar http://localhost:8000/api con port-forward

# --- ESTADO DE SESIÓN ---
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

# --- FUNCIONES AUXILIARES ---
def show_login_page():
    """Muestra la página de login."""
    st.title("🔐 Login Agente Personal")
    username = st.text_input("Usuario")
    password = st.text_input("Contraseña", type="password")
    if st.button("Entrar"):
        # Codificar credenciales para Basic Auth
        credentials = f"{username}:{password}"
        encoded_credentials = base64.b64encode(credentials.encode("utf-8")).decode("utf-8")
        
        # Probar autenticación
        headers = {"Authorization": f"Basic {encoded_credentials}"}
        try:
            response = requests.get(f"{BACKEND_URL}/health", headers=headers, timeout=5)
            if response.status_code == 200:
                st.session_state.logged_in = True
                st.session_state.headers = headers
                st.success("Login exitoso!")
                st.rerun()
            else:
                st.error("Usuario o contraseña incorrectos.")
        except requests.exceptions.RequestException as e:
            st.error(f"No se pudo conectar al backend: {e}")

def show_main_app():
    """Muestra la aplicación principal después del login."""
    st.title("💬 Tu Agente de IA Personal")
    
    # Sidebar para ingestión de datos
    st.sidebar.title("📚 Aprender de tus Datos")
    uploaded_file = st.sidebar.file_uploader("Sube un archivo (PDF o TXT)", type=["pdf", "txt"])
    if uploaded_file is not None:
        with st.sidebar.spinner("Procesando archivo..."):
            files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
            try:
                response = requests.post(f"{BACKEND_URL}/ingest", files=files, headers=st.session_state.headers)
                if response.status_code == 200:
                    st.sidebar.success(response.json().get("message"))
                else:
                    st.sidebar.error(f"Error: {response.json().get('detail')}")
            except requests.exceptions.RequestException as e:
                st.sidebar.error(f"Error de conexión: {e}")

    # Lógica del chat
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("¿En qué puedo ayudarte?"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            try:
                response = requests.post(f"{BACKEND_URL}/chat", json={"prompt": prompt}, headers=st.session_state.headers)
                if response.status_code == 200:
                    full_response = response.json().get("response", "No pude generar una respuesta.")
                    message_placeholder.markdown(full_response)
                    st.session_state.messages.append({"role": "assistant", "content": full_response})
                else:
                    error_detail = response.json().get("detail", "Error desconocido.")
                    message_placeholder.error(f"Error del backend: {error_detail}")
            except requests.exceptions.RequestException as e:
                message_placeholder.error(f"No se pudo contactar al backend: {e}")

# --- FLUJO PRINCIPAL ---
if not st.session_state.logged_in:
    show_login_page()
else:
    show_main_app()
