import streamlit as st
import requests
from urllib.parse import urlparse, parse_qs

# --- CONFIGURACIÓN ---
# VOLVEMOS A TU URL ORIGINAL, que funciona en tu entorno
BACKEND_URL = "https://amael-ia.richardx.dev/api"

# --- ESTADO DE SESIÓN ---
if "jwt_token" not in st.session_state:
    st.session_state.jwt_token = None
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- LÓGICA DE LOGIN ---
jwt_token = st.query_params.get("token")
error = st.query_params.get("error")

if jwt_token:
    st.session_state.jwt_token = jwt_token # Ya no es una lista, no necesita [0]
    st.query_params.clear() # La nueva forma de limpiar los parámetros

if error:
    st.error("No tienes permiso para acceder. Contacta al administrador.")
    st.stop()

if not st.session_state.jwt_token:
    st.title("🔐 Login Requerido")
    st.write("Por favor, inicia sesión para continuar.")
    # El botón redirige al endpoint de login de nuestro backend
    st.link_button("Iniciar sesión con Google", f"{BACKEND_URL}/auth/login")
    st.stop()

# --- LÓGICA DE LA APLICACIÓN PRINCIPAL ---
st.title("💬 Tu Agente de IA Personal")

headers = {"Authorization": f"Bearer {st.session_state.jwt_token}"}

# Sidebar para ingestión de datos
st.sidebar.title("📚 Aprender de tus Datos")
uploaded_file = st.sidebar.file_uploader("Sube un archivo (PDF o TXT)", type=["pdf", "txt"])

if uploaded_file is not None:
    with st.spinner("Procesando archivo..."):
        files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
        try:
            response = requests.post(f"{BACKEND_URL}/ingest", files=files, headers=headers)
            if response.status_code == 200:
                st.sidebar.success(response.json().get("message"))
            else:
                st.sidebar.error(f"Error: {response.json().get('detail')}")
        except requests.exceptions.RequestException as e:
            st.sidebar.error(f"Error de conexión: {e}")

# Lógica del chat
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
            response = requests.post(f"{BACKEND_URL}/chat", json={"prompt": prompt}, headers=headers)
            if response.status_code == 200:
                full_response = response.json().get("response", "No pude generar una respuesta.")
                message_placeholder.markdown(full_response)
                st.session_state.messages.append({"role": "assistant", "content": full_response})
            else:
                error_detail = response.json().get("detail", "Error desconocido.")
                message_placeholder.error(f"Error del backend: {error_detail}")
        except requests.exceptions.RequestException as e:
            message_placeholder.error(f"No se pudo contactar al backend: {e}")