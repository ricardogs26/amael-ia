import streamlit as st
import requests
import base64
from io import BytesIO

# --- CONFIGURACIÓN ---
# VOLVEMOS A TU URL ORIGINAL, que funciona en tu entorno
BACKEND_URL = "http://amael-ia.richardx.dev/api"

# --- ESTADO DE SESIÓN ---
# Inicializa todo el estado de la sesión aquí, al principio del script
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "headers" not in st.session_state:
    st.session_state.headers = None
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- LÓGICA DE LOGIN ---
if not st.session_state.logged_in:
    st.title("🔐 Login Agente Personal")
    username = st.text_input("Usuario")
    password = st.text_input("Contraseña", type="password")
    if st.button("Entrar"):
        credentials = f"{username}:{password}"
        encoded_credentials = base64.b64encode(credentials.encode("utf-8")).decode("utf-8")
        
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

# --- LÓGICA DE LA APLICACIÓN PRINCIPAL ---
else:
    st.title("💬 Amael IA Personal")
    
    # Botón para cerrar sesión
    if st.sidebar.button("Cerrar Sesión"):
        st.session_state.logged_in = False
        st.session_state.headers = None
        st.rerun()

    # Sidebar para ingestión de datos
    st.sidebar.title("📚 Aprender de tus Datos")
    uploaded_file = st.sidebar.file_uploader("Sube un archivo (PDF o TXT)", type=["pdf", "txt"])
    
    if uploaded_file is not None:
        # El spinner ahora está en el flujo principal, lo que evita el error
        with st.spinner("Procesando archivo..."):
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
