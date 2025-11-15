import streamlit as st
import requests
from urllib.parse import urlparse, parse_qs

# --- CONFIGURACIÓN ---
BACKEND_URL = "https://amael-ia.richardx.dev/api"

# --- ESTADO DE SESIÓN ---
# Se inicializa el estado de la sesión si no existe
if "jwt_token" not in st.session_state:
    st.session_state.jwt_token = None
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- LÓGICA DE LOGIN (SE EJECUTA SIEMPRE) ---
jwt_token = st.query_params.get("token")
error = st.query_params.get("error")

if jwt_token:
    st.session_state.jwt_token = jwt_token
    st.query_params.clear() # Limpia la URL para que no quede el token

if error:
    st.error("No tienes permiso para acceder. Contacta al administrador.")
    st.stop()

# --- FLUJO PRINCIPAL DE LA APLICACIÓN ---
if not st.session_state.jwt_token:
    # --- PANTALLA DE LOGIN (SI NO HAY TOKEN) ---
    st.title("🔐 Login Requerido")
    st.write("Por favor, inicia sesión para continuar.")
    st.link_button("Iniciar sesión con Google", f"{BACKEND_URL}/auth/login")
    st.stop() # Detiene la ejecución aquí si no está logueado

# --- AQUÍ EMPIEZA LA APLICACIÓN SI EL USUARIO ESTÁ LOGUEADO (ELSE) ---
else:
    # 1. Definir las cabeceras de autenticación para usar en todas las peticiones
    headers = {"Authorization": f"Bearer {st.session_state.jwt_token}"}

    # 2. Título principal de la aplicación
    st.title("💬 Tu Agente de IA Personal")

    # 3. BARRA LATERAL (SIDEBAR)
    # Botón para nueva conversación
    if st.sidebar.button("🔄 Nueva Conversación"):
        st.session_state.messages = []
        st.success("¡Conversación reiniciada! Puedes empezar de nuevo.")
        st.rerun()

    # Botón para cerrar sesión
    if st.sidebar.button("Cerrar Sesión"):
        # Al poner el token a None, la próxima recarga mostrará la pantalla de login
        st.session_state.jwt_token = None
        st.rerun()

    # Sección para subir archivos y aprender de ellos
    st.sidebar.title("📚 Aprender de tus Datos")
    uploaded_file = st.sidebar.file_uploader("Sube un archivo (PDF o TXT)", type=["pdf", "txt"])

    if uploaded_file is not None:
        with st.sidebar.spinner("Procesando archivo..."):
            files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
            try:
                response = requests.post(f"{BACKEND_URL}/ingest", files=files, headers=headers)
                if response.status_code == 200:
                    st.sidebar.success(response.json().get("message"))
                else:
                    st.sidebar.error(f"Error: {response.json().get('detail')}")
            except requests.exceptions.RequestException as e:
                st.sidebar.error(f"Error de conexión: {e}")

    # 4. ÁREA PRINCIPAL DE CHAT
    # Mostrar los mensajes anteriores del historial
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Input para que el usuario escriba un nuevo mensaje
    if prompt := st.chat_input("¿En qué puedo ayudarte?"):
        # Añadir mensaje del usuario al historial y mostrarlo en pantalla
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Obtener respuesta del asistente y mostrarla
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