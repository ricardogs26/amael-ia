# app/services/auth_service.py

import os
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import Flow
from googleapiclient.discovery import build
from app.core.config import settings # <-- CAMBIO: Importamos settings

# Ruta para almacenar el token.json (podría ser en un volumen persistente)
TOKEN_PATH = "token.json"

def get_auth_flow():
    """Crea un flujo de autenticación de OAuth usando la configuración de settings."""
    flow = Flow.from_client_config(
        client_config={
            "web": {
                "client_id": settings.google_client_id,
                "client_secret": settings.google_client_secret,
                "auth_uri": "https://accounts.google.com/o/oauth2/auth",
                "token_uri": "https://oauth2.googleapis.com/token"
            }
        },
        scopes=[
            'https://www.googleapis.com/auth/calendar',
            'https://www.googleapis.com/auth/gmail.readonly'
        ],
        redirect_uri=settings.redirect_uri # <-- CAMBIO: Usa settings
    )
    return flow

def get_user_credentials(user_email: str) -> Credentials | None:
    """
    Obtiene las credenciales de un usuario desde un almacenamiento (ej. BD o archivo).
    Esta es una función placeholder que debes implementar.
    Deberías buscar las credenciales guardadas para el usuario dado.
    """
    # EJEMPLO: Leer desde un archivo (NO RECOMENDADO PARA PRODUCCIÓN)
    # En producción, esto debería ser una búsqueda en tu base de datos.
    if os.path.exists(TOKEN_PATH):
        creds = Credentials.from_authorized_user_file(TOKEN_PATH)
        # Aquí deberías verificar que las credenciales pertenecen al user_email
        return creds
    
    return None

def save_user_credentials(user_email: str, creds: Credentials):
    """
    Guarda las credenciales de un usuario en un almacenamiento (ej. BD o archivo).
    Esta es una función placeholder que debes implementar.
    """
    # EJEMPLO: Guardar en un archivo (NO RECOMENDADO PARA PRODUCCIÓN)
    # En producción, esto debería ser una inserción/actualización en tu base de datos.
    with open(TOKEN_PATH, 'w') as token:
        token.write(creds.to_json())