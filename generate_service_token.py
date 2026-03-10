# generate_service_token.py
import os
from jose import jwt
from datetime import datetime, timedelta


# --- CONFIGURACIÓN ---
# Debe ser exactamente la misma SECRET_KEY y ALGORITHM que en tu main.py
# Ahora se requiere pasar como variable de exportación o entorno
SECRET_KEY = os.environ.get("JWT_SECRET_KEY")
if not SECRET_KEY:
    raise ValueError("Debes configurar la variable de entorno JWT_SECRET_KEY antes de ejecutar este script.")

ALGORITHM = "HS256"

# El email del "usuario de servicio" que añadiste a la lista blanca
SERVICE_USER_EMAIL = "bot-amael@richardx.dev"

# --- GENERACIÓN DEL TOKEN ---
def create_service_jwt_token(email: str):
    """Crea un token JWT para un servicio."""
    to_encode = {"sub": email}
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

if __name__ == "__main__":
    token = create_service_jwt_token(SERVICE_USER_EMAIL)
    print("--- Token JWT para el Bot de Amael-IA ---")
    print(token)
    print("----------------------------------------")
    print("Copia este token y úsalo en la configuración de tu bot.")

