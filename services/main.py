# services/auth-service/main.py
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.sessions import SessionMiddleware
from authlib.integrations.starlette_client import OAuth
from starlette.config import Config
from jose import jwt
from urllib.parse import urlencode
import os

# --- CONFIGURACIÓN ---
app = FastAPI(title="Auth Service")
# Mantén el middleware de CORS y sesión para el flujo de OAuth
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://amael-ia.richardx.dev"], # Especifica tu dominio
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(SessionMiddleware, secret_key=os.environ.get("SESSION_SECRET_KEY"))

config = Config(environ=os.environ)
oauth = OAuth(config)
oauth.register(
    name='google',
    server_metadata_url='https://accounts.google.com/.well-known/openid-configuration',
    client_kwargs={'scope': 'openid email profile'}
)

ALLOWED_EMAILS = [email.strip() for email in os.environ.get("ALLOWED_EMAILS_CSV", "").split(',') if email.strip()]
SECRET_KEY = os.environ.get("JWT_SECRET_KEY")
ALGORITHM = "HS256"

def create_jwt_token(email: str):
    to_encode = {"sub": email}
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

# --- ENDPOINTS DE AUTENTICACIÓN ---
@app.get('/login')
async def login(request: Request):
    redirect_uri = "https://amael-ia.richardx.dev/api/auth/callback" # Asegúrate de que esta URL sea pública
    return await oauth.google.authorize_redirect(request, redirect_uri)

@app.get('/callback')
async def auth_callback(request: Request):
    try:
        token = await oauth.google.authorize_access_token(request)
        user_info = token.get('userinfo')
        if user_info and user_info['email'] in ALLOWED_EMAILS:
            jwt_token = create_jwt_token(user_info['email'])
            frontend_url = "https://amael-ia.richardx.dev"
            
            params = {
                "token": jwt_token,
                "name": user_info.get('name', 'Usuario'),
                "picture": user_info.get('picture')
            }
            
            redirect_url = f"{frontend_url}?{urlencode(params)}"
            return Response(status_code=302, headers={"location": redirect_url})
        else:
            frontend_url = "https://amael-ia.richardx.dev"
            return Response(status_code=302, headers={"location": f"{frontend_url}?error=unauthorized"})

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error en la autenticación con Google: {e}")

@app.get("/health")
async def health_check():
    return {"status": "ok"}