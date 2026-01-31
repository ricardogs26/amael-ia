# services/image-analysis-service/main.py
from fastapi import FastAPI, HTTPException, Depends, UploadFile, File
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from jose import JWTError, jwt
import os
import requests
from PIL import Image
import numpy as np
import io

# --- CONFIGURACIÓN ---
app = FastAPI(title="Image Analysis Service")
security = HTTPBearer()
SECRET_KEY = os.environ.get("JWT_SECRET_KEY")
ALGORITHM = "HS256"
ALLOWED_EMAILS = [email.strip() for email in os.environ.get("ALLOWED_EMAILS_CSV", "").split(',') if email.strip()]

def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    # ... (misma función de validación de JWT) ...
    pass

@app.post("/analyze-image")
async def analyze_image(file: UploadFile = File(...), user: str = Depends(get_current_user)):
    try:
        # ... (copia la lógica de análisis de imagen de tu main.py original) ...
        # return {"analysis_result": predictions}
        pass
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error procesando la imagen: {e}")

@app.get("/health")
async def health_check():
    return {"status": "ok"}