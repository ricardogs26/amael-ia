from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.responses import RedirectResponse
from pydantic import BaseModel
from typing import Optional
from app.core.config import settings
from app.services.planner_service import organize_day_for_user
from app.services.auth_service import get_auth_flow, save_user_credentials, get_user_credentials
from app.models.schemas import OrganizeDayResponse
from google.oauth2.credentials import Credentials

app = FastAPI(title="Productivity Microservice")

# P7: OpenTelemetry — server-side spans for service map (tracing.py at PYTHONPATH=/app)
from tracing import instrument_app
instrument_app(app)

# --- Seguridad para comunicación interna ---
security = HTTPBearer()

async def verify_internal_call(credentials: HTTPAuthorizationCredentials = Depends(security)):
    # CAMBIO: Usamos el secreto desde el objeto settings
    if credentials.credentials != settings.internal_api_secret:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid internal API secret"
        )

# --- Endpoints ---
@app.get("/health")
async def health_check():
    return {"status": "ok"}

@app.post("/organize", response_model=OrganizeDayResponse, dependencies=[Depends(verify_internal_call)])
async def organize_day_endpoint(user_email: str):
    """
    Endpoint principal que orquesta la organización del día para un usuario.
    Espera un Bearer Token con el secreto interno en la cabecera Authorization.
    """
    try:
        result = await organize_day_for_user(user_email)
        return result
    except Exception as e:
        print(f"Error in productivity service for user {user_email}: {e}")
        raise HTTPException(status_code=500, detail="Failed to organize day.")


class CredentialsPayload(BaseModel):
    user_email: str
    token: Optional[str] = None
    refresh_token: str
    token_uri: str = "https://oauth2.googleapis.com/token"
    client_id: str
    client_secret: str
    scopes: list[str]


@app.post("/credentials", dependencies=[Depends(verify_internal_call)])
async def save_credentials(payload: CredentialsPayload):
    """Guarda los tokens OAuth de Google Calendar en Vault para el usuario."""
    try:
        creds = Credentials(
            token=payload.token,
            refresh_token=payload.refresh_token,
            token_uri=payload.token_uri,
            client_id=payload.client_id,
            client_secret=payload.client_secret,
            scopes=payload.scopes,
        )
        save_user_credentials(payload.user_email, creds)
        return {"ok": True, "user_email": payload.user_email}
    except Exception as e:
        print(f"Error saving credentials for {payload.user_email}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/credentials/status", dependencies=[Depends(verify_internal_call)])
async def credentials_status(user_email: str):
    """Verifica si el usuario tiene credenciales de Google Calendar en Vault."""
    creds = get_user_credentials(user_email)
    return {"connected": creds is not None and (creds.valid or creds.refresh_token is not None)}