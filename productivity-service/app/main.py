from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from app.core.config import settings  # <-- CAMBIO: Importamos settings
from app.services.planner_service import organize_day_for_user
from app.models.schemas import OrganizeDayResponse

app = FastAPI(title="Productivity Microservice")

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
        # Loguea el error para depuración
        print(f"Error in productivity service for user {user_email}: {e}")
        raise HTTPException(status_code=500, detail="Failed to organize day.")