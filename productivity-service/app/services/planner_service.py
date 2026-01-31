import json
import datetime
from googleapiclient.discovery import build
from langchain_ollama import OllamaLLM
from app.core.config import settings  # <-- CAMBIO: Importamos settings
from app.services.auth_service import get_user_credentials
from app.services.google_apis import get_todays_events, get_unread_emails

# CAMBIO: Inicializar el LLM usando la configuración de settings
llm = OllamaLLM(model=settings.model_name, base_url=settings.ollama_base_url)

# ... (el prompt PLANNING_PROMPT_TEMPLATE se mantiene igual) ...
PLANNING_PROMPT_TEMPLATE = """
Eres un asistente de productividad experto...
... (resto del prompt)
"""

def sync_plan_to_calendar(service, plan_data):
    """Crea los eventos del plan en el calendario de Google."""
    # ... (Implementación de la función) ...
    return 0 # Placeholder

async def organize_day_for_user(user_email: str):
    """Función principal que orquesta todo el proceso."""
    # 1. Obtener credenciales del usuario
    credentials = get_user_credentials(user_email)
    if not credentials:
        return {"summary": "Error: No se encontraron credenciales para este usuario. Por favor, autoriza el acceso a Google.", "tasks_created": 0}

    # 2. Obtener servicios de Google
    calendar_service = build('calendar', 'v3', credentials=credentials)
    gmail_service = build('gmail', 'v1', credentials=credentials)

    # 3. Recuperar datos
    events = get_todays_events(calendar_service)
    emails = get_unread_emails(gmail_service)

    if not events and not emails:
        return {"summary": "¡Tu día está completamente libre! No hay nada que organizar por ahora.", "tasks_created": 0}

    # 4. Analizar y planificar con IA
    context_string = f"EVENTOS: {events}\n\nEMAILS: {emails}" # Simplificado para el ejemplo
    final_prompt = PLANNING_PROMPT_TEMPLATE.format(context=context_string)
    
    response = llm.invoke(final_prompt)
    
    try:
        plan_data = json.loads(response)
    except json.JSONDecodeError:
        return {"summary": "Error: No pude generar un plan válido a partir de la información.", "tasks_created": 0}

    # 5. Sincronizar con el calendario
    created_tasks_count = sync_plan_to_calendar(calendar_service, plan_data)
    
    # 6. Devolver resumen
    summary = plan_data.get('summary', 'Tu día ha sido analizado.')
    return {"summary": summary, "tasks_created": created_tasks_count}