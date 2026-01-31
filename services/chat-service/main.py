# services/chat-service/main.py
from fastapi import FastAPI, HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from jose import JWTError, jwt
from pydantic import BaseModel
import os
import json
import httpx
from langchain_ollama import OllamaLLM
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings

# --- CONFIGURACIÓN ---
app = FastAPI(title="Chat Service")
security = HTTPBearer()
SECRET_KEY = os.environ.get("JWT_SECRET_KEY")
ALGORITHM = "HS256"
ALLOWED_EMAILS = [email.strip() for email in os.environ.get("ALLOWED_EMAILS_CSV", "").split(',') if email.strip()]

OLLAMA_BASE_URL = "http://ollama-service:11434"
MODEL_NAME = "glm4"
llm = OllamaLLM(model=MODEL_NAME, base_url=OLLAMA_BASE_URL)
CHROMA_BASE_DIR = "/chroma_data"
CHAT_HISTORIES_DIR = "/chat_histories"
PRODUCTIVITY_SERVICE_URL = "http://productivity-service:8001"
COMMAND_EXECUTOR_URL = "http://command-executor-service:8001/execute"

# ... (copia aquí la función run_kubectl_command, sanitize_email, get_history_path_for_id) ...

# ... (copia aquí el system_prompt_template) ...

class ChatRequest(BaseModel):
    prompt: str
    history: list[dict] = []

# ... (copia aquí las funciones get_current_user, get_user_vectorstore) ...

# --- ENDPOINT ---
@app.post("/chat")
async def chat_endpoint(request: ChatRequest, user: str = Depends(get_current_user)):
    if not request.prompt or request.prompt.strip() == "":
        return {"response": "¡Hola! ¿En qué puedo ayudarte? Envía tu consulta."}

    # Lógica de "organiza mi día" (llama a otro microservicio)
    if "organiza mi día" in request.prompt.lower():
        # ... (copia la lógica de llamada a PRODUCTIVITY_SERVICE_URL) ...
        # return {"response": final_response}

    # Lógica principal del chat
    history_id = user # Usa el email como ID para el historial
    history_path = get_history_path_for_id(history_id)
    history = []
    if os.path.exists(history_path):
        with open(history_path, "r") as f:
            history = json.load(f)

    user_vectorstore = get_user_vectorstore(user)
    retriever = user_vectorstore.as_retriever()
    relevant_docs = retriever.invoke(request.prompt)
    context = "\n".join([doc.page_content for doc in relevant_docs])

    conversation_history = "\n".join([f"Human: {msg['content']}" if msg['role'] == 'user' else f"Assistant: {msg['content']}" for msg in history])
    
    final_prompt = system_prompt_template.format(
        conversation_history=conversation_history,
        request_prompt=request.prompt
    )
    final_prompt = final_prompt.replace("<<<CONTEXT>>>", context)

    try:
        response = llm.invoke(final_prompt)
        final_response = response # Respuesta por defecto

        # Lógica de tool_calling (llama a otro microservicio)
        if "```json" in response and "tool_call" in response:
            # ... (copia la lógica para parsear el JSON y llamar a run_kubectl_command) ...
            # final_response = f"Aquí está la salida del comando `{command_key}`:\n\n```\n{command_result.get('output')}\n```"

        # Guardar historial
        updated_history = history + [{"role": "user", "content": request.prompt}, {"role": "assistant", "content": final_response}]
        with open(history_path, "w") as f:
            json.dump(updated_history, f)

        return {"response": final_response}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al contactar al modelo de IA: {e}")

@app.get("/health")
async def health_check():
    return {"status": "ok"}