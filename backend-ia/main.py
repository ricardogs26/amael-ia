# main.py
from fastapi.responses import HTMLResponse
from fastapi import FastAPI, HTTPException, Depends, status, Request, Response, UploadFile, File
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware import Middleware
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.sessions import SessionMiddleware
from pydantic import BaseModel
import requests
import os
import uuid
import json # <-- AÑADIDO: Para manejar el historial en formato JSON
from authlib.integrations.starlette_client import OAuth
from starlette.config import Config
from jose import JWTError, jwt
from urllib.parse import urlencode, quote_plus 

# --- IMPORTACIONES DE LANGCHAIN ---
from langchain_ollama import OllamaLLM, OllamaEmbeddings
from langchain_chroma import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader

# ... IMPORTACIONDES DE TENSOFLOW2
import base64
from PIL import Image
import numpy as np
import io

# --- CONFIGURACIÓN DE OAUTH Y JWT ---
config = Config(environ=os.environ)
oauth = OAuth(config)

oauth.register(
    name='google',
    server_metadata_url='https://accounts.google.com/.well-known/openid-configuration',
    client_kwargs={
        'scope': 'openid email profile'
    }
)

#--- LISTA BLANCA DE USUARIOS (desde ConfigMap) ---
allowed_emails_csv = os.environ.get("ALLOWED_EMAILS_CSV")
if not allowed_emails_csv:
    raise ValueError("La variable de entorno ALLOWED_EMAILS_CSV no está configurada.")
ALLOWED_EMAILS = [email.strip() for email in allowed_emails_csv.split(',') if email.strip()]

# --- FUNCIONES DE AUTENTICACIÓN JWT ---
SECRET_KEY = os.environ.get("JWT_SECRET_KEY")
ALGORITHM = "HS256"

security = HTTPBearer()

def create_jwt_token(email: str):
    to_encode = {"sub": email}
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    try:
        payload = jwt.decode(credentials.credentials, SECRET_KEY, algorithms=[ALGORITHM])
        email: str = payload.get("sub")
        if email is None or email not in ALLOWED_EMAILS:
            raise HTTPException(status_code=403, detail="Usuario no autorizado")
        return email
    except JWTError:
        raise HTTPException(status_code=401, detail="Token inválido o expirado")

# --- CONFIGURACIÓN E INICIALIZACIÓN DE IA ---
OLLAMA_BASE_URL = "http://ollama-service:11434"
MODEL_NAME = "glm4"
llm = OllamaLLM(model=MODEL_NAME, base_url=OLLAMA_BASE_URL)
# <-- CAMBIO IMPORTANTE: Usamos un directorio base, pero la inicialización de Chroma será por usuario.
CHROMA_BASE_DIR = "/chroma_data"
CHAT_HISTORIES_DIR = "/chat_histories" # <-- NUEVO: Directorio para los historiales

# --- FUNCIONES AUXILIARES MULTIUSUARIO ---

def sanitize_email(email: str) -> str:
    """Crea un nombre de directorio seguro a partir de un email."""
    return email.replace("@", "_at_").replace(".", "_dot_")

def get_user_vectorstore(user_email: str):
    """Carga o crea la base de datos vectorial para un usuario específico."""
    user_dir = os.path.join(CHROMA_BASE_DIR, sanitize_email(user_email))
    os.makedirs(user_dir, exist_ok=True)
    
    embeddings = OllamaEmbeddings(model=MODEL_NAME, base_url=OLLAMA_BASE_URL)
    vectorstore = Chroma(
        embedding_function=embeddings,
        persist_directory=user_dir
    )
    return vectorstore

def get_history_path_for_id(identifier: str) -> str:
    """Obtiene la ruta al archivo de historial para un identificador único (email o teléfono)."""
    # Sanitiza el identificador para que sea un nombre de archivo válido
    sanitized_id = identifier.replace("@", "_at_").replace(".", "_dot_").replace("-", "_dash_")
    return os.path.join(CHAT_HISTORIES_DIR, f"{sanitized_id}.json")

# --- MODELOS DE DATOS ---
class ChatRequest(BaseModel):
    prompt: str
    history: list[dict] = [] # Para recibir el historial del frontend
    user_id: str = None     # <-- NUEVO: Identificador único del usuario (ej. número de teléfono)

class ChatResponse(BaseModel):
    response: str

# --- CREACIÓN DE LA APLICACIÓN FASTAPI CON MIDDLEWARE ---
app = FastAPI(
    title="API Agente Personal",
    version="2.1",
    middleware=[
        Middleware(SessionMiddleware, secret_key=os.environ.get("SESSION_SECRET_KEY")),
        Middleware(
            CORSMiddleware,
            allow_origins=["*"], # En producción, restringe esto a tu dominio
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        ),
    ]
)

# --- ENDPOINTS DE AUTENTICACIÓN (Sin cambios) ---
@app.get('/api/auth/login')
async def login(request: Request):
    redirect_uri = "https://amael-ia.richardx.dev/api/auth/callback"
    return await oauth.google.authorize_redirect(request, redirect_uri)

@app.get('/api/auth/callback')
async def auth_callback(request: Request):
    try:
        token = await oauth.google.authorize_access_token(request)
        user_info = token.get('userinfo')
        if user_info and user_info['email'] in ALLOWED_EMAILS:
            jwt_token = create_jwt_token(user_info['email'])
            frontend_url = "https://amael-ia.richardx.dev"
            
            # --- CAMBIO IMPORTANTE AQUÍ ---
            # Extraemos la información del perfil
            user_name = user_info.get('name', 'Usuario')
            user_picture = user_info.get('picture')

            # Creamos un diccionario con los parámetros
            params = {
                "token": jwt_token,
                "name": user_name,
                "picture": user_picture
            }
            
            # Codificamos los parámetros para la URL
            redirect_url = f"{frontend_url}?{urlencode(params)}"
            
            # Redirigimos con el token y la información del perfil
            return Response(status_code=302, headers={"location": redirect_url})
        else:
            # Manejo de error también con redirección
            frontend_url = "https://amael-ia.richardx.dev"
            return Response(status_code=302, headers={"location": f"{frontend_url}?error=unauthorized"})

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error en la autenticación con Google: {e}")

# --- ENDPOINTS DE LA APLICACIÓN (ahora protegidos y multiusuario) ---
@app.post("/api/ingest")
async def ingest_data(file: UploadFile = File(...), user: str = Depends(get_current_user)):
    """Endpoint para subir y procesar documentos (PDF, TXT) para un usuario específico."""
    temp_file_path = f"/tmp/{uuid.uuid4()}-{file.filename}"
    try:
        with open(temp_file_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)

        if file.filename.endswith(".pdf"):
            loader = PyPDFLoader(temp_file_path)
        else:
            loader = TextLoader(temp_file_path)
        
        documents = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        texts = text_splitter.split_documents(documents)

        # <-- CAMBIO IMPORTANTE: Obtener el vectorstore del usuario y añadir documentos allí.
        user_vectorstore = get_user_vectorstore(user)
        user_vectorstore.add_documents(texts)
        
        return {"message": f"Usuario {user} ha ingerido el archivo correctamente en su perfil."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error procesando el archivo: {e}")
    finally:
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)

@app.post("/api/chat", response_model=ChatResponse, dependencies=[Depends(get_current_user)])
async def chat_endpoint(request: ChatRequest, user: str = Depends(get_current_user)):
    """Endpoint para chatear, ahora con soporte para múltiples usuarios por ID."""
    
    # --- DEFINICIÓN DEL PROMPT (AL PRINCIPIO DE LA FUNCIÓN) ---
    system_prompt_template = """
### PERSONAJE
Eres un asistente de IA avanzado, versátil y servicial. Tu objetivo es proporcionar respuestas precisas, claras y útiles. Adapta tu estilo de respuesta a la naturaleza de la pregunta del usuario.

### REGLAS DE INTERACCIÓN
1.  **Usa el Historial y el Contexto:** Analiza primero el `HISTORIAL DE LA CONVERSACIÓN` y luego el `CONTEXTO DE DOCUMENTOS` para fundamentar tu respuesta.
2.  **Sé Natural:** Para preguntas simples, saludos o conversaciones casuales, responde de forma natural y concisa, como lo haría una persona. **No es necesario usar títulos ni listas.**
3.  **Estructura cuando sea Necesario:** Para respuestas complejas, explicaciones técnicas o listas de pasos, utiliza Markdown para mejorar la claridad:
    *   Usa **negrita** para resaltar términos clave.
    *   Usa listas (`*` o `1.`) para presentar opciones o pasos.
    *   Usa bloques de código para comandos o ejemplos.
    *   Usa títulos (`##`) solo para dividir respuestas muy largas en secciones claras.
4.  **Sé Transparente:** Si el `CONTEXTO` no contiene la información, pero tu conocimiento general te permite responder, indícalo explícitamente (ej: "Aunque no lo encuentro en tus documentos, basándome en mi experiencia...").
5.  **No Inventes Datos Específicos:** Nunca inventes métricas, nombres de archivos o detalles que no estén en el `CONTEXTO` o el `HISTORIAL`.

### HISTORIAL DE LA CONVERSACIÓN
---
{conversation_history}
---

### CONTEXTO DE DOCUMENTOS
---
<<<CONTEXT>>>

**Pregunta del Usuario:**
{request_prompt}

**Respuesta del Asistente:**
"""

    # --- LÓGICA DEL ENDPOINT ---
    
    os.makedirs(CHAT_HISTORIES_DIR, exist_ok=True)

    # Determinar qué ID usar para el historial (email del usuario o user_id de WhatsApp)
    history_id = request.user_id if request.user_id else user
    history_path = get_history_path_for_id(history_id)

    # Cargar el historial del usuario específico desde el archivo
    history = []
    if os.path.exists(history_path):
        with open(history_path, "r") as f:
            history = json.load(f)

    # 1. Recuperar documentos relevantes del USUARIO AUTENTICADO (el bot)
    user_vectorstore = get_user_vectorstore(user)
    retriever = user_vectorstore.as_retriever()
    relevant_docs = retriever.invoke(request.prompt)
    context = "\n".join([doc.page_content for doc in relevant_docs])

    # 2. Construir el historial de conversación (usando el historial cargado del archivo)
    conversation_history = ""
    if history:
        history_lines = [f"Human: {msg['content']}" if msg['role'] == 'user' else f"Assistant: {msg['content']}" for msg in history]
        conversation_history = "\n".join(history_lines)

    # 3. Crear el prompt final
    final_prompt = system_prompt_template.format(
        conversation_history=conversation_history,
        request_prompt=request.prompt
    )
    final_prompt = final_prompt.replace("<<<CONTEXT>>>", context)

    try:
        response = llm.invoke(final_prompt)
        
        # Guardar el historial actualizado en el archivo del usuario correcto
        updated_history = history + [
            {"role": "user", "content": request.prompt},
            {"role": "assistant", "content": response}
        ]
        with open(history_path, "w") as f:
            json.dump(updated_history, f)

        return ChatResponse(response=response)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al contactar al modelo de IA: {e}")

# ... TensorFlow (sin cambios, ya que no es multiusuario por ahora)
@app.post("/api/analyze-image", dependencies=[Depends(get_current_user)])
async def analyze_image(file: UploadFile = File(...), user: str = Depends(get_current_user)):
    """Endpoint para analizar una imagen con TensorFlow."""
    try:
        image_data = await file.read()
        img = Image.open(io.BytesIO(image_data)).convert('RGB')
        img = img.resize((224, 224))
        
        image_array = np.array(img, dtype=np.float32) / 255.0
        image_batch = np.expand_dims(image_array, axis=0)

        payload = {
            "instances": image_batch.tolist()
        }

        tf_serving_url = os.environ.get("TF_SERVING_URL")
        response = requests.post(tf_serving_url, json=payload)

        if response.status_code == 200:
            predictions = response.json()['predictions'][0]
            return {"analysis_result": predictions}
        else:
            raise HTTPException(status_code=500, detail="Error en TensorFlow Serving")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error procesando la imagen: {e}")

@app.get("/api/health")
async def health_check():
    return {"status": "ok"}