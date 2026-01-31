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

import httpx
import magic # libreria para determinar el tipo de formato en un archivo a ingestar


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

# --- CONFIGURACIÓN DEL SERVICIO DE PRODUCTIVIDAD ---
PRODUCTIVITY_SERVICE_URL = "http://productivity-service:8001" # URL del nuevo servicio en la red de Docker/K8s
INTERNAL_API_SECRET = "g686GnXRZFfJ48Au1a1pGkVTxCQoEE" # Debe ser el mismo que en el .env del microservicio

# URL del nuevo servicio ejecutor dentro del clúster
COMMAND_EXECUTOR_URL = "http://command-executor-service:8001/execute"

async def run_kubectl_command(command_key: str, namespace: str = "amael-ia"):
    """Llama al servicio ejecutor de comandos de forma segura."""
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(COMMAND_EXECUTOR_URL, json={"command_key": command_key, "namespace": namespace})
            response.raise_for_status()
            return response.json()
    except httpx.HTTPStatusError as e:
        return {"error": f"Error calling executor service: {e.response.status_code}", "detail": e.response.json()}
    except Exception as e:
        return {"error": f"An unexpected error occurred: {e}"}

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

# --- DEFINICIÓN DEL PROMPT (MOVIDO AQUÍ - ANTES DE LAS FUNCIONES QUE LO USAN) ---
system_prompt_template = """
### PERSONAJE
Eres un asistente de IA avanzado con nombre de Amael creado por Ricardo Guzman, versátil y servicial. Tu objetivo es proporcionar respuestas precisas, claras y útiles. Adapta tu estilo de respuesta a la naturaleza de la pregunta del usuario, adaptate a su lenguaje.

### REGLAS DE INTERACCIÓN
    Usa el Historial y el Contexto: Analiza primero el HISTORIAL DE LA CONVERSACIÓN y luego el CONTEXTO DE DOCUMENTOS. 
    Sé Natural: Para preguntas simples, responde de forma natural. 
    Usa Herramientas cuando sea Necesario: Si el usuario te pide información sobre el estado del clúster (ej. "muéstrame los pods"), usa la herramienta run_kubectl_command. No inventes la respuesta. 
    Procesa la Salida de la Herramienta: Después de llamar a la herramienta, recibirás una respuesta. Usa esa respuesta para formatear una respuesta clara y útil para el usuario. Muestra la salida del comando en un bloque de código. 
    Sé Transparente: Si el CONTEXTO no contiene la información, pero tu conocimiento general te permite responder, indícalo explícitamente. 
    No Inventes Datos Específicos: Nunca inventes métricas, nombres de archivos o detalles que no estén en el CONTEXTO o el HISTORIAL. 
### HERRAMIENTAS DISPONIBLES
Tienes acceso a una herramienta para ejecutar comandos de Kubernetes en el namespace 'amael-ia'.
Para usarla, responde con un bloque de código JSON en el siguiente formato:
```json
{{
  "tool_call": {{
    "name": "run_kubectl_command",
    "arguments": {{
      "command_key": "get pods"
    }}
  }}
}}
```
    - Los `command_key` permitidos son: "get pods", "get services", "get deployments".

### INSTRUCCIONES
1.  **Analiza la Petición del Usuario:** Entiende la pregunta o solicitud del usuario.
2.  **Usa el Contexto Disponible:**
    - Primero, revisa el `HISTORIAL DE LA CONVERSACIÓN` para entender el contexto.
    - Luego, utiliza el `CONTEXTO DE DOCUMENTOS` para fundamentar tu respuesta con datos específicos.
3.  **Decide Cómo Responder:**
    - **¿El usuario pide información del clúster?** (ej: "muéstrame los pods"). Usa la herramienta `run_kubectl_command`. No inventes la respuesta.
    - **¿Es una pregunta general o conversacional?** (ej: "hola", "¿cómo estás?"). Responde de forma natural y directa.
    - **¿Es una pregunta compleja?** Estructura tu respuesta con Markdown (títulos, listas, negrita) para que sea clara y fácil de leer.
4.  **Sé Preciso y Transparente:** Basa tus respuestas en la información proporcionada. Si el contexto es insuficiente, admítelo. Si usas tu conocimiento general, indícalo.
5.  **No Inventes Datos Específicos:** Nunca inventes métricas, nombres de archivo o detalles que no estén en el `CONTEXTO` o el `HISTORIAL`.

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
        # 1. Leer el contenido del archivo en memoria primero
        content = await file.read()
        
        # 2. Determinar el tipo de archivo (MIME type) desde su contenido
        #    Esto es mucho más seguro que fiarse de la extensión.
        try:
            # Usamos la librería 'python-magic' para inspeccionar los bytes
            mime = magic.from_buffer(content, mime=True)
        except Exception as e:
            # Si magic falla, lanzamos un error genérico de tipo de archivo
            raise HTTPException(status_code=400, detail="No se pudo determinar el tipo de archivo. Asegúrate de que no esté corrupto.")

        # 3. Guardar el archivo temporalmente
        with open(temp_file_path, "wb") as buffer:
            buffer.write(content)

        # 4. Seleccionar el loader correcto basado en el MIME type detectado
        if mime == "application/pdf":
            loader = PyPDFLoader(temp_file_path)
        elif mime == "text/plain":
            loader = TextLoader(temp_file_path)
        else:
            # Si no es PDF ni TXT, devolvemos un error claro.
            # Por ejemplo, si es una imagen (mime == 'image/jpeg'), caerá aquí.
            raise HTTPException(
                status_code=400, 
                detail=f"Tipo de archivo no soportado: '{mime}'. Solo se permiten archivos PDF y TXT."
            )
        
        # 5. Procesar el documento como antes
        documents = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        texts = text_splitter.split_documents(documents)

        user_vectorstore = get_user_vectorstore(user)
        user_vectorstore.add_documents(texts)
        
        return {"message": f"Usuario {user} ha ingerido el archivo correctamente en su perfil."}

    except HTTPException as http_e:
        # Si ya es un HTTPException, simplemente la volvemos a lanzar
        raise http_e
    except Exception as e:
        # Para cualquier otro error inesperado
        raise HTTPException(status_code=500, detail=f"Error procesando el archivo: {e}")
    finally:
        # 6. Limpiar el archivo temporal
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)

@app.post("/api/chat", response_model=ChatResponse, dependencies=[Depends(get_current_user)])
async def chat_endpoint(request: ChatRequest, user: str = Depends(get_current_user)):
    """Endpoint para chatear, ahora con soporte para múltiples usuarios por ID y ejecución de herramientas."""
    
    # --- LÓGICA DEL ENDPOINT ---
     
    # Validar que el prompt no esté vacío o contenga solo espacios en blanco.
    if not request.prompt or request.prompt.strip() == "":
        # En lugar de dejar que la aplicación falle, devolvemos una respuesta amigable.
        # Esto hace que la experiencia sea mejor y no se registra un error en el bridge de WhatsApp.
        return ChatResponse(response="¡Hola! como te encuentras, envía tu consulta.")
    # Asegurarse de que el directorio de historiales exista   
    os.makedirs(CHAT_HISTORIES_DIR, exist_ok=True)

    # --- NUEVA LÓGICA PARA ORGANIZAR EL DÍA ---
    if "organiza mi día" in request.prompt.lower():
        try:
            headers = {"Authorization": f"Bearer {INTERNAL_API_SECRET}"}
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{PRODUCTIVITY_SERVICE_URL}/organize",
                    json={"user_email": user}, # Pasamos el email del usuario
                    headers=headers,
                    timeout=60.0 # La operación puede tardar
                )
                response.raise_for_status() # Lanza un error si la petición falló (ej. 500)
                
                result_data = response.json()
                summary = result_data.get("summary", "Tu día ha sido organizado.")
                tasks_created = result_data.get("tasks_created", 0)
                final_response = f"{summary}\n\nHe creado {tasks_created} nuevas tareas en tu calendario."

            return ChatResponse(response=final_response)

        except httpx.RequestError as e:
            print(f"Error contacting productivity service: {e}")
            raise HTTPException(status_code=503, detail="El servicio de productividad no está disponible en este momento.")
        except httpx.HTTPStatusError as e:
            print(f"Productivity service returned an error: {e.response.text}")
            raise HTTPException(status_code=500, detail="Ocurrió un error al organizar tu día.")


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
        # 4. Invocar al LLM
        response = llm.invoke(final_prompt)

        # --- LÓGICA PARA MANEJAR LLAMADAS A HERRAMIENTAS ---
        if "```json" in response and "tool_call" in response:
            try:
                # Extraer el JSON de la respuesta
                start_index = response.find("```json") + 7
                end_index = response.find("```", start_index)
                json_str = response[start_index:end_index].strip()
                tool_call_data = json.loads(json_str)

                if tool_call_data.get("tool_call", {}).get("name") == "run_kubectl_command":
                    command_key = tool_call_data["tool_call"]["arguments"]["command_key"]
                    print(f"Agent is requesting to execute command: {command_key}")
                    
                    # Llamar a la función que ejecuta el comando
                    command_result = await run_kubectl_command(command_key)
                    
                    # Formatear la respuesta final con el resultado del comando
                    final_response = f"Aquí está la salida del comando `{command_key}`:\n\n```\n{command_result.get('output', command_result.get('error'))}\n```"
                else:
                    final_response = "Lo siento, no reconozco esa herramienta."
            except (json.JSONDecodeError, KeyError) as e:
                final_response = f"Error al procesar la solicitud de la herramienta: {e}"
        else:
            # Si no hay llamada a herramienta, la respuesta es la del LLM directamente
            final_response = response

        # --- GUARDAR EL HISTORIAL ACTUALIZADO (AHORA DENTRO DEL TRY) ---
        updated_history = history + [
            {"role": "user", "content": request.prompt},
            {"role": "assistant", "content": final_response} # <-- Usar la respuesta final
        ]
        with open(history_path, "w") as f:
            json.dump(updated_history, f)

        return ChatResponse(response=final_response)

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