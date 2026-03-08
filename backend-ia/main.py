# main.py
from fastapi.responses import HTMLResponse
from fastapi import FastAPI, HTTPException, Depends, status, Request, Response, UploadFile, File
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware import Middleware
from fastapi.middleware.cors import CORSMiddleware
from prometheus_fastapi_instrumentator import Instrumentator
from prometheus_client import Counter, Histogram, Gauge
from starlette.middleware.sessions import SessionMiddleware
import time
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
from langchain_ollama import OllamaLLM, OllamaEmbeddings, ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader

# --- MEMORIA Y PERSISTENCIA ---
import redis
import psycopg2
from psycopg2 import pool
from minio import Minio

# ... IMPORTACIONDES DE TENSOFLOW2
import base64
from PIL import Image
import numpy as np
import io

# --- CONFIGURACIÓN DEL SERVICIO DE PRODUCTIVIDAD ---
PRODUCTIVITY_SERVICE_URL = "http://productivity-service:8001" # URL del nuevo servicio en la red de Docker/K8s
# Seguridad: Leído desde Kubernetes Secrets para cumplimiento de buenas prácticas
INTERNAL_API_SECRET = os.environ.get("INTERNAL_API_SECRET")

if not INTERNAL_API_SECRET:
    raise ValueError("Falta la variable de entorno 'INTERNAL_API_SECRET' montada en el Pod.")

# URL del nuevo servicio ejecutor dentro del clúster (ELIMINADO)

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

k8s_allowed_users_csv = os.environ.get("K8S_ALLOWED_USERS_CSV")
if not k8s_allowed_users_csv:
    print("Warning: Variable K8S_ALLOWED_USERS_CSV no encontrada. Usando ALLOWED_EMAILS como respaldo temporal.")
    K8S_ALLOWED_USERS = ALLOWED_EMAILS
else:
    K8S_ALLOWED_USERS = [u.strip() for u in k8s_allowed_users_csv.split(',') if u.strip()]


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
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://ollama-service:11434")
LLM_MODEL = os.getenv("LLM_MODEL", "qwen2.5:14b")       # Modelo principal
VISION_MODEL = os.getenv("LLM_VISION_MODEL", "qwen2.5-vl:7b") # Modelo Vision
EMBED_MODEL = "nomic-embed-text" 

# Modelos definitivos
llm = OllamaLLM(model=LLM_MODEL, base_url=OLLAMA_BASE_URL)
vision_llm = ChatOllama(model=VISION_MODEL, base_url=OLLAMA_BASE_URL)

# --- CONFIGURACIÓN DE CAPA DE DATOS (PHASE 2) ---
REDIS_HOST = os.getenv("REDIS_HOST", "redis-service")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))
redis_client = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)

QDRANT_URL = os.getenv("QDRANT_URL", "http://qdrant-service:6333")

POSTGRES_HOST = os.getenv("POSTGRES_HOST", "postgres-service")
POSTGRES_DB = os.getenv("POSTGRES_DB", "amael_db")
POSTGRES_USER = os.getenv("POSTGRES_USER", "amael_user")
POSTGRES_PASS = os.getenv("POSTGRES_PASSWORD", "amael_password_2026")

# Pool de conexiones para Postgres
try:
    postgres_pool = psycopg2.pool.SimpleConnectionPool(1, 10,
        user=POSTGRES_USER,
        password=POSTGRES_PASS,
        host=POSTGRES_HOST,
        database=POSTGRES_DB
    )
    print("PostgreSQL connection pool created successfully")
except Exception as e:
    print(f"Error creating PostgreSQL pool: {e}")
    postgres_pool = None

MINIO_URL = os.getenv("MINIO_URL", "minio-service:9000")
MINIO_ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY", "amael_admin")
MINIO_SECRET_KEY = os.getenv("MINIO_SECRET_KEY", "amael_minio_secret_key")

minio_client = Minio(
    MINIO_URL,
    access_key=MINIO_ACCESS_KEY,
    secret_key=MINIO_SECRET_KEY,
    secure=False
)
CHAT_HISTORIES_DIR = os.getenv("CHAT_HISTORIES_DIR", "/chat_histories")

# --- CUSTOM PROMETHEUS METRICS ---
LLM_TOKENS_TOTAL = Counter('amael_llm_tokens_total', 'Total tokens used by LLM', ['model', 'type'])
LLM_LATENCY_SECONDS = Histogram('amael_llm_latency_seconds', 'Latency of LLM requests in seconds', ['model'])
AGENT_STEPS_TOTAL = Counter('amael_agent_steps_total', 'Total steps taken by the agent')
AGENT_FAILURES_TOTAL = Counter('amael_agent_failures_total', 'Total failures in agent execution')
AGENT_TOOLS_USAGE_TOTAL = Counter('amael_agent_tools_usage_total', 'Total usage of agent tools', ['tool'])
RAG_HITS_TOTAL = Counter('amael_rag_hits_total', 'Total number of RAG hits')
RAG_MISS_TOTAL = Counter('amael_rag_miss_total', 'Total number of RAG misses')

# --- DEFINICIÓN DEL PROMPT (MOVIDO AQUÍ - ANTES DE LAS FUNCIONES QUE LO USAN) ---
system_prompt_template = """
### PERSONAJE
Eres un asistente de IA avanzado con nombre de Amael creado por Ricardo Guzman, versátil y servicial. Tu objetivo es proporcionar respuestas precisas, claras y útiles. Adapta tu estilo de respuesta a la naturaleza de la pregunta del usuario, adaptate a su lenguaje.

### REGLAS DE INTERACCIÓN
    Usa el Historial y el Contexto: Analiza primero el HISTORIAL DE LA CONVERSACIÓN y luego el CONTEXTO DE DOCUMENTOS. 
    Sé Natural: Para preguntas simples, responde de forma natural. 
    Sé Transparente: Si el CONTEXTO no contiene la información, pero tu conocimiento general te permite responder, indícalo explícitamente. 
    No Inventes Datos Específicos: Nunca inventes métricas, nombres de archivos o detalles que no estén en el CONTEXTO o el HISTORIAL. 

### INSTRUCCIONES
1.  **Analiza la Petición del Usuario:** Entiende la pregunta o solicitud del usuario.
2.  **Usa el Contexto Disponible:**
    - Primero, revisa el `HISTORIAL DE LA CONVERSACIÓN` para entender el contexto.
    - Luego, utiliza el `CONTEXTO DE DOCUMENTOS` para fundamentar tu respuesta con datos específicos.
3.  **Decide Cómo Responder:**
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
    """Carga o crea la base de datos vectorial para un usuario específico en Qdrant."""
    embeddings = OllamaEmbeddings(model=EMBED_MODEL, base_url=OLLAMA_BASE_URL)
    collection_name = sanitize_email(user_email)
    
    try:
        vectorstore = QdrantVectorStore.from_existing_collection(
            embedding=embeddings,
            collection_name=collection_name,
            url=QDRANT_URL,
        )
        return vectorstore
    except Exception as e:
        print(f"[Qdrant] Error o diferencia de dimensiones en {collection_name}: {e}. Recreando...")
        client = QdrantClient(url=QDRANT_URL)
        
        # Si la colección existe pero dio error (probablemente dimensiones), la borramos
        if client.collection_exists(collection_name):
            print(f"[Qdrant] Borrando colección existente {collection_name} para corregir configuración...")
            client.delete_collection(collection_name)
            
        client.create_collection(
            collection_name=collection_name,
            vectors_config={"size": 768, "distance": "Cosine"} # nomic-embed-text es de 768
        )
        
        return QdrantVectorStore(
            client=client,
            collection_name=collection_name,
            embedding=embeddings
        )

def init_db():
    """Inicializa las tablas necesarias en PostgreSQL."""
    if not postgres_pool: return
    conn = postgres_pool.getconn()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS chat_history (
                    id SERIAL PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
                CREATE INDEX IF NOT EXISTS idx_user_id ON chat_history(user_id);
            """)
            conn.commit()
    finally:
        postgres_pool.putconn(conn)


def save_chat_message(user_id: str, role: str, content: str):
    """Guarda un mensaje en Redis (caché) y Postgres (persistente)."""
    # 1. Guardar en Redis (Lista de los últimos 10 mensajes)
    redis_key = f"chat_history:{user_id}"
    message_json = json.dumps({"role": role, "content": content})
    redis_client.lpush(redis_key, message_json)
    redis_client.ltrim(redis_key, 0, 9) # Mantener solo los últimos 10
    
    # 2. Guardar en Postgres
    if postgres_pool:
        conn = postgres_pool.getconn()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    "INSERT INTO chat_history (user_id, role, content) VALUES (%s, %s, %s)",
                    (user_id, role, content)
                )
                conn.commit()
        finally:
            postgres_pool.putconn(conn)

def get_chat_history(user_id: str, limit: int = 10) -> list:
    """Intenta obtener el historial de Redis, si falla va a Postgres."""
    redis_key = f"chat_history:{user_id}"
    try:
        history = redis_client.lrange(redis_key, 0, limit - 1)
        if history:
            # Redis devuelve los más recientes primero, los invertimos para el prompt
            return [json.loads(m) for m in reversed(history)]
    except Exception as e:
        print(f"Redis error: {e}")

    # Fallback a Postgres
    if postgres_pool:
        conn = postgres_pool.getconn()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT role, content FROM chat_history WHERE user_id = %s ORDER BY timestamp DESC LIMIT %s",
                    (user_id, limit)
                )
                rows = cur.fetchall()
                # Voltear para orden cronológico
                return [{"role": r, "content": c} for r, c in reversed(rows)]
        finally:
            postgres_pool.putconn(conn)
    
    return []

def get_history_path_for_id(identifier: str) -> str:
    """Obtiene la ruta al archivo de historial para un identificador único (email o teléfono)."""
    # Sanitiza el identificador para que sea un nombre de archivo válido
    sanitized_id = identifier.replace("@", "_at_").replace(".", "_dot_").replace("-", "_dash_")
    return os.path.join(CHAT_HISTORIES_DIR, f"{sanitized_id}.json")

# --- MODELOS DE DATOS ---
class ChatRequest(BaseModel):
    prompt: str
    history: list[dict] = [] 
    user_id: str = None     
    image: str = None  # Base64 string

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

# Inicializar Instrumentator fuera del ciclo de vida para evitar RuntimeError
Instrumentator().instrument(app).expose(app)

# Llamar a la inicialización al arrancar
@app.on_event("startup")
async def startup_event():
    init_db()

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
        
        # 5. Procesar el documento
        documents = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        texts = text_splitter.split_documents(documents)

        # 6. Guardar en Qdrant
        user_vectorstore = get_user_vectorstore(user)
        user_vectorstore.add_documents(texts)
        
        # 7. Guardar el archivo original en MinIO para referencia futura
        bucket_name = sanitize_email(user).replace("_", "-") # MinIO buckets prefer dashes
        if not minio_client.bucket_exists(bucket_name):
            minio_client.make_bucket(bucket_name)
        
        with open(temp_file_path, "rb") as f:
            minio_client.put_object(
                bucket_name, 
                file.filename, 
                f, 
                length=os.path.getsize(temp_file_path),
                content_type=mime
            )
        
        return {"message": f"Archivo '{file.filename}' ingerido en Qdrant y respaldado en MinIO para el usuario {user}."}

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

    # --- NUEVA LÓGICA PARA GESTIÓN DE CLUSTER Y NEW RELIC ---
    k8s_keywords = ["kubernetes", "cluster", "clúster", "new relic", "newrelic", "pods", "pod", "deployments", "deployment", "logs", "métricas", "rendimiento", "cpu", "memoria", "eliminar", "borrar", "namespace", "namespaces", "espacio de nombres", "espacios de nombres"]
    if any(keyword in request.prompt.lower() for keyword in k8s_keywords):
        
        # VALIDACIÓN DE WHITELIST DE INFRAESTRUCTURA
        if user not in K8S_ALLOWED_USERS:
            print(f"Intento de acceso a k8s bloqueado para el usuario: {user}")
            return ChatResponse(response="Lo siento, tu usuario no cuenta con los privilegios de administrador requeridos para interactuar con la infraestructura del clúster o consultar métricas avanzadas. Puedo ayudarte con dudas regulares sobre conocimiento general.")

        try:
            K8S_AGENT_URL = "http://k8s-agent-service:8002"
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{K8S_AGENT_URL}/api/k8s-agent",
                    json={"query": request.prompt, "user_email": user},
                    timeout=120.0 # LangChain agentes pueden tardar pensando
                )
                response.raise_for_status()
                result_data = response.json()
                
                # Devolvemos la respuesta del Agente en lugar del LLM estándar
                # Guardamos en el historial pero como respuesta del sistema experto
                history_id = request.user_id if request.user_id else user
                history_path = get_history_path_for_id(history_id)
                history = []
                if os.path.exists(history_path):
                    with open(history_path, "r") as f:
                        history = json.load(f)
                        
                updated_history = history + [
                    {"role": "user", "content": request.prompt},
                    {"role": "assistant", "content": result_data["response"]} 
                ]
                with open(history_path, "w") as f:
                    json.dump(updated_history, f)
                    
                return ChatResponse(response=result_data["response"])

        except httpx.RequestError as e:
            print(f"Error contacting k8s-agent service: {e}")
            raise HTTPException(status_code=503, detail="El servicio de agente experto de Kubernetes no está disponible.")
        except httpx.HTTPStatusError as e:
            AGENT_FAILURES_TOTAL.inc() # Increment failure count
            print(f"K8s-agent returned an error: {e.response.text}")
            raise HTTPException(status_code=500, detail="Ocurrió un error al consultar el estado de tu cluster.")


    # Determinar qué ID usar para el historial
    history_id = request.user_id if request.user_id else user
    
    # --- MIGRACIÓN DE HISTORIAL (JSON -> DB) ---
    history = get_chat_history(history_id)
    if not history:
        history_path = get_history_path_for_id(history_id)
        if os.path.exists(history_path):
            print(f"Migrando historial JSON para {history_id}...")
            with open(history_path, "r") as f:
                old_history = json.load(f)
                for msg in old_history:
                    save_chat_message(history_id, msg['role'], msg['content'])
            # Opcional: renombrar el archivo para no migrarlo de nuevo
            os.rename(history_path, history_path + ".migrated")
            history = get_chat_history(history_id)

    # 1. Recuperar documentos relevantes del USUARIO AUTENTICADO desde QDRANT
    context = ""
    try:
        user_vectorstore = get_user_vectorstore(user)
        relevant_docs = user_vectorstore.similarity_search(request.prompt, k=3)
        if relevant_docs:
            RAG_HITS_TOTAL.inc()
        else:
            RAG_MISS_TOTAL.inc()
        context = "\n".join([doc.page_content for doc in relevant_docs])
    except Exception as e:
        RAG_MISS_TOTAL.inc()
        print(f"[RAG/Qdrant] Error: {e}")
        context = ""

    # 2. Construir el historial de conversación
    conversation_history = ""
    if history:
        history_lines = [f"Human: {msg['content']}" if msg['role'] == 'user' else f"Assistant: {msg['content']}" for msg in history]
        conversation_history = "\n".join(history_lines)

    # 3. Crear el prompt final
    # Si hay imagen, usamos Vision Model con ChatOllama
    if request.image:
        try:
            # Construir mensajes para ChatOllama
            content = [
                {"type": "text", "text": request.prompt},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{request.image}"},
                },
            ]
            
            # Incorporamos contexto RAG si existe
            system_content = "Eres Amael, un asistente avanzado. Analiza la imagen y el texto proporcionado."
            if context:
                system_content += f"\n\nContexto relevante:\n{context}"

            messages = [
                SystemMessage(content=system_content),
                HumanMessage(content=content)
            ]

            # Invocación multimodal
            response = vision_llm.invoke(messages)
            final_response = response.content

            # Guardar en historial
            save_chat_message(history_id, "user", request.prompt + " [Imagen enviada]")
            save_chat_message(history_id, "assistant", final_response)

            return ChatResponse(response=final_response)

        except Exception as e:
            print(f"Error en Vision LLM: {e}")
            raise HTTPException(status_code=500, detail=f"Error al procesar la imagen con Qwen-VL: {e}")

    # Si no hay imagen, seguimos con el flujo normal de OllamaLLM
    final_prompt = system_prompt_template.format(
        conversation_history=conversation_history,
        request_prompt=request.prompt
    )
    final_prompt = final_prompt.replace("<<<CONTEXT>>>", context)

    try:
        # 4. Invocar al LLM estándar
        start_time = time.time()
        response = llm.invoke(final_prompt)
        latency = time.time() - start_time
        
        final_response = response

        # Record metrics
        LLM_LATENCY_SECONDS.labels(model=LLM_MODEL).observe(latency)
        # Assuming we can estimate tokens roughly or get them from response
        # OllamaLLM doesn't easily return token counts in Invoke, but we can estimate
        # For now, let's increment by 1 for simplicity or estimate 1 token per 4 chars
        prompt_tokens = len(final_prompt) // 4
        completion_tokens = len(final_response) // 4
        LLM_TOKENS_TOTAL.labels(model=LLM_MODEL, type="prompt").inc(prompt_tokens)
        LLM_TOKENS_TOTAL.labels(model=LLM_MODEL, type="completion").inc(completion_tokens)

        # --- GUARDAR EN LA NUEVA CAPA DE DATOS ---
        save_chat_message(history_id, "user", request.prompt)
        save_chat_message(history_id, "assistant", final_response)

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
            
            # --- NUEVA LÓGICA DE DECODIFICACIÓN ---
            # Obtenemos las etiquetas de ImageNet si no están cacheadas
            if not hasattr(app.state, "imagenet_labels"):
                url = "https://storage.googleapis.com/download.tensorflow.org/data/imagenet_class_index.json"
                try:
                    labels_response = requests.get(url, timeout=5)
                    app.state.imagenet_labels = labels_response.json()
                except Exception as e:
                    print(f"Error descargando etiquetas ImageNet: {e}")
                    return {"analysis_result": predictions[:5]} # Fallback
            
            labels_dict = app.state.imagenet_labels
            
            # Obtener los top 5 índices con mayor probabilidad
            top_indices = np.argsort(predictions)[-5:][::-1]
            
            results = []
            for idx in top_indices:
                class_id, label = labels_dict[str(idx)]
                prob = float(predictions[idx])
                results.append({
                    "etiqueta": label.replace("_", " "),
                    "probabilidad": f"{round(prob * 100, 2)}%"
                })
                
            resumen_ia = ", ".join([f"{item['etiqueta']} ({item['probabilidad']})" for item in results])
            
            return {
                "analisis_detallado": results,
                "resumen_ia": resumen_ia
            }
        else:
            raise HTTPException(status_code=500, detail="Error en TensorFlow Serving")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error procesando la imagen: {e}")

@app.get("/api/health")
async def health_check():
    return {"status": "ok"}