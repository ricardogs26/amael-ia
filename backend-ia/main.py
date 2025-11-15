from fastapi import FastAPI, HTTPException, Depends, status, Request, Response, UploadFile, File
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware import Middleware
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.sessions import SessionMiddleware
from pydantic import BaseModel
import requests
import os
import uuid
from authlib.integrations.starlette_client import OAuth
from starlette.config import Config
from jose import JWTError, jwt

# --- IMPORTACIONES DE LANGCHAIN ---
from langchain_ollama import OllamaLLM, OllamaEmbeddings
from langchain_chroma import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader

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
MODEL_NAME = "llama3:8b"
CHROMA_PERSIST_DIR = "/chroma_data"

# --- INICIALIZACIÓN CON CLASES ACTUALIZADAS ---
llm = OllamaLLM(model=MODEL_NAME, base_url=OLLAMA_BASE_URL)
embeddings = OllamaEmbeddings(model=MODEL_NAME, base_url=OLLAMA_BASE_URL)

vectorstore = Chroma(
    embedding_function=embeddings,
    persist_directory=CHROMA_PERSIST_DIR
)

# --- MODELOS DE DATOS ---
class ChatRequest(BaseModel):
    prompt: str

class ChatResponse(BaseModel):
    response: str

# --- CREACIÓN DE LA APLICACIÓN FASTAPI CON MIDDLEWARE ---
app = FastAPI(
    title="API Agente Personal",
    version="2.1",
    middleware=[
        # Middleware 1: Sesiones (para OAuth)
        Middleware(SessionMiddleware, secret_key=os.environ.get("SESSION_SECRET_KEY")),
        # Middleware 2: CORS (para permitir peticiones del frontend)
        Middleware(
            CORSMiddleware,
            allow_origins=["*"], # En producción, restringe esto a tu dominio
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        ),
    ]
)
# --- ENDPOINTS DE AUTENTICACIÓN ---
@app.get('/api/auth/login')
async def login(request: Request):
    # CAMBIO CLAVE: Especificamos la URL pública de forma manual
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
            return Response(status_code=302, headers={"location": f"{frontend_url}?token={jwt_token}"})
        else:
            return Response(status_code=302, headers={"location": f"https://amael-ia.richardx.dev?error=unauthorized"})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error en la autenticación con Google: {e}")


# --- ENDPOINTS DE LA APLICACIÓN (ahora protegidos por JWT) ---
@app.post("/api/ingest")
async def ingest_data(file: UploadFile = File(...), user: str = Depends(get_current_user)):
    """Endpoint para subir y procesar documentos (PDF, TXT)."""
    temp_file_path = f"/tmp/{uuid.uuid4()}-{file.filename}"
    try:
        # El bloque 'with' maneja la apertura y cierre del archivo automáticamente
        with open(temp_file_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)

        # El resto del código se ejecuta DESPUÉS de que el archivo se ha cerrado
        if file.filename.endswith(".pdf"):
            loader = PyPDFLoader(temp_file_path)
        else:
            loader = TextLoader(temp_file_path)
        
        documents = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        texts = text_splitter.split_documents(documents)

        vectorstore.add_documents(texts)
        return {"message": f"Usuario {user} ha ingerido el archivo correctamente."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error procesando el archivo: {e}")
    finally:
        # El bloque 'finally' se ejecuta siempre, haya habido un error o no
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)

@app.post("/api/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest, user: str = Depends(get_current_user)):
    """Endpoint para chatear usando RAG."""
    retriever = vectorstore.as_retriever()
    relevant_docs = retriever.get_relevant_documents(request.prompt)
    context = "\n".join([doc.page_content for doc in relevant_docs])

    enriched_prompt = f"""
    Usa el siguiente contexto para responder a la pregunta al final.
    Si no sabes la respuesta basándote en el contexto, di que no lo sabes, pero intenta ser útil con tu conocimiento general.
    Contexto:
    {context}

    Pregunta:
    {request.prompt}
    """

    try:
        response = llm.invoke(enriched_prompt)
        return ChatResponse(response=response)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al contactar al modelo de IA: {e}")

@app.get("/api/health")
async def health_check():
    return {"status": "ok"}
