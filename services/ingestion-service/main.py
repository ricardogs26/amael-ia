# services/ingestion-service/main.py
from fastapi import FastAPI, HTTPException, Depends, status, UploadFile, File
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from jose import JWTError, jwt
import os
import uuid
import magic
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings

# --- CONFIGURACIÓN ---
app = FastAPI(title="Ingestion Service")
security = HTTPBearer()
SECRET_KEY = os.environ.get("JWT_SECRET_KEY")
ALGORITHM = "HS256"
ALLOWED_EMAILS = [email.strip() for email in os.environ.get("ALLOWED_EMAILS_CSV", "").split(',') if email.strip()]

OLLAMA_BASE_URL = "http://ollama-service:11434" # Asumiendo que Ollama está en K8s
MODEL_NAME = "glm4"
CHROMA_BASE_DIR = "/chroma_data"

def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    try:
        payload = jwt.decode(credentials.credentials, SECRET_KEY, algorithms=[ALGORITHM])
        email: str = payload.get("sub")
        if email is None or email not in ALLOWED_EMAILS:
            raise HTTPException(status_code=403, detail="Usuario no autorizado")
        return email
    except JWTError:
        raise HTTPException(status_code=401, detail="Token inválido o expirado")

def sanitize_email(email: str) -> str:
    return email.replace("@", "_at_").replace(".", "_dot_")

def get_user_vectorstore(user_email: str):
    user_dir = os.path.join(CHROMA_BASE_DIR, sanitize_email(user_email))
    os.makedirs(user_dir, exist_ok=True)
    embeddings = OllamaEmbeddings(model=MODEL_NAME, base_url=OLLAMA_BASE_URL)
    vectorstore = Chroma(embedding_function=embeddings, persist_directory=user_dir)
    return vectorstore

# --- ENDPOINT ---
@app.post("/ingest")
async def ingest_data(file: UploadFile = File(...), user: str = Depends(get_current_user)):
    temp_file_path = f"/tmp/{uuid.uuid4()}-{file.filename}"
    try:
        content = await file.read()
        mime = magic.from_buffer(content, mime=True)
        
        with open(temp_file_path, "wb") as buffer:
            buffer.write(content)

        if mime == "application/pdf":
            loader = PyPDFLoader(temp_file_path)
        elif mime == "text/plain":
            loader = TextLoader(temp_file_path)
        else:
            raise HTTPException(status_code=400, detail=f"Tipo de archivo no soportado: '{mime}'. Solo se permiten PDF y TXT.")
        
        documents = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        texts = text_splitter.split_documents(documents)

        user_vectorstore = get_user_vectorstore(user)
        user_vectorstore.add_documents(texts)
        
        return {"message": f"Archivo ingerido correctamente para el usuario {user}."}
    except HTTPException as http_e:
        raise http_e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error procesando el archivo: {e}")
    finally:
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)

@app.get("/health")
async def health_check():
    return {"status": "ok"}