from fastapi import FastAPI, HTTPException, Depends, status, UploadFile, File
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import requests
import secrets

# --- IMPORTACIONES ACTUALIZADAS ---
from langchain_ollama import OllamaLLM, OllamaEmbeddings
from langchain_chroma import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader
import os
import uuid

# --- CONFIGURACIÓN ---
app = FastAPI(title="API Agente Personal", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # En producción, restringe esto a tu dominio
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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

# --- AUTENTICACIÓN ---
security = HTTPBasic()
USUARIOS = {"admin": "password123"} # ¡CAMBIA ESTO!

def get_current_user(credentials: HTTPBasicCredentials = Depends(security)):
    if credentials.username in USUARIOS and USUARIOS[credentials.username] == credentials.password:
        return credentials.username
    raise HTTPException(status_code=401, detail="Credenciales incorrectas")

# --- MODELOS DE DATOS ---
class ChatRequest(BaseModel):
    prompt: str

class ChatResponse(BaseModel):
    response: str

# --- ENDPOINTS ---
@app.post("/api/ingest", dependencies=[Depends(get_current_user)])
async def ingest_data(file: UploadFile = File(...)):
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
        return {"message": f"Se han ingerido {len(texts)} fragmentos del archivo {file.filename}."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error procesando el archivo: {e}")
    finally:
        # El bloque 'finally' se ejecuta siempre, haya habido un error o no
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)

@app.post("/api/chat", response_model=ChatResponse, dependencies=[Depends(get_current_user)])
async def chat_endpoint(request: ChatRequest):
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
