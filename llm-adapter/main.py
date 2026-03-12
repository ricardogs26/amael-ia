import os
import httpx
import time
import uuid
import json
import logging
from fastapi import FastAPI, HTTPException, Request, Depends, status
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any, Union
from prometheus_fastapi_instrumentator import Instrumentator
from prometheus_client import Counter, Histogram

# Configuración de Logging
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s in %(module)s: %(message)s')
logger = logging.getLogger(__name__)

app = FastAPI(title="LLM Adapter: OpenAI to Ollama")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuración
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://ollama-service.default.svc.cluster.local:11434")
DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "glm4")
ADAPTER_API_KEY = os.getenv("ADAPTER_API_KEY", None)

# --- CUSTOM PROMETHEUS METRICS ---
LLM_TOKENS_TOTAL = Counter('amael_llm_tokens_total', 'Total tokens used by LLM', ['model', 'type', 'user'])
LLM_LATENCY_SECONDS = Histogram('amael_llm_latency_seconds', 'Latency of LLM requests in seconds', ['model', 'user'])

# --- SCHEMAS ---

class ChatMessage(BaseModel):
    role: str
    content: str
    name: Optional[str] = None

class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    stream: Optional[bool] = False
    temperature: Optional[float] = 1.0
    top_p: Optional[float] = 1.0
    max_tokens: Optional[int] = None
    stop: Optional[Union[str, List[str]]] = None
    presence_penalty: Optional[float] = 0.0
    frequency_penalty: Optional[float] = 0.0
    seed: Optional[int] = None
    user: Optional[str] = None
    # Support for legacy 'images' field if present (Ollama specific extension)
    images: Optional[List[str]] = None

class EmbeddingRequest(BaseModel):
    model: str
    input: Union[str, List[str]]
    user: Optional[str] = None

class EmbeddingResponse(BaseModel):
    object: str = "list"
    data: List[Dict[str, Any]]
    model: str
    usage: Dict[str, int]

# --- SECURITY ---

async def get_api_key(request: Request):
    if ADAPTER_API_KEY:
        auth_header = request.headers.get("Authorization")
        if not auth_header or not auth_header.startswith("Bearer "):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid or missing API Key",
            )
        token = auth_header.split(" ")[1]
        if token != ADAPTER_API_KEY:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid API Key",
            )
    return True

# --- ENDPOINTS ---

@app.get("/v1/models")
@app.get("/api/v1/models")
@app.get("/llm/v1/models")
async def list_models(authorized: bool = Depends(get_api_key)):
    """List available models from Ollama in OpenAI format."""
    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(f"{OLLAMA_BASE_URL}/api/tags")
            response.raise_for_status()
            ollama_models = response.json().get("models", [])
            
            data = []
            for m in ollama_models:
                data.append({
                    "id": m["name"],
                    "object": "model",
                    "created": int(time.time()), # Ollama doesn't always provide unix ts in easy format
                    "owned_by": "ollama"
                })
            
            return {
                "object": "list",
                "data": data
            }
        except Exception as e:
            logger.error(f"Error fetching models from Ollama: {e}")
            raise HTTPException(status_code=503, detail="Ollama service unavailable")

@app.post("/v1/chat/completions")
@app.post("/api/v1/chat/completions")
@app.post("/llm/v1/chat/completions")
async def create_chat_completion(request: ChatCompletionRequest, req_raw: Request, authorized: bool = Depends(get_api_key)):
    """
    Endpoint compatible con OpenAI SDK.
    Transforma la petición y la envía a Ollama.
    """
    
    # Captura de Identidad (X-User-Email para Frontend, X-User-Phone para WhatsApp)
    user_email = req_raw.headers.get("X-User-Email", "anonymous")
    user_phone = req_raw.headers.get("X-User-Phone", "none")
    
    user_identity = f"{user_email}:{user_phone}" if user_phone != "none" else user_email
    logger.info(f"[LLM-ADAPTER] Request received from identity: {user_identity} | Model: {request.model}")
    
    target_model = request.model

    # Payload para Ollama (API nativa /api/chat)
    ollama_payload = {
        "model": target_model,
        "messages": [msg.dict(exclude_none=True) for msg in request.messages],
        "stream": request.stream,
        "options": {
            "temperature": request.temperature,
            "top_p": request.top_p,
            "num_predict": request.max_tokens,
            "stop": request.stop if isinstance(request.stop, list) else ([request.stop] if request.stop else None),
            "presence_penalty": request.presence_penalty,
            "repeat_penalty": request.frequency_penalty + 1.0 if request.frequency_penalty != 0 else None, # Rough mapping
            "seed": request.seed,
        }
    }

    # Remove None options
    ollama_payload["options"] = {k: v for k, v in ollama_payload["options"].items() if v is not None}

    if request.images:
        ollama_payload["images"] = request.images

    ollama_url = f"{OLLAMA_BASE_URL}/api/chat"

    client = httpx.AsyncClient(timeout=120.0)
    try:
        if request.stream:
            # Manejo de Streaming (SSE)
            async def stream_generator():
                start_time = time.time()
                request_id = f"chatcmpl-{uuid.uuid4()}"
                try:
                    async with client.stream("POST", ollama_url, json=ollama_payload) as response:
                        response.raise_for_status()
                        async for line in response.aiter_lines():
                            if not line:
                                continue
                            
                            try:
                                data = json.loads(line)
                                if data.get("done"):
                                    prompt_tokens = data.get("prompt_eval_count", 0)
                                    completion_tokens = data.get("eval_count", 0)
                                    latency = time.time() - start_time
                                    
                                    LLM_LATENCY_SECONDS.labels(model=target_model, user=user_email).observe(latency)
                                    LLM_TOKENS_TOTAL.labels(model=target_model, type="prompt", user=user_email).inc(prompt_tokens)
                                    LLM_TOKENS_TOTAL.labels(model=target_model, type="completion", user=user_email).inc(completion_tokens)
                                    logger.info(f"[METRICS] Stream finalizado for {user_identity}: {prompt_tokens} prompt, {completion_tokens} completion tokens, {latency:.2f}s latency")
                                
                                content = data.get("message", {}).get("content", "")
                                chunk = {
                                    "id": request_id,
                                    "object": "chat.completion.chunk",
                                    "created": int(time.time()),
                                    "model": target_model,
                                    "choices": [{
                                        "index": 0,
                                        "delta": { "content": content },
                                        "finish_reason": "stop" if data.get("done") else None
                                    }]
                                }
                                # Add usage on the last chunk if done
                                if data.get("done"):
                                    chunk["usage"] = {
                                        "prompt_tokens": data.get("prompt_eval_count", 0),
                                        "completion_tokens": data.get("eval_count", 0),
                                        "total_tokens": data.get("prompt_eval_count", 0) + data.get("eval_count", 0)
                                    }

                                yield f"data: {json.dumps(chunk)}\n\n"
                            except Exception as e:
                                logger.error(f"Error parseando linea de Ollama: {e}")
                                continue
                    yield "data: [DONE]\n\n"
                finally:
                    await client.aclose()

            return StreamingResponse(stream_generator(), media_type="text/event-stream")

        else:
            # Petición síncrona (No streaming)
            start_time = time.time()
            async with client:
                response = await client.post(ollama_url, json=ollama_payload)
                latency = time.time() - start_time
                response.raise_for_status()
                ollama_data = response.json()

                # Record metrics
                LLM_LATENCY_SECONDS.labels(model=target_model, user=user_email).observe(latency)
                prompt_tokens = ollama_data.get("prompt_eval_count", 0)
                completion_tokens = ollama_data.get("eval_count", 0)
                LLM_TOKENS_TOTAL.labels(model=target_model, type="prompt", user=user_email).inc(prompt_tokens)
                LLM_TOKENS_TOTAL.labels(model=target_model, type="completion", user=user_email).inc(completion_tokens)
                logger.info(f"[METRICS] Petición síncrona finalizada for {user_identity}: {prompt_tokens} tokens, {latency:.2f}s")

                # Formatear respuesta como OpenAI
                openai_response = {
                    "id": "chatcmpl-local",
                    "object": "chat.completion",
                    "created": int(time.time()),
                    "model": target_model,
                    "choices": [{
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": ollama_data.get("message", {}).get("content", "")
                        },
                        "finish_reason": "stop"
                    }],
                    "usage": {
                        "prompt_tokens": prompt_tokens,
                        "completion_tokens": completion_tokens,
                        "total_tokens": prompt_tokens + completion_tokens
                    }
                }
                return JSONResponse(content=openai_response)

    except httpx.RequestError as exc:
        await client.aclose()
        raise HTTPException(status_code=503, detail=f"Error conectando con Ollama: {str(exc)}")
    except Exception as e:
        await client.aclose()
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/v1/embeddings")
@app.post("/api/v1/embeddings")
@app.post("/llm/v1/embeddings")
async def create_embedding(request: EmbeddingRequest, authorized: bool = Depends(get_api_key)):
    """Generate embeddings using Ollama."""
    inputs = [request.input] if isinstance(request.input, str) else request.input
    
    embeddings_data = []
    total_tokens = 0
    
    async with httpx.AsyncClient(timeout=60.0) as client:
        for i, text in enumerate(inputs):
            try:
                response = await client.post(
                    f"{OLLAMA_BASE_URL}/api/embeddings",
                    json={"model": request.model, "prompt": text}
                )
                response.raise_for_status()
                result = response.json()
                
                embeddings_data.append({
                    "object": "embedding",
                    "index": i,
                    "embedding": result["embedding"]
                })
                # Ollama doesn't always return token counts for embeddings, we estimate or put 0
                total_tokens += len(text.split()) # Very rough estimation
                
            except Exception as e:
                logger.error(f"Error generating embedding for index {i}: {e}")
                raise HTTPException(status_code=502, detail=f"Error from Ollama: {str(e)}")
                
    return {
        "object": "list",
        "data": embeddings_data,
        "model": request.model,
        "usage": {
            "prompt_tokens": total_tokens,
            "total_tokens": total_tokens
        }
    }

@app.get("/health")
@app.get("/v1/health")
@app.get("/llm/health")
async def health():
    return {"status": "healthy", "model": DEFAULT_MODEL}

# --- INSTRUMENTATION ---
Instrumentator().instrument(app).expose(app)
