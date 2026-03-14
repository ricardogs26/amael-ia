import os
import httpx
import time
import uuid
import json
import logging
import hvac
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

# --- VAULT CONFIGURATION ---
VAULT_ADDR = os.getenv("VAULT_ADDR", "http://vault.vault.svc.cluster.local:8200")
VAULT_ROLE = os.getenv("VAULT_ROLE", "llm-adapter")
VAULT_SECRET_PATH = "amael/llm-adapter"
K8S_TOKEN_PATH = "/var/run/secrets/kubernetes.io/serviceaccount/token"

def get_secret_from_vault():
    """Fetch ADAPTER_API_KEY from Vault using Kubernetes auth."""
    if not os.path.exists(K8S_TOKEN_PATH):
        logger.info("[VAULT] No K8s token found, skipping Vault secret retrieval.")
        return None
    
    try:
        client = hvac.Client(url=VAULT_ADDR)
        with open(K8S_TOKEN_PATH, 'r') as f:
            jwt = f.read().strip()
        
        client.auth.kubernetes.login(role=VAULT_ROLE, jwt=jwt)
        if not client.is_authenticated():
            logger.warning("[VAULT] Authentication to Vault failed.")
            return None
        
        response = client.secrets.kv.v2.read_secret_version(path=VAULT_SECRET_PATH, mount_point="secret")
        # response['data']['data'] contains the actual key-value pairs
        secrets = response.get("data", {}).get("data", {})
        key = secrets.get("ADAPTER_API_KEY")
        if key:
            logger.info("[VAULT] Successfully retrieved ADAPTER_API_KEY from Vault.")
            return key
    except Exception as e:
        logger.warning(f"[VAULT] Error retrieving secret from Vault: {e}")
    
    return None

# Prioritize Vault, then ENV
ADAPTER_API_KEY = get_secret_from_vault() or os.getenv("ADAPTER_API_KEY")

if not ADAPTER_API_KEY:
    logger.warning("[SECURITY] No ADAPTER_API_KEY found in Vault or ENV. Service will be public!")
else:
    logger.info("[SECURITY] ADAPTER_API_KEY loaded and active.")

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

# --- ANTHROPIC MESSAGES API SCHEMAS ---

class AnthropicContentBlock(BaseModel):
    type: str  # "text", "image", etc.
    text: Optional[str] = None

class AnthropicMessage(BaseModel):
    role: str  # "user" or "assistant"
    content: Union[str, List[Dict[str, Any]]]

class AnthropicMessagesRequest(BaseModel):
    model: str
    messages: List[AnthropicMessage]
    max_tokens: int = 4096
    system: Optional[str] = None
    stream: Optional[bool] = False
    temperature: Optional[float] = 1.0
    top_p: Optional[float] = None
    stop_sequences: Optional[List[str]] = None
    metadata: Optional[Dict[str, Any]] = None
    tools: Optional[List[Dict[str, Any]]] = None
    tool_choice: Optional[Dict[str, Any]] = None

class AnthropicCountTokensRequest(BaseModel):
    model: str
    messages: List[AnthropicMessage]
    system: Optional[str] = None
    tools: Optional[List[Dict[str, Any]]] = None

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
            try:
                async with client:
                    response = await client.post(ollama_url, json=ollama_payload)
                    latency = time.time() - start_time
                    
                    if response.status_code != 200:
                        error_detail = response.text
                        try:
                            error_json = response.json()
                            error_detail = error_json.get("error", error_detail)
                        except:
                            pass
                        logger.error(f"[OLLAMA-ERROR] Upstream returned {response.status_code} for {user_identity}: {error_detail}")
                        raise HTTPException(status_code=response.status_code, detail=f"Ollama error: {error_detail}")

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
                        "id": f"chatcmpl-{uuid.uuid4()}",
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
            except httpx.HTTPStatusError as e:
                logger.error(f"[OLLAMA-HTTP-ERROR] {user_identity} | {str(e)}")
                raise HTTPException(status_code=e.response.status_code, detail=f"Upstream HTTP error: {str(e)}")

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
                if response.status_code != 200:
                    logger.error(f"[OLLAMA-EMBED-ERROR] Upstream returned {response.status_code} for text index {i}")
                    raise HTTPException(status_code=response.status_code, detail=f"Ollama embeddings error: {response.text}")

                result = response.json()
                
                embeddings_data.append({
                    "object": "embedding",
                    "index": i,
                    "embedding": result["embedding"]
                })
                # Ollama doesn't always return token counts for embeddings, we estimate or put 0
                total_tokens += len(text.split()) # Very rough estimation
                
            except HTTPException:
                raise
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

def _anthropic_messages_to_ollama(messages: List[AnthropicMessage], system: Optional[str]) -> List[Dict]:
    """Convert Anthropic messages format to Ollama chat messages."""
    ollama_messages = []
    if system:
        ollama_messages.append({"role": "system", "content": system})
    for msg in messages:
        content = msg.content if isinstance(msg.content, str) else "".join(
            block.get("text", "") for block in msg.content if block.get("type") == "text"
        )
        ollama_messages.append({"role": msg.role, "content": content})
    return ollama_messages


@app.post("/v1/messages")
@app.post("/api/v1/messages")
@app.post("/llm/v1/messages")
async def anthropic_messages(request: AnthropicMessagesRequest, req_raw: Request, authorized: bool = Depends(get_api_key)):
    """Anthropic Messages API compatible endpoint — translates to Ollama."""
    user_email = req_raw.headers.get("X-User-Email", "anonymous")
    logger.info(f"[ANTHROPIC-ADAPTER] Request from {user_email} | Model: {request.model} | Stream: {request.stream}")

    ollama_messages = _anthropic_messages_to_ollama(request.messages, request.system)
    ollama_payload = {
        "model": request.model,
        "messages": ollama_messages,
        "stream": request.stream,
        "options": {
            "temperature": request.temperature,
            "num_predict": request.max_tokens,
        }
    }
    if request.top_p is not None:
        ollama_payload["options"]["top_p"] = request.top_p
    if request.stop_sequences:
        ollama_payload["options"]["stop"] = request.stop_sequences

    ollama_url = f"{OLLAMA_BASE_URL}/api/chat"
    msg_id = f"msg_{uuid.uuid4().hex[:24]}"

    client = httpx.AsyncClient(timeout=120.0)
    try:
        if request.stream:
            async def stream_generator():
                start_time = time.time()
                output_tokens = 0
                input_tokens = 0
                try:
                    # message_start event
                    yield f"event: message_start\ndata: {json.dumps({'type': 'message_start', 'message': {'id': msg_id, 'type': 'message', 'role': 'assistant', 'content': [], 'model': request.model, 'stop_reason': None, 'stop_sequence': None, 'usage': {'input_tokens': 0, 'output_tokens': 0}}})}\n\n"
                    yield f"event: content_block_start\ndata: {json.dumps({'type': 'content_block_start', 'index': 0, 'content_block': {'type': 'text', 'text': ''}})}\n\n"
                    yield f"event: ping\ndata: {json.dumps({'type': 'ping'})}\n\n"

                    async with client.stream("POST", ollama_url, json=ollama_payload) as response:
                        response.raise_for_status()
                        async for line in response.aiter_lines():
                            if not line:
                                continue
                            try:
                                data = json.loads(line)
                                if data.get("done"):
                                    input_tokens = data.get("prompt_eval_count", 0)
                                    output_tokens = data.get("eval_count", 0)
                                    latency = time.time() - start_time
                                    LLM_LATENCY_SECONDS.labels(model=request.model, user=user_email).observe(latency)
                                    LLM_TOKENS_TOTAL.labels(model=request.model, type="prompt", user=user_email).inc(input_tokens)
                                    LLM_TOKENS_TOTAL.labels(model=request.model, type="completion", user=user_email).inc(output_tokens)
                                    break
                                text = data.get("message", {}).get("content", "")
                                if text:
                                    yield f"event: content_block_delta\ndata: {json.dumps({'type': 'content_block_delta', 'index': 0, 'delta': {'type': 'text_delta', 'text': text}})}\n\n"
                            except Exception as e:
                                logger.error(f"[ANTHROPIC-STREAM] Parse error: {e}")
                                continue

                    yield f"event: content_block_stop\ndata: {json.dumps({'type': 'content_block_stop', 'index': 0})}\n\n"
                    yield f"event: message_delta\ndata: {json.dumps({'type': 'message_delta', 'delta': {'stop_reason': 'end_turn', 'stop_sequence': None}, 'usage': {'output_tokens': output_tokens}})}\n\n"
                    yield f"event: message_stop\ndata: {json.dumps({'type': 'message_stop'})}\n\n"
                finally:
                    await client.aclose()

            return StreamingResponse(stream_generator(), media_type="text/event-stream")

        else:
            start_time = time.time()
            async with client:
                response = await client.post(ollama_url, json=ollama_payload)
                latency = time.time() - start_time
                if response.status_code != 200:
                    raise HTTPException(status_code=response.status_code, detail=f"Ollama error: {response.text}")
                ollama_data = response.json()

            input_tokens = ollama_data.get("prompt_eval_count", 0)
            output_tokens = ollama_data.get("eval_count", 0)
            LLM_LATENCY_SECONDS.labels(model=request.model, user=user_email).observe(latency)
            LLM_TOKENS_TOTAL.labels(model=request.model, type="prompt", user=user_email).inc(input_tokens)
            LLM_TOKENS_TOTAL.labels(model=request.model, type="completion", user=user_email).inc(output_tokens)
            logger.info(f"[ANTHROPIC-ADAPTER] Completed for {user_email}: {input_tokens} in, {output_tokens} out, {latency:.2f}s")

            return JSONResponse(content={
                "id": msg_id,
                "type": "message",
                "role": "assistant",
                "content": [{"type": "text", "text": ollama_data.get("message", {}).get("content", "")}],
                "model": request.model,
                "stop_reason": "end_turn",
                "stop_sequence": None,
                "usage": {"input_tokens": input_tokens, "output_tokens": output_tokens}
            })

    except httpx.RequestError as exc:
        await client.aclose()
        raise HTTPException(status_code=503, detail=f"Error conectando con Ollama: {str(exc)}")
    except HTTPException:
        raise
    except Exception as e:
        await client.aclose()
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/v1/messages/count_tokens")
@app.post("/api/v1/messages/count_tokens")
@app.post("/llm/v1/messages/count_tokens")
async def anthropic_count_tokens(request: AnthropicCountTokensRequest, authorized: bool = Depends(get_api_key)):
    """Anthropic count_tokens compatible endpoint — estimates via Ollama prompt_eval_count."""
    ollama_messages = _anthropic_messages_to_ollama(request.messages, request.system)
    # Use num_predict=1 to get just the prompt_eval_count without generating output
    payload = {"model": request.model, "messages": ollama_messages, "stream": False, "options": {"num_predict": 1}}
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(f"{OLLAMA_BASE_URL}/api/chat", json=payload)
            if response.status_code == 200:
                data = response.json()
                input_tokens = data.get("prompt_eval_count", 0)
                return JSONResponse(content={"input_tokens": input_tokens})
    except Exception as e:
        logger.warning(f"[COUNT-TOKENS] Ollama call failed, using estimate: {e}")
    # Fallback: rough estimation
    total_chars = sum(
        len(m.content) if isinstance(m.content, str) else sum(len(b.get("text", "")) for b in m.content)
        for m in request.messages
    )
    if request.system:
        total_chars += len(request.system)
    return JSONResponse(content={"input_tokens": max(1, total_chars // 4)})


@app.get("/health")
@app.get("/v1/health")
@app.get("/llm/health")
async def health():
    return {"status": "healthy", "model": DEFAULT_MODEL}

# --- INSTRUMENTATION ---
Instrumentator().instrument(app).expose(app)
