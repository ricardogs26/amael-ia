import os
import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel
from typing import Optional, List, Dict, Any, Union
from prometheus_fastapi_instrumentator import Instrumentator
from prometheus_client import Counter, Histogram
import time

app = FastAPI(title="LLM Adapter: OpenAI to Ollama")

# Configuración
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://ollama-service.default.svc.cluster.local:11434")
DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "glm4") # Ajusta esto al nombre exacto de tu modelo en Ollama

# --- CUSTOM PROMETHEUS METRICS ---
LLM_TOKENS_TOTAL = Counter('amael_llm_tokens_total', 'Total tokens used by LLM', ['model', 'type'])
LLM_LATENCY_SECONDS = Histogram('amael_llm_latency_seconds', 'Latency of LLM requests in seconds', ['model'])

# Modelos de datos para simular la API de OpenAI (simplificado)
class ChatMessage(BaseModel):
    role: str
    content: str

class ChatCompletionRequest(BaseModel):
    model: str = DEFAULT_MODEL
    messages: List[ChatMessage]
    images: Optional[List[str]] = None # Base64 images
    stream: bool = False
    temperature: Optional[float] = 0.7

@app.post("/llm/v1/chat/completions")
async def create_chat_completion(request: ChatCompletionRequest):
    """
    Endpoint compatible con OpenAI SDK.
    Transforma la petición y la envía a Ollama.
    """
    
    # Mapeo del modelo: Si OpenClaw pide "gpt-4", lo cambiamos por tu modelo local
    target_model = DEFAULT_MODEL if request.model.startswith("gpt") else request.model

    # Payload para Ollama (API nativa /api/chat o /api/generate)
    # Usaremos /api/chat que es más moderna para chat models
    ollama_payload = {
        "model": target_model,
        "messages": [msg.dict() for msg in request.messages],
        "stream": request.stream,
        "options": {
            "temperature": request.temperature
        }
    }

    # Si se envían imágenes, las agregamos al payload de Ollama
    if request.images:
        ollama_payload["images"] = request.images

    ollama_url = f"{OLLAMA_BASE_URL}/api/chat"

    client = httpx.AsyncClient(timeout=120.0)
    try:
        if request.stream:
            # Manejo de Streaming (SSE)
            async def stream_generator():
                start_time = time.time()
                try:
                    async with client.stream("POST", ollama_url, json=ollama_payload) as response:
                        response.raise_for_status()
                        async for line in response.aiter_lines():
                            if not line:
                                continue
                            
                            import json
                            try:
                                data = json.loads(line)
                                if data.get("done"):
                                    prompt_tokens = data.get("prompt_eval_count", 0)
                                    completion_tokens = data.get("eval_count", 0)
                                    latency = time.time() - start_time
                                    
                                    LLM_LATENCY_SECONDS.labels(model=target_model).observe(latency)
                                    LLM_TOKENS_TOTAL.labels(model=target_model, type="prompt").inc(prompt_tokens)
                                    LLM_TOKENS_TOTAL.labels(model=target_model, type="completion").inc(completion_tokens)
                                    print(f"[METRICS] Stream finalizado: {prompt_tokens} prompt, {completion_tokens} completion tokens, {latency:.2f}s latency")
                                
                                content = data.get("message", {}).get("content", "")
                                chunk = {
                                    "id": "chatcmpl-local",
                                    "object": "chat.completion.chunk",
                                    "created": int(time.time()),
                                    "model": target_model,
                                    "choices": [{
                                        "index": 0,
                                        "delta": { "content": content },
                                        "finish_reason": "stop" if data.get("done") else None
                                    }]
                                }
                                yield f"data: {json.dumps(chunk)}\n\n"
                            except Exception as e:
                                print(f"Error parseando linea de Ollama: {e}")
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
                LLM_LATENCY_SECONDS.labels(model=target_model).observe(latency)
                prompt_tokens = ollama_data.get("prompt_eval_count", 0)
                completion_tokens = ollama_data.get("eval_count", 0)
                LLM_TOKENS_TOTAL.labels(model=target_model, type="prompt").inc(prompt_tokens)
                LLM_TOKENS_TOTAL.labels(model=target_model, type="completion").inc(completion_tokens)

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

@app.get("/llm/health")
async def health():
    return {"status": "healthy"}

# --- INSTRUMENTATION ---
Instrumentator().instrument(app).expose(app)
