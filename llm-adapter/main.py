import os
import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel
from typing import Optional, List, Dict, Any, Union

app = FastAPI(title="LLM Adapter: OpenAI to Ollama")

# Configuración
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://ollama-service.default.svc.cluster.local:11434")
DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "glm4") # Ajusta esto al nombre exacto de tu modelo en Ollama

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

    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            if request.stream:
                # Manejo de Streaming (SSE)
                async def stream_generator():
                    async with client.stream("POST", ollama_url, json=ollama_payload) as response:
                        response.raise_for_status()
                        async for line in response.aiter_lines():
                            # Ollama devuelve JSON por línea, hay que transformarlo a formato OpenAI SSE
                            import json
                            try:
                                data = json.loads(line)
                                # Estructura simulada de respuesta OpenAI stream chunk
                                chunk = {
                                    "id": "chatcmpl-local",
                                    "object": "chat.completion.chunk",
                                    "created": 1234567890,
                                    "model": target_model,
                                    "choices": [{
                                        "index": 0,
                                        "delta": {
                                            "content": data.get("message", {}).get("content", "")
                                        },
                                        "finish_reason": None
                                    }]
                                }
                                yield f"data: {json.dumps(chunk)}\n\n"
                            except Exception:
                                continue
                    yield "data: [DONE]\n\n"

                return StreamingResponse(stream_generator(), media_type="text/event-stream")

            else:
                # Petición síncrona (No streaming)
                response = await client.post(ollama_url, json=ollama_payload)
                response.raise_for_status()
                ollama_data = response.json()

                # Formatear respuesta como OpenAI
                openai_response = {
                    "id": "chatcmpl-local",
                    "object": "chat.completion",
                    "created": 1234567890,
                    "model": target_model,
                    "choices": [{
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": ollama_data.get("message", {}).get("content", "")
                        },
                        "finish_reason": "stop"
                    }],
                    "usage": { # Valores simulados, Ollama a veces devuelve esto en otro nivel
                        "prompt_tokens": 0,
                        "completion_tokens": 0,
                        "total_tokens": 0
                    }
                }
                return JSONResponse(content=openai_response)

    except httpx.RequestError as exc:
        raise HTTPException(status_code=503, detail=f"Error conectando con Ollama: {str(exc)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/llm/health")
async def health():
    return {"status": "healthy"}
