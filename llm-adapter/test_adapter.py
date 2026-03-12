import pytest
import respx
import httpx
import json
from main import app
from httpx import AsyncClient

@pytest.mark.asyncio
async def test_health():
    async with AsyncClient(app=app, base_url="http://test") as ac:
        response = await ac.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"

@pytest.mark.asyncio
@respx.mock
async def test_list_models():
    respx.get("http://ollama-service.default.svc.cluster.local:11434/api/tags").mock(
        return_value=httpx.Response(200, json={"models": [{"name": "llama3"}]})
    )
    
    async with AsyncClient(app=app, base_url="http://test") as ac:
        response = await ac.get("/v1/models")
    
    assert response.status_code == 200
    data = response.json()
    assert data["object"] == "list"
    assert data["data"][0]["id"] == "llama3"

@pytest.mark.asyncio
@respx.mock
async def test_chat_completion_sync():
    respx.post("http://ollama-service.default.svc.cluster.local:11434/api/chat").mock(
        return_value=httpx.Response(200, json={
            "model": "llama3",
            "message": {"role": "assistant", "content": "Hello!"},
            "done": True,
            "prompt_eval_count": 10,
            "eval_count": 5
        })
    )
    
    async with AsyncClient(app=app, base_url="http://test") as ac:
        response = await ac.post("/v1/chat/completions", json={
            "model": "llama3",
            "messages": [{"role": "user", "content": "Hi"}]
        })
    
    assert response.status_code == 200
    data = response.json()
    assert data["choices"][0]["message"]["content"] == "Hello!"
    assert data["usage"]["total_tokens"] == 15

@pytest.mark.asyncio
@respx.mock
async def test_embeddings():
    respx.post("http://ollama-service.default.svc.cluster.local:11434/api/embeddings").mock(
        return_value=httpx.Response(200, json={"embedding": [0.1, 0.2, 0.3]})
    )
    
    async with AsyncClient(app=app, base_url="http://test") as ac:
        response = await ac.post("/v1/embeddings", json={
            "model": "llama3",
            "input": "test text"
        })
    
    assert response.status_code == 200
    data = response.json()
    assert data["object"] == "list"
    assert data["data"][0]["embedding"] == [0.1, 0.2, 0.3]
