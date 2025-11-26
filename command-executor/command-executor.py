# command-executor.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import subprocess
import os

app = FastAPI(title="Command Executor Service")

# --- LISTA BLANCA DE COMANDOS PERMITIDOS ---
# Por ahora, solo permitimos comandos 'get' seguros.
# Puedes expandir esto según sea necesario.
ALLOWED_COMMANDS = {
    "get pods": "kubectl get pods",
    "get services": "kubectl get services",
    "get deployments": "kubectl get deployments",
    # Puedes añadir más, pero con mucho cuidado.
    # "describe pod": "kubectl describe pod",
}

class ExecuteRequest(BaseModel):
    command_key: str # La clave de la lista blanca, ej: "get pods"
    namespace: str = "amael-ia" # Namespace por defecto

@app.post("/execute")
async def execute_command(request: ExecuteRequest):
    command_key = request.command_key
    namespace = request.namespace

    if command_key not in ALLOWED_COMMANDS:
        raise HTTPException(status_code=403, detail=f"Command '{command_key}' is not allowed.")

    # Construye el comando final a ejecutar
    base_command = ALLOWED_COMMANDS[command_key]
    full_command = f"{base_command} -n {namespace}"

    print(f"Executing command: {full_command}")

    try:
        # Ejecuta el comando de forma segura
        result = subprocess.run(
            full_command,
            shell=True, # Necesario para kubectl
            capture_output=True,
            text=True,
            check=True # Lanza una excepción si el comando falla
        )
        return {
            "output": result.stdout,
            "error": result.stderr
        }
    except subprocess.CalledProcessError as e:
        raise HTTPException(status_code=500, detail=f"Error executing command: {e.stderr}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {e}")

@app.get("/health")
async def health_check():
    return {"status": "ok"}
