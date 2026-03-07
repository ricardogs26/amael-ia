from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
import os
import subprocess
import requests
import json
from langchain_ollama import OllamaLLM
from langchain.agents import initialize_agent, AgentType, Tool
from langchain.prompts import PromptTemplate

app = FastAPI(title="K8s Agentic AI Service")

# --- CONFIGURACIÓN ---
OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://ollama-service:11434")
MODEL_NAME = os.environ.get("MODEL_NAME", "glm4")

NEW_RELIC_ACCOUNT_ID = os.environ.get("NEW_RELIC_ACCOUNT_ID", "YOUR_ACCOUNT_ID")
NEW_RELIC_API_KEY = os.environ.get("NEW_RELIC_API_KEY", "YOUR_API_KEY")
NEW_RELIC_GRAPHQL_URL = "https://api.newrelic.com/graphql"

# Inicializar LLM
llm = OllamaLLM(model=MODEL_NAME, base_url=OLLAMA_BASE_URL)

# --- HERRAMIENTAS DEL AGENTE (TOOLS) ---

def run_kubectl_tool(command: str) -> str:
    """Ejecuta un comando kubectl seguro en el clúster. 
    Acepta comandos de lectura y borrado de pods, services, deployments y eventos.
    El namespace siempre será 'amael-ia'."""
    allowed_starts = ["get pods", "get services", "get deployments", "get events", "describe pod", "logs", "top pods", "top nodes", "delete pod", "delete pods", "delete deployments"]
    
    if not any(command.startswith(cmd) for cmd in allowed_starts):
        return "Error: Comando kubectl no permitido. Usa solo get pods, describe pod, delete pod, etc."
    
    # Prevenir inyección de comandos
    if ";" in command or "&" in command or "|" in command:
        return "Error: Caracteres inválidos en el comando."

    full_command = f"kubectl {command} -n amael-ia"
    print(f"Agent ejecutando: {full_command}")
    try:
        result = subprocess.run(full_command, shell=True, capture_output=True, text=True, timeout=10)
        return result.stdout if result.returncode == 0 else f"Error ejecutando kubectl: {result.stderr}"
    except Exception as e:
        return f"Error inesperado al ejecutar kubectl: {str(e)}"

def query_new_relic_tool(query_alias: str) -> str:
    """Ejecuta una consulta NRQL predefinida en New Relic para obtener métricas del clúster o contenedores."""
    if NEW_RELIC_ACCOUNT_ID == "YOUR_ACCOUNT_ID" or NEW_RELIC_API_KEY == "YOUR_API_KEY":
        return "Aviso: Las credenciales de New Relic no están configuradas correctamente. No se pueden obtener métricas."

    # Diccionario de consultas seguras y validadas
    safe_queries = {
        "estatus_cluster": "SELECT latest(cpuUsedCores) as 'Cores Usados', latest(cpuAllocatableCores) as 'Cores Totales', latest(memoryWorkingSetBytes)/1000000000 as 'RAM Usada (GB)', latest(memoryAllocatableBytes)/1000000000 as 'RAM Total (GB)' FROM K8sNodeSample FACET nodeName",
        "cpu_cluster": "SELECT latest(cpuUsedCores) as 'Cores Usados', latest(cpuAllocatableCores) as 'Cores Totales' FROM K8sNodeSample FACET nodeName",
        "ram_cluster": "SELECT latest(memoryWorkingSetBytes)/1000000000 as 'GB Usados', latest(memoryAllocatableBytes)/1000000000 as 'GB Totales' FROM K8sNodeSample FACET nodeName",
        "cpu_pods": "SELECT latest(cpuCoresUtilization) as 'Uso CPU %' FROM K8sContainerSample FACET podName LIMIT 10",
        "ram_pods": "SELECT latest(memoryUsedBytes)/1000000 as 'MB Usados' FROM K8sContainerSample FACET podName LIMIT 10"
    }
    
    # Limpiar el input por si la IA envía comillas extras
    query_alias_clean = query_alias.strip("'\" \n").lower()
    
    if query_alias_clean not in safe_queries:
        available = ", ".join(safe_queries.keys())
        return f"Error: La consulta '{query_alias_clean}' no es válida. Debes usar exactamente uno de estos identificadores: {available}"
        
    nrql_query = safe_queries[query_alias_clean]

    headers = {
        "Content-Type": "application/json",
        "API-Key": NEW_RELIC_API_KEY
    }
    
    graphql_query = f"""
    {{
      actor {{
        account(id: {NEW_RELIC_ACCOUNT_ID}) {{
          nrql(query: "{nrql_query}") {{
            results
          }}
        }}
      }}
    }}
    """
    
    try:
        response = requests.post(NEW_RELIC_GRAPHQL_URL, json={"query": graphql_query}, headers=headers, timeout=10)
        if response.status_code == 200:
            data = response.json()
            if "errors" in data:
                return f"Error en la consulta NRQL: {data['errors']}"
            return json.dumps(data["data"]["actor"]["account"]["nrql"]["results"])
        else:
            return f"Error HTTP {response.status_code} al consultar New Relic."
    except Exception as e:
        return f"Excepción al conectar con New Relic: {str(e)}"

# Definir herramientas para LangChain
tools = [
    Tool(
        name="Kubernetes_Query",
        func=run_kubectl_tool,
        description="Útil para interactuar con el clúster (ej. 'get pods', 'describe pod X', 'logs Y', 'delete pod Z'). Ingresa SOLO el resto del comando kubectl sin la palabra kubectl."
    ),
    Tool(
        name="New_Relic_Query",
        func=query_new_relic_tool,
        description="Útil para obtener métricas desde New Relic. SOLO PUEDES INGRESAR COMO INPUT UNO DE ESTOS 5 VALORES LITERAMENTE: 'estatus_cluster', 'cpu_cluster', 'ram_cluster', 'cpu_pods', o 'ram_pods'."
    )
]

# Inicializar Agente
agent = initialize_agent(
    tools, 
    llm, 
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, 
    verbose=True,
    handle_parsing_errors=True,
    max_iterations=5,
    early_stopping_method="generate"
)

agent_prompt = """Eres un experto en DevOps e Inteligencia Artificial que administra un clúster de Kubernetes llamado 'MicroK8s' usando New Relic para monitorización.

Tu objetivo es responder a las peticiones del usuario sobre la infraestructura utilizando las herramientas a tu disposición. 
Si el usuario pregunta por el estado de los pods, deployments, servicios, logs, o te pide ELIMINAR/BORRAR algún recurso, usa 'Kubernetes_Query'.
Si el usuario pregunta por el estatus general, salud, o métricas de rendimiento del clúster (como recursos consumidos de CPU o RAM compartidos o por nodo), usa 'New_Relic_Query'.
    
IMPORTANTE PARA New_Relic_Query: NO DEBES INVENTAR CONSULTAS NRQL. Solo puedes pasarle a la herramienta UNO de los siguientes 5 identificadores exactos (alias):
- Para un estatus general o resumen completo del clúster (CPU y RAM totales y usados): usa 'estatus_cluster'
- Para CPU general del cluster: usa 'cpu_cluster'
- Para Memoria/RAM general del cluster: usa 'ram_cluster'
- Para CPU específica de cada pod: usa 'cpu_pods'
- Para Memoria/RAM específica de cada pod: usa 'ram_pods'

Cuando recibas la información de métricas (especialmente estatus general), analízala y redáctala de manera fluida, profesional y detallada. Redondea los números a máximo dos decimales para que sean legibles y explica claramente los límites (ej. 'Tenemos X Cores en total en el nodo Y, de los cuales se usan Z'). NO des respuestas ambiguas, da un reporte del estado real y funcional.

SI EL USUARIO PIDE ELIMINAR PODS, no uses selectores de field-selector, usa nombres específicos de pods o selectores de labels si es necesario, o pide al usuario que sea más específico si la consulta es muy genérica. Ejemplo válido: `delete pod mi-pod` o `delete pods -l app=mi-app`. Para eliminar pods por estado que no sea running usando field-selectors está deshabilitado en tu nivel, explícaselo al usuario amablemente y ofrécele listar los pods primero.

¡IMPORTANTE!: Si al ejecutar una herramienta recibes un mensaje de "Error" o "Excepción", NO LO INTENTES DE NUEVO ni trates de adivinar comandos. Detente inmediatamente, explícale al usuario exactamente cuál fue el problema que recibiste y finaliza tu respuesta.

REGLA ESTRICTA DE FORMATO: Para comunicarte con el usuario o hacerle preguntas (por ejemplo, si necesitas que te dé el nombre del pod), DEBES USAR OBLIGATORIAMENTE el formato al final de tu respuesta:
Final Answer: [Tu mensaje para el usuario]
¡NUNCA uses una herramienta llamada 'None' o 'Action' para hablar con el usuario!

Pregunta del usuario: {query}"""

class AgentRequest(BaseModel):
    query: str
    user_email: str = "unknown"

@app.post("/api/k8s-agent")
async def chat_with_agent(request: AgentRequest):
    print(f"Recibiendo petición de {request.user_email}: {request.query}")
    try:
        final_prompt = agent_prompt.format(query=request.query)
        response = agent.run(final_prompt)
        return {"response": response}
    except Exception as e:
        error_msg = str(e)
        print(f"Agent error: {error_msg}")
        return {"response": f"El agente tuvo dificultades entendiendo la petición o la herramienta. Error: {error_msg[:100]}... Intenta ser más específico."}

@app.get("/health")
async def health_check():
    return {"status": "ok"}
