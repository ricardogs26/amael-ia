from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
import os
from kubernetes import client, config
import requests
import json
from langchain_ollama import OllamaLLM
from langchain.agents import initialize_agent, AgentType, Tool
from langchain.prompts import PromptTemplate
from langchain.callbacks.base import BaseCallbackHandler
from prometheus_fastapi_instrumentator import Instrumentator
from prometheus_client import Counter
from typing import Dict, Any

app = FastAPI(title="K8s Agentic AI Service")

# --- MÉTRICAS PROMETHEUS ---
AGENT_STEPS_TOTAL = Counter('amael_agent_steps_total', 'Total steps taken by the agent')
AGENT_TOOLS_USAGE_TOTAL = Counter('amael_agent_tools_usage_total', 'Total usage of agent tools', ['tool'])
AGENT_REQUESTS_TOTAL = Counter('amael_agent_requests_total', 'Total requests to the agent')

# --- CALLBACK PARA MÉTRICAS ---
class PrometheusMetricsCallback(BaseCallbackHandler):
    def on_tool_start(self, serialized: Dict[str, Any], input_str: str, **kwargs: Any) -> Any:
        tool_name = serialized.get("name", "unknown")
        # Incrementar contador de herramienta
        AGENT_TOOLS_USAGE_TOTAL.labels(tool=tool_name).inc()
    
    def on_agent_action(self, action: Any, **kwargs: Any) -> Any:
        # Incrementar contador de pasos (iteraciones Reason/Act)
        AGENT_STEPS_TOTAL.inc()

metrics_callback = PrometheusMetricsCallback()

# --- MODELOS DE DATOS ---
class AgentRequest(BaseModel):
    query: str
    user_email: str = "unknown"


# --- CONFIGURACIÓN ---
OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://ollama-service:11434")
MODEL_NAME = os.environ.get("MODEL_NAME", "qwen2.5:14b")

NEW_RELIC_ACCOUNT_ID = os.environ.get("NEW_RELIC_ACCOUNT_ID", "YOUR_ACCOUNT_ID")
NEW_RELIC_API_KEY = os.environ.get("NEW_RELIC_API_KEY", "YOUR_API_KEY")
NEW_RELIC_GRAPHQL_URL = "https://api.newrelic.com/graphql"

# Inicializar LLM
llm = OllamaLLM(model=MODEL_NAME, base_url=OLLAMA_BASE_URL)

# --- HERRAMIENTAS DEL AGENTE (TOOLS) KUBERNETES NATIVO ---
# Cargar configuración de Kubernetes
try:
    config.load_incluster_config()
except:
    print("Aviso: No se pudo cargar configuración de K8s In-Cluster, probando local Kubeconfig...")
    try:
        config.load_kube_config()
    except:
         print("Aviso: Falló la carga de config local de Kubernetes.")

v1 = client.CoreV1Api()
apps_v1 = client.AppsV1Api()

def list_k8s_pods(ns: str = "") -> str:
    """Útil para listar los pods en el cluster. Input: (opcional) el nombre del namespace (ej. 'kube-system'). Si está vacío, usa 'amael-ia'. IMPORTANTE: después de obtener la lista, ANALIZA el estado de cada pod y reporta activamente cuáles tienen problemas."""
    ns = ns.strip("'\" \n")
    if not ns or ns.lower() == "none" or ns.lower() == "null":
        ns = "amael-ia"
    try:
        pods = v1.list_namespaced_pod(ns)
        result = f"Pods en namespace {ns}:\n"
        failed_pods = []
        for pod in pods.items:
            state = pod.status.phase if pod.status.phase else "Unknown"
            result += f"- Nombre: {pod.metadata.name}, Estado: {state}\n"
            # Detectar pods en estado no saludable
            if state in ["Failed", "Unknown", "Error", "CrashLoopBackOff"] or state not in ["Running", "Pending", "Succeeded"]:
                failed_pods.append(pod.metadata.name)
            elif pod.status.container_statuses:
                for cs in pod.status.container_statuses:
                    if cs.state and cs.state.waiting and cs.state.waiting.reason in ["CrashLoopBackOff", "Error", "OOMKilled"]:
                        failed_pods.append(pod.metadata.name)
        
        if failed_pods:
            result += f"\n** ALERTA: Se detectaron {len(failed_pods)} pods con estado no saludable: {', '.join(failed_pods)} **"
            result += f"\n[ACCION SUGERIDA]: Estos pods están fallidos y consumen recursos. Te sugeriero eliminarlos para liberar recursos. Puedo hacerlo ahora si me lo confirmas."
        else:
            result += "\n** Todos los pods están en estado saludable. **"
        return result
    except Exception as e:
        return f"Error al listar pods: {str(e)}"

def get_pod_logs(pod_name: str) -> str:
    """Útil para obtener los logs de un pod específico."""
    pod_name = pod_name.strip("'\" \n")
    try:
        logs = v1.read_namespaced_pod_log(name=pod_name, namespace="amael-ia", tail_lines=50)
        return f"Logs de {pod_name}:\n{logs}"
    except Exception as e:
        return f"Error al obtener logs del pod {pod_name}: {str(e)}"

def list_k8s_namespaces(query: str = "") -> str:
    """Útil para obtener la lista de todos los namespaces del clúster y su estado actual."""
    try:
        namespaces = v1.list_namespace()
        result = "Namespaces en el clúster:\n"
        for ns in namespaces.items:
            status = ns.status.phase
            result += f"- Namespace: {ns.metadata.name}, Estado: {status}\n"
        return result
    except Exception as e:
        return f"Error al listar namespaces: {str(e)}"

def inspect_namespace(ns_name: str) -> str:
    """Útil para obtener detalles y estado de un namespace específico. Input OBLIGATORIO: el nombre del namespace."""
    ns_name = ns_name.strip("'\" \n")
    try:
        ns = v1.read_namespace(ns_name)
        status = ns.status.phase
        return f"Detalles del namespace '{ns_name}':\n- Estado: {status}\n- UID: {ns.metadata.uid}\n- Creación: {ns.metadata.creation_timestamp}"
    except Exception as e:
        return f"Error al inspeccionar el namespace {ns_name}: {str(e)}"

def delete_k8s_pod(pod_name: str) -> str:
    """Útil para eliminar un pod específico."""
    pod_name = pod_name.strip("'\" \\n")
    try:
        v1.delete_namespaced_pod(name=pod_name, namespace="amael-ia")
        return f"El pod {pod_name} ha sido eliminado exitosamente (Kubernetes lo recreará si es parte de un deploy)."
    except Exception as e:
        return f"Error al intentar eliminar el pod {pod_name}: {str(e)}"


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
        name="Listar_Namespaces",
        func=list_k8s_namespaces,
        description="Útil para listar todos los namespaces del clúster y su estado. Input: opcional (ej. un espacio en blanco)."
    ),
    Tool(
        name="Detalle_Namespace",
        func=inspect_namespace,
        description="Útil para ver el estado o detalles de un namespace específico. Input OBLIGATORIO: el nombre del namespace exacto."
    ),
    Tool(
        name="Listar_Pods",
        func=list_k8s_pods,
        description="Útil para listar los pods actuales y su estado en el cluster. Input: opcional el nombre del namespace (ej. 'kube-system'). Si no se indica, usa 'amael-ia'."
    ),
    Tool(
        name="Obtener_Logs_Pod",
        func=get_pod_logs,
        description="Útil para leer los logs de un pod. Input OBLIGATORIO: exactamente el nombre del pod."
    ),
    Tool(
        name="Eliminar_Pod",
        func=delete_k8s_pod,
        description="Útil para borrar o reiniciar un pod. Input OBLIGATORIO: exactamente el nombre del pod."
    ),
    Tool(
        name="New_Relic_Query",
        func=query_new_relic_tool,
        description="Útil para obtener métricas desde New Relic. SOLO PUEDES INGRESAR COMO INPUT UNO DE ESTOS 5 VALORES LITERAMENTE: 'estatus_cluster', 'cpu_cluster', 'ram_cluster', 'cpu_pods', o 'ram_pods'."
    )
]

# Inicializar Agente con más iteraciones para tareas complejas
agent = initialize_agent(
    tools, 
    llm, 
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, 
    verbose=True,
    handle_parsing_errors=True,
    max_iterations=8,
    early_stopping_method="generate"
)

def extract_final_answer(raw_response: str) -> str:
    """Extrae solo el texto después de 'Final Answer:' para mostrar al usuario texto limpio."""
    marker = "Final Answer:"
    if marker in raw_response:
        # Tomar solo lo que viene después del marcador
        return raw_response.split(marker)[-1].strip()
    
    # Si el modelo filtró pensamiento interno sin dar Final Answer, limpiarlo
    lines_to_remove = ["Thought:", "Action:", "Action Input:", "Observation:"]
    cleaned_lines = []
    for line in raw_response.split("\n"):
        if not any(line.strip().startswith(prefix) for prefix in lines_to_remove):
            cleaned_lines.append(line)
    cleaned = "\n".join(cleaned_lines).strip()
    return cleaned if cleaned else raw_response

@app.post("/api/k8s-agent")
async def chat_with_agent(request: AgentRequest):
    AGENT_REQUESTS_TOTAL.inc()
    print(f"Recibiendo petición de {request.user_email}: {request.query}")
    try:
        final_prompt = agent_prompt.format(query=request.query)
        # Ejecutar el agente con el callback de métricas
        raw_response = agent.run(final_prompt, callbacks=[metrics_callback])
        clean_response = extract_final_answer(raw_response)
        return {"response": clean_response}
    except Exception as e:
        error_msg = str(e)
        print(f"Agent error: {error_msg}")
        return {"response": f"El agente tuvo dificultades entendiendo la petición o la herramienta. Error: {error_msg[:100]}... Intenta ser más específico."}

@app.get("/health")
async def health_check():
    return {"status": "ok"}

agent_prompt = """Eres un SRE (Site Reliability Engineer) Senior y experto en Kubernetes que administra un clúster 'MicroK8s'. Eres un AGENTE AUTÓNOMO con capacidad de toma de decisiones y ejecución, NO un simple chatbot.

## Tu comportamiento es de resolución de problemas (Troubleshooting SRE Real):
- NO sugieras eliminar un pod fallido sin antes investigar por qué falló.
- Si un usuario pide revisar el clúster o listar pods:
   1. Usa 'Listar_Pods'.
   2. Si detectas pods con problemas ("Failed", "Unknown", "CrashLoopBackOff", "Error"):
      - INFORMA al usuario del problema.
      - **DEDUCCIÓN SRE:** Usa INMEDIATAMENTE la herramienta 'Obtener_Logs_Pod' en los pods fallidos para investigar la causa raíz. 
      - EXPLICA la causa raíz encontrada en los logs al usuario.
      - SOLO DESPUÉS de explicar la causa, propón soluciones (ej. "Podemos eliminar el pod para que se reinice, ¿procedo?").

## Herramientas disponibles:
- 'Listar_Namespaces': Muestra todos los namespaces en el clúster.
- 'Detalle_Namespace': Input OBLIGATORIO: nombre exacto del namespace. Para ver estado y detalles del namespace.
- 'Listar_Pods': Muestra estado actual de los pods.
- 'Obtener_Logs_Pod': Input OBLIGATORIO: nombre exacto del pod. Para ver por qué un pod está fallando.
- 'Eliminar_Pod': Input OBLIGATORIO: nombre exacto del pod. Para borrar pods (útil para forzar reinicio).
- 'New_Relic_Query': Extrae métricas de CPU/RAM.

## IMPORTANTE para New_Relic_Query: NUNCA inventes consultas NRQL. Solo usa uno de estos 5 alias: 'estatus_cluster', 'cpu_cluster', 'ram_cluster', 'cpu_pods', 'ram_pods'.

## PERMISOS Y AUTONOMÍA (CAPACIDAD DE EJECUCIÓN):
- TÚ TIENES PERMISO TOTAL para usar todas las herramientas, incluida 'Eliminar_Pod'.
- NUNCA digas "no tengo la capacidad de ejecutar comandos" o "hazlo tú desde tu terminal". Tienes la herramienta, úsala si el usuario lo autoriza.
- Eres el operador. Si el usuario te confirma que elimines un pod, lanza la acción inmediatamente con tu herramienta.

## Formato de comunicación:
Termina SIEMPRE tu respuesta hacia el usuario con:
Final Answer: [Tu mensaje detallado y profesional para el usuario]

Pregunta del usuario: {query}"""

# --- INSTRUMENTACIÓN ---
Instrumentator().instrument(app).expose(app)
