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
PROMETHEUS_URL = os.environ.get("PROMETHEUS_URL", "http://kube-prometheus-stack-prometheus.observability.svc.cluster.local:9090")
GRAFANA_URL = os.environ.get("GRAFANA_URL", "http://kube-prometheus-stack-grafana.observability.svc.cluster.local:80")
GRAFANA_USER = os.environ.get("GRAFANA_USER", "admin")
GRAFANA_PASSWORD = os.environ.get("GRAFANA_PASSWORD", "admin")

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

def query_prometheus(query: str) -> str:
    """Útil para ejecutar consultas PromQL en Prometheus y obtener métricas del clúster. 
    Ejemplo de input: 'sum(rate(container_cpu_usage_seconds_total{namespace="amael-ia"}[5m])) by (pod)'"""
    query = query.strip("'\" \n")
    try:
        response = requests.get(f"{PROMETHEUS_URL}/api/v1/query", params={"query": query}, timeout=10)
        if response.status_code == 200:
            data = response.json()
            if data["status"] == "success":
                results = data["data"]["result"]
                # Simplificar la respuesta para el LLM
                simplified = []
                for res in results:
                    metric = res.get("metric", {})
                    value = res.get("value", [None, None])[1]
                    simplified.append({"metric": metric, "value": value})
                return json.dumps(simplified[:10]) # Limitar a 10 resultados para no saturar contexto
            else:
                return f"Error en Prometheus: {data.get('error', 'Unknown error')}"
        else:
            return f"Error HTTP {response.status_code} al consultar Prometheus."
    except Exception as e:
        return f"Excepción al conectar con Prometheus: {str(e)}"

def list_grafana_dashboards(query: str = "") -> str:
    """Útil para buscar dashboards en Grafana consultando los ConfigMaps del clúster. 
    Retorna una lista de dashboards disponibles para monitoreo."""
    try:
        v1 = client.CoreV1Api()
        # Buscar ConfigMaps etiquetados como dashboards de grafana en el namespace de observabilidad
        cms = v1.list_namespaced_config_map(
            namespace="observability", 
            label_selector="grafana_dashboard=1"
        )
        
        if not cms.items:
            return "No se encontraron dashboards registrados en el clúster (vía ConfigMaps)."
            
        result = "Dashboards de Kubernetes encontrados en el sistema:\n"
        for cm in cms.items:
            title = cm.metadata.name.replace("kube-prometheus-stack-", "").replace("-", " ").title()
            result += f"- {title} (ConfigMap: {cm.metadata.name})\n"
        
        result += "\nNota: Puedes acceder a ellos en la interfaz de Grafana (grafana.richardx.dev)."
        return result
    except Exception as e:
        return f"Excepción al listar dashboards vía K8s API: {str(e)}"

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
    ),
    Tool(
        name="Prometheus_Query",
        func=query_prometheus,
        description="Útil para ejecutar consultas PromQL y obtener métricas avanzadas (CPU, RAM, red, etc.) desde Prometheus."
    ),
    Tool(
        name="Listar_Grafana_Dashboards",
        func=list_grafana_dashboards,
        description="Útil para buscar dashboards en Grafana y obtener sus URLs."
    )
]

# Inicializar Agente con más iteraciones para tareas complejas
agent = initialize_agent(
    tools, 
    llm, 
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, 
    verbose=True,
    handle_parsing_errors="Check your output format. Remember to use 'Action:' and 'Action Input:' or 'Final Answer:'.",
    max_iterations=10,
    early_stopping_method="generate",
    agent_kwargs={
        "prefix": """Eres un SRE (Site Reliability Engineer) Senior y experto en Kubernetes. 
Tu objetivo es resolver problemas técnicos en el clúster de forma autónoma.
TIENES PERMISO para ejecutar acciones como listar pods, ver logs, eliminar pods y consultar métricas en Prometheus/New Relic.

Debes seguir SIEMPRE este formato EXACTO:
Thought: Describe tu razonamiento sobre qué hacer a continuación.
Action: El nombre de la herramienta a usar (debe ser una de las herramientas listadas abajo).
Action Input: El parámetro de entrada para la herramienta.
Observation: El resultado de la herramienta (esto lo recibirás tú).
... (puedes repetir Thought/Action/Action Input/Observation varias veces)
Thought: Cuando tengas la respuesta final.
Final Answer: La respuesta detallada y profesional para el usuario.

Herramientas disponibles:""",
        "suffix": """Pregunta del usuario: {input}
{agent_scratchpad}"""
    }
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
        # Ejecutar el agente con el callback de métricas
        raw_response = agent.run(request.query, callbacks=[metrics_callback])
        clean_response = extract_final_answer(raw_response)
        return {"response": clean_response}
    except Exception as e:
        error_msg = str(e)
        print(f"Agent error: {error_msg}")
        return {"response": f"El agente tuvo dificultades entendiendo la petición o la herramienta. Error: {error_msg[:150]}... Intenta ser más específico."}

@app.get("/health")
async def health_check():
    return {"status": "ok"}

# --- INSTRUMENTACIÓN ---
Instrumentator().instrument(app).expose(app)
