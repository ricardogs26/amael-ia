from fastapi import FastAPI, HTTPException, Request, Header
from pydantic import BaseModel
import os
import re
import logging
from kubernetes import client, config, stream
import requests
import json

logging.basicConfig(level=logging.INFO)
from langchain_ollama import OllamaLLM
from langchain.agents import initialize_agent, AgentType, Tool
from langchain.prompts import PromptTemplate
from langchain.callbacks.base import BaseCallbackHandler
from prometheus_fastapi_instrumentator import Instrumentator
from prometheus_client import Counter
from typing import Dict, Any

app = FastAPI(title="K8s Agentic AI Service")

# P7: OpenTelemetry — server-side spans so Tempo can build the service map
from tracing import instrument_app
instrument_app(app)

# --- CONOCIMIENTO DE VAULT ---
_VAULT_KNOWLEDGE = ""
_vault_kb_path = os.path.join(os.path.dirname(__file__), "vault_knowledge.md")
try:
    with open(_vault_kb_path) as _f:
        _VAULT_KNOWLEDGE = _f.read().replace("{", "{{").replace("}", "}}")
    logging.info("[VAULT_KB] Conocimiento de Vault cargado correctamente.")
except FileNotFoundError:
    logging.warning(f"[VAULT_KB] No se encontró {_vault_kb_path}. El agente no tendrá contexto de Vault.")

# --- CONOCIMIENTO DE MÉTRICAS PROMETHEUS ---
_METRICS_KNOWLEDGE = ""
_metrics_kb_path = os.path.join(os.path.dirname(__file__), "metrics_knowledge.md")
try:
    with open(_metrics_kb_path) as _f:
        _METRICS_KNOWLEDGE = _f.read().replace("{", "{{").replace("}", "}}")
    logging.info("[METRICS_KB] Conocimiento de métricas cargado correctamente.")
except FileNotFoundError:
    logging.warning(f"[METRICS_KB] No se encontró {_metrics_kb_path}.")

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
INTERNAL_API_SECRET = os.environ.get("INTERNAL_API_SECRET")
BACKEND_SRE_URL = os.environ.get("BACKEND_SRE_URL", "http://backend-service:8000/api/sre/query")

# --- LISTA BLANCA DE K8S ---
k8s_allowed_csv = os.environ.get("K8S_ALLOWED_USERS_CSV", "")
K8S_ALLOWED_USERS = [u.strip() for u in k8s_allowed_csv.split(',') if u.strip()]

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

def consult_vault_knowledge(query: str = "") -> str:
    """Usa esta herramienta para CUALQUIER pregunta sobre HashiCorp Vault: claves de unseal,
    secretos almacenados, políticas, roles, autenticación Kubernetes, tokens OAuth, arquitectura,
    comandos operacionales o troubleshooting de Vault. NO uses esta herramienta para problemas
    de pods o métricas de Kubernetes."""
    if not _VAULT_KNOWLEDGE:
        return "No se encontró la base de conocimiento de Vault."
    # Devuelve el conocimiento completo; el LLM extrae la sección relevante
    return _VAULT_KNOWLEDGE.replace("{{", "{").replace("}}", "}")


def consult_knowledge_base(issue_keywords: str = "") -> str:
    """Útil para consultar guías de resolución de problemas de Kubernetes y servicios de la aplicación
    (pods fallidos, errores de contenedores, problemas de red, etc.) mediante búsqueda vectorial.
    NO usar para preguntas sobre Vault — usa Consultar_Vault en su lugar."""
    try:
        # Petición a la API de RAG del backend
        response = requests.post(BACKEND_SRE_URL, json={
            "query": issue_keywords or "all"
        }, timeout=10)
        
        if response.status_code == 200:
            return response.json().get("response", "No se encontró información relevante.")
        else:
            return f"Error al consultar el backend (HTTP {response.status_code}): {response.text}"
    except Exception as e:
        return f"Error al consultar la base de conocimiento: {str(e)}"

# --- SEGURIDAD: Ejecutar_Comando_Contenedor ---
# Solo comandos de diagnóstico de solo lectura están permitidos.
# Cualquier comando destructivo o metacaracter de shell es bloqueado.
_ALLOWED_EXEC_COMMANDS = {
    "ls", "ps", "df", "du", "cat", "head", "tail",
    "find", "grep", "env", "id", "whoami", "uname",
    "date", "uptime", "free", "stat", "wc", "printenv",
}
# Metacaracteres que permiten inyección de shell
_BLOCKED_SHELL_METACHARACTERS = re.compile(r'[;&|`$><\\(){}\[\]!\n\r]')


def run_command_in_pod(input_str: str) -> str:
    """Comandos de diagnóstico de SOLO LECTURA dentro de un pod.
    Input: 'nombre-pod, comando'. Solo se permiten: ls, ps, df, du, cat, head,
    tail, find, grep, env, id, whoami, uname, date, uptime, free, stat, wc."""
    try:
        if "," not in input_str:
            return "Error: Formato inválido. Usa: 'nombre-pod, comando'. Ej: 'backend-pod, ls -la /tmp'"

        parts = input_str.split(",", 1)
        pod_name = parts[0].strip("'\" \n")
        command = parts[1].strip("'\" \n")

        # 1. Bloquear metacaracteres de shell (previene inyección)
        if _BLOCKED_SHELL_METACHARACTERS.search(command):
            logging.warning(
                f"[SECURITY] pod_exec bloqueado por metacaracteres. pod={pod_name!r} cmd={command!r}"
            )
            return "Error de seguridad: el comando contiene caracteres no permitidos (;, &, |, $, >, <, etc.)."

        # 2. Validar que el comando base esté en la whitelist
        tokens = command.split()
        if not tokens:
            return "Error: comando vacío."
        base_cmd = tokens[0].lower()
        if base_cmd not in _ALLOWED_EXEC_COMMANDS:
            logging.warning(
                f"[SECURITY] pod_exec bloqueado por comando no permitido. pod={pod_name!r} cmd={command!r}"
            )
            allowed_list = ", ".join(sorted(_ALLOWED_EXEC_COMMANDS))
            return (
                f"Error: el comando '{base_cmd}' no está permitido. "
                f"Solo se aceptan comandos de solo lectura: {allowed_list}."
            )

        # 3. Audit log de toda ejecución autorizada
        logging.info(f"[AUDIT] pod_exec pod={pod_name!r} namespace=amael-ia cmd={command!r}")

        exec_command = ["/bin/sh", "-c", command]
        resp = stream.stream(
            v1.connect_get_namespaced_pod_exec,
            pod_name,
            "amael-ia",
            command=exec_command,
            stderr=True,
            stdin=False,
            stdout=True,
            tty=False,
        )
        return f"Salida del comando en {pod_name}:\n{resp if resp else '(sin salida)'}"
    except Exception as e:
        return f"Error al ejecutar comando en el pod: {str(e)}"

# --- SISTEMA DE MANEJO DE MEDIA ---
# Almacenamos el contenido base64 fuera del contexto del LLM para evitar errores de parseo
latest_media_captured = None

def capture_grafana_screenshot(dashboard: str = "recursos") -> str:
    """Útil para obtener una captura de pantalla visual de Grafana. 
    Parámetro 'dashboard': puede ser 'recursos' (clúster) o 'rag' (rendimiento de RAG).
    Retorna un marcador que será reemplazado automáticamente por la imagen al final."""
    global latest_media_captured
    
    # Mapeo de nombres amigables a UIDs y rutas de Grafana
    DASHBOARD_MAP = {
        "recursos": "efa86fd1d0c121a26444b636a3f509a8/k8s-resources-cluster",
        "rag": "amael-rag/3-amael-rag-performance"
    }
    
    # Normalizar input
    db_key = "recursos"
    if "rag" in dashboard.lower() or "performance" in dashboard.lower():
        db_key = "rag"
    
    target_path = DASHBOARD_MAP.get(db_key, DASHBOARD_MAP["recursos"])
    
    GRAFANA_INTERNAL_URL = f"http://kube-prometheus-stack-grafana.observability.svc.cluster.local/d/{target_path}?orgId=1&refresh=10s"
    BRIDGE_URL = "http://whatsapp-bridge-service:3000/screenshot"
    
    print(f"Solicitando captura de dashboard: {db_key} -> {GRAFANA_INTERNAL_URL}")
    
    try:
        response = requests.post(BRIDGE_URL, json={
            "url": GRAFANA_INTERNAL_URL,
            "username": GRAFANA_USER,
            "password": GRAFANA_PASSWORD
        }, timeout=60)
        
        if response.status_code == 200:
            base64_data = response.json().get("base64")
            # Guardamos la data real en la variable global
            latest_media_captured = base64_data
            return f"ÉXITO: Captura de pantalla del dashboard '{db_key}' generada. El marcador [MEDIA_PLACEHOLDER] ha sido activado. Por favor, termina tu respuesta mencionando que adjuntas la captura."
        else:
            return f"Error al solicitar captura al bridge: {response.text}"
    except Exception as e:
        return f"Excepción al intentar capturar pantalla: {str(e)}"

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
    ),
    Tool(
        name="Capturar_Imagen_Grafana",
        func=capture_grafana_screenshot,
        description="Útil cuando el usuario pide ver una imagen de Grafana. Input: 'recursos' para consumo de CPU/RAM o 'rag' para métricas de rendimiento del RAG."
    ),
    Tool(
        name="Consultar_Vault",
        func=consult_vault_knowledge,
        description="USA ESTA HERRAMIENTA para cualquier pregunta sobre HashiCorp Vault: dónde están las claves de unseal, cómo dessellar, qué secretos hay, políticas, roles, autenticación Kubernetes, tokens OAuth de Google, arquitectura de Vault, comandos vault, troubleshooting de Vault. Input: la pregunta o tema sobre Vault."
    ),
    Tool(
        name="Consultar_Base_Conocimiento",
        func=consult_knowledge_base,
        description="USA ESTA HERRAMIENTA ante fallas de pods, errores de contenedores o comportamientos anómalos en Kubernetes. Contiene guías de remediación. NO usar para preguntas de Vault. Input: nombre del servicio o error."
    ),
    Tool(
        name="Ejecutar_Comando_Contenedor",
        func=run_command_in_pod,
        description="Ejecuta comandos de diagnóstico de SOLO LECTURA dentro de un pod (ls, ps, df, du, cat, head, tail, find, grep, env, id, whoami). NO admite comandos destructivos ni metacaracteres de shell. Input OBLIGATORIO: nombre del pod y comando separados por coma. Ej: 'pod-name, ls -la /tmp'"
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
        "prefix": (
            "Eres un SRE (Site Reliability Engineer) Senior y experto en Kubernetes y HashiCorp Vault.\n"
            "Tu objetivo es resolver problemas técnicos en el clúster de forma autónoma.\n"
            "TIENES PERMISO para ejecutar acciones como listar pods, ver logs, eliminar pods y consultar métricas en Prometheus/New Relic.\n\n"
            "=== CONOCIMIENTO DE HASHICORP VAULT ===\n"
            + _VAULT_KNOWLEDGE +
            "\n=== FIN CONOCIMIENTO VAULT ===\n\n"
            "=== MÉTRICAS PROMETHEUS — NOMBRES Y QUERIES CORRECTOS ===\n"
            + _METRICS_KNOWLEDGE +
            "\n=== FIN MÉTRICAS ===\n\n"
            "REGLA IMPORTANTE: Para cualquier pregunta sobre Vault (claves de unseal, secretos, políticas,\n"
            "roles, autenticación, tokens OAuth, arquitectura, comandos vault, etc.) DEBES usar la\n"
            "herramienta Consultar_Vault. NUNCA uses Consultar_Base_Conocimiento para temas de Vault.\n"
            "REGLA IMPORTANTE: Para preguntas de métricas o solicitudes HTTP, usa SIEMPRE los nombres\n"
            "de métricas y queries PromQL del bloque MÉTRICAS PROMETHEUS de arriba con Prometheus_Query.\n\n"
            "Debes seguir SIEMPRE este formato EXACTO:\n"
            "Thought: Describe tu razonamiento sobre qué hacer a continuación.\n"
            "Si detectas un pod con problemas (CrashLoopBackOff, Error, o que no inicializa), o ante CUALQUIER reporte de que un servicio no funciona (incluso en Running), tu PRIMERA acción DEBE SER Consultar_Base_Conocimiento para ver si hay una solución conocida.\n"
            "Si la guía recomienda un comando de limpieza, usa Ejecutar_Comando_Contenedor y luego Eliminar_Pod para reiniciarlo.\n\n"
            "Action: El nombre de la herramienta a usar (debe ser una de las herramientas listadas abajo).\n"
            "Action Input: El parámetro de entrada para la herramienta.\n"
            "Observation: El resultado de la herramienta (esto lo recibirás tú).\n"
            "... (puedes repetir Thought/Action/Action Input/Observation varias veces)\n"
            "Thought: Cuando tengas la respuesta final.\n"
            "Final Answer: La respuesta detallada y profesional para el usuario. Explica qué problema encontraste en la base de conocimiento y qué acciones correctivas aplicaste. SI HAS USADO 'Capturar_Imagen_Grafana', DEBES INCLUIR EL TEXTO EXACTO '[MEDIA_PLACEHOLDER]' EN TU RESPUESTA FINAL.\n\n"
            "Herramientas disponibles:"
        ),
        "suffix": """Pregunta del usuario: {input}
{agent_scratchpad}"""
    }
)

def extract_final_answer(raw_response: str) -> str:
    """Extrae solo el texto después de 'Final Answer:' para mostrar al usuario texto limpio."""
    global latest_media_captured
    
    marker = "Final Answer:"
    if marker in raw_response:
        res = raw_response.split(marker)[-1].strip()
    else:
        # Si el modelo filtró pensamiento interno sin dar Final Answer, limpiarlo
        lines_to_remove = ["Thought:", "Action:", "Action Input:", "Observation:"]
        cleaned_lines = []
        for line in raw_response.split("\n"):
            if not any(line.strip().startswith(prefix) for prefix in lines_to_remove):
                cleaned_lines.append(line)
        res = "\n".join(cleaned_lines).strip()
    
    # Si tenemos media capturada, la adjuntamos al final si el agente no lo hizo
    if latest_media_captured:
        res += f"\n\n[MEDIA:{latest_media_captured}]"
        # Limpiar para la siguiente petición
        latest_media_captured = None
        
    return res if res else raw_response

_VAULT_KEYWORDS = {
    "vault", "unseal", "dessellar", "sellar", "seal", "claves", "unseal key",
    "secret", "secreto", "policy", "política", "rol", "role", "kv", "hvac",
    "token oauth", "google token", "productivity", "auth/kubernetes",
    "amael-productivity", "vault.root", "root token",
}

def _is_vault_question(query: str) -> bool:
    q = query.lower()
    return any(kw in q for kw in _VAULT_KEYWORDS)


@app.post("/api/k8s-agent")
async def chat_with_agent(request: AgentRequest, req: Request):
    AGENT_REQUESTS_TOTAL.inc()
    global latest_media_captured
    latest_media_captured = None # Reset al inicio de cada petición

    # P4-1: Validar INTERNAL_API_SECRET en cabecera Authorization
    if INTERNAL_API_SECRET:
        auth_header = req.headers.get("Authorization", "")
        token = auth_header.removeprefix("Bearer ").strip()
        if token != INTERNAL_API_SECRET:
            logging.warning(f"[SECURITY] Petición rechazada a /api/k8s-agent: secret inválido desde {req.client.host}")
            raise HTTPException(status_code=403, detail="Acceso no autorizado.")

    logging.info(f"Recibiendo petición de {request.user_email}: {request.query[:80]}")

    # --- VALIDACIÓN DE SEGURIDAD ---
    if K8S_ALLOWED_USERS and request.user_email not in K8S_ALLOWED_USERS:
        print(f"BLOQUEO: El usuario {request.user_email} NO está en la whitelist de K8s.")
        return {"response": "Lo siento, no tienes permisos para acceder a información del clúster de Kubernetes o dashboards de rendimiento."}

    # --- RESPUESTA DIRECTA PARA PREGUNTAS DE VAULT (sin pasar por el agente LLM) ---
    if _is_vault_question(request.query) and _VAULT_KNOWLEDGE:
        logging.info(f"[VAULT_KB] Pregunta de Vault detectada, respondiendo desde KB local.")
        vault_raw = _VAULT_KNOWLEDGE.replace("{{", "{").replace("}}", "}")
        prompt = (
            f"Eres un experto en HashiCorp Vault. Basándote ÚNICAMENTE en el siguiente "
            f"documento de referencia, responde la pregunta del usuario de forma concisa y precisa.\n\n"
            f"DOCUMENTO:\n{vault_raw}\n\n"
            f"PREGUNTA: {request.query}\n\n"
            f"RESPUESTA:"
        )
        try:
            answer = llm.invoke(prompt)
            return {"response": str(answer).strip()}
        except Exception as exc:
            logging.error(f"[VAULT_KB] Error invocando LLM para pregunta de Vault: {exc}")
            # Fallback: devolver la sección relevante directamente
            return {"response": vault_raw}

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
