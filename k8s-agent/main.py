from __future__ import annotations
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
import os
import re
import time
import json
import logging
import threading
import concurrent.futures
import dataclasses
import glob as _glob
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional, List

logging.basicConfig(level=logging.INFO)
from kubernetes import client, config, stream
import requests
from langchain_ollama import OllamaLLM
from langchain.agents import initialize_agent, AgentType, Tool
from langchain.callbacks.base import BaseCallbackHandler
from prometheus_fastapi_instrumentator import Instrumentator
from prometheus_client import Counter, Gauge, Histogram
from typing import Dict, Any

app = FastAPI(title="K8s Agentic AI Service — Autonomous SRE")

from tracing import instrument_app
instrument_app(app)

# ─────────────────────────────────────────────────────────────────────────────
# KNOWLEDGE BASES
# ─────────────────────────────────────────────────────────────────────────────
def _load_kb(filename: str, label: str) -> str:
    path = os.path.join(os.path.dirname(__file__), filename)
    try:
        with open(path) as f:
            content = f.read().replace("{", "{{").replace("}", "}}")
        logging.info(f"[{label}] Cargado correctamente.")
        return content
    except FileNotFoundError:
        logging.warning(f"[{label}] No se encontró {path}.")
        return ""

_VAULT_KNOWLEDGE   = _load_kb("vault_knowledge.md",   "VAULT_KB")
_METRICS_KNOWLEDGE = _load_kb("metrics_knowledge.md", "METRICS_KB")

# ─────────────────────────────────────────────────────────────────────────────
# MÉTRICAS PROMETHEUS
# ─────────────────────────────────────────────────────────────────────────────
AGENT_STEPS_TOTAL       = Counter('amael_agent_steps_total',       'Total steps taken by the agent')
AGENT_TOOLS_USAGE_TOTAL = Counter('amael_agent_tools_usage_total', 'Tool usage', ['tool'])
AGENT_REQUESTS_TOTAL    = Counter('amael_agent_requests_total',    'Requests to /api/k8s-agent')
SRE_NOTIFY_TOTAL        = Counter('amael_sre_notify_total',        'WhatsApp SRE alerts', ['severity'])

# P1 — Loop
SRE_LOOP_RUNS_TOTAL    = Counter('amael_sre_loop_runs_total',          'SRE loop iterations', ['result'])
SRE_ANOMALIES_DETECTED = Counter('amael_sre_anomalies_detected_total', 'Anomalies detected',  ['severity', 'issue_type'])
SRE_ACTIONS_TAKEN      = Counter('amael_sre_actions_taken_total',      'Remediation actions', ['action', 'result'])
SRE_CB_STATE           = Gauge  ('amael_sre_circuit_breaker_state',    '0=closed 1=open 2=half_open')

# P2 — Diagnosis
SRE_DIAGNOSIS_CONFIDENCE = Histogram(
    'amael_sre_diagnosis_confidence',
    'LLM diagnosis confidence score',
    buckets=[0.3, 0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0],
)
SRE_DIAGNOSIS_LLM_TOTAL   = Counter('amael_sre_diagnosis_llm_total',         'LLM scorer calls', ['result'])
SRE_RESTART_LIMIT_HIT     = Counter('amael_sre_restart_limit_hit_total',     'Restart limit reached')
SRE_RUNBOOK_HITS_TOTAL    = Counter('amael_sre_runbook_hits_total',           'Runbook matches found')

# P3 — Verification, learning, LangGraph
SRE_VERIFICATION_TOTAL      = Counter('amael_sre_verification_total',         'Post-action verifications', ['result'])
SRE_LEARNING_ADJUSTED_TOTAL = Counter('amael_sre_learning_adjusted_total',    'Confidence adjusted by history')
SRE_LANGGRAPH_REQUESTS      = Counter('amael_sre_langgraph_requests_total',   'LangGraph agent requests', ['result'])

# P4 — Proactive monitoring, correlation, maintenance, auto-runbooks
SRE_METRIC_ANOMALIES_TOTAL  = Counter('amael_sre_metric_anomalies_total',     'Proactive metric anomalies', ['issue_type'])
SRE_CORRELATION_GROUPED     = Counter('amael_sre_correlation_grouped_total',  'Multi-pod anomalies correlated')
SRE_AUTO_RUNBOOK_SAVED      = Counter('amael_sre_auto_runbook_saved_total',   'Auto-generated runbook entries saved')
SRE_MAINTENANCE_ACTIVE      = Gauge  ('amael_sre_maintenance_active',         '1 if a maintenance window is active')

# P5 — Predictive alerting, rollback, SLO, postmortem, WhatsApp commands
SRE_TREND_ANOMALIES_TOTAL   = Counter('amael_sre_trend_anomalies_total',      'Predictive trend anomalies', ['issue_type'])
SRE_SLO_VIOLATIONS_TOTAL    = Counter('amael_sre_slo_violations_total',       'SLO budget violations', ['service'])
SRE_ROLLBACK_TOTAL          = Counter('amael_sre_rollback_total',             'Automated rollbacks', ['result'])
SRE_POSTMORTEM_TOTAL        = Counter('amael_sre_postmortem_total',           'Auto-generated postmortems')
SRE_WA_COMMANDS_TOTAL       = Counter('amael_sre_wa_commands_total',          'WhatsApp SRE commands received', ['command'])

# ─────────────────────────────────────────────────────────────────────────────
# MODELOS
# ─────────────────────────────────────────────────────────────────────────────
class AgentRequest(BaseModel):
    query: str
    user_email: str = "unknown"


class SRECommandRequest(BaseModel):
    """P5-E: WhatsApp SRE command routed from whatsapp-bridge."""
    command: str        # e.g. "status", "incidents", "slo", "maintenance on 60"
    phone:   str = ""   # sender phone number (for reply routing)

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURACIÓN — operacional
# ─────────────────────────────────────────────────────────────────────────────
OLLAMA_BASE_URL      = os.environ.get("OLLAMA_BASE_URL",      "http://ollama-service:11434")
MODEL_NAME           = os.environ.get("MODEL_NAME",           "qwen2.5:14b")
DEFAULT_NAMESPACE    = os.environ.get("DEFAULT_NAMESPACE",    "amael-ia")
PROMETHEUS_URL       = os.environ.get("PROMETHEUS_URL",
    "http://kube-prometheus-stack-prometheus.observability.svc.cluster.local:9090")
GRAFANA_USER         = os.environ.get("GRAFANA_USER",  "admin")
GRAFANA_PASSWORD     = os.environ.get("GRAFANA_PASSWORD", "admin")
INTERNAL_API_SECRET  = os.environ.get("INTERNAL_API_SECRET")
BACKEND_SRE_URL      = os.environ.get("BACKEND_SRE_URL", "http://backend-service:8000/api/sre/query")
WHATSAPP_BRIDGE_URL  = os.environ.get("WHATSAPP_BRIDGE_URL", "http://whatsapp-bridge-service:3000")
OWNER_PHONE          = os.environ.get("OWNER_PHONE", "")

# P1 — Loop
SRE_LOOP_ENABLED  = os.environ.get("SRE_LOOP_ENABLED",  "true").lower() == "true"
SRE_LOOP_INTERVAL = int(os.environ.get("SRE_LOOP_INTERVAL", "60"))
OBSERVE_NAMESPACES = [
    ns.strip()
    for ns in os.environ.get("SRE_OBSERVE_NAMESPACES", "amael-ia,vault,observability,kong").split(",")
    if ns.strip()
]

# P1 — PostgreSQL
POSTGRES_HOST = os.environ.get("POSTGRES_HOST", "postgres-service")
POSTGRES_DB   = os.environ.get("POSTGRES_DB",   "amael_db")
POSTGRES_USER = os.environ.get("POSTGRES_USER", "amael_user")
POSTGRES_PASS = os.environ.get("POSTGRES_PASSWORD", "")

# P2 — Qdrant runbooks
QDRANT_URL = os.environ.get("QDRANT_URL", "http://qdrant-service:6333")
SRE_RUNBOOKS_COLLECTION = "sre_runbooks"

# P3 — Lease / leader election
_POD_NAME      = os.environ.get("POD_NAME", "k8s-agent-single")
_LEASE_NAME    = "sre-agent-leader"
_LEASE_DURATION_S = 90  # seconds a lease is valid before expiry

# P4 — Proactive metric thresholds
SRE_CPU_THRESHOLD    = float(os.environ.get("SRE_CPU_THRESHOLD",    "0.85"))
SRE_MEMORY_THRESHOLD = float(os.environ.get("SRE_MEMORY_THRESHOLD", "0.85"))
_MAINTENANCE_KEY     = "sre:maintenance:active"

# P5 — Predictive alerting & SLO
SRE_MEMORY_LEAK_RATE_BYTES = int(os.environ.get("SRE_MEMORY_LEAK_RATE_BYTES", str(1024 * 1024)))  # 1 MB/s
_SLO_TARGETS: List[dict] = []  # loaded at startup from SLO_TARGETS_JSON env var

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURACIÓN — política (desde sre-agent-policy ConfigMap)
# ─────────────────────────────────────────────────────────────────────────────
k8s_allowed_csv = os.environ.get("K8S_ALLOWED_USERS_CSV", "")
K8S_ALLOWED_USERS = [u.strip() for u in k8s_allowed_csv.split(',') if u.strip()]

_PROTECTED_CSV = os.environ.get("SRE_PROTECTED_DEPLOYMENTS",
                                "postgres-deployment,ollama-deployment,vault-0")
PROTECTED_DEPLOYMENTS = {d.strip() for d in _PROTECTED_CSV.split(',') if d.strip()}

AUTO_HEAL_MIN_SEVERITY   = os.environ.get("SRE_AUTO_HEAL_MIN_SEVERITY",   "HIGH")
CONFIDENCE_THRESHOLD     = float(os.environ.get("SRE_CONFIDENCE_THRESHOLD",     "0.75"))
MAX_RESTARTS_PER_RESOURCE = int(os.environ.get("SRE_MAX_RESTARTS_PER_RESOURCE", "3"))
RESTART_WINDOW_MINUTES   = int(os.environ.get("SRE_RESTART_WINDOW_MINUTES",    "15"))

_SEVERITY_RANK = {"CRITICAL": 3, "HIGH": 2, "MEDIUM": 1, "LOW": 0}

# ─────────────────────────────────────────────────────────────────────────────
# REDIS
# ─────────────────────────────────────────────────────────────────────────────
_redis = None
_dedup_cache: dict[str, float] = {}
_DEDUP_TTL = 600

try:
    import redis as _redis_lib
    _redis = _redis_lib.Redis.from_url(
        os.environ.get("REDIS_URL", "redis://redis-service:6379/0"),
        decode_responses=True, socket_connect_timeout=2, socket_timeout=2,
    )
    _redis.ping()
    logging.info("[REDIS] Conexión establecida.")
except Exception as _e:
    logging.warning(f"[REDIS] No disponible ({_e}). Usando fallback en memoria.")
    _redis = None


def _is_duplicate_incident(key: str) -> bool:
    if _redis:
        return _redis.exists(f"sre:incident:{key}") == 1
    now = time.time()
    if key in _dedup_cache and now - _dedup_cache[key] < _DEDUP_TTL:
        return True
    _dedup_cache.pop(key, None)
    return False


def _mark_incident(key: str):
    if _redis:
        _redis.set(f"sre:incident:{key}", "1", ex=_DEDUP_TTL)
    else:
        _dedup_cache[key] = time.time()


# P2 — restart counter (guardrail de límite de reinicios)
def _check_restart_limit(resource_name: str, namespace: str) -> bool:
    """True si el recurso alcanzó el máximo de reinicios automáticos en la ventana."""
    if not _redis:
        return False
    key   = f"sre:restarts:{namespace}:{resource_name}"
    count = int(_redis.get(key) or 0)
    return count >= MAX_RESTARTS_PER_RESOURCE


def _record_restart(resource_name: str, namespace: str):
    """Incrementa el contador de reinicios automáticos (con TTL de ventana)."""
    if not _redis:
        return
    key = f"sre:restarts:{namespace}:{resource_name}"
    pipe = _redis.pipeline()
    pipe.incr(key)
    pipe.expire(key, RESTART_WINDOW_MINUTES * 60)
    pipe.execute()


# ─────────────────────────────────────────────────────────────────────────────
# POSTGRESQL
# ─────────────────────────────────────────────────────────────────────────────
_postgres_pool = None
_last_pool_attempt = 0.0
_POOL_RETRY_INTERVAL = 30


def get_postgres_pool():
    global _postgres_pool, _last_pool_attempt
    if _postgres_pool is not None:
        return _postgres_pool
    now = time.time()
    if now - _last_pool_attempt < _POOL_RETRY_INTERVAL:
        return None
    _last_pool_attempt = now
    try:
        import psycopg2.pool as pg_pool
        _postgres_pool = pg_pool.SimpleConnectionPool(
            1, 5,
            host=POSTGRES_HOST, database=POSTGRES_DB,
            user=POSTGRES_USER, password=POSTGRES_PASS,
        )
        logging.info("[PG] Pool PostgreSQL creado.")
        return _postgres_pool
    except Exception as e:
        logging.error(f"[PG] Error creando pool: {e}")
        return None


def init_sre_db():
    pool = get_postgres_pool()
    if not pool:
        logging.warning("[SRE_DB] PostgreSQL no disponible.")
        return
    conn = pool.getconn()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS sre_incidents (
                    id            SERIAL PRIMARY KEY,
                    incident_key  TEXT UNIQUE,
                    created_at    TIMESTAMPTZ DEFAULT now(),
                    namespace     TEXT,
                    resource_name TEXT,
                    resource_type TEXT,
                    issue_type    TEXT,
                    severity      TEXT,
                    details       TEXT,
                    root_cause    TEXT,
                    confidence    FLOAT,
                    action_taken  TEXT,
                    action_result TEXT,
                    notified      BOOLEAN DEFAULT false
                );
            """)
            cur.execute(
                "CREATE INDEX IF NOT EXISTS idx_sre_ts ON sre_incidents(created_at DESC);"
            )
            cur.execute(
                "CREATE INDEX IF NOT EXISTS idx_sre_issue ON sre_incidents(issue_type, namespace);"
            )
            # P5-D: Postmortem table
            cur.execute("""
                CREATE TABLE IF NOT EXISTS sre_postmortems (
                    id                  SERIAL PRIMARY KEY,
                    incident_key        TEXT UNIQUE,
                    created_at          TIMESTAMPTZ DEFAULT now(),
                    namespace           TEXT,
                    resource_name       TEXT,
                    issue_type          TEXT,
                    impact              TEXT,
                    timeline            TEXT,
                    root_cause_summary  TEXT,
                    resolution          TEXT,
                    prevention          TEXT,
                    action_items        TEXT,
                    raw_json            TEXT
                );
            """)
            cur.execute(
                "CREATE INDEX IF NOT EXISTS idx_pm_ts ON sre_postmortems(created_at DESC);"
            )
            conn.commit()
        logging.info("[SRE_DB] Tablas sre_incidents y sre_postmortems listas.")
    except Exception as e:
        logging.error(f"[SRE_DB] Error init: {e}")
        conn.rollback()
    finally:
        pool.putconn(conn)


def store_incident(incident_key: str, namespace: str, resource_name: str,
                   resource_type: str, issue_type: str, severity: str,
                   details: str, root_cause: str, confidence: float,
                   action_taken: str, action_result: str, notified: bool = False):
    pool = get_postgres_pool()
    if not pool:
        return
    conn = pool.getconn()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO sre_incidents
                    (incident_key, namespace, resource_name, resource_type,
                     issue_type, severity, details, root_cause, confidence,
                     action_taken, action_result, notified)
                VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                ON CONFLICT (incident_key) DO NOTHING;
            """, (incident_key, namespace, resource_name, resource_type,
                  issue_type, severity, details, root_cause, confidence,
                  action_taken, action_result, notified))
            conn.commit()
    except Exception as e:
        logging.error(f"[SRE_DB] store_incident error: {e}")
        conn.rollback()
    finally:
        pool.putconn(conn)


def _update_incident_verification(incident_key: str, verification_result: str):
    """Update sre_incidents.action_result with post-action verification outcome."""
    pool = get_postgres_pool()
    if not pool:
        return
    conn = pool.getconn()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                UPDATE sre_incidents
                SET action_result = COALESCE(action_result, '') || ' [verify:' || %s || ']'
                WHERE incident_key = %s;
            """, (verification_result, incident_key))
            conn.commit()
    except Exception as e:
        logging.error(f"[VERIFY_DB] Error: {e}")
        conn.rollback()
    finally:
        pool.putconn(conn)


def get_historical_success_rate(issue_type: str, owner_name: str, namespace: str) -> Optional[float]:
    """
    Returns the rollout_restart success rate [0.0–1.0] for the same issue_type+resource
    based on the last 10 incidents. Returns None if insufficient data.
    """
    pool = get_postgres_pool()
    if not pool or not owner_name:
        return None
    conn = pool.getconn()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT action_result
                FROM sre_incidents
                WHERE issue_type     = %s
                  AND namespace      = %s
                  AND resource_name  LIKE %s
                  AND action_taken   = 'ROLLOUT_RESTART'
                ORDER BY created_at DESC
                LIMIT 10;
            """, (issue_type, namespace, f"{owner_name}%"))
            rows = cur.fetchall()
        if not rows:
            return None
        total     = len(rows)
        successes = sum(
            1 for r in rows
            if r[0] and '✅' in r[0] and 'verify:unresolved' not in r[0]
        )
        return round(successes / total, 3)
    except Exception as e:
        logging.warning(f"[LEARNING] Error en historical query: {e}")
        return None
    finally:
        pool.putconn(conn)


def get_learning_stats() -> dict:
    """Returns aggregate statistics for the learning endpoint."""
    pool = get_postgres_pool()
    if not pool:
        return {}
    conn = pool.getconn()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT issue_type, action_taken,
                       COUNT(*) AS total,
                       SUM(CASE WHEN action_result LIKE '%✅%' AND
                                     action_result NOT LIKE '%verify:unresolved%'
                                THEN 1 ELSE 0 END) AS successes,
                       AVG(confidence) AS avg_confidence
                FROM sre_incidents
                WHERE created_at > now() - INTERVAL '7 days'
                GROUP BY issue_type, action_taken
                ORDER BY total DESC;
            """)
            rows = cur.fetchall()
        return [
            {
                "issue_type":      r[0],
                "action":          r[1],
                "total":           r[2],
                "successes":       r[3],
                "success_rate":    round(r[3] / r[2], 2) if r[2] else 0,
                "avg_confidence":  round(float(r[4]), 2) if r[4] else None,
            }
            for r in rows
        ]
    except Exception as e:
        logging.error(f"[LEARNING_STATS] Error: {e}")
        return []
    finally:
        pool.putconn(conn)


def get_recent_incidents(limit: int = 20) -> list:
    pool = get_postgres_pool()
    if not pool:
        return []
    conn = pool.getconn()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT incident_key, created_at, namespace, resource_name,
                       issue_type, severity, root_cause, confidence,
                       action_taken, action_result
                FROM sre_incidents
                ORDER BY created_at DESC LIMIT %s;
            """, (limit,))
            rows = cur.fetchall()
            return [
                {
                    "incident_key": r[0],
                    "created_at":   r[1].isoformat() if r[1] else None,
                    "namespace":    r[2],
                    "resource":     r[3],
                    "issue_type":   r[4],
                    "severity":     r[5],
                    "root_cause":   r[6],
                    "confidence":   r[7],
                    "action":       r[8],
                    "result":       r[9],
                }
                for r in rows
            ]
    except Exception as e:
        logging.error(f"[SRE_DB] get_recent_incidents error: {e}")
        return []
    finally:
        pool.putconn(conn)


# ─────────────────────────────────────────────────────────────────────────────
# QDRANT — runbooks locales (P2 #12)
# ─────────────────────────────────────────────────────────────────────────────
_qdrant_client = None


def _get_embedding(text: str) -> list:
    """Genera embedding con nomic-embed-text vía Ollama (768 dims)."""
    try:
        resp = requests.post(
            f"{OLLAMA_BASE_URL}/api/embeddings",
            json={"model": "nomic-embed-text", "prompt": text},
            timeout=15,
        )
        if resp.status_code == 200:
            return resp.json().get("embedding", [])
    except Exception as e:
        logging.warning(f"[EMBED] Error: {e}")
    return []


def init_runbooks_qdrant():
    """Inicializa el cliente Qdrant y crea/llena la colección sre_runbooks si no existe."""
    global _qdrant_client
    try:
        from qdrant_client import QdrantClient
        from qdrant_client.http.models import Distance, VectorParams, PointStruct

        _qdrant_client = QdrantClient(url=QDRANT_URL, timeout=10)

        # Crear colección si no existe
        if not _qdrant_client.collection_exists(SRE_RUNBOOKS_COLLECTION):
            _qdrant_client.create_collection(
                collection_name=SRE_RUNBOOKS_COLLECTION,
                vectors_config=VectorParams(size=768, distance=Distance.COSINE),
            )
            logging.info(f"[RUNBOOKS] Colección '{SRE_RUNBOOKS_COLLECTION}' creada.")
            _ingest_runbooks()
        else:
            info = _qdrant_client.get_collection(SRE_RUNBOOKS_COLLECTION)
            count = info.points_count or 0
            if count == 0:
                logging.info("[RUNBOOKS] Colección vacía, ingresando runbooks.")
                _ingest_runbooks()
            else:
                logging.info(f"[RUNBOOKS] Colección lista con {count} documentos.")
    except Exception as e:
        logging.warning(f"[RUNBOOKS] Qdrant no disponible ({e}). RAG de runbooks deshabilitado.")
        _qdrant_client = None


def _ingest_runbooks():
    """Lee todos los *.md de runbooks/ y los indexa en Qdrant."""
    from qdrant_client.http.models import PointStruct

    runbooks_dir = os.path.join(os.path.dirname(__file__), "runbooks")
    files = _glob.glob(os.path.join(runbooks_dir, "*.md"))
    if not files:
        logging.warning("[RUNBOOKS] No se encontraron archivos .md en runbooks/")
        return

    points = []
    for idx, filepath in enumerate(files):
        try:
            with open(filepath) as f:
                content = f.read()
            # Generar embedding del contenido completo
            vector = _get_embedding(content[:2000])  # primeros 2000 chars
            if not vector:
                logging.warning(f"[RUNBOOKS] Sin embedding para {filepath}")
                continue
            name = os.path.basename(filepath).replace(".md", "")
            points.append(PointStruct(
                id=idx + 1,
                vector=vector,
                payload={
                    "name":    name,
                    "content": content,
                    "file":    os.path.basename(filepath),
                },
            ))
            logging.info(f"[RUNBOOKS] Indexado: {name}")
        except Exception as e:
            logging.warning(f"[RUNBOOKS] Error indexando {filepath}: {e}")

    if points:
        _qdrant_client.upsert(
            collection_name=SRE_RUNBOOKS_COLLECTION, points=points
        )
        logging.info(f"[RUNBOOKS] {len(points)} runbooks indexados en Qdrant.")


def search_runbooks(query: str, score_threshold: float = 0.55) -> str:
    """Busca el runbook más relevante para la query. Retorna el contenido o ''."""
    if _qdrant_client is None:
        return ""
    try:
        query_vec = _get_embedding(query)
        if not query_vec:
            return ""
        # qdrant-client >= 1.7: query_points() reemplaza search()
        try:
            response = _qdrant_client.query_points(
                collection_name=SRE_RUNBOOKS_COLLECTION,
                query=query_vec,
                limit=1,
                score_threshold=score_threshold,
            )
            results = response.points
        except AttributeError:
            # Fallback para qdrant-client < 1.7
            results = _qdrant_client.search(
                collection_name=SRE_RUNBOOKS_COLLECTION,
                query_vector=query_vec,
                limit=1,
                score_threshold=score_threshold,
            )
        if results:
            SRE_RUNBOOK_HITS_TOTAL.inc()
            content = results[0].payload.get("content", "")
            name    = results[0].payload.get("name", "")
            logging.debug(f"[RUNBOOKS] Match: {name} (score={results[0].score:.2f})")
            return content
    except Exception as e:
        logging.warning(f"[RUNBOOKS] Error en búsqueda: {e}")
    return ""


# ─────────────────────────────────────────────────────────────────────────────
# P5-B: ROLLOUT UNDO
# ─────────────────────────────────────────────────────────────────────────────

def _was_recently_deployed(deployment_name: str, namespace: str,
                            within_minutes: int = 30) -> bool:
    """
    Returns True if the deployment had a new rollout in the last N minutes.
    Detected via the 'Progressing' condition lastUpdateTime (changed by image updates,
    not by our rollout_restart annotations).
    """
    try:
        deploy  = apps_v1.read_namespaced_deployment(deployment_name, namespace)
        now     = datetime.now(timezone.utc)
        for cond in (deploy.status.conditions or []):
            if cond.type == "Progressing" and cond.last_update_time:
                t      = cond.last_update_time
                t_utc  = t if t.tzinfo else t.replace(tzinfo=timezone.utc)
                if (now - t_utc).total_seconds() < within_minutes * 60:
                    return True
    except Exception:
        pass
    return False


def rollout_undo_deployment(input_str: str) -> str:
    """
    P5-B: Rolls back a deployment to its previous revision by finding the
    previous ReplicaSet and patching the deployment template to match it.
    Equivalent to `kubectl rollout undo deployment/<name>`.
    """
    deployment_name, namespace = _parse_two(input_str)
    if deployment_name in PROTECTED_DEPLOYMENTS:
        return f"Error: '{deployment_name}' está protegido."
    try:
        logging.info(f"[AUDIT] rollout_undo deployment={deployment_name!r} ns={namespace!r}")
        deploy      = apps_v1.read_namespaced_deployment(deployment_name, namespace)
        current_rev = int(
            (deploy.metadata.annotations or {}).get("deployment.kubernetes.io/revision", "1")
        )
        if current_rev <= 1:
            return f"No hay revisión anterior para '{deployment_name}' (rev actual={current_rev})."

        target_rev = str(current_rev - 1)
        rs_list    = apps_v1.list_namespaced_replica_set(namespace)
        target_rs  = None
        for rs in rs_list.items:
            if not rs.metadata.owner_references:
                continue
            for ref in rs.metadata.owner_references:
                if ref.kind == "Deployment" and ref.name == deployment_name:
                    rs_rev = (rs.metadata.annotations or {}).get(
                        "deployment.kubernetes.io/revision", "0"
                    )
                    if rs_rev == target_rev:
                        target_rs = rs
                        break
            if target_rs:
                break

        if target_rs is None:
            return f"No se encontró revisión {target_rev} de '{deployment_name}'."

        # Build template patch from the previous RS
        template = apps_v1.api_client.sanitize_for_serialization(target_rs.spec.template)
        # Remove restartedAt annotation to avoid triggering another rollout_restart cycle
        ann = (template.get("metadata") or {}).get("annotations") or {}
        ann.pop("kubectl.kubernetes.io/restartedAt", None)

        apps_v1.patch_namespaced_deployment(
            deployment_name, namespace, {"spec": {"template": template}}
        )
        SRE_ROLLBACK_TOTAL.labels(result="ok").inc()
        return (f"✅ Rollback a revisión {target_rev} iniciado en "
                f"'{deployment_name}' ({namespace}).")
    except Exception as e:
        SRE_ROLLBACK_TOTAL.labels(result="error").inc()
        return f"Error al hacer rollback de '{deployment_name}': {e}"


# ─────────────────────────────────────────────────────────────────────────────
# P5-C: SLO / ERROR BUDGET TRACKING
# ─────────────────────────────────────────────────────────────────────────────

def load_slo_targets():
    """Loads SLO targets from SLO_TARGETS_JSON environment variable at startup."""
    global _SLO_TARGETS
    raw = os.environ.get("SLO_TARGETS_JSON", "[]")
    try:
        _SLO_TARGETS = json.loads(raw)
        logging.info(f"[SLO] {len(_SLO_TARGETS)} SLO targets cargados.")
    except json.JSONDecodeError as e:
        logging.warning(f"[SLO] Error parseando SLO_TARGETS_JSON: {e}")
        _SLO_TARGETS = []


def observe_slo() -> List[Anomaly]:
    """
    P5-C: Checks if the current error rate is burning the SLO error budget
    faster than 2× the sustainable rate. Alerts before budget exhaustion.
    """
    if not _SLO_TARGETS:
        return []
    anomalies: List[Anomaly] = []

    for slo in _SLO_TARGETS:
        handler      = slo.get("handler", "")
        target       = slo.get("availability", 0.995)
        window_h     = slo.get("window_h", 24)
        error_budget = 1.0 - target   # e.g. 0.005 for 99.5%
        if error_budget <= 0:
            continue

        err_q = (
            f'sum(rate(http_requests_total{{namespace="amael-ia",'
            f'handler=~"{handler}",status=~"5.."}}[{window_h}h]))'
            f' / sum(rate(http_requests_total{{namespace="amael-ia",'
            f'handler=~"{handler}"}}[{window_h}h]))'
        )
        try:
            resp = requests.get(f"{PROMETHEUS_URL}/api/v1/query",
                                params={"query": err_q}, timeout=8)
            if resp.status_code != 200 or resp.json()["status"] != "success":
                continue
            results = resp.json()["data"]["result"]
            if not results:
                continue
            current_err_rate = float(results[0]["value"][1])
            burn_rate = current_err_rate / error_budget

            if burn_rate < 2.0:
                continue   # Within acceptable range

            severity = "CRITICAL" if burn_rate >= 5.0 else "HIGH"
            # Estimate remaining budget hours
            consumed = current_err_rate * window_h
            remaining_h = max(0.0, (error_budget - consumed) / error_budget * window_h)
            service = slo.get("service", handler)

            anomalies.append(Anomaly(
                issue_type="SLO_BUDGET_BURNING",
                severity=severity,
                namespace="amael-ia",
                resource_name=handler,
                resource_type="endpoint",
                owner_name=service,
                owner_kind="Deployment",
                details=(f"SLO '{handler}': burn_rate={burn_rate:.1f}× "
                         f"(objetivo: {target:.1%}, error actual: {current_err_rate:.2%}). "
                         f"Budget restante ≈{remaining_h:.1f}h."),
                dedup_key=f"amael-ia:{handler}:SLO_BURN",
            ))
            SRE_SLO_VIOLATIONS_TOTAL.labels(service=service).inc()
        except Exception as e:
            logging.debug(f"[SLO] Error para '{handler}': {e}")

    if anomalies:
        logging.info(f"[SLO] {len(anomalies)} violación(es) de SLO detectadas.")
    return anomalies


# ─────────────────────────────────────────────────────────────────────────────
# P5-A: PREDICTIVE TREND ALERTING
# ─────────────────────────────────────────────────────────────────────────────

def observe_trends() -> List[Anomaly]:
    """
    P5-A: Uses predict_linear() and deriv() to detect trends before incidents.
    Detects: DISK_EXHAUSTION_PREDICTED, MEMORY_LEAK_PREDICTED, ERROR_RATE_ESCALATING.
    """
    anomalies: List[Anomaly] = []
    ns_regex = "|".join(OBSERVE_NAMESPACES)

    # ── Disk exhaustion in < 4 hours ─────────────────────────────────────────
    disk_q = (
        'predict_linear(node_filesystem_avail_bytes{mountpoint="/"}[1h], 4 * 3600) < 0'
    )
    try:
        resp = requests.get(f"{PROMETHEUS_URL}/api/v1/query",
                            params={"query": disk_q}, timeout=8)
        if resp.status_code == 200 and resp.json()["status"] == "success":
            for r in resp.json()["data"]["result"]:
                instance = r["metric"].get("instance", "unknown")
                anomalies.append(Anomaly(
                    issue_type="DISK_EXHAUSTION_PREDICTED",
                    severity="HIGH",
                    namespace="",
                    resource_name=instance,
                    resource_type="node",
                    owner_name="",
                    owner_kind="Node",
                    details=(f"Nodo '{instance}': disco agotará espacio en <4 horas "
                             f"según predict_linear (tendencia de 1h)."),
                    dedup_key=f"node:{instance}:DISK_EXHAUSTION_PREDICTED",
                ))
                SRE_TREND_ANOMALIES_TOTAL.labels(issue_type="DISK_EXHAUSTION_PREDICTED").inc()
    except Exception as e:
        logging.debug(f"[TRENDS] Disk predict error: {e}")

    # ── Memory leak: steady positive deriv > threshold ────────────────────────
    mem_leak_q = (
        f'deriv(container_memory_working_set_bytes{{container!="",namespace=~"{ns_regex}"}}[15m])'
        f' > {SRE_MEMORY_LEAK_RATE_BYTES}'
    )
    try:
        resp = requests.get(f"{PROMETHEUS_URL}/api/v1/query",
                            params={"query": mem_leak_q}, timeout=8)
        if resp.status_code == 200 and resp.json()["status"] == "success":
            for r in resp.json()["data"]["result"]:
                pod       = r["metric"].get("pod", "")
                container = r["metric"].get("container", "")
                ns        = r["metric"].get("namespace", "")
                val       = float(r["value"][1])
                if not pod or not ns:
                    continue
                anomalies.append(Anomaly(
                    issue_type="MEMORY_LEAK_PREDICTED",
                    severity="MEDIUM",
                    namespace=ns,
                    resource_name=pod,
                    resource_type="pod",
                    owner_name=_guess_owner_from_pod_name(pod),
                    owner_kind="Deployment",
                    details=(f"Pod '{pod}' (container: {container}): memoria creciendo "
                             f"sostenidamente a {val/1024/1024:.2f} MB/s (15m). Posible leak."),
                    dedup_key=f"{ns}:{pod}:MEMORY_LEAK_PREDICTED",
                ))
                SRE_TREND_ANOMALIES_TOTAL.labels(issue_type="MEMORY_LEAK_PREDICTED").inc()
    except Exception as e:
        logging.debug(f"[TRENDS] Memory leak error: {e}")

    # ── Error rate escalation: rising 5xx trend ───────────────────────────────
    err_trend_q = (
        'deriv(sum(rate(http_requests_total{namespace="amael-ia",status=~"5.."}[5m]))[15m:1m])'
        ' > 0.001'
    )
    try:
        resp = requests.get(f"{PROMETHEUS_URL}/api/v1/query",
                            params={"query": err_trend_q}, timeout=8)
        if resp.status_code == 200 and resp.json()["status"] == "success":
            for r in resp.json()["data"]["result"]:
                val = float(r["value"][1])
                anomalies.append(Anomaly(
                    issue_type="ERROR_RATE_ESCALATING",
                    severity="HIGH",
                    namespace="amael-ia",
                    resource_name="backend-ia",
                    resource_type="endpoint",
                    owner_name="backend-ia-deployment",
                    owner_kind="Deployment",
                    details=(f"Tasa de errores 5xx en escalada sostenida "
                             f"(deriv={val:.4f} req/s²). Posible degradación progresiva."),
                    dedup_key="amael-ia:backend-ia:ERROR_RATE_ESCALATING",
                ))
                SRE_TREND_ANOMALIES_TOTAL.labels(issue_type="ERROR_RATE_ESCALATING").inc()
    except Exception as e:
        logging.debug(f"[TRENDS] Error rate escalation error: {e}")

    if anomalies:
        logging.info(f"[TRENDS] {len(anomalies)} anomalía(s) predictiva(s) detectadas.")
    return anomalies


# ─────────────────────────────────────────────────────────────────────────────
# P5-D: POSTMORTEM AUTO-GENERATION
# ─────────────────────────────────────────────────────────────────────────────

def _get_incident_by_key(incident_key: str) -> Optional[dict]:
    pool = get_postgres_pool()
    if not pool:
        return None
    conn = pool.getconn()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT namespace, resource_name, issue_type, severity,
                       root_cause, confidence, action_taken, action_result, created_at
                FROM sre_incidents WHERE incident_key = %s LIMIT 1;
            """, (incident_key,))
            row = cur.fetchone()
        if not row:
            return None
        return {
            "namespace": row[0], "resource": row[1], "issue_type": row[2],
            "severity": row[3], "root_cause": row[4], "confidence": row[5],
            "action": row[6], "result": row[7], "created_at": row[8],
        }
    except Exception:
        return None
    finally:
        pool.putconn(conn)


def _generate_postmortem_sync(incident_key: str, deployment_name: str,
                               namespace: str, issue_type: str) -> Optional[dict]:
    """Calls LLM to generate a structured postmortem JSON. Returns None on failure."""
    incident = _get_incident_by_key(incident_key)
    if not incident:
        return None

    prompt = (
        f"Eres un SRE Senior. Genera un postmortem conciso en español para este incidente "
        f"resuelto en Kubernetes.\n\n"
        f"INCIDENTE:\n"
        f"- Tipo: {issue_type}\n"
        f"- Recurso: {deployment_name} (namespace: {namespace})\n"
        f"- Severidad: {incident.get('severity', 'UNKNOWN')}\n"
        f"- Causa raíz: {incident.get('root_cause', 'desconocida')}\n"
        f"- Acción aplicada: {incident.get('action', 'desconocida')}\n\n"
        f"Responde ÚNICAMENTE con un JSON válido sin texto adicional:\n"
        f'{{"impact": "descripción del impacto operacional en una oración",'
        f'"timeline": "detección → diagnóstico → remediación → resolución",'
        f'"root_cause_summary": "causa raíz en una oración clara",'
        f'"resolution": "qué se hizo para resolver el incidente",'
        f'"prevention": "medida concreta para evitar recurrencia",'
        f'"action_items": ["acción 1", "acción 2"]}}\n\nJSON:'
    )
    try:
        raw   = llm.invoke(prompt)
        match = re.search(r'\{.*\}', raw, re.DOTALL)
        if not match:
            return None
        return json.loads(match.group())
    except Exception as e:
        logging.warning(f"[POSTMORTEM] LLM error: {e}")
        return None


def _store_postmortem(incident_key: str, deployment_name: str,
                      namespace: str, issue_type: str, data: dict):
    pool = get_postgres_pool()
    if not pool:
        return
    conn = pool.getconn()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO sre_postmortems
                    (incident_key, namespace, resource_name, issue_type,
                     impact, timeline, root_cause_summary, resolution,
                     prevention, action_items, raw_json)
                VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                ON CONFLICT (incident_key) DO NOTHING;
            """, (
                incident_key, namespace, deployment_name, issue_type,
                data.get("impact", ""),
                data.get("timeline", ""),
                data.get("root_cause_summary", ""),
                data.get("resolution", ""),
                data.get("prevention", ""),
                json.dumps(data.get("action_items", [])),
                json.dumps(data),
            ))
            conn.commit()
    except Exception as e:
        logging.error(f"[POSTMORTEM_DB] Error: {e}")
        conn.rollback()
    finally:
        pool.putconn(conn)


def _generate_and_store_postmortem(incident_key: str, deployment_name: str,
                                    namespace: str, issue_type: str):
    """P5-D: Calls LLM (with 60s timeout) to generate and persist a postmortem."""
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(
            _generate_postmortem_sync, incident_key, deployment_name, namespace, issue_type
        )
        try:
            data = future.result(timeout=60)
            if data:
                _store_postmortem(incident_key, deployment_name, namespace, issue_type, data)
                SRE_POSTMORTEM_TOTAL.inc()
                logging.info(f"[POSTMORTEM] Generado para '{deployment_name}' ({issue_type}).")
        except concurrent.futures.TimeoutError:
            logging.warning("[POSTMORTEM] LLM timeout (60s) al generar postmortem.")
        except Exception as e:
            logging.warning(f"[POSTMORTEM] Error: {e}")


def get_recent_postmortems(limit: int = 10) -> list:
    pool = get_postgres_pool()
    if not pool:
        return []
    conn = pool.getconn()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT incident_key, created_at, namespace, resource_name,
                       issue_type, root_cause_summary, resolution, prevention, action_items
                FROM sre_postmortems
                ORDER BY created_at DESC LIMIT %s;
            """, (limit,))
            rows = cur.fetchall()
        return [
            {
                "incident_key":      r[0],
                "created_at":        r[1].isoformat() if r[1] else None,
                "namespace":         r[2],
                "resource":          r[3],
                "issue_type":        r[4],
                "root_cause_summary": r[5],
                "resolution":        r[6],
                "prevention":        r[7],
                "action_items":      json.loads(r[8]) if r[8] else [],
            }
            for r in rows
        ]
    except Exception as e:
        logging.error(f"[POSTMORTEM_DB] get_recent error: {e}")
        return []
    finally:
        pool.putconn(conn)


# ─────────────────────────────────────────────────────────────────────────────
# P4-C: MAINTENANCE WINDOWS
# ─────────────────────────────────────────────────────────────────────────────

def _is_maintenance_active() -> bool:
    """Returns True if a maintenance window is currently active (Redis key exists)."""
    if _redis:
        active = bool(_redis.exists(_MAINTENANCE_KEY))
        SRE_MAINTENANCE_ACTIVE.set(1 if active else 0)
        return active
    SRE_MAINTENANCE_ACTIVE.set(0)
    return False


def _activate_maintenance(input_str: str = "60") -> str:
    """Activates maintenance window. Input: duration in minutes (default 60)."""
    try:
        minutes = int(input_str.strip() or "60")
        minutes = max(1, min(minutes, 480))  # clamp 1 min–8 h
    except ValueError:
        minutes = 60
    if _redis:
        _redis.set(_MAINTENANCE_KEY, "1", ex=minutes * 60)
        SRE_MAINTENANCE_ACTIVE.set(1)
        msg = f"✅ Ventana de mantenimiento activada por {minutes} minutos."
        logging.info(f"[MAINTENANCE] {msg}")
        return msg
    return "❌ Redis no disponible — ventana no persistida."


def _deactivate_maintenance(input_str: str = "") -> str:
    """Deactivates the active maintenance window."""
    if _redis:
        _redis.delete(_MAINTENANCE_KEY)
        SRE_MAINTENANCE_ACTIVE.set(0)
        msg = "✅ Ventana de mantenimiento desactivada."
        logging.info(f"[MAINTENANCE] {msg}")
        return msg
    return "❌ Redis no disponible."


# ─────────────────────────────────────────────────────────────────────────────
# KUBERNETES CLIENT
# ─────────────────────────────────────────────────────────────────────────────
try:
    config.load_incluster_config()
except Exception:
    logging.warning("Config in-cluster no disponible, probando kubeconfig local...")
    try:
        config.load_kube_config()
    except Exception:
        logging.warning("Kubeconfig local también falló.")

v1       = client.CoreV1Api()
apps_v1  = client.AppsV1Api()
coord_v1 = client.CoordinationV1Api()  # P3-D: leader election

# LLM
llm = OllamaLLM(model=MODEL_NAME, base_url=OLLAMA_BASE_URL)


# ─────────────────────────────────────────────────────────────────────────────
# DATACLASSES
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class PodStatus:
    name:              str
    namespace:         str
    phase:             str
    restart_count:     int
    waiting_reason:    str
    last_state_reason: str
    owner_name:        str
    owner_kind:        str
    start_time:        Optional[datetime] = None


@dataclass
class NodeStatus:
    name:            str
    ready:           bool
    memory_pressure: bool
    disk_pressure:   bool
    pid_pressure:    bool


@dataclass
class ClusterSnapshot:
    timestamp: datetime
    pods:      List[PodStatus]
    nodes:     List[NodeStatus]


@dataclass
class Anomaly:
    issue_type:    str
    severity:      str
    namespace:     str
    resource_name: str
    resource_type: str
    owner_name:    str
    owner_kind:    str
    details:       str
    dedup_key:     str


@dataclass
class ActionPlan:
    action:           str
    target_name:      str
    target_namespace: str
    reason:           str
    auto_execute:     bool


# P2 — Diagnóstico estructurado
@dataclass
class Diagnosis:
    issue_type:          str
    root_cause:          str
    root_cause_category: str   # DB_ERROR | OOM | CONFIG_ERROR | DEPENDENCY | NETWORK | IMAGE_ERROR | RESOURCE_LIMIT | UNKNOWN
    confidence:          float  # 0.0–1.0
    severity:            str
    recommended_action:  str    # ROLLOUT_RESTART | NOTIFY_HUMAN | SCALE_UP | FIX_IMAGE
    evidence:            List[str] = field(default_factory=list)
    source:              str = "deterministic"  # "llm" | "deterministic"


# ─────────────────────────────────────────────────────────────────────────────
# P3-A: POST-ACTION VERIFICATION
# ─────────────────────────────────────────────────────────────────────────────
_sre_scheduler = None  # set in startup(); used for one-shot verification jobs


def _is_deployment_healthy(deployment_name: str, namespace: str) -> bool:
    """True if all desired replicas are ready."""
    try:
        deploy  = apps_v1.read_namespaced_deployment(deployment_name, namespace)
        desired = deploy.spec.replicas or 1
        ready   = deploy.status.ready_replicas or 0
        return ready >= desired
    except Exception:
        return False


def _run_verification_job(incident_key: str, deployment_name: str,
                           namespace: str, issue_type: str):
    """
    Runs ~5 min after a ROLLOUT_RESTART to check if the anomaly resolved.
    P5-B: Auto-rollback if unresolved and a recent deploy was the trigger.
    P5-D: Generate postmortem if resolved (background thread).
    """
    try:
        resolved = _is_deployment_healthy(deployment_name, namespace)
        label    = "resolved" if resolved else "unresolved"
        SRE_VERIFICATION_TOTAL.labels(result=label).inc()
        _update_incident_verification(incident_key, label)

        if resolved:
            logging.info(f"[VERIFY] '{deployment_name}' estable — {issue_type} resuelto.")
            # P5-D: Generate postmortem in background (non-blocking)
            threading.Thread(
                target=_generate_and_store_postmortem,
                args=(incident_key, deployment_name, namespace, issue_type),
                daemon=True,
            ).start()
        else:
            logging.warning(f"[VERIFY] '{deployment_name}' SIGUE inestable tras el reinicio.")
            # P5-B: Auto-rollback if a recent deploy introduced the regression
            if _was_recently_deployed(deployment_name, namespace, within_minutes=30):
                logging.info(
                    f"[ROLLBACK] Deploy reciente detectado — iniciando rollback de "
                    f"'{deployment_name}'."
                )
                rollback_result = rollout_undo_deployment(f"{deployment_name}, {namespace}")
                _send_sre_notification(
                    f"↩️ AUTO-ROLLBACK: '{deployment_name}' ({namespace}) — "
                    f"reinicio no resolvió {issue_type} y había un deploy reciente. "
                    f"Resultado: {rollback_result}",
                    severity="HIGH",
                )
            else:
                _send_sre_notification(
                    f"⚠️ POST-VERIFY: '{deployment_name}' ({namespace}) no se resolvió "
                    f"después del reinicio automático ({issue_type}). "
                    f"Inspección manual requerida.",
                    severity="HIGH",
                )
    except Exception as e:
        SRE_VERIFICATION_TOTAL.labels(result="error").inc()
        logging.error(f"[VERIFY] Error en verificación: {e}")


def _schedule_verification(incident_key: str, deployment_name: str,
                            namespace: str, issue_type: str, delay_s: int = 300):
    """Schedules a one-shot verification job via APScheduler."""
    if _sre_scheduler is None:
        return
    import datetime as _dt
    run_at = _dt.datetime.now(_dt.timezone.utc) + _dt.timedelta(seconds=delay_s)
    try:
        _sre_scheduler.add_job(
            _run_verification_job,
            trigger="date",
            run_date=run_at,
            args=[incident_key, deployment_name, namespace, issue_type],
            id=f"verify:{incident_key}",
            replace_existing=True,
        )
        logging.info(
            f"[VERIFY] Verificación programada en {delay_s}s para '{deployment_name}'."
        )
    except Exception as e:
        logging.warning(f"[VERIFY] Error al programar verificación: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# P3-B: HISTORICAL CONFIDENCE ADJUSTMENT
# ─────────────────────────────────────────────────────────────────────────────

def adjust_confidence_with_history(diagnosis: "Diagnosis", anomaly: "Anomaly") -> "Diagnosis":
    """
    Blends LLM/deterministic confidence with historical success rate
    for the same issue_type + deployment.
    Formula: new_confidence = 0.7 × diagnosis.confidence + 0.3 × historical_rate
    Only applied when ≥3 historical data points exist.
    """
    rate = get_historical_success_rate(
        anomaly.issue_type, anomaly.owner_name, anomaly.namespace
    )
    if rate is None:
        return diagnosis

    blended = round(0.7 * diagnosis.confidence + 0.3 * rate, 3)
    blended = max(0.0, min(1.0, blended))

    SRE_LEARNING_ADJUSTED_TOTAL.inc()
    logging.info(
        f"[LEARNING] {anomaly.owner_name}: confianza {diagnosis.confidence:.0%} → "
        f"{blended:.0%} (hist_rate={rate:.0%})"
    )
    return dataclasses.replace(
        diagnosis,
        confidence=blended,
        source=f"{diagnosis.source}+history",
    )


# ─────────────────────────────────────────────────────────────────────────────
# P3-D: KUBERNETES LEASE — LEADER ELECTION
# ─────────────────────────────────────────────────────────────────────────────

def _try_acquire_lease() -> bool:
    """
    Tries to acquire the Kubernetes Lease 'sre-agent-leader'.
    Returns True if this pod is the current leader (or lease check fails — fail-open).
    Returns False if another pod holds a valid lease.
    """
    now = datetime.now(timezone.utc)
    try:
        try:
            lease   = coord_v1.read_namespaced_lease(_LEASE_NAME, DEFAULT_NAMESPACE)
            spec    = lease.spec
            holder  = spec.holder_identity or ""
            renew   = spec.renew_time
            duration = spec.lease_duration_seconds or _LEASE_DURATION_S

            if holder != _POD_NAME and renew is not None:
                # Ensure renew is timezone-aware
                renew_aware = renew if renew.tzinfo else renew.replace(tzinfo=timezone.utc)
                age = (now - renew_aware).total_seconds()
                if age < duration:
                    logging.debug(f"[LEASE] Líder actual: '{holder}'. Saltando iteración.")
                    return False

            # Expired or ours — renew
            lease.spec.holder_identity = _POD_NAME
            lease.spec.renew_time      = now
            coord_v1.replace_namespaced_lease(_LEASE_NAME, DEFAULT_NAMESPACE, lease)
            return True

        except client.ApiException as api_err:
            if api_err.status == 404:
                # Create lease for the first time
                new_lease = client.V1Lease(
                    metadata=client.V1ObjectMeta(
                        name=_LEASE_NAME, namespace=DEFAULT_NAMESPACE
                    ),
                    spec=client.V1LeaseSpec(
                        holder_identity=_POD_NAME,
                        acquire_time=now,
                        renew_time=now,
                        lease_duration_seconds=_LEASE_DURATION_S,
                    ),
                )
                coord_v1.create_namespaced_lease(DEFAULT_NAMESPACE, new_lease)
                return True
            raise

    except Exception as e:
        # Fail-open: if lease check errors, run the loop anyway
        logging.warning(f"[LEASE] Error ({e}). Asumiendo líder (fail-open).")
        return True


# ─────────────────────────────────────────────────────────────────────────────
# CIRCUIT BREAKER
# ─────────────────────────────────────────────────────────────────────────────
class CircuitBreaker:
    CLOSED    = "closed"
    OPEN      = "open"
    HALF_OPEN = "half_open"
    _STATE_METRIC = {CLOSED: 0, OPEN: 1, HALF_OPEN: 2}

    def __init__(self, failure_threshold: int = 5, recovery_timeout: int = 600):
        self.failure_threshold = failure_threshold
        self.recovery_timeout  = recovery_timeout
        self._failures         = 0
        self._state            = self.CLOSED
        self._opened_at: float = 0.0
        self._lock             = threading.Lock()

    @property
    def state(self) -> str:
        with self._lock:
            if self._state == self.OPEN:
                if time.time() - self._opened_at > self.recovery_timeout:
                    self._state = self.HALF_OPEN
                    logging.info("[CB] Circuito en HALF_OPEN.")
            SRE_CB_STATE.set(self._STATE_METRIC.get(self._state, 0))
            return self._state

    def is_open(self) -> bool:
        return self.state == self.OPEN

    def record_success(self):
        with self._lock:
            self._failures = 0
            if self._state in (self.OPEN, self.HALF_OPEN):
                self._state = self.CLOSED
                logging.info("[CB] Circuito CLOSED.")
            SRE_CB_STATE.set(0)

    def record_failure(self):
        with self._lock:
            self._failures += 1
            if self._failures >= self.failure_threshold and self._state == self.CLOSED:
                self._state     = self.OPEN
                self._opened_at = time.time()
                logging.warning(
                    f"[CB] Circuito OPEN tras {self._failures} fallos. "
                    f"Pausa {self.recovery_timeout}s."
                )
                SRE_CB_STATE.set(1)


_circuit_breaker = CircuitBreaker(
    failure_threshold=int(os.environ.get("SRE_CB_FAILURE_THRESHOLD", "5")),
    recovery_timeout =int(os.environ.get("SRE_CB_RECOVERY_TIMEOUT_S", "600")),
)
_loop_lock = threading.Lock()


# ─────────────────────────────────────────────────────────────────────────────
# PROMETHEUS CALLBACK
# ─────────────────────────────────────────────────────────────────────────────
class PrometheusMetricsCallback(BaseCallbackHandler):
    def on_tool_start(self, serialized, input_str, **kwargs):
        AGENT_TOOLS_USAGE_TOTAL.labels(tool=serialized.get("name", "unknown")).inc()
    def on_agent_action(self, action, **kwargs):
        AGENT_STEPS_TOTAL.inc()

metrics_callback = PrometheusMetricsCallback()

# ─────────────────────────────────────────────────────────────────────────────
# HELPER
# ─────────────────────────────────────────────────────────────────────────────
def _parse_two(input_str: str, default_second: str = DEFAULT_NAMESPACE):
    input_str = input_str.strip("'\" \n")
    if "," in input_str:
        parts = input_str.split(",", 1)
        return parts[0].strip(), parts[1].strip()
    return input_str, default_second


# ─────────────────────────────────────────────────────────────────────────────
# HERRAMIENTAS — OBSERVABILIDAD
# ─────────────────────────────────────────────────────────────────────────────
def list_k8s_namespaces(query: str = "") -> str:
    try:
        nss = v1.list_namespace()
        return "Namespaces:\n" + "".join(
            f"- {n.metadata.name}: {n.status.phase}\n" for n in nss.items
        )
    except Exception as e:
        return f"Error: {e}"


def inspect_namespace(ns_name: str) -> str:
    ns_name = ns_name.strip("'\" \n")
    try:
        ns = v1.read_namespace(ns_name)
        return (f"Namespace '{ns_name}':\n"
                f"- Estado: {ns.status.phase}\n"
                f"- UID: {ns.metadata.uid}\n"
                f"- Creación: {ns.metadata.creation_timestamp}")
    except Exception as e:
        return f"Error: {e}"


def list_k8s_pods(input_str: str = "") -> str:
    ns = input_str.strip("'\" \n") or DEFAULT_NAMESPACE
    try:
        pods = v1.list_namespaced_pod(ns)
        result = f"Pods en '{ns}':\n"
        failed = []
        for pod in pods.items:
            phase = pod.status.phase or "Unknown"
            rc, wr = 0, ""
            if pod.status.container_statuses:
                for cs in pod.status.container_statuses:
                    rc += cs.restart_count or 0
                    if cs.state and cs.state.waiting:
                        wr = cs.state.waiting.reason or ""
            status = wr if wr else phase
            result += f"- {pod.metadata.name}: {status} (reinicios: {rc})\n"
            if (phase in ("Failed", "Unknown") or
                    wr in ("CrashLoopBackOff", "ImagePullBackOff", "ErrImagePull",
                           "OOMKilled", "Error") or rc >= 5):
                failed.append(pod.metadata.name)
        if failed:
            result += (f"\n** ALERTA: {len(failed)} pod(s) con problemas: "
                       f"{', '.join(failed)} **\n"
                       "[ACCION]: Usa Describir_Pod para obtener la causa raíz.")
        else:
            result += "\n** Todos los pods están saludables. **"
        return result
    except Exception as e:
        return f"Error listando pods en '{ns}': {e}"


def get_pod_logs(input_str: str) -> str:
    pod_name, namespace = _parse_two(input_str)
    try:
        logs = v1.read_namespaced_pod_log(name=pod_name, namespace=namespace, tail_lines=50)
        return f"Logs de '{pod_name}':\n{logs}"
    except Exception as e:
        return f"Error: {e}"


def describe_pod(input_str: str) -> str:
    pod_name, namespace = _parse_two(input_str)
    try:
        pod = v1.read_namespaced_pod(pod_name, namespace)
        result = (f"Pod: {pod.metadata.name} | Namespace: {namespace}\n"
                  f"Phase: {pod.status.phase} | Node: {pod.spec.node_name}\n")
        if pod.status.container_statuses:
            for cs in pod.status.container_statuses:
                result += f"\nContainer: {cs.name} | Ready: {cs.ready} | Reinicios: {cs.restart_count}\n"
                if cs.state.waiting:
                    result += f"  Waiting: {cs.state.waiting.reason}: {cs.state.waiting.message or ''}\n"
                elif cs.state.running:
                    result += f"  Running desde: {cs.state.running.started_at}\n"
                elif cs.state.terminated:
                    t = cs.state.terminated
                    result += f"  Terminated: {t.reason} exit={t.exit_code}\n"
                if cs.last_state and cs.last_state.terminated:
                    lt = cs.last_state.terminated
                    result += f"  Last state: {lt.reason} exit={lt.exit_code} @ {lt.finished_at}\n"
        if pod.spec.containers:
            for c in pod.spec.containers:
                if c.resources:
                    req = c.resources.requests or {}
                    lim = c.resources.limits   or {}
                    result += (f"  Requests: cpu={req.get('cpu','?')} mem={req.get('memory','?')}\n"
                               f"  Limits:   cpu={lim.get('cpu','?')} mem={lim.get('memory','?')}\n")
        events = v1.list_namespaced_event(namespace,
                    field_selector=f"involvedObject.name={pod_name}")
        if events.items:
            sorted_evs = sorted(events.items,
                key=lambda e: (e.last_timestamp or e.event_time or e.metadata.creation_timestamp),
                reverse=True)
            result += "\nEvents (últimos 10):\n"
            for ev in sorted_evs[:10]:
                result += f"  [{ev.type}] {ev.reason}: {ev.message} ({ev.last_timestamp})\n"
        else:
            result += "\nEvents: ninguno reciente.\n"
        return result
    except Exception as e:
        return f"Error describiendo pod '{pod_name}': {e}"


def get_k8s_events(input_str: str = "") -> str:
    input_str = input_str.strip("'\" \n")
    if "," in input_str:
        parts = input_str.split(",", 1)
        namespace, field_selector = parts[0].strip() or DEFAULT_NAMESPACE, parts[1].strip()
    else:
        namespace, field_selector = input_str or DEFAULT_NAMESPACE, ""
    try:
        kwargs: dict = {"namespace": namespace}
        if field_selector:
            kwargs["field_selector"] = field_selector
        events = v1.list_namespaced_event(**kwargs)
        if not events.items:
            return f"No hay eventos recientes en '{namespace}'."
        sorted_evs = sorted(events.items,
            key=lambda e: (e.last_timestamp or e.event_time or e.metadata.creation_timestamp),
            reverse=True)
        result = f"Eventos en '{namespace}' (últimos 20):\n"
        for ev in sorted_evs[:20]:
            obj = ev.involved_object
            result += f"  [{ev.type}] {obj.kind}/{obj.name} — {ev.reason}: {ev.message}\n"
        return result
    except Exception as e:
        return f"Error obteniendo eventos de '{namespace}': {e}"


def get_node_status(query: str = "") -> str:
    try:
        nodes = v1.list_node()
        result = "Estado de nodos:\n"
        alerts = []
        for node in nodes.items:
            name       = node.metadata.name
            conds      = {c.type: c.status for c in (node.status.conditions or [])}
            ready      = conds.get("Ready",          "Unknown")
            mem_p      = conds.get("MemoryPressure",  "False")
            disk_p     = conds.get("DiskPressure",    "False")
            pid_p      = conds.get("PIDPressure",     "False")
            alloc      = node.status.allocatable or {}
            cap        = node.status.capacity    or {}
            icon = "✅" if ready == "True" else "🚨"
            result += (f"\n{icon} {name}: Ready={ready} | MemPressure={mem_p} | "
                       f"DiskPressure={disk_p} | PIDPressure={pid_p}\n"
                       f"  CPU alloc={alloc.get('cpu','?')} / {cap.get('cpu','?')}\n"
                       f"  RAM alloc={alloc.get('memory','?')} / {cap.get('memory','?')}\n")
            if ready != "True":          alerts.append(f"🚨 CRITICAL: Nodo '{name}' NOT READY")
            if mem_p  == "True":         alerts.append(f"⚠️ HIGH: Nodo '{name}' MemoryPressure")
            if disk_p == "True":         alerts.append(f"⚠️ HIGH: Nodo '{name}' DiskPressure")
        if alerts:
            result += "\n** ALERTAS **\n" + "\n".join(alerts)
        else:
            result += "\n** Todos los nodos saludables. **"
        return result
    except Exception as e:
        return f"Error: {e}"


def list_k8s_deployments(input_str: str = "") -> str:
    ns = input_str.strip("'\" \n") or DEFAULT_NAMESPACE
    try:
        deploys = apps_v1.list_namespaced_deployment(ns)
        if not deploys.items:
            return f"No hay deployments en '{ns}'."
        result = f"Deployments en '{ns}':\n"
        for d in deploys.items:
            desired = d.spec.replicas or 0
            ready   = d.status.ready_replicas or 0
            avail   = d.status.available_replicas or 0
            icon = "✅" if ready == desired and desired > 0 else "⚠️"
            result += f"{icon} {d.metadata.name}: desired={desired} ready={ready} avail={avail}\n"
        return result
    except Exception as e:
        return f"Error: {e}"


# ─────────────────────────────────────────────────────────────────────────────
# HERRAMIENTAS — MÉTRICAS
# ─────────────────────────────────────────────────────────────────────────────
_PROMETHEUS_ALIASES: dict[str, str] = {
    "cpu_pods":     'sum(rate(container_cpu_usage_seconds_total{namespace="amael-ia",container!=""}[5m])) by (pod)',
    "ram_pods":     'sum(container_memory_working_set_bytes{namespace="amael-ia",container!=""}) by (pod)',
    "cpu_node":     'sum(rate(node_cpu_seconds_total{mode!="idle"}[5m])) by (instance)',
    "ram_node":     'node_memory_MemAvailable_bytes / node_memory_MemTotal_bytes',
    "http_errors":  'sum(rate(http_requests_total{namespace="amael-ia",status=~"5.."}[5m])) by (handler)',
    "restart_rate": 'increase(kube_pod_container_status_restarts_total{namespace="amael-ia"}[15m])',
}


def query_prometheus(query: str) -> str:
    query    = query.strip("'\" \n")
    resolved = _PROMETHEUS_ALIASES.get(query.lower(), query)
    try:
        resp = requests.get(f"{PROMETHEUS_URL}/api/v1/query",
                            params={"query": resolved}, timeout=10)
        if resp.status_code == 200:
            data = resp.json()
            if data["status"] == "success":
                simplified = [
                    {"metric": r.get("metric", {}), "value": r.get("value", [None, None])[1]}
                    for r in data["data"]["result"]
                ]
                return json.dumps(simplified[:15])
            return f"Error Prometheus: {data.get('error')}"
        return f"HTTP {resp.status_code}"
    except Exception as e:
        return f"Error: {e}"


def list_grafana_dashboards(query: str = "") -> str:
    try:
        cms = v1.list_namespaced_config_map(
            namespace="observability", label_selector="grafana_dashboard=1")
        if not cms.items:
            return "No se encontraron dashboards."
        result = "Dashboards en Grafana:\n"
        for cm in cms.items:
            title = cm.metadata.name.replace("kube-prometheus-stack-", "").replace("-", " ").title()
            result += f"- {title} ({cm.metadata.name})\n"
        return result + "\nAcceso: grafana.richardx.dev"
    except Exception as e:
        return f"Error: {e}"


# ─────────────────────────────────────────────────────────────────────────────
# HERRAMIENTAS — SELF-HEALING
# ─────────────────────────────────────────────────────────────────────────────
def rollout_restart_deployment(input_str: str) -> str:
    deployment_name, namespace = _parse_two(input_str)
    if deployment_name in PROTECTED_DEPLOYMENTS:
        return (f"Error: '{deployment_name}' está protegido "
                f"({', '.join(PROTECTED_DEPLOYMENTS)}).")
    patch = {"spec": {"template": {"metadata": {"annotations": {
        "kubectl.kubernetes.io/restartedAt": datetime.now(timezone.utc).isoformat()
    }}}}}
    try:
        logging.info(f"[AUDIT] rollout_restart deployment={deployment_name!r} ns={namespace!r}")
        apps_v1.patch_namespaced_deployment(deployment_name, namespace, patch)
        return f"✅ Rollout restart iniciado en '{deployment_name}' ({namespace})."
    except Exception as e:
        return f"Error al reiniciar '{deployment_name}': {e}"


def delete_k8s_pod(input_str: str) -> str:
    pod_name, namespace = _parse_two(input_str)
    try:
        logging.info(f"[AUDIT] delete_pod pod={pod_name!r} ns={namespace!r}")
        v1.delete_namespaced_pod(name=pod_name, namespace=namespace)
        return f"Pod '{pod_name}' eliminado en '{namespace}'."
    except Exception as e:
        return f"Error al eliminar '{pod_name}': {e}"


# ─────────────────────────────────────────────────────────────────────────────
# HERRAMIENTA — NOTIFICACIÓN
# ─────────────────────────────────────────────────────────────────────────────
def notify_whatsapp_sre(input_str: str) -> str:
    message = input_str.strip("'\" \n")
    if not WHATSAPP_BRIDGE_URL or not OWNER_PHONE:
        return "WHATSAPP_BRIDGE_URL o OWNER_PHONE no configurados."
    try:
        resp = requests.post(
            f"{WHATSAPP_BRIDGE_URL}/send",
            json={"phone": OWNER_PHONE, "message": f"[SRE] {message}"},
            timeout=10,
        )
        if resp.status_code == 200:
            SRE_NOTIFY_TOTAL.labels(severity="alert").inc()
            return f"✅ Alerta SRE enviada a {OWNER_PHONE}."
        return f"Error HTTP {resp.status_code}: {resp.text[:80]}"
    except Exception as e:
        return f"Error: {e}"


def _send_sre_notification(message: str, severity: str = "INFO"):
    if not WHATSAPP_BRIDGE_URL or not OWNER_PHONE:
        logging.info(f"[SRE_NOTIFY] {severity}: {message}")
        return
    try:
        requests.post(
            f"{WHATSAPP_BRIDGE_URL}/send",
            json={"phone": OWNER_PHONE, "message": f"[SRE {severity}] {message}"},
            timeout=10,
        )
        SRE_NOTIFY_TOTAL.labels(severity=severity.lower()).inc()
    except Exception as e:
        logging.warning(f"[SRE_NOTIFY] {e}")


# ─────────────────────────────────────────────────────────────────────────────
# HERRAMIENTA — DIAGNÓSTICO EN CONTENEDOR
# ─────────────────────────────────────────────────────────────────────────────
_ALLOWED_EXEC_COMMANDS = {
    "ls", "ps", "df", "du", "cat", "head", "tail",
    "find", "grep", "env", "id", "whoami", "uname",
    "date", "uptime", "free", "stat", "wc", "printenv",
}
_BLOCKED_SHELL_META = re.compile(r'[;&|`$><\\(){}\[\]!\n\r]')


def run_command_in_pod(input_str: str) -> str:
    try:
        if "," not in input_str:
            return "Error: Usa 'pod_name, comando'."
        parts    = input_str.split(",", 1)
        pod_name = parts[0].strip("'\" \n")
        command  = parts[1].strip("'\" \n")
        if _BLOCKED_SHELL_META.search(command):
            logging.warning(f"[SECURITY] pod_exec bloqueado: pod={pod_name!r} cmd={command!r}")
            return "Error: el comando contiene caracteres no permitidos."
        tokens = command.split()
        if not tokens:
            return "Error: comando vacío."
        base_cmd = tokens[0].lower()
        if base_cmd not in _ALLOWED_EXEC_COMMANDS:
            return (f"Error: '{base_cmd}' no permitido. "
                    f"Válidos: {', '.join(sorted(_ALLOWED_EXEC_COMMANDS))}.")
        logging.info(f"[AUDIT] pod_exec pod={pod_name!r} ns={DEFAULT_NAMESPACE!r} cmd={command!r}")
        resp = stream.stream(
            v1.connect_get_namespaced_pod_exec,
            pod_name, DEFAULT_NAMESPACE,
            command=["/bin/sh", "-c", command],
            stderr=True, stdin=False, stdout=True, tty=False,
        )
        return f"Salida en '{pod_name}':\n{resp or '(sin salida)'}"
    except Exception as e:
        return f"Error: {e}"


# ─────────────────────────────────────────────────────────────────────────────
# HERRAMIENTAS — KB Y VAULT
# ─────────────────────────────────────────────────────────────────────────────
def consult_vault_knowledge(query: str = "") -> str:
    if not _VAULT_KNOWLEDGE:
        return "No se encontró la base de conocimiento de Vault."
    return _VAULT_KNOWLEDGE.replace("{{", "{").replace("}}", "}")


def consult_knowledge_base(issue_keywords: str = "") -> str:
    """Busca en runbooks locales (Qdrant) primero, luego en el backend como fallback."""
    # P2: buscar en runbooks locales
    local = search_runbooks(issue_keywords)
    if local:
        return f"[Runbook local]\n{local[:1500]}"
    # Fallback al backend RAG
    try:
        resp = requests.post(BACKEND_SRE_URL,
                             json={"query": issue_keywords or "all"}, timeout=10)
        if resp.status_code == 200:
            return resp.json().get("response", "Sin resultados.")
        return f"Backend error HTTP {resp.status_code}"
    except Exception as e:
        return f"Error: {e}"


# ─────────────────────────────────────────────────────────────────────────────
# HERRAMIENTA — CAPTURA GRAFANA
# ─────────────────────────────────────────────────────────────────────────────
latest_media_captured = None


def capture_grafana_screenshot(dashboard: str = "recursos") -> str:
    global latest_media_captured
    DASHBOARD_MAP = {
        "recursos": "efa86fd1d0c121a26444b636a3f509a8/k8s-resources-cluster",
        "rag":      "amael-rag/3-amael-rag-performance",
    }
    db_key      = "rag" if "rag" in dashboard.lower() else "recursos"
    target_path = DASHBOARD_MAP[db_key]
    url = (f"http://kube-prometheus-stack-grafana.observability.svc.cluster.local"
           f"/d/{target_path}?orgId=1&refresh=10s")
    try:
        resp = requests.post(
            f"{WHATSAPP_BRIDGE_URL}/screenshot",
            json={"url": url, "username": GRAFANA_USER, "password": GRAFANA_PASSWORD},
            timeout=60,
        )
        if resp.status_code == 200:
            latest_media_captured = resp.json().get("base64")
            return (f"Captura del dashboard '{db_key}' generada. "
                    "Incluye [MEDIA_PLACEHOLDER] en tu respuesta final.")
        return f"Error: {resp.text}"
    except Exception as e:
        return f"Error: {e}"


# ─────────────────────────────────────────────────────────────────────────────
# LISTA DE HERRAMIENTAS
# ─────────────────────────────────────────────────────────────────────────────
tools = [
    Tool(name="Listar_Namespaces",          func=list_k8s_namespaces,
         description="Lista todos los namespaces del clúster. Input: vacío."),
    Tool(name="Detalle_Namespace",           func=inspect_namespace,
         description="Detalle de un namespace. Input: nombre del namespace."),
    Tool(name="Listar_Pods",                func=list_k8s_pods,
         description="Lista pods y su estado. Detecta problemas. Input: namespace."),
    Tool(name="Describir_Pod",              func=describe_pod,
         description=(
             "Descripción completa: containers, recursos, Events. "
             "ÚSALA ante cualquier pod con problemas. "
             "Input: 'pod_name' o 'pod_name, namespace'.")),
    Tool(name="Obtener_Logs_Pod",           func=get_pod_logs,
         description="Últimos 50 logs de un pod. Input: 'pod_name' o 'pod_name, namespace'."),
    Tool(name="Obtener_Eventos",            func=get_k8s_events,
         description="Eventos recientes de un namespace. Input: 'ns' o 'ns, field_selector'."),
    Tool(name="Estado_Nodos",               func=get_node_status,
         description="Estado de nodos: Ready, MemPressure, DiskPressure. Input: vacío."),
    Tool(name="Listar_Deployments",         func=list_k8s_deployments,
         description="Lista deployments y réplicas. Input: namespace."),
    Tool(name="Reiniciar_Deployment",       func=rollout_restart_deployment,
         description=(
             "Reinicia deployment con RollingUpdate (zero-downtime). "
             "Input: 'deployment_name' o 'deployment_name, namespace'.")),
    Tool(name="Eliminar_Pod",               func=delete_k8s_pod,
         description="Elimina un pod. Input: 'pod_name' o 'pod_name, namespace'."),
    Tool(name="Prometheus_Query",           func=query_prometheus,
         description=(
             "PromQL en Prometheus. Aliases: cpu_pods, ram_pods, cpu_node, "
             "ram_node, http_errors, restart_rate.")),
    Tool(name="Listar_Grafana_Dashboards",  func=list_grafana_dashboards,
         description="Lista dashboards de Grafana. Input: vacío."),
    Tool(name="Capturar_Imagen_Grafana",    func=capture_grafana_screenshot,
         description="Captura visual de Grafana. Input: 'recursos' o 'rag'."),
    Tool(name="Consultar_Vault",            func=consult_vault_knowledge,
         description="Base de conocimiento de HashiCorp Vault."),
    Tool(name="Consultar_Base_Conocimiento", func=consult_knowledge_base,
         description=(
             "Busca runbooks locales y guías de remediación. "
             "Primero busca en runbooks locales (Qdrant), luego en backend RAG. "
             "NO usar para Vault. Input: nombre del servicio o tipo de error.")),
    Tool(name="Ejecutar_Comando_Contenedor", func=run_command_in_pod,
         description="Comandos read-only en pod. Input: 'pod_name, comando'."),
    Tool(name="Notificar_WhatsApp",         func=notify_whatsapp_sre,
         description="Envía alerta SRE por WhatsApp. Input: mensaje."),
    # P5-B: Rollback tool
    Tool(name="Revertir_Deployment",        func=rollout_undo_deployment,
         description=(
             "P5-B: Rollback de un deployment a la revisión anterior "
             "(equivale a kubectl rollout undo). Usar cuando un deploy nuevo rompe la app. "
             "Input: 'deployment_name' o 'deployment_name, namespace'.")),
    # P4-C: Maintenance window tools
    Tool(name="Activar_Mantenimiento",      func=_activate_maintenance,
         description=(
             "Activa una ventana de mantenimiento: el loop SRE pausa acciones automáticas. "
             "Input: duración en minutos (ej: 60). Por defecto 60 min.")),
    Tool(name="Desactivar_Mantenimiento",   func=_deactivate_maintenance,
         description="Desactiva la ventana de mantenimiento activa. Input: vacío."),
]

# ─────────────────────────────────────────────────────────────────────────────
# AGENTE LANGCHAIN REACT (fallback clásico)
# ─────────────────────────────────────────────────────────────────────────────
_SRE_SYSTEM_PROMPT = (
    "Eres un SRE Senior experto en Kubernetes y HashiCorp Vault.\n"
    "Tu objetivo es diagnosticar y resolver problemas del clúster.\n\n"
    "=== HASHICORP VAULT ===\n"
    + _VAULT_KNOWLEDGE +
    "\n=== FIN VAULT ===\n\n"
    "=== MÉTRICAS PROMETHEUS ===\n"
    + _METRICS_KNOWLEDGE +
    "\n=== FIN MÉTRICAS ===\n\n"
    "REGLAS:\n"
    "1. Pods con problemas → Describir_Pod primero (tiene Events con causa raíz).\n"
    "2. Vault → Consultar_Vault siempre.\n"
    "3. Runbooks y soluciones → Consultar_Base_Conocimiento.\n"
    "4. Reiniciar → Reiniciar_Deployment (no Eliminar_Pod salvo necesidad).\n"
    "5. Problema crítico → Notificar_WhatsApp.\n\n"
    "FORMATO:\n"
    "Thought: | Action: | Action Input: | Observation:\n"
    "Final Answer: [respuesta detallada. Si usaste Capturar_Imagen_Grafana: '[MEDIA_PLACEHOLDER]']\n\n"
    "Herramientas disponibles:"
)

agent = initialize_agent(
    tools, llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
    handle_parsing_errors="Use 'Action:' and 'Action Input:' or 'Final Answer:'.",
    max_iterations=10,
    early_stopping_method="generate",
    agent_kwargs={
        "prefix": _SRE_SYSTEM_PROMPT,
        "suffix": "Pregunta: {input}\n{agent_scratchpad}",
    },
)

# ─────────────────────────────────────────────────────────────────────────────
# P3-C: LANGGRAPH AGENT (principal)
# Usa create_react_agent con ChatOllama + bind_tools.
# El agente clásico de arriba actúa como fallback si LangGraph falla.
# ─────────────────────────────────────────────────────────────────────────────
_LANGGRAPH_ENABLED = False
_langgraph_agent   = None

try:
    from langgraph.prebuilt import create_react_agent as _create_react_agent
    from langchain_ollama import ChatOllama as _ChatOllama
    from langchain_core.messages import HumanMessage as _HumanMessage, SystemMessage as _SystemMessage

    _chat_llm = _ChatOllama(model=MODEL_NAME, base_url=OLLAMA_BASE_URL)

    # create_react_agent acepta la lista de Tool estándar de LangChain
    _langgraph_agent = _create_react_agent(
        _chat_llm,
        tools,
        prompt=_SystemMessage(content=_SRE_SYSTEM_PROMPT),
    )
    _LANGGRAPH_ENABLED = True
    logging.info("[LangGraph] Agente ReAct compilado correctamente (P3-C).")
except ImportError as _e:
    logging.warning(f"[LangGraph] langgraph no instalado ({_e}). Usando agente LangChain clásico.")
except Exception as _e:
    logging.warning(f"[LangGraph] Error al compilar grafo ({_e}). Usando agente LangChain clásico.")


# ─────────────────────────────────────────────────────────────────────────────
# LOOP — DIAGNÓSTICO LLM (P2 #11)
# ─────────────────────────────────────────────────────────────────────────────

def _deterministic_diagnosis(anomaly: Anomaly) -> Diagnosis:
    """Diagnóstico de fallback 100% determinístico, sin LLM."""
    ISSUE_MAP = {
        "CRASH_LOOP":        ("DEPENDENCY",     "ROLLOUT_RESTART", 0.70),
        "OOM_KILLED":        ("RESOURCE_LIMIT", "ROLLOUT_RESTART", 0.80),
        "IMAGE_PULL_ERROR":  ("IMAGE_ERROR",    "NOTIFY_HUMAN",    0.95),
        "POD_FAILED":        ("UNKNOWN",        "ROLLOUT_RESTART", 0.60),
        "POD_PENDING_STUCK": ("RESOURCE_LIMIT", "NOTIFY_HUMAN",    0.65),
        "HIGH_RESTARTS":     ("DEPENDENCY",     "ROLLOUT_RESTART", 0.65),
        "NODE_NOT_READY":       ("UNKNOWN",        "NOTIFY_HUMAN",    0.90),
        "NODE_MEMORY_PRESSURE": ("RESOURCE_LIMIT", "NOTIFY_HUMAN",    0.90),
        "NODE_DISK_PRESSURE":   ("RESOURCE_LIMIT", "NOTIFY_HUMAN",    0.90),
        # P4-A — metric-based anomalies
        "HIGH_CPU":             ("RESOURCE_LIMIT", "NOTIFY_HUMAN",    0.75),
        "HIGH_MEMORY":          ("RESOURCE_LIMIT", "ROLLOUT_RESTART", 0.70),
        "HIGH_ERROR_RATE":      ("DEPENDENCY",     "NOTIFY_HUMAN",    0.75),
        # P5-A — predictive / trend anomalies
        "DISK_EXHAUSTION_PREDICTED": ("RESOURCE_LIMIT", "NOTIFY_HUMAN", 0.85),
        "MEMORY_LEAK_PREDICTED":     ("RESOURCE_LIMIT", "ROLLOUT_RESTART", 0.70),
        "ERROR_RATE_ESCALATING":     ("DEPENDENCY",     "NOTIFY_HUMAN", 0.70),
        # P5-C — SLO violations
        "SLO_BUDGET_BURNING":        ("DEPENDENCY",     "NOTIFY_HUMAN", 0.90),
    }
    cat, action, conf = ISSUE_MAP.get(
        anomaly.issue_type, ("UNKNOWN", "NOTIFY_HUMAN", 0.50)
    )
    return Diagnosis(
        issue_type=anomaly.issue_type,
        root_cause=anomaly.details,
        root_cause_category=cat,
        confidence=conf,
        severity=anomaly.severity,
        recommended_action=action,
        evidence=[anomaly.details],
        source="deterministic",
    )


def _diagnose_llm_sync(anomaly: Anomaly) -> Optional[Diagnosis]:
    """
    Diagnóstico asistido por LLM. Recopila evidencia (logs, events, runbook)
    y pide a qwen2.5:14b un JSON estructurado. Fallback a None en cualquier error.
    """
    evidence_parts = []

    # 1. Pod logs (últimas 30 líneas)
    if anomaly.resource_type == "pod":
        try:
            logs = v1.read_namespaced_pod_log(
                name=anomaly.resource_name,
                namespace=anomaly.namespace,
                tail_lines=30,
            )
            if logs:
                evidence_parts.append(f"LOGS (últimas 30 líneas):\n{logs[-1500:]}")
        except Exception:
            pass

        # 2. Events del pod
        try:
            evs = v1.list_namespaced_event(
                anomaly.namespace,
                field_selector=f"involvedObject.name={anomaly.resource_name}",
            )
            sorted_evs = sorted(
                evs.items,
                key=lambda e: (e.last_timestamp or e.metadata.creation_timestamp),
                reverse=True,
            )
            ev_lines = "\n".join(
                f"[{ev.type}] {ev.reason}: {ev.message}"
                for ev in sorted_evs[:5]
            )
            if ev_lines:
                evidence_parts.append(f"EVENTS K8S:\n{ev_lines}")
        except Exception:
            pass

    # 3. Runbook local (máximo relevante)
    runbook = search_runbooks(
        f"{anomaly.issue_type} {anomaly.details[:100]}"
    )
    if runbook:
        evidence_parts.append(f"RUNBOOK RELEVANTE:\n{runbook[:600]}")

    evidence_text = "\n\n".join(evidence_parts) or "Sin evidencia adicional disponible."

    prompt = f"""Eres un SRE experto. Analiza el siguiente incidente de Kubernetes y genera un diagnóstico estructurado.

INCIDENTE:
- Tipo: {anomaly.issue_type}
- Severidad: {anomaly.severity}
- Recurso: {anomaly.resource_name} (namespace: {anomaly.namespace})
- Descripción: {anomaly.details}

EVIDENCIA:
{evidence_text}

Responde ÚNICAMENTE con un objeto JSON válido, sin texto adicional antes ni después:
{{
  "issue_type": "{anomaly.issue_type}",
  "root_cause": "descripción concisa de la causa raíz en una oración",
  "root_cause_category": "uno de: DB_ERROR, OOM, CONFIG_ERROR, DEPENDENCY, NETWORK, IMAGE_ERROR, RESOURCE_LIMIT, UNKNOWN",
  "confidence": 0.85,
  "severity": "{anomaly.severity}",
  "recommended_action": "uno de: ROLLOUT_RESTART, NOTIFY_HUMAN, SCALE_UP, FIX_IMAGE",
  "evidence_summary": ["hecho clave 1", "hecho clave 2"]
}}

JSON:"""

    try:
        raw = llm.invoke(prompt)
        # Extraer el bloque JSON de la respuesta
        json_match = re.search(r'\{[^{}]*\}', raw, re.DOTALL)
        if not json_match:
            # Intentar con bloques anidados
            json_match = re.search(r'\{.*\}', raw, re.DOTALL)
        if not json_match:
            logging.warning(f"[DIAGNOSIS] No se encontró JSON en respuesta LLM: {raw[:100]}")
            return None

        data = json.loads(json_match.group())

        confidence = float(data.get("confidence", 0.5))
        # Validar rango
        confidence = max(0.0, min(1.0, confidence))

        return Diagnosis(
            issue_type=data.get("issue_type", anomaly.issue_type),
            root_cause=data.get("root_cause", anomaly.details),
            root_cause_category=data.get("root_cause_category", "UNKNOWN"),
            confidence=confidence,
            severity=data.get("severity", anomaly.severity),
            recommended_action=data.get("recommended_action", "NOTIFY_HUMAN"),
            evidence=data.get("evidence_summary", []),
            source="llm",
        )
    except json.JSONDecodeError as e:
        logging.warning(f"[DIAGNOSIS] JSON inválido del LLM: {e}")
        return None
    except Exception as e:
        logging.warning(f"[DIAGNOSIS] Error inesperado: {e}")
        return None


def diagnose_with_llm(anomaly: Anomaly, timeout_s: int = 30) -> Diagnosis:
    """
    Intenta diagnóstico LLM (con timeout). Fallback a determinístico si falla.
    Solo usa LLM para CRITICAL y HIGH (conserva recursos GPU).
    """
    min_rank = _SEVERITY_RANK.get("HIGH", 2)
    sev_rank = _SEVERITY_RANK.get(anomaly.severity, 0)

    if sev_rank < min_rank:
        # MEDIUM / LOW → siempre determinístico
        return _deterministic_diagnosis(anomaly)

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(_diagnose_llm_sync, anomaly)
        try:
            result = future.result(timeout=timeout_s)
            if result is not None:
                SRE_DIAGNOSIS_LLM_TOTAL.labels(result="ok").inc()
                SRE_DIAGNOSIS_CONFIDENCE.observe(result.confidence)
                logging.info(
                    f"[DIAGNOSIS] LLM: {result.root_cause_category} "
                    f"confidence={result.confidence:.0%} action={result.recommended_action}"
                )
                return result
        except concurrent.futures.TimeoutError:
            logging.warning(f"[DIAGNOSIS] LLM timeout ({timeout_s}s). Usando diagnóstico determinístico.")
            SRE_DIAGNOSIS_LLM_TOTAL.labels(result="timeout").inc()
        except Exception as e:
            logging.warning(f"[DIAGNOSIS] LLM error: {e}. Usando diagnóstico determinístico.")
            SRE_DIAGNOSIS_LLM_TOTAL.labels(result="error").inc()

    fallback = _deterministic_diagnosis(anomaly)
    SRE_DIAGNOSIS_CONFIDENCE.observe(fallback.confidence)
    return fallback


# ─────────────────────────────────────────────────────────────────────────────
# P4-A: PROACTIVE METRICS MONITORING
# ─────────────────────────────────────────────────────────────────────────────

def _guess_owner_from_pod_name(pod_name: str) -> str:
    """Heuristically derives deployment name from K8s pod naming convention.
    Pattern: {deployment}-{rs_hash}-{pod_hash} → returns deployment part."""
    parts = pod_name.rsplit("-", 2)
    if len(parts) >= 3:
        return "-".join(parts[:-2])
    if len(parts) == 2:
        return parts[0]
    return pod_name


def observe_metrics() -> List[Anomaly]:
    """
    P4-A: Queries Prometheus for resource pressure anomalies (proactive).
    Detects HIGH_CPU, HIGH_MEMORY, and HIGH_ERROR_RATE before they cause incidents.
    """
    anomalies: List[Anomaly] = []
    ns_regex = "|".join(OBSERVE_NAMESPACES)

    # ── CPU ratio: usage / request > threshold ────────────────────────────────
    cpu_q = (
        f'sum by (pod, namespace) ('
        f'  rate(container_cpu_usage_seconds_total{{container!="",namespace=~"{ns_regex}"}}[5m])'
        f') / ignoring(container) group_left sum by (pod, namespace) ('
        f'  kube_pod_container_resource_requests{{resource="cpu",container!="",namespace=~"{ns_regex}"}}'
        f') > {SRE_CPU_THRESHOLD}'
    )
    try:
        resp = requests.get(f"{PROMETHEUS_URL}/api/v1/query",
                            params={"query": cpu_q}, timeout=8)
        if resp.status_code == 200 and resp.json()["status"] == "success":
            for r in resp.json()["data"]["result"]:
                pod = r["metric"].get("pod", "")
                ns  = r["metric"].get("namespace", "")
                val = float(r["value"][1])
                if not pod or not ns:
                    continue
                anomalies.append(Anomaly(
                    issue_type="HIGH_CPU", severity="MEDIUM",
                    namespace=ns, resource_name=pod, resource_type="pod",
                    owner_name=_guess_owner_from_pod_name(pod), owner_kind="Deployment",
                    details=(f"Pod '{pod}' usa {val:.0%} de su CPU solicitada "
                             f"(umbral: {SRE_CPU_THRESHOLD:.0%})."),
                    dedup_key=f"{ns}:{pod}:HIGH_CPU",
                ))
                SRE_METRIC_ANOMALIES_TOTAL.labels(issue_type="HIGH_CPU").inc()
    except Exception as e:
        logging.debug(f"[METRICS] CPU query error: {e}")

    # ── Memory ratio: working_set / limit > threshold ─────────────────────────
    mem_q = (
        f'sum by (pod, namespace) ('
        f'  container_memory_working_set_bytes{{container!="",namespace=~"{ns_regex}"}}'
        f') / ignoring(container) group_left sum by (pod, namespace) ('
        f'  kube_pod_container_resource_limits{{resource="memory",container!="",namespace=~"{ns_regex}"}}'
        f') > {SRE_MEMORY_THRESHOLD}'
    )
    try:
        resp = requests.get(f"{PROMETHEUS_URL}/api/v1/query",
                            params={"query": mem_q}, timeout=8)
        if resp.status_code == 200 and resp.json()["status"] == "success":
            for r in resp.json()["data"]["result"]:
                pod = r["metric"].get("pod", "")
                ns  = r["metric"].get("namespace", "")
                val = float(r["value"][1])
                if not pod or not ns:
                    continue
                anomalies.append(Anomaly(
                    issue_type="HIGH_MEMORY", severity="HIGH",
                    namespace=ns, resource_name=pod, resource_type="pod",
                    owner_name=_guess_owner_from_pod_name(pod), owner_kind="Deployment",
                    details=(f"Pod '{pod}' usa {val:.0%} de su memory limit "
                             f"(umbral: {SRE_MEMORY_THRESHOLD:.0%}). Riesgo de OOMKill."),
                    dedup_key=f"{ns}:{pod}:HIGH_MEMORY",
                ))
                SRE_METRIC_ANOMALIES_TOTAL.labels(issue_type="HIGH_MEMORY").inc()
    except Exception as e:
        logging.debug(f"[METRICS] Memory query error: {e}")

    # ── HTTP 5xx error rate > 1% ──────────────────────────────────────────────
    err_q = (
        'sum by (handler) (rate(http_requests_total{namespace="amael-ia",status=~"5.."}[5m]))'
        ' / sum by (handler) (rate(http_requests_total{namespace="amael-ia"}[5m]))'
        ' > 0.01'
    )
    try:
        resp = requests.get(f"{PROMETHEUS_URL}/api/v1/query",
                            params={"query": err_q}, timeout=8)
        if resp.status_code == 200 and resp.json()["status"] == "success":
            for r in resp.json()["data"]["result"]:
                handler = r["metric"].get("handler", "unknown")
                val     = float(r["value"][1])
                severity = "CRITICAL" if val > 0.10 else "HIGH"
                anomalies.append(Anomaly(
                    issue_type="HIGH_ERROR_RATE", severity=severity,
                    namespace="amael-ia", resource_name=handler,
                    resource_type="endpoint", owner_name="backend-ia-deployment",
                    owner_kind="Deployment",
                    details=(f"Endpoint '{handler}' tiene {val:.1%} de errores 5xx "
                             f"(umbral: 1%)."),
                    dedup_key=f"amael-ia:{handler}:HIGH_ERROR_RATE",
                ))
                SRE_METRIC_ANOMALIES_TOTAL.labels(issue_type="HIGH_ERROR_RATE").inc()
    except Exception as e:
        logging.debug(f"[METRICS] Error rate query error: {e}")

    if anomalies:
        logging.info(f"[METRICS] {len(anomalies)} anomalía(s) proactiva(s) detectadas.")
    return anomalies


# ─────────────────────────────────────────────────────────────────────────────
# P4-B: ANOMALY CORRELATION
# ─────────────────────────────────────────────────────────────────────────────

def correlate_anomalies(anomalies: List[Anomaly]) -> List[Anomaly]:
    """
    P4-B: Groups anomalies from the same deployment with the same issue_type
    into a single aggregate anomaly to reduce notification noise.
    """
    if len(anomalies) <= 1:
        return anomalies

    groups: dict = {}
    ungrouped: List[Anomaly] = []

    for a in anomalies:
        if a.owner_kind == "Deployment" and a.owner_name and a.resource_type == "pod":
            key = (a.owner_name, a.namespace, a.issue_type)
            groups.setdefault(key, []).append(a)
        else:
            ungrouped.append(a)

    result = list(ungrouped)
    for (owner, ns, issue_type), group in groups.items():
        if len(group) == 1:
            result.append(group[0])
            continue
        # Multiple pods → create a single aggregate anomaly
        pod_names  = [a.resource_name for a in group]
        worst      = max(group, key=lambda a: _SEVERITY_RANK.get(a.severity, 0))
        suffix     = "..." if len(pod_names) > 3 else ""
        result.append(Anomaly(
            issue_type=issue_type,
            severity=worst.severity,
            namespace=ns,
            resource_name=group[0].resource_name,
            resource_type="pod",
            owner_name=owner,
            owner_kind="Deployment",
            details=(f"{len(pod_names)} pods de '{owner}' con {issue_type}: "
                     f"{', '.join(pod_names[:3])}{suffix}."),
            dedup_key=f"{ns}:{owner}:{issue_type}:multi",
        ))
        SRE_CORRELATION_GROUPED.inc()
        logging.debug(f"[CORRELATE] {len(pod_names)} pods → 1 anomalía: {owner}/{issue_type}")

    return result


# ─────────────────────────────────────────────────────────────────────────────
# P4-D: AUTO-GENERATED RUNBOOKS
# ─────────────────────────────────────────────────────────────────────────────

def _maybe_save_runbook_entry(anomaly: Anomaly, diagnosis: Diagnosis):
    """
    P4-D: If the LLM diagnosed an issue with high confidence (≥0.80)
    and no existing runbook covers it well, saves a new auto-generated
    runbook entry to Qdrant to fill knowledge gaps over time.
    """
    if _qdrant_client is None:
        return
    if diagnosis.source != "llm" or diagnosis.confidence < 0.80:
        return  # Only persist high-confidence LLM-sourced diagnoses

    # Avoid duplicates: check if a strong runbook match already exists
    existing = search_runbooks(
        f"{anomaly.issue_type} {anomaly.namespace}", score_threshold=0.75
    )
    if existing:
        return  # Good coverage already exists

    content = (
        f"# Auto-runbook: {anomaly.issue_type}\n\n"
        f"**Generado automáticamente por el agente SRE — "
        f"confianza {diagnosis.confidence:.0%}**\n\n"
        f"## Descripción\n{anomaly.details}\n\n"
        f"## Causa raíz identificada\n"
        f"{diagnosis.root_cause} (categoría: `{diagnosis.root_cause_category}`)\n\n"
        f"## Acción recomendada\n`{diagnosis.recommended_action}`\n\n"
        f"## Evidencia\n"
        + "\n".join(f"- {e}" for e in diagnosis.evidence[:5])
        + "\n"
    )
    try:
        from qdrant_client.http.models import PointStruct
        vector = _get_embedding(content[:2000])
        if not vector:
            return
        # Stable ID based on issue_type + namespace — allows safe upsert
        point_id = (abs(hash(f"{anomaly.issue_type}:{anomaly.namespace}")) % 10**9) + 10000
        _qdrant_client.upsert(
            collection_name=SRE_RUNBOOKS_COLLECTION,
            points=[PointStruct(
                id=point_id,
                vector=vector,
                payload={
                    "name":           f"auto_{anomaly.issue_type.lower()}_{anomaly.namespace}",
                    "content":        content,
                    "file":           "auto_generated",
                    "auto_generated": True,
                    "generated_at":   datetime.now(timezone.utc).isoformat(),
                    "confidence":     diagnosis.confidence,
                },
            )],
        )
        SRE_AUTO_RUNBOOK_SAVED.inc()
        logging.info(
            f"[RUNBOOKS] Auto-runbook guardado: '{anomaly.issue_type}' "
            f"en '{anomaly.namespace}' (confidence={diagnosis.confidence:.0%})."
        )
    except Exception as e:
        logging.warning(f"[RUNBOOKS] Error guardando auto-runbook: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# LOOP — OBSERVE / DETECT / DECIDE / ACT (P1 + P2 guardrails)
# ─────────────────────────────────────────────────────────────────────────────

def observe_cluster() -> ClusterSnapshot:
    pods:  List[PodStatus]  = []
    nodes: List[NodeStatus] = []

    for ns in OBSERVE_NAMESPACES:
        try:
            k8s_pods = v1.list_namespaced_pod(ns)
            for pod in k8s_pods.items:
                phase, rc, wr, lsr = pod.status.phase or "Unknown", 0, "", ""
                if pod.status.container_statuses:
                    for cs in pod.status.container_statuses:
                        rc += cs.restart_count or 0
                        if cs.state and cs.state.waiting:
                            wr = cs.state.waiting.reason or ""
                        if cs.last_state and cs.last_state.terminated:
                            lsr = cs.last_state.terminated.reason or ""

                owner_name, owner_kind = "", ""
                if pod.metadata.owner_references:
                    for ref in pod.metadata.owner_references:
                        if ref.kind in ("ReplicaSet", "StatefulSet", "DaemonSet"):
                            owner_kind = ref.kind
                            owner_name = ref.name
                            if ref.kind == "ReplicaSet":
                                try:
                                    rs = apps_v1.read_namespaced_replica_set(ref.name, ns)
                                    if rs.metadata.owner_references:
                                        for rr in rs.metadata.owner_references:
                                            if rr.kind == "Deployment":
                                                owner_name, owner_kind = rr.name, "Deployment"
                                except Exception:
                                    pass
                            break

                pods.append(PodStatus(
                    name=pod.metadata.name, namespace=ns, phase=phase,
                    restart_count=rc, waiting_reason=wr, last_state_reason=lsr,
                    owner_name=owner_name, owner_kind=owner_kind,
                    start_time=pod.status.start_time,
                ))
        except Exception as e:
            logging.warning(f"[OBSERVE] Error en '{ns}': {e}")

    try:
        for node in v1.list_node().items:
            conds = {c.type: c.status for c in (node.status.conditions or [])}
            nodes.append(NodeStatus(
                name=node.metadata.name,
                ready=conds.get("Ready",          "Unknown") == "True",
                memory_pressure=conds.get("MemoryPressure", "False") == "True",
                disk_pressure=conds.get("DiskPressure",   "False") == "True",
                pid_pressure=conds.get("PIDPressure",    "False") == "True",
            ))
    except Exception as e:
        logging.warning(f"[OBSERVE] Error en nodos: {e}")

    return ClusterSnapshot(timestamp=datetime.now(timezone.utc), pods=pods, nodes=nodes)


def detect_anomalies(snapshot: ClusterSnapshot) -> List[Anomaly]:
    anomalies: List[Anomaly] = []

    for pod in snapshot.pods:
        kp = f"{pod.namespace}:{pod.name}"
        if pod.waiting_reason == "CrashLoopBackOff":
            anomalies.append(Anomaly("CRASH_LOOP", "CRITICAL", pod.namespace, pod.name,
                "pod", pod.owner_name, pod.owner_kind,
                f"Pod '{pod.name}' en CrashLoopBackOff ({pod.restart_count} reinicios). "
                f"Deployment: {pod.owner_name or 'sin owner'}.",
                f"{kp}:CRASH_LOOP"))
        elif pod.waiting_reason in ("ImagePullBackOff", "ErrImagePull"):
            anomalies.append(Anomaly("IMAGE_PULL_ERROR", "HIGH", pod.namespace, pod.name,
                "pod", pod.owner_name, pod.owner_kind,
                f"Pod '{pod.name}' no puede descargar su imagen ({pod.waiting_reason}).",
                f"{kp}:IMAGE_PULL"))
        elif pod.last_state_reason == "OOMKilled":
            anomalies.append(Anomaly("OOM_KILLED", "HIGH", pod.namespace, pod.name,
                "pod", pod.owner_name, pod.owner_kind,
                f"Pod '{pod.name}' terminado por OOMKilled. Memory limit excedido.",
                f"{kp}:OOM"))
        elif pod.phase == "Failed":
            anomalies.append(Anomaly("POD_FAILED", "HIGH", pod.namespace, pod.name,
                "pod", pod.owner_name, pod.owner_kind,
                f"Pod '{pod.name}' en estado Failed.",
                f"{kp}:FAILED"))
        elif pod.phase == "Pending" and pod.start_time:
            age = (datetime.now(timezone.utc) - pod.start_time).total_seconds()
            if age > 300:
                anomalies.append(Anomaly("POD_PENDING_STUCK", "MEDIUM", pod.namespace, pod.name,
                    "pod", pod.owner_name, pod.owner_kind,
                    f"Pod '{pod.name}' lleva {int(age/60)}min en Pending.",
                    f"{kp}:PENDING"))
        elif pod.restart_count >= 5 and not pod.waiting_reason:
            anomalies.append(Anomaly("HIGH_RESTARTS", "MEDIUM", pod.namespace, pod.name,
                "pod", pod.owner_name, pod.owner_kind,
                f"Pod '{pod.name}' tiene {pod.restart_count} reinicios (actualmente Running).",
                f"{kp}:HIGH_RESTARTS"))

    for node in snapshot.nodes:
        kn = f"node:{node.name}"
        if not node.ready:
            anomalies.append(Anomaly("NODE_NOT_READY", "CRITICAL", "", node.name, "node",
                "", "", f"Nodo '{node.name}' NOT READY. Clúster afectado.", f"{kn}:NOT_READY"))
        if node.memory_pressure:
            anomalies.append(Anomaly("NODE_MEMORY_PRESSURE", "HIGH", "", node.name, "node",
                "", "", f"Nodo '{node.name}' con MemoryPressure.", f"{kn}:MEM_PRESSURE"))
        if node.disk_pressure:
            anomalies.append(Anomaly("NODE_DISK_PRESSURE", "HIGH", "", node.name, "node",
                "", "", f"Nodo '{node.name}' con DiskPressure.", f"{kn}:DISK_PRESSURE"))

    return anomalies


def decide_action(anomaly: Anomaly, diagnosis: Optional[Diagnosis] = None) -> ActionPlan:
    """
    Política de decisión con guardrails P2 + aprendizaje histórico P3-B:
    1. Usa diagnosis.confidence si está disponible (LLM o determinístico).
    2. Ajusta confianza con historial de incidentes (P3-B).
    3. Chequea restart limit antes de ROLLOUT_RESTART.
    4. Solo actúa en namespaces autorizados y deployments no protegidos.
    """
    diag = diagnosis or _deterministic_diagnosis(anomaly)
    # P3-B: blend with historical success rate
    diag = adjust_confidence_with_history(diag, anomaly)

    min_rank = _SEVERITY_RANK.get(AUTO_HEAL_MIN_SEVERITY, 2)
    sev_rank = _SEVERITY_RANK.get(anomaly.severity, 0)
    in_heal_ns     = anomaly.namespace == DEFAULT_NAMESPACE
    has_deployment = (anomaly.owner_kind == "Deployment" and
                      anomaly.owner_name and
                      anomaly.owner_name not in PROTECTED_DEPLOYMENTS)
    is_node        = anomaly.resource_type == "node"
    meets_severity = sev_rank >= min_rank
    meets_confidence = diag.confidence >= CONFIDENCE_THRESHOLD

    # Nodos, endpoints, y tipos específicos → siempre notificar (sin auto-heal)
    _notify_only_types = {
        "IMAGE_PULL_ERROR", "HIGH_CPU", "HIGH_ERROR_RATE",
        # P5: predictive and SLO anomalies always need human judgment
        "DISK_EXHAUSTION_PREDICTED", "ERROR_RATE_ESCALATING", "SLO_BUDGET_BURNING",
    }
    if is_node or anomaly.issue_type in _notify_only_types or anomaly.resource_type in ("endpoint", "namespace", "node"):
        return ActionPlan(
            action="NOTIFY_HUMAN", target_name=anomaly.resource_name,
            target_namespace=anomaly.namespace,
            reason=f"{anomaly.issue_type} requiere intervención manual. {anomaly.details}",
            auto_execute=True,
        )

    if anomaly.issue_type == "POD_PENDING_STUCK":
        return ActionPlan(
            action="NOTIFY_HUMAN", target_name=anomaly.resource_name,
            target_namespace=anomaly.namespace,
            reason=f"Pod en Pending: revisar scheduling y recursos. {anomaly.details}",
            auto_execute=True,
        )

    # Auto-heal si cumple todos los criterios
    if has_deployment and in_heal_ns and meets_severity and meets_confidence:
        # P2: Guardrail de restart limit
        if _check_restart_limit(anomaly.owner_name, anomaly.namespace):
            SRE_RESTART_LIMIT_HIT.inc()
            return ActionPlan(
                action="NOTIFY_HUMAN", target_name=anomaly.owner_name,
                target_namespace=anomaly.namespace,
                reason=(f"Límite de {MAX_RESTARTS_PER_RESOURCE} reinicios automáticos alcanzado "
                        f"en {RESTART_WINDOW_MINUTES}min para '{anomaly.owner_name}'. "
                        "Intervención manual requerida."),
                auto_execute=True,
            )
        return ActionPlan(
            action="ROLLOUT_RESTART", target_name=anomaly.owner_name,
            target_namespace=anomaly.namespace,
            reason=(f"Auto-remediación: {anomaly.issue_type} | "
                    f"causa={diag.root_cause_category} | "
                    f"confianza={diag.confidence:.0%} | "
                    f"fuente={diag.source}"),
            auto_execute=True,
        )

    # No cumple umbrales → notificar con contexto de diagnóstico
    reason_parts = []
    if not in_heal_ns:
        reason_parts.append(f"namespace '{anomaly.namespace}' no es heal namespace")
    if not has_deployment:
        reason_parts.append(f"sin deployment owner o deployment protegido")
    if not meets_severity:
        reason_parts.append(f"severidad {anomaly.severity} < {AUTO_HEAL_MIN_SEVERITY}")
    if not meets_confidence:
        reason_parts.append(f"confianza {diag.confidence:.0%} < {CONFIDENCE_THRESHOLD:.0%}")

    return ActionPlan(
        action="NOTIFY_HUMAN", target_name=anomaly.resource_name,
        target_namespace=anomaly.namespace,
        reason=f"No auto-healable: {', '.join(reason_parts)}. Causa: {diag.root_cause}",
        auto_execute=True,
    )


def execute_sre_action(plan: ActionPlan, anomaly: Anomaly,
                       diagnosis: Optional[Diagnosis] = None,
                       incident_key: str = "") -> str:
    if plan.action == "ROLLOUT_RESTART":
        result = rollout_restart_deployment(f"{plan.target_name}, {plan.target_namespace}")
        ok = "✅" in result
        SRE_ACTIONS_TAKEN.labels(action="rollout_restart", result="ok" if ok else "error").inc()
        if ok:
            _record_restart(plan.target_name, plan.target_namespace)
            confidence_str = f" (confianza: {diagnosis.confidence:.0%})" if diagnosis else ""
            _send_sre_notification(
                f"✅ Auto-remediado: {anomaly.issue_type} en "
                f"'{plan.target_name}' ({plan.target_namespace}){confidence_str}.",
                severity="INFO",
            )
            # P3-A: Schedule verification in 5 min
            if incident_key:
                _schedule_verification(
                    incident_key, plan.target_name,
                    plan.target_namespace, anomaly.issue_type,
                )
        return result

    if plan.action == "NOTIFY_HUMAN":
        diag = diagnosis
        root_cause_str = f"\nCausa: {diag.root_cause} ({diag.root_cause_category})" if diag else ""
        conf_str       = f"\nConfianza: {diag.confidence:.0%} ({diag.source})"       if diag else ""
        msg = (f"⚠️ {anomaly.severity}: {anomaly.issue_type}\n"
               f"Recurso: {anomaly.resource_name} ({anomaly.namespace})"
               f"{root_cause_str}{conf_str}\n"
               f"Motivo: {plan.reason}")
        _send_sre_notification(msg, severity=anomaly.severity)
        SRE_ACTIONS_TAKEN.labels(action="notify_human", result="ok").inc()
        return f"Notificación enviada: {anomaly.issue_type}"

    return "NO_ACTION"


def _run_loop_iteration():
    logging.info("[SRE_LOOP] Iniciando iteración.")

    # P4-C: Skip the entire iteration during a maintenance window
    if _is_maintenance_active():
        SRE_LOOP_RUNS_TOTAL.labels(result="maintenance").inc()
        logging.info("[SRE_LOOP] Ventana de mantenimiento activa. Loop pausado.")
        return

    try:
        # ── OBSERVE K8S ──────────────────────────────────────────────────────
        snapshot = observe_cluster()
        logging.info(
            f"[SRE_LOOP] {len(snapshot.pods)} pods en "
            f"{len(OBSERVE_NAMESPACES)} ns, {len(snapshot.nodes)} nodo(s)."
        )

        # ── OBSERVE METRICS (P4-A) ────────────────────────────────────────────
        metric_anomalies = observe_metrics()

        # ── OBSERVE TRENDS (P5-A) ─────────────────────────────────────────────
        trend_anomalies  = observe_trends()

        # ── OBSERVE SLO (P5-C) ───────────────────────────────────────────────
        slo_anomalies    = observe_slo()

        # ── DETECT ───────────────────────────────────────────────────────────
        anomalies = detect_anomalies(snapshot) + metric_anomalies + trend_anomalies + slo_anomalies

        # ── CORRELATE (P4-B) ─────────────────────────────────────────────────
        anomalies = correlate_anomalies(anomalies)

        if not anomalies:
            SRE_LOOP_RUNS_TOTAL.labels(result="ok_clean").inc()
            _circuit_breaker.record_success()
            return

        logging.info(f"[SRE_LOOP] {len(anomalies)} anomalía(s).")

        for anomaly in anomalies:
            if _is_duplicate_incident(anomaly.dedup_key):
                continue
            _mark_incident(anomaly.dedup_key)
            SRE_ANOMALIES_DETECTED.labels(
                severity=anomaly.severity, issue_type=anomaly.issue_type
            ).inc()
            logging.warning(
                f"[SRE_LOOP] [{anomaly.severity}] {anomaly.issue_type} — "
                f"{anomaly.namespace}/{anomaly.resource_name}"
            )

            # ── DIAGNOSE (P2) ─────────────────────────────────────────────
            diagnosis = diagnose_with_llm(anomaly)

            # ── AUTO-RUNBOOK (P4-D) ────────────────────────────────────────
            _maybe_save_runbook_entry(anomaly, diagnosis)

            # ── DECIDE (P2 + P3-B guardrails) ─────────────────────────────
            plan = decide_action(anomaly, diagnosis)

            # ── ACT ───────────────────────────────────────────────────────
            incident_key = (f"{anomaly.dedup_key}:"
                            f"{datetime.now(timezone.utc).strftime('%Y%m%d%H')}")
            action_result = execute_sre_action(plan, anomaly, diagnosis, incident_key)
            logging.info(f"[SRE_LOOP] '{plan.action}': {action_result}")

            # ── STORE (P1 + P2 root_cause/confidence) ─────────────────────
            store_incident(
                incident_key=incident_key,
                namespace=anomaly.namespace,
                resource_name=anomaly.resource_name,
                resource_type=anomaly.resource_type,
                issue_type=anomaly.issue_type,
                severity=anomaly.severity,
                details=anomaly.details,
                root_cause=diagnosis.root_cause,
                confidence=diagnosis.confidence,
                action_taken=plan.action,
                action_result=action_result[:200],
                notified=(plan.action == "NOTIFY_HUMAN"),
            )

        SRE_LOOP_RUNS_TOTAL.labels(result="ok").inc()
        _circuit_breaker.record_success()

    except Exception as e:
        logging.error(f"[SRE_LOOP] Error: {e}", exc_info=True)
        SRE_LOOP_RUNS_TOTAL.labels(result="error").inc()
        _circuit_breaker.record_failure()


def sre_autonomous_loop():
    # P3-D: Only the leader pod runs the SRE loop
    if not _try_acquire_lease():
        SRE_LOOP_RUNS_TOTAL.labels(result="skipped_not_leader").inc()
        return
    if _circuit_breaker.is_open():
        SRE_LOOP_RUNS_TOTAL.labels(result="skipped_cb").inc()
        return
    if not _loop_lock.acquire(blocking=False):
        SRE_LOOP_RUNS_TOTAL.labels(result="skipped_running").inc()
        return
    try:
        _run_loop_iteration()
    finally:
        _loop_lock.release()


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS ENDPOINT CONVERSACIONAL
# ─────────────────────────────────────────────────────────────────────────────
def extract_final_answer(raw_response: str) -> str:
    global latest_media_captured
    if "Final Answer:" in raw_response:
        res = raw_response.split("Final Answer:")[-1].strip()
    else:
        skip = {"Thought:", "Action:", "Action Input:", "Observation:"}
        res  = "\n".join(
            l for l in raw_response.split("\n")
            if not any(l.strip().startswith(p) for p in skip)
        ).strip()
    if latest_media_captured:
        res += f"\n\n[MEDIA:{latest_media_captured}]"
        latest_media_captured = None
    return res or raw_response


_VAULT_KEYWORDS = {
    "vault", "unseal", "dessellar", "sellar", "seal", "unseal key",
    "secret", "secreto", "policy", "política", "rol", "role", "kv", "hvac",
    "token oauth", "google token", "productivity", "auth/kubernetes",
    "amael-productivity", "vault.root", "root token",
}


def _is_vault_question(query: str) -> bool:
    return any(kw in query.lower() for kw in _VAULT_KEYWORDS)


# ─────────────────────────────────────────────────────────────────────────────
# STARTUP / SHUTDOWN
# ─────────────────────────────────────────────────────────────────────────────
@app.on_event("startup")
async def startup():
    # P1: tabla sre_incidents
    init_sre_db()

    # P2: runbooks en Qdrant
    init_runbooks_qdrant()

    # P5-C: SLO targets
    load_slo_targets()

    # P1: loop autónomo
    if SRE_LOOP_ENABLED:
        from apscheduler.schedulers.background import BackgroundScheduler
        scheduler = BackgroundScheduler(daemon=True)
        scheduler.add_job(
            sre_autonomous_loop,
            trigger="interval",
            seconds=SRE_LOOP_INTERVAL,
            max_instances=1,
            coalesce=True,
            id="sre_loop",
        )
        scheduler.start()
        app.state.sre_scheduler = scheduler
        # P3-A: expose scheduler for one-shot verification jobs
        global _sre_scheduler
        _sre_scheduler = scheduler
        logging.info(
            f"[SRE_LOOP] Iniciado — interval={SRE_LOOP_INTERVAL}s | "
            f"ns={OBSERVE_NAMESPACES} | min_severity={AUTO_HEAL_MIN_SEVERITY} | "
            f"confidence_threshold={CONFIDENCE_THRESHOLD} | "
            f"max_restarts={MAX_RESTARTS_PER_RESOURCE}/{RESTART_WINDOW_MINUTES}min"
        )
    else:
        logging.info("[SRE_LOOP] Deshabilitado.")


@app.on_event("shutdown")
async def shutdown():
    if hasattr(app.state, "sre_scheduler"):
        app.state.sre_scheduler.shutdown(wait=False)
        logging.info("[SRE_LOOP] Scheduler detenido.")


# ─────────────────────────────────────────────────────────────────────────────
# ENDPOINTS
# ─────────────────────────────────────────────────────────────────────────────
@app.post("/api/k8s-agent")
async def chat_with_agent(request: AgentRequest, req: Request):
    AGENT_REQUESTS_TOTAL.inc()
    global latest_media_captured
    latest_media_captured = None

    if INTERNAL_API_SECRET:
        auth_header = req.headers.get("Authorization", "")
        token = auth_header.removeprefix("Bearer ").strip()
        if token != INTERNAL_API_SECRET:
            logging.warning(f"[SECURITY] Secret inválido desde {req.client.host}")
            raise HTTPException(status_code=403, detail="Acceso no autorizado.")

    logging.info(f"Petición de {request.user_email}: {request.query[:80]}")

    if K8S_ALLOWED_USERS and request.user_email not in K8S_ALLOWED_USERS:
        return {"response": "No tienes permisos para acceder al clúster."}

    if _is_vault_question(request.query) and _VAULT_KNOWLEDGE:
        vault_raw = _VAULT_KNOWLEDGE.replace("{{", "{").replace("}}", "}")
        prompt = (
            f"Eres un experto en HashiCorp Vault. Basándote ÚNICAMENTE en el siguiente "
            f"documento, responde de forma concisa.\n\n"
            f"DOCUMENTO:\n{vault_raw}\n\nPREGUNTA: {request.query}\n\nRESPUESTA:"
        )
        try:
            return {"response": str(llm.invoke(prompt)).strip()}
        except Exception as exc:
            logging.error(f"[VAULT_KB] Error: {exc}")
            return {"response": vault_raw}

    # P3-C: LangGraph agent (primary), fallback to LangChain classic
    if _LANGGRAPH_ENABLED and _langgraph_agent is not None:
        try:
            from langchain_core.messages import HumanMessage as _HM
            result = _langgraph_agent.invoke(
                {"messages": [_HM(content=request.query)]},
                config={"callbacks": [metrics_callback]},
            )
            msgs  = result.get("messages", [])
            final = msgs[-1].content if msgs else ""
            SRE_LANGGRAPH_REQUESTS.labels(result="ok").inc()
            return {"response": extract_final_answer(final)}
        except Exception as e:
            logging.warning(f"[LangGraph] Error en invocación: {e}. Fallback a LangChain clásico.")
            SRE_LANGGRAPH_REQUESTS.labels(result="fallback").inc()

    try:
        raw_response = agent.run(request.query, callbacks=[metrics_callback])
        return {"response": extract_final_answer(raw_response)}
    except Exception as e:
        logging.error(f"Agent error: {e}")
        return {"response": f"Error procesando la petición: {str(e)[:150]}"}


@app.get("/api/sre/incidents")
async def get_sre_incidents(limit: int = 20):
    incidents = get_recent_incidents(limit=min(limit, 100))
    return {"incidents": incidents, "count": len(incidents)}


@app.get("/api/sre/loop/status")
async def get_loop_status():
    return {
        "loop_enabled":              SRE_LOOP_ENABLED,
        "loop_interval_s":           SRE_LOOP_INTERVAL,
        "observe_namespaces":        OBSERVE_NAMESPACES,
        "auto_heal_min_severity":    AUTO_HEAL_MIN_SEVERITY,
        "confidence_threshold":      CONFIDENCE_THRESHOLD,
        "max_restarts_per_resource": MAX_RESTARTS_PER_RESOURCE,
        "restart_window_minutes":    RESTART_WINDOW_MINUTES,
        "circuit_breaker":           _circuit_breaker.state,
        "protected_deployments":     list(PROTECTED_DEPLOYMENTS),
        "qdrant_runbooks":           _qdrant_client is not None,
        # P3
        "langgraph_enabled":         _LANGGRAPH_ENABLED,
        "leader_pod":                _POD_NAME,
        # P4
        "maintenance_active":        _is_maintenance_active(),
        "cpu_threshold":             SRE_CPU_THRESHOLD,
        "memory_threshold":          SRE_MEMORY_THRESHOLD,
        # P5
        "slo_targets_count":         len(_SLO_TARGETS),
        "memory_leak_rate_bytes_s":  SRE_MEMORY_LEAK_RATE_BYTES,
    }


@app.get("/api/sre/learning/stats")
async def get_learning_stats_endpoint():
    """P3-B: 7-day aggregate stats per (issue_type, action): success rate, avg confidence."""
    stats = get_learning_stats()
    return {"stats": stats, "window_days": 7}


# ─── P4-C: Maintenance window endpoints ──────────────────────────────────────
@app.get("/api/sre/maintenance")
async def get_maintenance():
    active = _is_maintenance_active()
    ttl    = _redis.ttl(_MAINTENANCE_KEY) if (active and _redis) else None
    return {"active": active, "ttl_seconds": ttl}


@app.post("/api/sre/maintenance")
async def post_maintenance(req: Request):
    body    = await req.json()
    minutes = body.get("duration_minutes", 60)
    return {"result": _activate_maintenance(str(minutes))}


@app.delete("/api/sre/maintenance")
async def delete_maintenance():
    return {"result": _deactivate_maintenance()}


# ─── P5-D: Postmortems endpoint ──────────────────────────────────────────────
@app.get("/api/sre/postmortems")
async def get_postmortems(limit: int = 10):
    """Returns the most recent auto-generated postmortems."""
    postmortems = get_recent_postmortems(limit=min(limit, 50))
    return {"postmortems": postmortems, "count": len(postmortems)}


# ─── P5-C: SLO status endpoint ───────────────────────────────────────────────
@app.get("/api/sre/slo/status")
async def get_slo_status():
    """Returns current SLO targets and live burn rates from Prometheus."""
    results = []
    for slo in _SLO_TARGETS:
        handler      = slo.get("handler", "")
        target       = slo.get("availability", 0.995)
        window_h     = slo.get("window_h", 24)
        error_budget = 1.0 - target
        entry = {
            "service":       slo.get("service", handler),
            "handler":       handler,
            "target":        target,
            "window_h":      window_h,
            "error_budget":  error_budget,
            "burn_rate":     None,
            "status":        "unknown",
        }
        if error_budget > 0:
            err_q = (
                f'sum(rate(http_requests_total{{namespace="amael-ia",'
                f'handler=~"{handler}",status=~"5.."}}[{window_h}h]))'
                f' / sum(rate(http_requests_total{{namespace="amael-ia",'
                f'handler=~"{handler}"}}[{window_h}h]))'
            )
            try:
                resp = requests.get(f"{PROMETHEUS_URL}/api/v1/query",
                                    params={"query": err_q}, timeout=8)
                if resp.status_code == 200 and resp.json()["status"] == "success":
                    rs = resp.json()["data"]["result"]
                    if rs:
                        current = float(rs[0]["value"][1])
                        burn    = current / error_budget
                        entry["burn_rate"] = round(burn, 3)
                        entry["status"]    = (
                            "critical" if burn >= 5.0 else
                            "warning"  if burn >= 2.0 else "ok"
                        )
            except Exception:
                pass
        results.append(entry)
    return {"slo_targets": results, "count": len(results)}


# ─── P5-E: WhatsApp SRE command endpoint ─────────────────────────────────────
@app.post("/api/sre/command")
async def sre_command(body: SRECommandRequest):
    """
    P5-E: Receives /sre commands from whatsapp-bridge and returns a text reply.
    Supported commands: status, incidents, postmortems, slo, maintenance [on <min>|off].
    """
    cmd = body.command.strip().lower()
    SRE_WA_COMMANDS_TOTAL.labels(command=cmd.split()[0] if cmd else "empty").inc()

    if not cmd or cmd == "ayuda":
        return {"reply": (
            "📋 *Comandos SRE disponibles:*\n"
            "• `status` — estado del loop y circuit breaker\n"
            "• `incidents` — últimos 5 incidentes\n"
            "• `postmortems` — últimos 3 postmortems\n"
            "• `slo` — estado de SLOs\n"
            "• `maintenance on <min>` — activar mantenimiento\n"
            "• `maintenance off` — desactivar mantenimiento"
        )}

    if cmd == "status":
        cb    = _circuit_breaker.state
        maint = _is_maintenance_active()
        return {"reply": (
            f"🤖 *SRE Agent — Estado*\n"
            f"• Loop: {'✅ activo' if SRE_LOOP_ENABLED else '❌ inactivo'}\n"
            f"• Circuit Breaker: {cb}\n"
            f"• Mantenimiento: {'⚙️ activo' if maint else '✅ inactivo'}\n"
            f"• Namespaces: {', '.join(OBSERVE_NAMESPACES)}\n"
            f"• SLOs configurados: {len(_SLO_TARGETS)}"
        )}

    if cmd == "incidents":
        incidents = get_recent_incidents(limit=5)
        if not incidents:
            return {"reply": "✅ Sin incidentes recientes."}
        lines = ["📋 *Últimos incidentes:*"]
        for inc in incidents:
            ts = inc.get("created_at", "")[:16] if inc.get("created_at") else "?"
            lines.append(
                f"• [{ts}] {inc['issue_type']} — {inc['resource']} "
                f"({inc['severity']}) → {inc['action']}"
            )
        return {"reply": "\n".join(lines)}

    if cmd == "postmortems":
        pms = get_recent_postmortems(limit=3)
        if not pms:
            return {"reply": "📭 Sin postmortems generados aún."}
        lines = ["📝 *Últimos postmortems:*"]
        for pm in pms:
            ts = pm.get("created_at", "")[:10] if pm.get("created_at") else "?"
            lines.append(
                f"• [{ts}] {pm['issue_type']} — {pm['resource']}\n"
                f"  Causa: {(pm.get('root_cause_summary') or '')[:80]}"
            )
        return {"reply": "\n".join(lines)}

    if cmd == "slo":
        slo_data = []
        for slo in _SLO_TARGETS:
            slo_data.append(
                f"• {slo.get('service', slo.get('handler', '?'))}: "
                f"target {slo.get('availability', 0):.1%}"
            )
        if not slo_data:
            return {"reply": "📭 No hay SLO targets configurados (SLO_TARGETS_JSON vacío)."}
        return {"reply": "📊 *SLO Targets:*\n" + "\n".join(slo_data)}

    if cmd.startswith("maintenance"):
        parts = cmd.split()
        if len(parts) >= 2 and parts[1] == "off":
            return {"reply": _deactivate_maintenance()}
        if len(parts) >= 3 and parts[1] == "on":
            return {"reply": _activate_maintenance(parts[2])}
        if len(parts) == 2 and parts[1] == "on":
            return {"reply": _activate_maintenance("60")}
        active = _is_maintenance_active()
        return {"reply": f"🔧 Mantenimiento: {'activo' if active else 'inactivo'}. "
                         f"Usa: maintenance on <min> / maintenance off"}

    return {"reply": f"❓ Comando desconocido: '{cmd}'. Escribe `ayuda` para ver opciones."}


@app.get("/health")
async def health_check():
    return {"status": "ok"}


Instrumentator().instrument(app).expose(app)
