"""
conftest.py — Fixtures compartidos para el test suite del Agente SRE Autónomo.
Proporciona objetos de datos base (Anomaly, Diagnosis, ClusterSnapshot) y
mocks pre-configurados para dependencias externas (Redis, PostgreSQL, K8s, Prometheus).
"""
from __future__ import annotations
import sys
import types
import pytest
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch


# ─────────────────────────────────────────────────────────────────────────────
# Stub de módulos pesados antes de importar main.py
# ─────────────────────────────────────────────────────────────────────────────

def _stub_heavy_modules():
    """Crea stubs mínimos para que main.py pueda importarse sin servicios reales."""

    # kubernetes
    k8s = types.ModuleType("kubernetes")
    k8s.client = MagicMock()
    k8s.config = MagicMock()
    k8s.stream = MagicMock()
    sys.modules.setdefault("kubernetes", k8s)
    sys.modules.setdefault("kubernetes.client", k8s.client)
    sys.modules.setdefault("kubernetes.config", k8s.config)
    sys.modules.setdefault("kubernetes.stream", k8s.stream)

    # langchain / langchain_ollama
    for mod in [
        "langchain", "langchain.agents", "langchain.callbacks",
        "langchain.callbacks.base", "langchain_ollama",
        "langchain_core", "langchain_core.messages",
        "langgraph", "langgraph.prebuilt",
    ]:
        sys.modules.setdefault(mod, MagicMock())

    # redis
    redis_mod = types.ModuleType("redis")
    redis_cls = MagicMock()
    redis_cls.from_url.return_value = MagicMock()
    redis_mod.Redis = redis_cls
    sys.modules.setdefault("redis", redis_mod)

    # psycopg2
    pg = types.ModuleType("psycopg2")
    pg.pool = MagicMock()
    sys.modules.setdefault("psycopg2", pg)
    sys.modules.setdefault("psycopg2.pool", pg.pool)

    # prometheus_client
    prom = types.ModuleType("prometheus_client")
    prom.Counter   = MagicMock(return_value=MagicMock())
    prom.Gauge     = MagicMock(return_value=MagicMock())
    prom.Histogram = MagicMock(return_value=MagicMock())
    sys.modules.setdefault("prometheus_client", prom)

    # prometheus_fastapi_instrumentator
    instr = types.ModuleType("prometheus_fastapi_instrumentator")
    instr.Instrumentator = MagicMock()
    sys.modules.setdefault("prometheus_fastapi_instrumentator", instr)

    # apscheduler
    aps = types.ModuleType("apscheduler")
    sys.modules.setdefault("apscheduler", aps)
    sys.modules.setdefault("apscheduler.schedulers", MagicMock())
    sys.modules.setdefault("apscheduler.schedulers.background", MagicMock())

    # qdrant_client
    qdrant = types.ModuleType("qdrant_client")
    qdrant.QdrantClient = MagicMock()
    qdrant.http = MagicMock()
    qdrant.http.models = MagicMock()
    sys.modules.setdefault("qdrant_client", qdrant)
    sys.modules.setdefault("qdrant_client.http", qdrant.http)
    sys.modules.setdefault("qdrant_client.http.models", qdrant.http.models)

    # tracing (local)
    tracing = types.ModuleType("tracing")
    tracing.instrument_app = lambda app: None
    sys.modules["tracing"] = tracing

    # opentelemetry
    for mod in [
        "opentelemetry", "opentelemetry.sdk", "opentelemetry.exporter",
        "opentelemetry.instrumentation", "opentelemetry.instrumentation.fastapi",
        "opentelemetry.instrumentation.requests",
        "opentelemetry.exporter.otlp", "opentelemetry.exporter.otlp.proto",
        "opentelemetry.exporter.otlp.proto.grpc",
        "opentelemetry.sdk.trace", "opentelemetry.sdk.trace.export",
    ]:
        sys.modules.setdefault(mod, MagicMock())


_stub_heavy_modules()


# ─────────────────────────────────────────────────────────────────────────────
# Importar dataclasses de main.py después de los stubs
# ─────────────────────────────────────────────────────────────────────────────
import importlib
import main as _main_module

from main import (
    Anomaly, Diagnosis, ActionPlan, ClusterSnapshot, PodStatus, NodeStatus,
    CircuitBreaker,
    _parse_two, _guess_owner_from_pod_name,
    detect_anomalies, correlate_anomalies,
    _deterministic_diagnosis, adjust_confidence_with_history,
    decide_action,
    _is_duplicate_incident, _mark_incident,
    _check_restart_limit,
    _is_maintenance_active,
    load_slo_targets,
    observe_slo, observe_trends, observe_metrics,
    rollout_undo_deployment, rollout_restart_deployment,
    _was_recently_deployed,
    search_runbooks,
    _maybe_save_runbook_entry,
)


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures de objetos de datos
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def crash_loop_anomaly():
    return Anomaly(
        issue_type="CRASH_LOOP",
        severity="CRITICAL",
        namespace="amael-ia",
        resource_name="backend-ia-abc-xyz",
        resource_type="pod",
        owner_name="backend-ia-deployment",
        owner_kind="Deployment",
        details="Pod 'backend-ia-abc-xyz' en CrashLoopBackOff (10 reinicios).",
        dedup_key="amael-ia:backend-ia-abc-xyz:CRASH_LOOP",
    )


@pytest.fixture
def oom_anomaly():
    return Anomaly(
        issue_type="OOM_KILLED",
        severity="HIGH",
        namespace="amael-ia",
        resource_name="backend-ia-oom-pod",
        resource_type="pod",
        owner_name="backend-ia-deployment",
        owner_kind="Deployment",
        details="Pod 'backend-ia-oom-pod' terminado por OOMKilled.",
        dedup_key="amael-ia:backend-ia-oom-pod:OOM",
    )


@pytest.fixture
def image_pull_anomaly():
    return Anomaly(
        issue_type="IMAGE_PULL_ERROR",
        severity="HIGH",
        namespace="amael-ia",
        resource_name="frontend-bad-image",
        resource_type="pod",
        owner_name="frontend-deployment",
        owner_kind="Deployment",
        details="Pod 'frontend-bad-image' no puede descargar imagen (ErrImagePull).",
        dedup_key="amael-ia:frontend-bad-image:IMAGE_PULL",
    )


@pytest.fixture
def slo_anomaly():
    return Anomaly(
        issue_type="SLO_BUDGET_BURNING",
        severity="CRITICAL",
        namespace="amael-ia",
        resource_name="/api/chat",
        resource_type="endpoint",
        owner_name="backend-ia",
        owner_kind="Deployment",
        details="SLO '/api/chat': burn_rate=6.0× (objetivo: 99.5%).",
        dedup_key="amael-ia:/api/chat:SLO_BURN",
    )


@pytest.fixture
def low_confidence_diagnosis(crash_loop_anomaly):
    return Diagnosis(
        issue_type="CRASH_LOOP",
        root_cause="Dependency failed",
        root_cause_category="DEPENDENCY",
        confidence=0.40,
        severity="CRITICAL",
        recommended_action="ROLLOUT_RESTART",
        evidence=["CrashLoopBackOff detected"],
        source="llm",
    )


@pytest.fixture
def high_confidence_diagnosis(crash_loop_anomaly):
    return Diagnosis(
        issue_type="CRASH_LOOP",
        root_cause="Dependency failed at startup",
        root_cause_category="DEPENDENCY",
        confidence=0.90,
        severity="CRITICAL",
        recommended_action="ROLLOUT_RESTART",
        evidence=["CrashLoopBackOff", "connection refused in logs"],
        source="llm",
    )


@pytest.fixture
def healthy_cluster_snapshot():
    pods = [
        PodStatus(
            name="backend-ia-abc", namespace="amael-ia", phase="Running",
            restart_count=0, waiting_reason="", last_state_reason="",
            owner_name="backend-ia-deployment", owner_kind="Deployment",
            start_time=datetime.now(timezone.utc),
        )
    ]
    nodes = [
        NodeStatus(name="lab-home", ready=True, memory_pressure=False,
                   disk_pressure=False, pid_pressure=False)
    ]
    return ClusterSnapshot(timestamp=datetime.now(timezone.utc), pods=pods, nodes=nodes)


@pytest.fixture
def unhealthy_cluster_snapshot():
    pods = [
        PodStatus(
            name="backend-ia-crash", namespace="amael-ia", phase="Running",
            restart_count=12, waiting_reason="CrashLoopBackOff", last_state_reason="Error",
            owner_name="backend-ia-deployment", owner_kind="Deployment",
            start_time=datetime.now(timezone.utc),
        ),
        PodStatus(
            name="frontend-oom", namespace="amael-ia", phase="Running",
            restart_count=3, waiting_reason="", last_state_reason="OOMKilled",
            owner_name="frontend-deployment", owner_kind="Deployment",
            start_time=datetime.now(timezone.utc),
        ),
    ]
    nodes = [
        NodeStatus(name="lab-home", ready=True, memory_pressure=False,
                   disk_pressure=False, pid_pressure=False)
    ]
    return ClusterSnapshot(timestamp=datetime.now(timezone.utc), pods=pods, nodes=nodes)


@pytest.fixture
def mock_redis():
    """Redis mock con comportamiento de incidente/restart."""
    r = MagicMock()
    r.exists.return_value = 0
    r.get.return_value = None
    pipeline = MagicMock()
    pipeline.__enter__ = MagicMock(return_value=pipeline)
    pipeline.__exit__ = MagicMock(return_value=False)
    r.pipeline.return_value = pipeline
    return r


@pytest.fixture
def mock_pg_pool():
    """Pool PostgreSQL simulado con cursor que retorna datos genéricos."""
    conn = MagicMock()
    cur = MagicMock()
    cur.__enter__ = MagicMock(return_value=cur)
    cur.__exit__ = MagicMock(return_value=False)
    conn.cursor.return_value = cur
    cur.fetchall.return_value = []
    cur.fetchone.return_value = None
    pool = MagicMock()
    pool.getconn.return_value = conn
    return pool, conn, cur
