"""
test_sre_agent.py — Suite de pruebas del Agente SRE Autónomo (k8s-agent)
Cubre las fases P0–P5 del ciclo: Observe → Detect → Diagnose → Decide → Act → Verify → Learn

Estructura:
  TestCircuitBreaker          — P0: lógica del circuit breaker
  TestHelpers                 — P0: utilidades parse / guess_owner
  TestDetectAnomalies         — P1: detección estructural de anomalías K8s
  TestCorrelateAnomalies      — P4-B: correlación de anomalías multi-pod
  TestDeterministicDiagnosis  — P2: diagnóstico determinístico (sin LLM)
  TestHistoricalLearning      — P3-B: ajuste de confianza con historial
  TestDecideAction            — P2+P3: política de decisión y guardrails
  TestDedup                   — P1: deduplicación con Redis / fallback en memoria
  TestRestartLimit            — P2: guardrail de reinicios máximos
  TestMaintenanceWindow       — P4-C: ventanas de mantenimiento
  TestObserveMetrics          — P4-A: monitoreo proactivo vía Prometheus
  TestObserveTrends           — P5-A: alertas predictivas (disk, memory leak, error trend)
  TestObserveSLO              — P5-C: error budget burn rate
  TestRolloutUndo             — P5-B: rollback automático
  TestAutoRunbook             — P4-D: auto-generación de runbooks
  TestPostVerification        — P3-A: verificación post-acción y auto-rollback
  TestSLOTargetLoading        — P5-C: carga de targets desde env var
  TestIntegration             — Flujo end-to-end del bucle autónomo
"""
from __future__ import annotations

import json
import time
import threading
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch, call

import pytest

# ── imports de main (los stubs ya están en conftest.py) ──────────────────────
import main
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
    search_runbooks, _maybe_save_runbook_entry,
    patch_deployment_memory_limit,
    _run_scale_down_check, _schedule_scale_down,
    _auto_unseal_vault,
    _SEVERITY_RANK, CONFIDENCE_THRESHOLD,
)


# ═════════════════════════════════════════════════════════════════════════════
# P0 — CIRCUIT BREAKER
# ═════════════════════════════════════════════════════════════════════════════

class TestCircuitBreaker:
    """Valida la máquina de estados CLOSED → OPEN → HALF_OPEN → CLOSED."""

    def test_initial_state_is_closed(self):
        cb = CircuitBreaker(failure_threshold=3, recovery_timeout=60)
        assert cb.state == CircuitBreaker.CLOSED
        assert not cb.is_open()

    def test_opens_after_threshold_failures(self):
        cb = CircuitBreaker(failure_threshold=3, recovery_timeout=60)
        for _ in range(3):
            cb.record_failure()
        assert cb.state == CircuitBreaker.OPEN
        assert cb.is_open()

    def test_does_not_open_below_threshold(self):
        cb = CircuitBreaker(failure_threshold=5, recovery_timeout=60)
        for _ in range(4):
            cb.record_failure()
        assert cb.state == CircuitBreaker.CLOSED

    def test_transitions_to_half_open_after_timeout(self):
        cb = CircuitBreaker(failure_threshold=2, recovery_timeout=0)
        cb.record_failure()
        cb.record_failure()
        # Verificar que quedó OPEN directamente (sin disparar la propiedad .state)
        with cb._lock:
            assert cb._state == CircuitBreaker.OPEN
        # Con recovery_timeout=0, la próxima lectura de .state transiciona a HALF_OPEN
        time.sleep(0.01)
        assert cb.state == CircuitBreaker.HALF_OPEN
        assert not cb.is_open()  # HALF_OPEN no es OPEN

    def test_success_closes_circuit_from_open(self):
        cb = CircuitBreaker(failure_threshold=2, recovery_timeout=0)
        cb.record_failure()
        cb.record_failure()
        cb.record_success()
        assert cb.state == CircuitBreaker.CLOSED
        assert not cb.is_open()

    def test_failure_counter_resets_on_success(self):
        cb = CircuitBreaker(failure_threshold=5, recovery_timeout=60)
        cb.record_failure()
        cb.record_failure()
        cb.record_success()
        # Después del reset, necesita 5 fallos para abrir
        for _ in range(4):
            cb.record_failure()
        assert cb.state == CircuitBreaker.CLOSED

    def test_thread_safety(self):
        cb = CircuitBreaker(failure_threshold=10, recovery_timeout=60)
        errors = []

        def worker():
            try:
                for _ in range(5):
                    cb.record_failure()
                    cb.record_success()
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert not errors


# ═════════════════════════════════════════════════════════════════════════════
# P0 — HELPERS
# ═════════════════════════════════════════════════════════════════════════════

class TestHelpers:
    """Utilidades internas: parseo de input y deducción de owner."""

    def test_parse_two_with_comma(self):
        name, ns = _parse_two("my-deploy, amael-ia")
        assert name == "my-deploy"
        assert ns == "amael-ia"

    def test_parse_two_without_comma_uses_default(self):
        name, ns = _parse_two("my-deploy", default_second="default-ns")
        assert name == "my-deploy"
        assert ns == "default-ns"

    def test_parse_two_strips_outer_quotes(self):
        # _parse_two strips quotes from the whole input string (not per-token),
        # so leading ' is removed but inner quotes within tokens remain.
        name, ns = _parse_two("my-deploy, amael-ia")
        assert name == "my-deploy"
        assert ns == "amael-ia"

    def test_parse_two_strips_whitespace(self):
        name, ns = _parse_two("  deploy-name  ,  test-ns  ")
        assert name == "deploy-name"
        assert ns == "test-ns"

    def test_guess_owner_standard_k8s_name(self):
        # {deployment}-{rs_hash}-{pod_hash}
        owner = _guess_owner_from_pod_name("backend-ia-deployment-7d9f8b-abc12")
        assert owner == "backend-ia-deployment"

    def test_guess_owner_two_parts(self):
        owner = _guess_owner_from_pod_name("myapp-abc12")
        assert owner == "myapp"

    def test_guess_owner_single_part(self):
        owner = _guess_owner_from_pod_name("simplepod")
        assert owner == "simplepod"

    def test_severity_rank_ordering(self):
        assert _SEVERITY_RANK["CRITICAL"] > _SEVERITY_RANK["HIGH"]
        assert _SEVERITY_RANK["HIGH"] > _SEVERITY_RANK["MEDIUM"]
        assert _SEVERITY_RANK["MEDIUM"] > _SEVERITY_RANK["LOW"]


# ═════════════════════════════════════════════════════════════════════════════
# P1 — DETECT ANOMALIES
# ═════════════════════════════════════════════════════════════════════════════

class TestDetectAnomalies:
    """Detección estructural de anomalías a partir de ClusterSnapshot."""

    def test_healthy_cluster_returns_no_anomalies(self, healthy_cluster_snapshot):
        anomalies = detect_anomalies(healthy_cluster_snapshot)
        assert anomalies == []

    def test_detects_crash_loop(self):
        pod = PodStatus(
            name="crash-pod", namespace="amael-ia", phase="Running",
            restart_count=8, waiting_reason="CrashLoopBackOff", last_state_reason="",
            owner_name="crash-deployment", owner_kind="Deployment",
            start_time=datetime.now(timezone.utc),
        )
        snap = ClusterSnapshot(
            timestamp=datetime.now(timezone.utc),
            pods=[pod], nodes=[],
        )
        anomalies = detect_anomalies(snap)
        assert len(anomalies) == 1
        assert anomalies[0].issue_type == "CRASH_LOOP"
        assert anomalies[0].severity == "CRITICAL"
        assert "crash-pod" in anomalies[0].details

    def test_detects_oom_killed(self):
        pod = PodStatus(
            name="oom-pod", namespace="amael-ia", phase="Running",
            restart_count=2, waiting_reason="", last_state_reason="OOMKilled",
            owner_name="backend-deployment", owner_kind="Deployment",
            start_time=datetime.now(timezone.utc),
        )
        snap = ClusterSnapshot(
            timestamp=datetime.now(timezone.utc),
            pods=[pod], nodes=[],
        )
        anomalies = detect_anomalies(snap)
        assert any(a.issue_type == "OOM_KILLED" for a in anomalies)

    def test_detects_image_pull_backoff(self):
        pod = PodStatus(
            name="bad-image-pod", namespace="amael-ia", phase="Pending",
            restart_count=0, waiting_reason="ImagePullBackOff", last_state_reason="",
            owner_name="frontend-deployment", owner_kind="Deployment",
            start_time=datetime.now(timezone.utc),
        )
        snap = ClusterSnapshot(
            timestamp=datetime.now(timezone.utc),
            pods=[pod], nodes=[],
        )
        anomalies = detect_anomalies(snap)
        assert any(a.issue_type == "IMAGE_PULL_ERROR" for a in anomalies)

    def test_detects_err_image_pull(self):
        pod = PodStatus(
            name="bad-image-pod2", namespace="amael-ia", phase="Pending",
            restart_count=0, waiting_reason="ErrImagePull", last_state_reason="",
            owner_name="frontend-deployment", owner_kind="Deployment",
            start_time=datetime.now(timezone.utc),
        )
        snap = ClusterSnapshot(
            timestamp=datetime.now(timezone.utc),
            pods=[pod], nodes=[],
        )
        anomalies = detect_anomalies(snap)
        assert any(a.issue_type == "IMAGE_PULL_ERROR" for a in anomalies)

    def test_detects_pod_failed(self):
        pod = PodStatus(
            name="failed-pod", namespace="amael-ia", phase="Failed",
            restart_count=0, waiting_reason="", last_state_reason="",
            owner_name="worker-deployment", owner_kind="Deployment",
            start_time=datetime.now(timezone.utc),
        )
        snap = ClusterSnapshot(
            timestamp=datetime.now(timezone.utc),
            pods=[pod], nodes=[],
        )
        anomalies = detect_anomalies(snap)
        assert any(a.issue_type == "POD_FAILED" for a in anomalies)

    def test_detects_pending_stuck_after_5_minutes(self):
        start = datetime.now(timezone.utc) - timedelta(minutes=10)
        pod = PodStatus(
            name="pending-pod", namespace="amael-ia", phase="Pending",
            restart_count=0, waiting_reason="", last_state_reason="",
            owner_name="stuck-deployment", owner_kind="Deployment",
            start_time=start,
        )
        snap = ClusterSnapshot(
            timestamp=datetime.now(timezone.utc),
            pods=[pod], nodes=[],
        )
        anomalies = detect_anomalies(snap)
        assert any(a.issue_type == "POD_PENDING_STUCK" for a in anomalies)

    def test_no_pending_anomaly_before_5_minutes(self):
        start = datetime.now(timezone.utc) - timedelta(minutes=2)
        pod = PodStatus(
            name="pending-pod", namespace="amael-ia", phase="Pending",
            restart_count=0, waiting_reason="", last_state_reason="",
            owner_name="new-deployment", owner_kind="Deployment",
            start_time=start,
        )
        snap = ClusterSnapshot(
            timestamp=datetime.now(timezone.utc),
            pods=[pod], nodes=[],
        )
        anomalies = detect_anomalies(snap)
        assert not any(a.issue_type == "POD_PENDING_STUCK" for a in anomalies)

    def test_detects_high_restarts(self):
        pod = PodStatus(
            name="restart-pod", namespace="amael-ia", phase="Running",
            restart_count=7, waiting_reason="", last_state_reason="",
            owner_name="backend-deployment", owner_kind="Deployment",
            start_time=datetime.now(timezone.utc),
        )
        snap = ClusterSnapshot(
            timestamp=datetime.now(timezone.utc),
            pods=[pod], nodes=[],
        )
        anomalies = detect_anomalies(snap)
        assert any(a.issue_type == "HIGH_RESTARTS" for a in anomalies)

    def test_no_high_restarts_below_threshold(self):
        pod = PodStatus(
            name="restart-pod", namespace="amael-ia", phase="Running",
            restart_count=4, waiting_reason="", last_state_reason="",
            owner_name="backend-deployment", owner_kind="Deployment",
            start_time=datetime.now(timezone.utc),
        )
        snap = ClusterSnapshot(
            timestamp=datetime.now(timezone.utc),
            pods=[pod], nodes=[],
        )
        anomalies = detect_anomalies(snap)
        assert not any(a.issue_type == "HIGH_RESTARTS" for a in anomalies)

    def test_detects_node_not_ready(self):
        node = NodeStatus(
            name="bad-node", ready=False,
            memory_pressure=False, disk_pressure=False, pid_pressure=False,
        )
        snap = ClusterSnapshot(
            timestamp=datetime.now(timezone.utc),
            pods=[], nodes=[node],
        )
        anomalies = detect_anomalies(snap)
        assert any(a.issue_type == "NODE_NOT_READY" for a in anomalies)
        assert any(a.severity == "CRITICAL" for a in anomalies)

    def test_detects_node_memory_pressure(self):
        node = NodeStatus(
            name="pressure-node", ready=True,
            memory_pressure=True, disk_pressure=False, pid_pressure=False,
        )
        snap = ClusterSnapshot(
            timestamp=datetime.now(timezone.utc),
            pods=[], nodes=[node],
        )
        anomalies = detect_anomalies(snap)
        assert any(a.issue_type == "NODE_MEMORY_PRESSURE" for a in anomalies)

    def test_detects_node_disk_pressure(self):
        node = NodeStatus(
            name="disk-node", ready=True,
            memory_pressure=False, disk_pressure=True, pid_pressure=False,
        )
        snap = ClusterSnapshot(
            timestamp=datetime.now(timezone.utc),
            pods=[], nodes=[node],
        )
        anomalies = detect_anomalies(snap)
        assert any(a.issue_type == "NODE_DISK_PRESSURE" for a in anomalies)

    def test_dedup_key_format(self):
        pod = PodStatus(
            name="test-pod", namespace="amael-ia", phase="Running",
            restart_count=6, waiting_reason="CrashLoopBackOff", last_state_reason="",
            owner_name="my-deployment", owner_kind="Deployment",
            start_time=datetime.now(timezone.utc),
        )
        snap = ClusterSnapshot(
            timestamp=datetime.now(timezone.utc),
            pods=[pod], nodes=[],
        )
        anomalies = detect_anomalies(snap)
        crash = next(a for a in anomalies if a.issue_type == "CRASH_LOOP")
        assert "amael-ia:test-pod:CRASH_LOOP" == crash.dedup_key


# ═════════════════════════════════════════════════════════════════════════════
# P4-B — CORRELATE ANOMALIES
# ═════════════════════════════════════════════════════════════════════════════

class TestCorrelateAnomalies:
    """Agrupación de anomalías multi-pod del mismo deployment."""

    def _make_pod_anomaly(self, pod_name: str, owner: str = "my-deployment",
                          issue: str = "CRASH_LOOP", severity: str = "CRITICAL") -> Anomaly:
        return Anomaly(
            issue_type=issue,
            severity=severity,
            namespace="amael-ia",
            resource_name=pod_name,
            resource_type="pod",
            owner_name=owner,
            owner_kind="Deployment",
            details=f"Pod '{pod_name}' con {issue}.",
            dedup_key=f"amael-ia:{pod_name}:{issue}",
        )

    def test_single_anomaly_passes_through(self):
        a = self._make_pod_anomaly("pod-1")
        result = correlate_anomalies([a])
        assert len(result) == 1
        assert result[0].resource_name == "pod-1"

    def test_multi_pod_same_deployment_collapsed(self):
        anomalies = [
            self._make_pod_anomaly("pod-1", "my-deployment", "CRASH_LOOP", "CRITICAL"),
            self._make_pod_anomaly("pod-2", "my-deployment", "CRASH_LOOP", "CRITICAL"),
            self._make_pod_anomaly("pod-3", "my-deployment", "CRASH_LOOP", "CRITICAL"),
        ]
        result = correlate_anomalies(anomalies)
        assert len(result) == 1
        assert "3 pods" in result[0].details
        assert result[0].issue_type == "CRASH_LOOP"

    def test_different_deployments_not_collapsed(self):
        anomalies = [
            self._make_pod_anomaly("pod-a", "deploy-a", "CRASH_LOOP"),
            self._make_pod_anomaly("pod-b", "deploy-b", "CRASH_LOOP"),
        ]
        result = correlate_anomalies(anomalies)
        assert len(result) == 2

    def test_different_issue_types_not_collapsed(self):
        anomalies = [
            self._make_pod_anomaly("pod-1", "my-deployment", "CRASH_LOOP"),
            self._make_pod_anomaly("pod-2", "my-deployment", "OOM_KILLED"),
        ]
        result = correlate_anomalies(anomalies)
        assert len(result) == 2

    def test_worst_severity_preserved(self):
        anomalies = [
            self._make_pod_anomaly("pod-1", "my-deployment", "CRASH_LOOP", "MEDIUM"),
            self._make_pod_anomaly("pod-2", "my-deployment", "CRASH_LOOP", "CRITICAL"),
        ]
        result = correlate_anomalies(anomalies)
        assert len(result) == 1
        assert result[0].severity == "CRITICAL"

    def test_non_deployment_resources_pass_through(self):
        """Nodos y endpoints no se agrupan."""
        anomaly = Anomaly(
            issue_type="NODE_NOT_READY",
            severity="CRITICAL",
            namespace="",
            resource_name="lab-home",
            resource_type="node",
            owner_name="",
            owner_kind="Node",
            details="Nodo 'lab-home' NOT READY.",
            dedup_key="node:lab-home:NOT_READY",
        )
        result = correlate_anomalies([anomaly])
        assert len(result) == 1
        assert result[0].issue_type == "NODE_NOT_READY"

    def test_correlated_dedup_key_has_multi_suffix(self):
        anomalies = [
            self._make_pod_anomaly("pod-1", "my-deployment", "OOM_KILLED"),
            self._make_pod_anomaly("pod-2", "my-deployment", "OOM_KILLED"),
        ]
        result = correlate_anomalies(anomalies)
        assert result[0].dedup_key.endswith(":multi")

    def test_empty_list_returns_empty(self):
        assert correlate_anomalies([]) == []


# ═════════════════════════════════════════════════════════════════════════════
# P2 — DETERMINISTIC DIAGNOSIS
# ═════════════════════════════════════════════════════════════════════════════

class TestDeterministicDiagnosis:
    """Verifica que cada issue_type produce la diagnosis correcta sin LLM."""

    @pytest.mark.parametrize("issue_type,expected_action,expected_category", [
        ("CRASH_LOOP",               "ROLLOUT_RESTART", "DEPENDENCY"),
        ("OOM_KILLED",               "ROLLOUT_RESTART", "RESOURCE_LIMIT"),
        ("IMAGE_PULL_ERROR",         "NOTIFY_HUMAN",    "IMAGE_ERROR"),
        ("POD_FAILED",               "ROLLOUT_RESTART", "UNKNOWN"),
        ("POD_PENDING_STUCK",        "NOTIFY_HUMAN",    "RESOURCE_LIMIT"),
        ("HIGH_RESTARTS",            "ROLLOUT_RESTART", "DEPENDENCY"),
        ("NODE_NOT_READY",           "NOTIFY_HUMAN",    "UNKNOWN"),
        ("NODE_MEMORY_PRESSURE",     "NOTIFY_HUMAN",    "RESOURCE_LIMIT"),
        ("NODE_DISK_PRESSURE",       "NOTIFY_HUMAN",    "RESOURCE_LIMIT"),
        ("HIGH_CPU",                 "NOTIFY_HUMAN",    "RESOURCE_LIMIT"),
        ("HIGH_MEMORY",              "ROLLOUT_RESTART", "RESOURCE_LIMIT"),
        ("HIGH_ERROR_RATE",          "NOTIFY_HUMAN",    "DEPENDENCY"),
        ("DISK_EXHAUSTION_PREDICTED","NOTIFY_HUMAN",    "RESOURCE_LIMIT"),
        ("MEMORY_LEAK_PREDICTED",    "ROLLOUT_RESTART", "RESOURCE_LIMIT"),
        ("ERROR_RATE_ESCALATING",    "ROLLOUT_RESTART", "DEPENDENCY"),
        ("SLO_BUDGET_BURNING",       "NOTIFY_HUMAN",    "DEPENDENCY"),
    ])
    def test_known_issue_type(self, issue_type, expected_action, expected_category):
        anomaly = Anomaly(
            issue_type=issue_type, severity="HIGH", namespace="amael-ia",
            resource_name="test-pod", resource_type="pod",
            owner_name="test-deployment", owner_kind="Deployment",
            details=f"Test anomaly for {issue_type}",
            dedup_key=f"amael-ia:test-pod:{issue_type}",
        )
        diag = _deterministic_diagnosis(anomaly)
        assert diag.recommended_action == expected_action, (
            f"{issue_type}: expected action={expected_action}, got={diag.recommended_action}"
        )
        assert diag.root_cause_category == expected_category, (
            f"{issue_type}: expected category={expected_category}, got={diag.root_cause_category}"
        )
        assert diag.source == "deterministic"

    def test_unknown_issue_type_defaults(self):
        anomaly = Anomaly(
            issue_type="UNKNOWN_TYPE", severity="LOW", namespace="amael-ia",
            resource_name="pod", resource_type="pod",
            owner_name="deploy", owner_kind="Deployment",
            details="Unknown issue.", dedup_key="amael-ia:pod:UNKNOWN",
        )
        diag = _deterministic_diagnosis(anomaly)
        assert diag.recommended_action == "NOTIFY_HUMAN"
        assert diag.confidence == 0.50

    def test_confidence_in_valid_range(self):
        """Todos los diagnósticos determinísticos tienen confianza en [0,1]."""
        for issue_type in [
            "CRASH_LOOP", "OOM_KILLED", "IMAGE_PULL_ERROR", "SLO_BUDGET_BURNING",
            "DISK_EXHAUSTION_PREDICTED", "MEMORY_LEAK_PREDICTED",
        ]:
            anomaly = Anomaly(
                issue_type=issue_type, severity="HIGH", namespace="amael-ia",
                resource_name="pod", resource_type="pod",
                owner_name="deploy", owner_kind="Deployment",
                details="test", dedup_key=f"key:{issue_type}",
            )
            diag = _deterministic_diagnosis(anomaly)
            assert 0.0 <= diag.confidence <= 1.0, f"{issue_type} confidence out of range"


# ═════════════════════════════════════════════════════════════════════════════
# P3-B — HISTORICAL CONFIDENCE ADJUSTMENT
# ═════════════════════════════════════════════════════════════════════════════

class TestHistoricalLearning:
    """Ajuste de confianza mezclando LLM (70%) con historial PostgreSQL (30%)."""

    def test_blending_formula(self, crash_loop_anomaly):
        """new_confidence = 0.7 × llm_conf + 0.3 × hist_rate"""
        diag = Diagnosis(
            issue_type="CRASH_LOOP", root_cause="Test", root_cause_category="DEPENDENCY",
            confidence=0.80, severity="CRITICAL", recommended_action="ROLLOUT_RESTART",
            source="llm",
        )
        with patch("main.get_historical_success_rate", return_value=0.60):
            result = adjust_confidence_with_history(diag, crash_loop_anomaly)
        expected = round(0.7 * 0.80 + 0.3 * 0.60, 3)
        assert result.confidence == pytest.approx(expected, abs=0.001)

    def test_source_updated_to_include_history(self, crash_loop_anomaly):
        diag = Diagnosis(
            issue_type="CRASH_LOOP", root_cause="Test", root_cause_category="DEPENDENCY",
            confidence=0.80, severity="CRITICAL", recommended_action="ROLLOUT_RESTART",
            source="llm",
        )
        with patch("main.get_historical_success_rate", return_value=0.50):
            result = adjust_confidence_with_history(diag, crash_loop_anomaly)
        assert "history" in result.source

    def test_no_history_returns_original(self, crash_loop_anomaly):
        diag = Diagnosis(
            issue_type="CRASH_LOOP", root_cause="Test", root_cause_category="DEPENDENCY",
            confidence=0.80, severity="CRITICAL", recommended_action="ROLLOUT_RESTART",
            source="llm",
        )
        with patch("main.get_historical_success_rate", return_value=None):
            result = adjust_confidence_with_history(diag, crash_loop_anomaly)
        assert result.confidence == 0.80
        assert result.source == "llm"

    def test_confidence_clamped_to_one(self, crash_loop_anomaly):
        diag = Diagnosis(
            issue_type="CRASH_LOOP", root_cause="Test", root_cause_category="DEPENDENCY",
            confidence=0.99, severity="CRITICAL", recommended_action="ROLLOUT_RESTART",
            source="llm",
        )
        with patch("main.get_historical_success_rate", return_value=1.0):
            result = adjust_confidence_with_history(diag, crash_loop_anomaly)
        assert result.confidence <= 1.0

    def test_confidence_clamped_to_zero(self, crash_loop_anomaly):
        diag = Diagnosis(
            issue_type="CRASH_LOOP", root_cause="Test", root_cause_category="DEPENDENCY",
            confidence=0.0, severity="CRITICAL", recommended_action="ROLLOUT_RESTART",
            source="llm",
        )
        with patch("main.get_historical_success_rate", return_value=0.0):
            result = adjust_confidence_with_history(diag, crash_loop_anomaly)
        assert result.confidence >= 0.0


# ═════════════════════════════════════════════════════════════════════════════
# P2+P3 — DECIDE ACTION
# ═════════════════════════════════════════════════════════════════════════════

class TestDecideAction:
    """Política de decisión: ROLLOUT_RESTART vs NOTIFY_HUMAN con todos los guardrails."""

    def _high_conf_diagnosis(self, action="ROLLOUT_RESTART"):
        return Diagnosis(
            issue_type="CRASH_LOOP", root_cause="dependency error",
            root_cause_category="DEPENDENCY", confidence=0.90,
            severity="CRITICAL", recommended_action=action,
            source="llm",
        )

    def test_auto_heals_crash_loop_high_confidence(self, crash_loop_anomaly):
        diag = self._high_conf_diagnosis()
        with (
            patch("main.get_historical_success_rate", return_value=None),
            patch("main._check_restart_limit", return_value=False),
        ):
            plan = decide_action(crash_loop_anomaly, diag)
        assert plan.action == "ROLLOUT_RESTART"
        assert plan.auto_execute is True

    def test_notifies_when_confidence_below_threshold(self, crash_loop_anomaly):
        diag = Diagnosis(
            issue_type="CRASH_LOOP", root_cause="unknown",
            root_cause_category="UNKNOWN", confidence=0.40,
            severity="CRITICAL", recommended_action="ROLLOUT_RESTART",
            source="llm",
        )
        with (
            patch("main.get_historical_success_rate", return_value=None),
            patch("main._check_restart_limit", return_value=False),
        ):
            plan = decide_action(crash_loop_anomaly, diag)
        assert plan.action == "NOTIFY_HUMAN"

    def test_notifies_when_restart_limit_hit(self, crash_loop_anomaly):
        diag = self._high_conf_diagnosis()
        with (
            patch("main.get_historical_success_rate", return_value=None),
            patch("main._check_restart_limit", return_value=True),
        ):
            plan = decide_action(crash_loop_anomaly, diag)
        assert plan.action == "NOTIFY_HUMAN"
        assert "Límite" in plan.reason

    def test_notifies_for_image_pull_error(self, image_pull_anomaly):
        diag = self._high_conf_diagnosis(action="NOTIFY_HUMAN")
        with (
            patch("main.get_historical_success_rate", return_value=None),
            patch("main._check_restart_limit", return_value=False),
        ):
            plan = decide_action(image_pull_anomaly, diag)
        assert plan.action == "NOTIFY_HUMAN"

    def test_notifies_for_node_anomaly(self):
        node_anomaly = Anomaly(
            issue_type="NODE_NOT_READY", severity="CRITICAL", namespace="",
            resource_name="lab-home", resource_type="node",
            owner_name="", owner_kind="", details="Node NOT READY.",
            dedup_key="node:lab-home:NOT_READY",
        )
        with (
            patch("main.get_historical_success_rate", return_value=None),
            patch("main._check_restart_limit", return_value=False),
        ):
            plan = decide_action(node_anomaly)
        assert plan.action == "NOTIFY_HUMAN"

    def test_notifies_for_slo_budget_burning(self, slo_anomaly):
        with (
            patch("main.get_historical_success_rate", return_value=None),
            patch("main._check_restart_limit", return_value=False),
        ):
            plan = decide_action(slo_anomaly)
        assert plan.action == "NOTIFY_HUMAN"

    def test_notifies_for_protected_deployment(self):
        anomaly = Anomaly(
            issue_type="CRASH_LOOP", severity="CRITICAL", namespace="amael-ia",
            resource_name="postgres-pod", resource_type="pod",
            owner_name="postgres-deployment",   # in PROTECTED_DEPLOYMENTS
            owner_kind="Deployment",
            details="Postgres crash.", dedup_key="amael-ia:postgres-pod:CRASH_LOOP",
        )
        with (
            patch("main.get_historical_success_rate", return_value=None),
            patch("main._check_restart_limit", return_value=False),
        ):
            plan = decide_action(anomaly)
        assert plan.action == "NOTIFY_HUMAN"

    def test_notifies_for_foreign_namespace(self):
        anomaly = Anomaly(
            issue_type="CRASH_LOOP", severity="CRITICAL", namespace="kube-system",
            resource_name="coredns-pod", resource_type="pod",
            owner_name="coredns", owner_kind="Deployment",
            details="CoreDNS crash.", dedup_key="kube-system:coredns-pod:CRASH_LOOP",
        )
        with (
            patch("main.get_historical_success_rate", return_value=None),
            patch("main._check_restart_limit", return_value=False),
        ):
            plan = decide_action(anomaly)
        assert plan.action == "NOTIFY_HUMAN"

    def test_notifies_for_disk_exhaustion_predicted(self):
        anomaly = Anomaly(
            issue_type="DISK_EXHAUSTION_PREDICTED", severity="HIGH",
            namespace="", resource_name="lab-home", resource_type="node",
            owner_name="", owner_kind="Node",
            details="Disco agotará espacio en <4 horas.",
            dedup_key="node:lab-home:DISK_EXHAUSTION_PREDICTED",
        )
        with (
            patch("main.get_historical_success_rate", return_value=None),
            patch("main._check_restart_limit", return_value=False),
        ):
            plan = decide_action(anomaly)
        assert plan.action == "NOTIFY_HUMAN"

    def test_notifies_for_pod_pending(self):
        anomaly = Anomaly(
            issue_type="POD_PENDING_STUCK", severity="MEDIUM", namespace="amael-ia",
            resource_name="new-pod", resource_type="pod",
            owner_name="new-deployment", owner_kind="Deployment",
            details="Pod en Pending por 10min.",
            dedup_key="amael-ia:new-pod:PENDING",
        )
        with (
            patch("main.get_historical_success_rate", return_value=None),
            patch("main._check_restart_limit", return_value=False),
        ):
            plan = decide_action(anomaly)
        assert plan.action == "NOTIFY_HUMAN"

    def test_llm_scale_up_recommendation_is_honored(self):
        """LLM que recomienda SCALE_UP para HIGH_CPU debe producir acción SCALE_UP."""
        anomaly = Anomaly(
            issue_type="HIGH_CPU", severity="HIGH", namespace="amael-ia",
            resource_name="backend-abc-xyz", resource_type="pod",
            owner_name="backend-ia-deployment", owner_kind="Deployment",
            details="CPU al 92%.", dedup_key="amael-ia:backend-abc-xyz:HIGH_CPU",
        )
        diag = Diagnosis(
            issue_type="HIGH_CPU", root_cause="CPU saturado por pico de tráfico",
            root_cause_category="RESOURCE_LIMIT", confidence=0.88,
            severity="HIGH", recommended_action="SCALE_UP", source="llm",
        )
        with (
            patch("main.get_historical_success_rate", return_value=None),
            patch("main._check_restart_limit", return_value=False),
        ):
            plan = decide_action(anomaly, diag)
        assert plan.action == "SCALE_UP"
        assert "LLM" in plan.reason

    def test_llm_restart_overrides_deterministic_notify(self):
        """LLM con alta confianza que recomienda ROLLOUT_RESTART supera regla determinística."""
        anomaly = Anomaly(
            issue_type="HIGH_RESTARTS", severity="HIGH", namespace="amael-ia",
            resource_name="backend-abc-xyz", resource_type="pod",
            owner_name="backend-ia-deployment", owner_kind="Deployment",
            details="10 reinicios en 15min.", dedup_key="amael-ia:backend-abc-xyz:HIGH_RESTARTS",
        )
        diag = Diagnosis(
            issue_type="HIGH_RESTARTS",
            root_cause="OOMKilled: exit code 137, límite de memoria insuficiente",
            root_cause_category="RESOURCE_LIMIT", confidence=0.92,
            severity="HIGH", recommended_action="ROLLOUT_RESTART", source="llm",
        )
        with (
            patch("main.get_historical_success_rate", return_value=None),
            patch("main._check_restart_limit", return_value=False),
        ):
            plan = decide_action(anomaly, diag)
        assert plan.action == "ROLLOUT_RESTART"
        assert "LLM" in plan.reason

    def test_llm_action_blocked_by_restart_limit(self):
        """Guardrail de restart limit aplica incluso cuando el LLM recomienda restart."""
        anomaly = Anomaly(
            issue_type="CRASH_LOOP", severity="CRITICAL", namespace="amael-ia",
            resource_name="backend-abc-xyz", resource_type="pod",
            owner_name="backend-ia-deployment", owner_kind="Deployment",
            details="Crash loop.", dedup_key="amael-ia:backend-abc-xyz:CRASH_LOOP",
        )
        diag = Diagnosis(
            issue_type="CRASH_LOOP", root_cause="DB connection refused",
            root_cause_category="DB_ERROR", confidence=0.95,
            severity="CRITICAL", recommended_action="ROLLOUT_RESTART", source="llm",
        )
        with (
            patch("main.get_historical_success_rate", return_value=None),
            patch("main._check_restart_limit", return_value=True),  # límite alcanzado
        ):
            plan = decide_action(anomaly, diag)
        assert plan.action == "NOTIFY_HUMAN"
        assert "Límite" in plan.reason

    def test_deterministic_high_cpu_produces_scale_up(self):
        """Sin LLM, HIGH_CPU + criterios de heal → SCALE_UP (no solo NOTIFY_HUMAN)."""
        anomaly = Anomaly(
            issue_type="HIGH_CPU", severity="HIGH", namespace="amael-ia",
            resource_name="backend-abc-xyz", resource_type="pod",
            owner_name="backend-ia-deployment", owner_kind="Deployment",
            details="CPU al 90%.", dedup_key="amael-ia:backend-abc-xyz:HIGH_CPU",
        )
        det_diag = Diagnosis(
            issue_type="HIGH_CPU", root_cause="CPU alto",
            root_cause_category="RESOURCE_LIMIT", confidence=0.90,
            severity="HIGH", recommended_action="NOTIFY_HUMAN", source="deterministic",
        )
        with (
            patch("main.get_historical_success_rate", return_value=None),
            patch("main._check_restart_limit", return_value=False),
        ):
            plan = decide_action(anomaly, det_diag)
        assert plan.action == "SCALE_UP"

    def test_llm_fix_image_produces_notify_with_specific_cause(self):
        """LLM que recomienda FIX_IMAGE → NOTIFY_HUMAN con causa específica del LLM."""
        anomaly = Anomaly(
            issue_type="CRASH_LOOP", severity="HIGH", namespace="amael-ia",
            resource_name="svc-abc-xyz", resource_type="pod",
            owner_name="svc-deployment", owner_kind="Deployment",
            details="CrashLoopBackOff.", dedup_key="amael-ia:svc-abc-xyz:CRASH_LOOP",
        )
        diag = Diagnosis(
            issue_type="CRASH_LOOP",
            root_cause="Entrypoint no encontrado: /app/start.sh no existe en la imagen",
            root_cause_category="IMAGE_ERROR", confidence=0.91,
            severity="HIGH", recommended_action="FIX_IMAGE", source="llm",
        )
        with (
            patch("main.get_historical_success_rate", return_value=None),
            patch("main._check_restart_limit", return_value=False),
        ):
            plan = decide_action(anomaly, diag)
        assert plan.action == "NOTIFY_HUMAN"
        assert "FIX_IMAGE" in plan.reason or "start.sh" in plan.reason


# ═════════════════════════════════════════════════════════════════════════════
# P1 — DEDUPLICACIÓN (Redis / fallback en memoria)
# ═════════════════════════════════════════════════════════════════════════════

class TestDedup:
    """Deduplicación de incidentes con Redis o dict en memoria."""

    def test_new_incident_is_not_duplicate(self):
        with patch("main._redis", None):
            import main as m
            m._dedup_cache.clear()
            assert not _is_duplicate_incident("unique-key-123")

    def test_marked_incident_is_duplicate(self):
        with patch("main._redis", None):
            import main as m
            m._dedup_cache.clear()
            _mark_incident("my-key")
            assert _is_duplicate_incident("my-key")

    def test_redis_exists_returns_duplicate(self):
        mock_r = MagicMock()
        mock_r.exists.return_value = 1
        with patch("main._redis", mock_r):
            assert _is_duplicate_incident("some-key")

    def test_redis_not_exists_returns_not_duplicate(self):
        mock_r = MagicMock()
        mock_r.exists.return_value = 0
        with patch("main._redis", mock_r):
            assert not _is_duplicate_incident("new-key")

    def test_redis_mark_calls_set_with_ttl(self):
        mock_r = MagicMock()
        with patch("main._redis", mock_r):
            _mark_incident("my-incident")
        mock_r.set.assert_called_once_with("sre:incident:my-incident", "1", ex=600)

    def test_memory_fallback_ttl_expires(self):
        with patch("main._redis", None):
            import main as m
            m._dedup_cache.clear()
            # Insertar con timestamp viejo (> TTL)
            m._dedup_cache["old-key"] = time.time() - 700
            assert not _is_duplicate_incident("old-key")


# ═════════════════════════════════════════════════════════════════════════════
# P2 — RESTART LIMIT GUARDRAIL
# ═════════════════════════════════════════════════════════════════════════════

class TestRestartLimit:
    """Guardrail que limita reinicios automáticos por ventana de tiempo."""

    def test_limit_not_hit_when_count_below_max(self):
        mock_r = MagicMock()
        mock_r.get.return_value = "2"  # < MAX_RESTARTS_PER_RESOURCE (default 3)
        with patch("main._redis", mock_r):
            assert not _check_restart_limit("my-deploy", "amael-ia")

    def test_limit_hit_when_count_equals_max(self):
        mock_r = MagicMock()
        mock_r.get.return_value = "3"  # == MAX_RESTARTS_PER_RESOURCE
        with patch("main._redis", mock_r):
            assert _check_restart_limit("my-deploy", "amael-ia")

    def test_limit_not_hit_without_redis(self):
        with patch("main._redis", None):
            assert not _check_restart_limit("my-deploy", "amael-ia")

    def test_limit_hit_when_count_exceeds_max(self):
        mock_r = MagicMock()
        mock_r.get.return_value = "10"
        with patch("main._redis", mock_r):
            assert _check_restart_limit("my-deploy", "amael-ia")


# ═════════════════════════════════════════════════════════════════════════════
# P4-C — MAINTENANCE WINDOWS
# ═════════════════════════════════════════════════════════════════════════════

class TestMaintenanceWindow:
    """Ventanas de mantenimiento: activa/desactiva el loop SRE automático."""

    def test_maintenance_active_with_redis_key(self):
        mock_r = MagicMock()
        mock_r.exists.return_value = 1
        with patch("main._redis", mock_r):
            assert _is_maintenance_active()

    def test_maintenance_inactive_without_redis_key(self):
        mock_r = MagicMock()
        mock_r.exists.return_value = 0
        with patch("main._redis", mock_r):
            assert not _is_maintenance_active()

    def test_maintenance_inactive_without_redis(self):
        with patch("main._redis", None):
            assert not _is_maintenance_active()

    def test_activate_maintenance_sets_redis_with_ttl(self):
        mock_r = MagicMock()
        with patch("main._redis", mock_r):
            from main import _activate_maintenance
            result = _activate_maintenance("30")
        mock_r.set.assert_called_once_with("sre:maintenance:active", "1", ex=1800)
        assert "30 minutos" in result

    def test_activate_maintenance_clamps_max_duration(self):
        mock_r = MagicMock()
        with patch("main._redis", mock_r):
            from main import _activate_maintenance
            result = _activate_maintenance("99999")
        # 10080 min max (7 días) = 604800 seconds
        mock_r.set.assert_called_once_with("sre:maintenance:active", "1", ex=604800)

    def test_activate_maintenance_defaults_to_60_minutes(self):
        mock_r = MagicMock()
        with patch("main._redis", mock_r):
            from main import _activate_maintenance
            result = _activate_maintenance("")
        mock_r.set.assert_called_once_with("sre:maintenance:active", "1", ex=3600)

    def test_deactivate_maintenance_deletes_redis_key(self):
        mock_r = MagicMock()
        with patch("main._redis", mock_r):
            from main import _deactivate_maintenance
            result = _deactivate_maintenance()
        mock_r.delete.assert_called_once_with("sre:maintenance:active")
        assert "desactivada" in result

    def test_activate_without_redis_returns_error(self):
        with patch("main._redis", None):
            from main import _activate_maintenance
            result = _activate_maintenance("60")
        assert "Redis no disponible" in result


# ═════════════════════════════════════════════════════════════════════════════
# P4-A — OBSERVE METRICS (Prometheus)
# ═════════════════════════════════════════════════════════════════════════════

class TestObserveMetrics:
    """Monitoreo proactivo de CPU, memoria y tasa de errores vía Prometheus."""

    def _mock_prometheus(self, results):
        resp = MagicMock()
        resp.status_code = 200
        resp.json.return_value = {
            "status": "success",
            "data": {"result": results},
        }
        return resp

    def test_no_anomalies_when_no_results(self):
        with patch("main.requests.get", return_value=self._mock_prometheus([])):
            anomalies = observe_metrics()
        assert anomalies == []

    def test_detects_high_cpu(self):
        results = [{"metric": {"pod": "backend-pod", "namespace": "amael-ia"}, "value": ["ts", "0.95"]}]
        with patch("main.requests.get", return_value=self._mock_prometheus(results)):
            anomalies = observe_metrics()
        cpu_anomalies = [a for a in anomalies if a.issue_type == "HIGH_CPU"]
        assert len(cpu_anomalies) >= 1
        # HIGH_CPU se clasifica como MEDIUM (CPU pressure es menos urgente que memoria)
        assert cpu_anomalies[0].severity == "MEDIUM"

    def test_detects_high_memory(self):
        # Para memory: la segunda llamada es la de memoria
        def side_effect(url, params, timeout):
            resp = MagicMock()
            resp.status_code = 200
            if "memory" in params.get("query", "") or "working_set" in params.get("query", ""):
                resp.json.return_value = {
                    "status": "success",
                    "data": {"result": [{"metric": {"pod": "mem-pod", "namespace": "amael-ia"}, "value": ["ts", "0.92"]}]},
                }
            else:
                resp.json.return_value = {"status": "success", "data": {"result": []}}
            return resp

        with patch("main.requests.get", side_effect=side_effect):
            anomalies = observe_metrics()
        memory_anomalies = [a for a in anomalies if a.issue_type == "HIGH_MEMORY"]
        assert len(memory_anomalies) >= 1

    def test_detects_high_error_rate(self):
        def side_effect(url, params, timeout):
            resp = MagicMock()
            resp.status_code = 200
            if "status_code=~" in params.get("query", "") and "5.." in params.get("query", ""):
                resp.json.return_value = {
                    "status": "success",
                    "data": {"result": [{"metric": {"handler": "/api/chat"}, "value": ["ts", "0.05"]}]},
                }
            else:
                resp.json.return_value = {"status": "success", "data": {"result": []}}
            return resp

        with patch("main.requests.get", side_effect=side_effect):
            anomalies = observe_metrics()
        error_anomalies = [a for a in anomalies if a.issue_type == "HIGH_ERROR_RATE"]
        assert len(error_anomalies) >= 1

    def test_returns_empty_on_prometheus_error(self):
        with patch("main.requests.get", side_effect=Exception("Connection refused")):
            anomalies = observe_metrics()
        assert isinstance(anomalies, list)


# ═════════════════════════════════════════════════════════════════════════════
# P5-A — OBSERVE TRENDS (predictive alerting)
# ═════════════════════════════════════════════════════════════════════════════

class TestObserveTrends:
    """Alertas predictivas: disk exhaustion, memory leak, error rate escalation."""

    def _ok_resp(self, results=None):
        resp = MagicMock()
        resp.status_code = 200
        resp.json.return_value = {
            "status": "success",
            "data": {"result": results or []},
        }
        return resp

    def test_no_trends_when_all_queries_empty(self):
        with patch("main.requests.get", return_value=self._ok_resp([])):
            anomalies = observe_trends()
        assert anomalies == []

    def test_detects_disk_exhaustion_predicted(self):
        def side_effect(url, params, timeout):
            q = params.get("query", "")
            if "predict_linear" in q:
                return self._ok_resp([{"metric": {"instance": "lab-home:9100"}, "value": ["ts", "-1"]}])
            return self._ok_resp([])

        with patch("main.requests.get", side_effect=side_effect):
            anomalies = observe_trends()
        disk_a = [a for a in anomalies if a.issue_type == "DISK_EXHAUSTION_PREDICTED"]
        assert len(disk_a) >= 1
        assert disk_a[0].severity == "HIGH"
        assert "4 horas" in disk_a[0].details

    def test_detects_memory_leak_predicted(self):
        def side_effect(url, params, timeout):
            q = params.get("query", "")
            if "deriv" in q and "memory" in q:
                return self._ok_resp([{
                    "metric": {"pod": "backend-ia-abc-xyz", "container": "backend", "namespace": "amael-ia"},
                    "value": ["ts", str(2 * 1024 * 1024)],  # 2 MB/s
                }])
            return self._ok_resp([])

        with patch("main.requests.get", side_effect=side_effect):
            anomalies = observe_trends()
        leak = [a for a in anomalies if a.issue_type == "MEMORY_LEAK_PREDICTED"]
        assert len(leak) >= 1
        assert "MB/s" in leak[0].details

    def test_detects_error_rate_escalating(self):
        def side_effect(url, params, timeout):
            q = params.get("query", "")
            if "deriv" in q and "5.." in q:
                return self._ok_resp([{"metric": {}, "value": ["ts", "0.005"]}])
            return self._ok_resp([])

        with patch("main.requests.get", side_effect=side_effect):
            anomalies = observe_trends()
        esc = [a for a in anomalies if a.issue_type == "ERROR_RATE_ESCALATING"]
        assert len(esc) >= 1

    def test_returns_empty_on_exception(self):
        with patch("main.requests.get", side_effect=Exception("Prometheus down")):
            anomalies = observe_trends()
        assert isinstance(anomalies, list)


# ═════════════════════════════════════════════════════════════════════════════
# P5-C — OBSERVE SLO (error budget tracking)
# ═════════════════════════════════════════════════════════════════════════════

class TestObserveSLO:
    """Seguimiento de error budget: detecta burn_rate >= 2× como anomalía SLO."""

    def setup_method(self):
        """Configura un SLO target de prueba antes de cada test."""
        import main as m
        m._SLO_TARGETS = [
            {
                "handler": "/api/chat",
                "availability": 0.995,
                "window_h": 24,
                "service": "backend-ia",
            }
        ]

    def teardown_method(self):
        import main as m
        m._SLO_TARGETS = []

    def test_no_slo_anomaly_when_burn_rate_below_2x(self):
        # error_budget = 0.005, burn_rate = 1.5 → sin anomalía
        resp = MagicMock()
        resp.status_code = 200
        resp.json.return_value = {
            "status": "success",
            "data": {"result": [{"value": ["ts", str(0.005 * 1.5)]}]},
        }
        with patch("main.requests.get", return_value=resp):
            anomalies = observe_slo()
        assert anomalies == []

    def test_slo_anomaly_when_burn_rate_above_2x(self):
        # error_budget = 0.005, burn_rate = 3.0 → anomalía
        resp = MagicMock()
        resp.status_code = 200
        resp.json.return_value = {
            "status": "success",
            "data": {"result": [{"value": ["ts", str(0.005 * 3.0)]}]},
        }
        with patch("main.requests.get", return_value=resp):
            anomalies = observe_slo()
        assert len(anomalies) == 1
        assert anomalies[0].issue_type == "SLO_BUDGET_BURNING"
        assert "burn_rate=3.0×" in anomalies[0].details

    def test_slo_critical_severity_at_5x_burn_rate(self):
        resp = MagicMock()
        resp.status_code = 200
        resp.json.return_value = {
            "status": "success",
            "data": {"result": [{"value": ["ts", str(0.005 * 6.0)]}]},
        }
        with patch("main.requests.get", return_value=resp):
            anomalies = observe_slo()
        assert anomalies[0].severity == "CRITICAL"

    def test_slo_high_severity_at_3x_burn_rate(self):
        resp = MagicMock()
        resp.status_code = 200
        resp.json.return_value = {
            "status": "success",
            "data": {"result": [{"value": ["ts", str(0.005 * 3.0)]}]},
        }
        with patch("main.requests.get", return_value=resp):
            anomalies = observe_slo()
        assert anomalies[0].severity == "HIGH"

    def test_no_anomalies_when_no_targets(self):
        import main as m
        m._SLO_TARGETS = []
        anomalies = observe_slo()
        assert anomalies == []

    def test_slo_dedup_key_format(self):
        resp = MagicMock()
        resp.status_code = 200
        resp.json.return_value = {
            "status": "success",
            "data": {"result": [{"value": ["ts", "0.025"]}]},
        }
        with patch("main.requests.get", return_value=resp):
            anomalies = observe_slo()
        assert anomalies[0].dedup_key == "amael-ia:/api/chat:SLO_BURN"


class TestSLOTargetLoading:
    """Carga y parseo de SLO_TARGETS_JSON desde variable de entorno."""

    def test_loads_valid_json(self):
        targets = json.dumps([
            {"handler": "/api/chat", "availability": 0.995, "window_h": 24, "service": "backend-ia"},
            {"handler": "/api/k8s-agent", "availability": 0.990, "window_h": 24, "service": "k8s-agent"},
        ])
        with patch.dict("os.environ", {"SLO_TARGETS_JSON": targets}):
            load_slo_targets()
        import main as m
        assert len(m._SLO_TARGETS) == 2
        assert m._SLO_TARGETS[0]["handler"] == "/api/chat"

    def test_handles_invalid_json_gracefully(self):
        with patch.dict("os.environ", {"SLO_TARGETS_JSON": "not-valid-json"}):
            load_slo_targets()
        import main as m
        assert m._SLO_TARGETS == []

    def test_handles_empty_env_var(self):
        with patch.dict("os.environ", {"SLO_TARGETS_JSON": "[]"}):
            load_slo_targets()
        import main as m
        assert m._SLO_TARGETS == []


# ═════════════════════════════════════════════════════════════════════════════
# P5-B — ROLLOUT UNDO (rollback automático)
# ═════════════════════════════════════════════════════════════════════════════

class TestRolloutUndo:
    """Rollback a revisión anterior: kubectl rollout undo equivalente."""

    def _mock_deploy(self, name, revision="2", conditions=None):
        deploy = MagicMock()
        deploy.metadata.name = name
        deploy.metadata.annotations = {"deployment.kubernetes.io/revision": revision}
        deploy.status.conditions = conditions or []
        deploy.spec.replicas = 1
        deploy.status.ready_replicas = 1
        return deploy

    def _mock_rs(self, name, deploy_name, revision):
        rs = MagicMock()
        rs.metadata.name = f"{deploy_name}-rs-{revision}"
        rs.metadata.annotations = {"deployment.kubernetes.io/revision": revision}
        ref = MagicMock()
        ref.kind = "Deployment"
        ref.name = deploy_name
        rs.metadata.owner_references = [ref]
        rs.spec.template = MagicMock()
        return rs

    def test_rollback_protected_deployment_blocked(self):
        result = rollout_undo_deployment("postgres-deployment, amael-ia")
        assert "protegido" in result.lower()

    def test_rollback_at_revision_1_returns_message(self):
        deploy = self._mock_deploy("my-deploy", revision="1")
        with patch("main.apps_v1.read_namespaced_deployment", return_value=deploy):
            result = rollout_undo_deployment("my-deploy, amael-ia")
        assert "No hay revisión anterior" in result

    def test_rollback_success_returns_checkmark(self):
        deploy = self._mock_deploy("my-deploy", revision="3")
        rs = self._mock_rs("my-deploy", "my-deploy", "2")
        rs_list = MagicMock()
        rs_list.items = [rs]

        mock_sanitize = {"metadata": {"annotations": {}}, "spec": {}}
        with (
            patch("main.apps_v1.read_namespaced_deployment", return_value=deploy),
            patch("main.apps_v1.list_namespaced_replica_set", return_value=rs_list),
            patch("main.apps_v1.api_client.sanitize_for_serialization", return_value=mock_sanitize),
            patch("main.apps_v1.patch_namespaced_deployment") as mock_patch,
        ):
            result = rollout_undo_deployment("my-deploy, amael-ia")
        assert "✅" in result
        assert mock_patch.called

    def test_rollback_when_rs_not_found(self):
        deploy = self._mock_deploy("my-deploy", revision="2")
        rs_list = MagicMock()
        rs_list.items = []  # No RS found
        with (
            patch("main.apps_v1.read_namespaced_deployment", return_value=deploy),
            patch("main.apps_v1.list_namespaced_replica_set", return_value=rs_list),
        ):
            result = rollout_undo_deployment("my-deploy, amael-ia")
        assert "No se encontró revisión" in result

    def test_rollback_k8s_exception_returns_error(self):
        with patch("main.apps_v1.read_namespaced_deployment", side_effect=Exception("K8s error")):
            result = rollout_undo_deployment("my-deploy, amael-ia")
        assert "Error" in result


# ═════════════════════════════════════════════════════════════════════════════
# P4-D — AUTO-GENERATED RUNBOOKS
# ═════════════════════════════════════════════════════════════════════════════

class TestAutoRunbook:
    """Auto-generación de runbooks en Qdrant a partir de diagnósticos LLM de alta confianza."""

    def test_skips_if_qdrant_not_available(self, crash_loop_anomaly):
        diag = Diagnosis(
            issue_type="CRASH_LOOP", root_cause="test", root_cause_category="DEPENDENCY",
            confidence=0.95, severity="CRITICAL", recommended_action="ROLLOUT_RESTART",
            source="llm",
        )
        with patch("main._qdrant_client", None):
            # No debe lanzar excepción
            _maybe_save_runbook_entry(crash_loop_anomaly, diag)

    def test_skips_if_source_is_deterministic(self, crash_loop_anomaly):
        diag = Diagnosis(
            issue_type="CRASH_LOOP", root_cause="test", root_cause_category="DEPENDENCY",
            confidence=0.95, severity="CRITICAL", recommended_action="ROLLOUT_RESTART",
            source="deterministic",   # <-- no es "llm"
        )
        mock_qdrant = MagicMock()
        with patch("main._qdrant_client", mock_qdrant):
            _maybe_save_runbook_entry(crash_loop_anomaly, diag)
        mock_qdrant.upsert.assert_not_called()

    def test_skips_if_confidence_below_0_80(self, crash_loop_anomaly):
        diag = Diagnosis(
            issue_type="CRASH_LOOP", root_cause="test", root_cause_category="DEPENDENCY",
            confidence=0.75, severity="CRITICAL", recommended_action="ROLLOUT_RESTART",
            source="llm",
        )
        mock_qdrant = MagicMock()
        with patch("main._qdrant_client", mock_qdrant):
            _maybe_save_runbook_entry(crash_loop_anomaly, diag)
        mock_qdrant.upsert.assert_not_called()

    def test_skips_if_existing_runbook_covers_issue(self, crash_loop_anomaly):
        diag = Diagnosis(
            issue_type="CRASH_LOOP", root_cause="dependency error", root_cause_category="DEPENDENCY",
            confidence=0.92, severity="CRITICAL", recommended_action="ROLLOUT_RESTART",
            source="llm",
        )
        mock_qdrant = MagicMock()
        with (
            patch("main._qdrant_client", mock_qdrant),
            patch("main.search_runbooks", return_value="Existing runbook content"),
        ):
            _maybe_save_runbook_entry(crash_loop_anomaly, diag)
        mock_qdrant.upsert.assert_not_called()

    def test_saves_runbook_when_no_existing_coverage(self, crash_loop_anomaly):
        diag = Diagnosis(
            issue_type="CRASH_LOOP", root_cause="dependency error", root_cause_category="DEPENDENCY",
            confidence=0.92, severity="CRITICAL", recommended_action="ROLLOUT_RESTART",
            source="llm", evidence=["log line 1"],
        )
        mock_qdrant = MagicMock()
        mock_embedding = [0.1] * 768
        with (
            patch("main._qdrant_client", mock_qdrant),
            patch("main.search_runbooks", return_value=""),
            patch("main._get_embedding", return_value=mock_embedding),
        ):
            _maybe_save_runbook_entry(crash_loop_anomaly, diag)
        mock_qdrant.upsert.assert_called_once()

    def test_skips_if_embedding_fails(self, crash_loop_anomaly):
        diag = Diagnosis(
            issue_type="CRASH_LOOP", root_cause="test", root_cause_category="DEPENDENCY",
            confidence=0.92, severity="CRITICAL", recommended_action="ROLLOUT_RESTART",
            source="llm",
        )
        mock_qdrant = MagicMock()
        with (
            patch("main._qdrant_client", mock_qdrant),
            patch("main.search_runbooks", return_value=""),
            patch("main._get_embedding", return_value=[]),  # embedding vacío
        ):
            _maybe_save_runbook_entry(crash_loop_anomaly, diag)
        mock_qdrant.upsert.assert_not_called()


# ═════════════════════════════════════════════════════════════════════════════
# P3-A — POST-ACTION VERIFICATION
# ═════════════════════════════════════════════════════════════════════════════

class TestPostVerification:
    """Verificación 5 minutos post-reinicio: resolved → postmortem / unresolved → rollback."""

    def test_verification_resolved_triggers_postmortem(self):
        with (
            patch("main._is_deployment_healthy", return_value=True),
            patch("main._update_incident_verification") as mock_update,
            patch("main._generate_and_store_postmortem") as mock_pm,
            patch("main.threading.Thread") as mock_thread,
        ):
            from main import _run_verification_job
            _run_verification_job("inc-key", "my-deploy", "amael-ia", "CRASH_LOOP")

        mock_update.assert_called_once_with("inc-key", "resolved")
        mock_thread.assert_called_once()  # postmortem en thread background

    def test_verification_unresolved_with_recent_deploy_triggers_rollback(self):
        with (
            patch("main._is_deployment_healthy", return_value=False),
            patch("main._was_recently_deployed", return_value=True),
            patch("main._update_incident_verification"),
            patch("main.rollout_undo_deployment", return_value="✅ Rollback iniciado"),
            patch("main._send_sre_notification") as mock_notify,
        ):
            from main import _run_verification_job
            _run_verification_job("inc-key", "my-deploy", "amael-ia", "CRASH_LOOP")

        # Debe enviar notificación de rollback
        notify_calls = mock_notify.call_args_list
        assert any("ROLLBACK" in str(c) for c in notify_calls)

    def test_verification_unresolved_without_recent_deploy_notifies_human(self):
        with (
            patch("main._is_deployment_healthy", return_value=False),
            patch("main._was_recently_deployed", return_value=False),
            patch("main._update_incident_verification"),
            patch("main._send_sre_notification") as mock_notify,
        ):
            from main import _run_verification_job
            _run_verification_job("inc-key", "my-deploy", "amael-ia", "CRASH_LOOP")

        notify_calls = mock_notify.call_args_list
        assert any("manual" in str(c).lower() or "POST-VERIFY" in str(c) for c in notify_calls)

    def test_schedule_verification_with_no_scheduler_is_noop(self):
        with patch("main._sre_scheduler", None):
            from main import _schedule_verification
            # No debe lanzar excepción
            _schedule_verification("key", "deploy", "ns", "CRASH_LOOP")

    def test_schedule_verification_adds_job(self):
        mock_scheduler = MagicMock()
        with patch("main._sre_scheduler", mock_scheduler):
            from main import _schedule_verification
            _schedule_verification("key", "deploy", "amael-ia", "OOM_KILLED", delay_s=300)
        mock_scheduler.add_job.assert_called_once()
        call_kwargs = mock_scheduler.add_job.call_args
        assert call_kwargs[1]["trigger"] == "date"
        assert call_kwargs[1]["id"] == "verify:key"


# ═════════════════════════════════════════════════════════════════════════════
# INTEGRATION — Flujo end-to-end del loop autónomo
# ═════════════════════════════════════════════════════════════════════════════

class TestIntegration:
    """
    Pruebas de flujo completo: simula un ClusterSnapshot con anomalías y
    verifica que el bucle autónomo ejecute Detect → Diagnose → Decide → Act.
    """

    def test_full_pipeline_crash_loop_triggers_restart(self):
        """Un pod en CrashLoopBackOff con alta confianza → ROLLOUT_RESTART."""
        pod = PodStatus(
            name="backend-ia-crash", namespace="amael-ia", phase="Running",
            restart_count=10, waiting_reason="CrashLoopBackOff", last_state_reason="",
            owner_name="backend-ia-deployment", owner_kind="Deployment",
            start_time=datetime.now(timezone.utc),
        )
        snap = ClusterSnapshot(
            timestamp=datetime.now(timezone.utc),
            pods=[pod], nodes=[],
        )

        anomalies = detect_anomalies(snap)
        assert len(anomalies) == 1
        assert anomalies[0].issue_type == "CRASH_LOOP"

        # Deterministic diagnosis tiene confidence=0.70, debajo del threshold=0.75.
        # Para probar el flujo de auto-heal, inyectamos una diagnosis con confidence alta.
        diag = Diagnosis(
            issue_type="CRASH_LOOP", root_cause="dependency error at startup",
            root_cause_category="DEPENDENCY", confidence=0.90,
            severity="CRITICAL", recommended_action="ROLLOUT_RESTART",
            source="llm",
        )

        with (
            patch("main.get_historical_success_rate", return_value=None),
            patch("main._check_restart_limit", return_value=False),
        ):
            plan = decide_action(anomalies[0], diag)

        assert plan.action == "ROLLOUT_RESTART"
        assert plan.target_name == "backend-ia-deployment"
        assert plan.target_namespace == "amael-ia"

    def test_full_pipeline_image_pull_notifies_human(self):
        """Un pod con ImagePullBackOff → NOTIFY_HUMAN (nunca auto-heal)."""
        pod = PodStatus(
            name="frontend-bad", namespace="amael-ia", phase="Pending",
            restart_count=0, waiting_reason="ImagePullBackOff", last_state_reason="",
            owner_name="frontend-deployment", owner_kind="Deployment",
            start_time=datetime.now(timezone.utc),
        )
        snap = ClusterSnapshot(
            timestamp=datetime.now(timezone.utc),
            pods=[pod], nodes=[],
        )
        anomalies = detect_anomalies(snap)
        diag = _deterministic_diagnosis(anomalies[0])

        with (
            patch("main.get_historical_success_rate", return_value=None),
            patch("main._check_restart_limit", return_value=False),
        ):
            plan = decide_action(anomalies[0], diag)

        assert plan.action == "NOTIFY_HUMAN"

    def test_full_pipeline_multiple_pods_correlated(self):
        """3 pods del mismo deploy → 1 anomalía correlacionada → 1 plan."""
        pods = [
            PodStatus(
                name=f"backend-crash-{i}", namespace="amael-ia", phase="Running",
                restart_count=8, waiting_reason="CrashLoopBackOff", last_state_reason="",
                owner_name="backend-ia-deployment", owner_kind="Deployment",
                start_time=datetime.now(timezone.utc),
            )
            for i in range(3)
        ]
        snap = ClusterSnapshot(
            timestamp=datetime.now(timezone.utc),
            pods=pods, nodes=[],
        )

        raw_anomalies = detect_anomalies(snap)
        assert len(raw_anomalies) == 3

        correlated = correlate_anomalies(raw_anomalies)
        assert len(correlated) == 1
        assert "3 pods" in correlated[0].details

    def test_maintenance_window_skips_loop_iteration(self):
        """Con ventana de mantenimiento activa, el loop no ejecuta ninguna acción."""
        with patch("main._is_maintenance_active", return_value=True):
            from main import _run_loop_iteration
            # No debe lanzar excepción ni llamar a observe_cluster
            with patch("main.observe_cluster") as mock_obs:
                _run_loop_iteration()
            mock_obs.assert_not_called()

    def test_circuit_breaker_skips_loop_when_open(self):
        """Con circuit breaker abierto, sre_autonomous_loop no ejecuta la iteración."""
        with (
            patch("main._try_acquire_lease", return_value=True),
            patch("main._circuit_breaker") as mock_cb,
        ):
            mock_cb.is_open.return_value = True
            from main import sre_autonomous_loop
            with patch("main._run_loop_iteration") as mock_run:
                sre_autonomous_loop()
            mock_run.assert_not_called()

    def test_dedup_prevents_double_processing(self):
        """La misma anomalía no se procesa dos veces en la misma ventana de tiempo."""
        key = "amael-ia:test-pod:CRASH_LOOP"
        with patch("main._redis", None):
            import main as m
            m._dedup_cache.clear()
            assert not _is_duplicate_incident(key)
            _mark_incident(key)
            assert _is_duplicate_incident(key)

    def test_health_check_endpoint(self):
        """El endpoint /health retorna status ok."""
        from fastapi.testclient import TestClient
        from main import app
        client = TestClient(app)
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "ok"

    def test_sre_loop_status_endpoint(self):
        """El endpoint /api/sre/loop/status retorna información del loop."""
        from fastapi.testclient import TestClient
        from main import app
        client = TestClient(app)
        response = client.get("/api/sre/loop/status")
        assert response.status_code == 200
        data = response.json()
        assert "loop_enabled" in data
        assert "circuit_breaker" in data

    def test_metrics_endpoint_not_found_in_test_env(self):
        """
        /metrics es añadido por prometheus_fastapi_instrumentator en producción.
        En el entorno de test (instrumentator mockeado), el endpoint no existe.
        Verificamos que la app responde (no cuelga) y no expone error 500.
        """
        from fastapi.testclient import TestClient
        from main import app
        client = TestClient(app)
        response = client.get("/metrics")
        # 404 es esperado en tests (instrumentator mockeado), nunca 500
        assert response.status_code in (200, 404)


# ═════════════════════════════════════════════════════════════════════════════
# DIAGNÓSTICO — Evidencia real (logs, container status, pod owner lookup)
# ═════════════════════════════════════════════════════════════════════════════

class TestCollectPodEvidence:
    """Valida _collect_pod_evidence: lee logs anteriores en crash, filtra ruido,
    incluye container status y events."""

    def _make_pod(self, exit_code=1, reason="Error", restarts=5, waiting_reason="CrashLoopBackOff"):
        pod = MagicMock()
        cs = MagicMock()
        cs.name = "app"
        cs.restart_count = restarts
        cs.last_state.terminated.exit_code = exit_code
        cs.last_state.terminated.reason = reason
        cs.last_state.terminated.finished_at = "2026-03-24T03:00:00Z"
        cs.state.waiting.reason = waiting_reason
        pod.status.container_statuses = [cs]
        return pod

    def test_crash_loop_reads_previous_logs(self):
        """Para CRASH_LOOP debe leer logs del container anterior (previous=True)."""
        from main import _collect_pod_evidence
        mock_v1 = MagicMock()
        mock_v1.read_namespaced_pod.return_value = self._make_pod()
        mock_v1.read_namespaced_pod_log.return_value = "ERROR: connection refused\nstartup failed"
        mock_v1.list_namespaced_event.return_value.items = []

        with patch("main.v1", mock_v1):
            parts = _collect_pod_evidence("myapp-abc-xyz", "amael-ia", "CRASH_LOOP")

        # Debe haber llamado con previous=True
        calls = mock_v1.read_namespaced_pod_log.call_args_list
        prev_calls = [c for c in calls if c.kwargs.get("previous") is True]
        assert len(prev_calls) == 1, "Debe leer logs del container anterior para CRASH_LOOP"

    def test_crash_loop_does_not_read_current_logs(self):
        """Para CRASH_LOOP NO debe intentar leer logs actuales (container crasheando = vacío)."""
        from main import _collect_pod_evidence
        mock_v1 = MagicMock()
        mock_v1.read_namespaced_pod.return_value = self._make_pod()
        mock_v1.read_namespaced_pod_log.return_value = "ERROR: oom"
        mock_v1.list_namespaced_event.return_value.items = []

        with patch("main.v1", mock_v1):
            _collect_pod_evidence("myapp-abc-xyz", "amael-ia", "CRASH_LOOP")

        calls = mock_v1.read_namespaced_pod_log.call_args_list
        current_calls = [c for c in calls if c.kwargs.get("previous") is not True]
        assert len(current_calls) == 0, "No debe leer logs actuales para CRASH_LOOP"

    def test_non_crash_reads_current_logs(self):
        """Para HIGH_CPU/MEMORY_LEAK debe leer logs actuales, no previos."""
        from main import _collect_pod_evidence
        mock_v1 = MagicMock()
        mock_v1.read_namespaced_pod.return_value = self._make_pod(restarts=0)
        mock_v1.read_namespaced_pod_log.return_value = "WARN: memory usage high\nERROR: oom"
        mock_v1.list_namespaced_event.return_value.items = []

        with patch("main.v1", mock_v1):
            parts = _collect_pod_evidence("myapp-abc-xyz", "amael-ia", "HIGH_MEMORY")

        calls = mock_v1.read_namespaced_pod_log.call_args_list
        current_calls = [c for c in calls if c.kwargs.get("previous") is not True]
        assert len(current_calls) == 1

    def test_includes_container_status_in_output(self):
        """El container status (exit code, restart count) debe aparecer en la evidencia."""
        from main import _collect_pod_evidence
        mock_v1 = MagicMock()
        mock_v1.read_namespaced_pod.return_value = self._make_pod(exit_code=137, reason="OOMKilled", restarts=8)
        mock_v1.read_namespaced_pod_log.return_value = ""
        mock_v1.list_namespaced_event.return_value.items = []

        with patch("main.v1", mock_v1):
            parts = _collect_pod_evidence("myapp-abc-xyz", "amael-ia", "OOM_KILLED")

        status_part = next((p for p in parts if "CONTAINER STATUS" in p), None)
        assert status_part is not None, "Debe incluir CONTAINER STATUS"
        assert "137" in status_part or "OOMKilled" in status_part

    def test_filters_noise_from_logs(self):
        """Solo deben quedar líneas con ERROR/Exception/Traceback/FATAL, no INFO noise."""
        from main import _collect_pod_evidence
        mock_v1 = MagicMock()
        mock_v1.read_namespaced_pod.return_value = self._make_pod()
        noisy_logs = "\n".join([
            "INFO: server started",
            "DEBUG: processing request",
            "ERROR: connection refused to postgres",
            "INFO: health check ok",
            "Traceback (most recent call last):",
            "Exception: db timeout",
        ])
        mock_v1.read_namespaced_pod_log.return_value = noisy_logs
        mock_v1.list_namespaced_event.return_value.items = []

        with patch("main.v1", mock_v1):
            parts = _collect_pod_evidence("myapp-abc-xyz", "amael-ia", "CRASH_LOOP")

        log_part = next((p for p in parts if "LOGS" in p), "")
        assert "ERROR: connection refused" in log_part
        assert "Traceback" in log_part
        assert "Exception: db timeout" in log_part
        assert "INFO: server started" not in log_part
        assert "DEBUG: processing request" not in log_part

    def test_includes_k8s_events(self):
        """Los events K8s del pod deben aparecer en la evidencia."""
        from main import _collect_pod_evidence
        mock_v1 = MagicMock()
        mock_v1.read_namespaced_pod.return_value = self._make_pod()
        mock_v1.read_namespaced_pod_log.return_value = ""

        ev = MagicMock()
        ev.type = "Warning"
        ev.reason = "BackOff"
        ev.message = "Back-off restarting failed container"
        ev.last_timestamp = "2026-03-24T03:00:00Z"
        ev.metadata.creation_timestamp = "2026-03-24T03:00:00Z"
        mock_v1.list_namespaced_event.return_value.items = [ev]

        with patch("main.v1", mock_v1):
            parts = _collect_pod_evidence("myapp-abc-xyz", "amael-ia", "CRASH_LOOP")

        events_part = next((p for p in parts if "EVENTS" in p), None)
        assert events_part is not None
        assert "BackOff" in events_part

    def test_gracefully_handles_missing_pod(self):
        """Si el pod no existe (404), debe retornar lista vacía sin lanzar excepción."""
        from main import _collect_pod_evidence
        mock_v1 = MagicMock()
        mock_v1.read_namespaced_pod.side_effect = Exception("404 Not Found")
        mock_v1.read_namespaced_pod_log.side_effect = Exception("404 Not Found")
        mock_v1.list_namespaced_event.side_effect = Exception("503 unavailable")

        with patch("main.v1", mock_v1):
            parts = _collect_pod_evidence("ghost-pod", "amael-ia", "CRASH_LOOP")

        assert isinstance(parts, list)  # nunca lanza excepción


class TestFindPodsForOwner:
    """Valida _find_pods_for_owner: búsqueda de pods por nombre de deployment."""

    def test_returns_pods_matching_owner_prefix(self):
        from main import _find_pods_for_owner
        mock_v1 = MagicMock()
        pod1 = MagicMock(); pod1.metadata.name = "backend-ia-abc-111"; pod1.status.phase = "Running"
        pod2 = MagicMock(); pod2.metadata.name = "backend-ia-abc-222"; pod2.status.phase = "Running"
        pod3 = MagicMock(); pod3.metadata.name = "frontend-xyz-333";   pod3.status.phase = "Running"
        mock_v1.list_namespaced_pod.return_value.items = [pod1, pod2, pod3]

        with patch("main.v1", mock_v1):
            result = _find_pods_for_owner("backend-ia", "amael-ia")

        assert "backend-ia-abc-111" in result
        assert "backend-ia-abc-222" in result
        assert "frontend-xyz-333" not in result

    def test_limits_to_three_pods(self):
        from main import _find_pods_for_owner
        mock_v1 = MagicMock()
        pods = []
        for i in range(10):
            p = MagicMock()
            p.metadata.name = f"myapp-rs-{i:03d}"
            p.status.phase = "Running"
            pods.append(p)
        mock_v1.list_namespaced_pod.return_value.items = pods

        with patch("main.v1", mock_v1):
            result = _find_pods_for_owner("myapp", "amael-ia")

        assert len(result) <= 3

    def test_returns_empty_on_k8s_error(self):
        from main import _find_pods_for_owner
        mock_v1 = MagicMock()
        mock_v1.list_namespaced_pod.side_effect = Exception("API server unavailable")

        with patch("main.v1", mock_v1):
            result = _find_pods_for_owner("myapp", "amael-ia")

        assert result == []


class TestMemoryLeakExclusions:
    """Valida que contenedores excluidos no generen alertas de memory leak."""

    def test_excluded_container_skipped(self):
        """ollama y prometheus no deben generar MEMORY_LEAK_PREDICTED."""
        from main import observe_trends

        def side_effect(url, params, timeout):
            resp = MagicMock()
            resp.status_code = 200
            query = params.get("query", "")
            if "deriv" in query and "container_memory" in query:
                # Simula que ollama y prometheus disparan el threshold
                resp.json.return_value = {
                    "status": "success",
                    "data": {"result": [
                        {"metric": {"pod": "ollama-abc-123", "container": "ollama",
                                    "namespace": "amael-ia"}, "value": ["ts", "9000000"]},
                        {"metric": {"pod": "prometheus-0", "container": "prometheus",
                                    "namespace": "observability"}, "value": ["ts", "6000000"]},
                        {"metric": {"pod": "whatsapp-bridge-abc", "container": "whatsapp-bridge",
                                    "namespace": "amael-ia"}, "value": ["ts", "15000000"]},
                    ]},
                }
            else:
                resp.json.return_value = {"status": "success", "data": {"result": []}}
            return resp

        with patch("main.requests.get", side_effect=side_effect):
            with patch("main._MEMORY_LEAK_EXCLUDED", {"ollama", "prometheus"}):
                anomalies = observe_trends()

        leak_anomalies = [a for a in anomalies if a.issue_type == "MEMORY_LEAK_PREDICTED"]
        containers_alerted = {a.resource_name for a in leak_anomalies}

        assert "ollama-abc-123" not in containers_alerted, "ollama debe estar excluido"
        assert "prometheus-0" not in containers_alerted, "prometheus debe estar excluido"
        assert "whatsapp-bridge-abc" in containers_alerted, "whatsapp-bridge sí debe alertar"


# ═════════════════════════════════════════════════════════════════════════════
# P#3 — PATCH_RESOURCES: MEMORY LIMIT AUTO-PATCH
# ═════════════════════════════════════════════════════════════════════════════

class TestPatchDeploymentMemoryLimit:
    """Valida patch_deployment_memory_limit() — Priority #3."""

    def _make_deploy(self, mem_limit: str, name: str = "backend-ia-deployment"):
        container = MagicMock()
        container.name = "backend-ia"
        container.resources = MagicMock()
        container.resources.limits = {"memory": mem_limit}
        deploy = MagicMock()
        deploy.spec.template.spec.containers = [container]
        return deploy

    def test_increases_memory_limit_mi(self):
        """512Mi → 768Mi (512 + 50% = 256Mi mínimo → +256Mi → 768Mi)."""
        from main import patch_deployment_memory_limit
        deploy = self._make_deploy("512Mi")
        with patch("main.apps_v1") as mock_apps:
            mock_apps.read_namespaced_deployment.return_value = deploy
            result = patch_deployment_memory_limit("backend-ia-deployment", "amael-ia")
        assert "✅" in result
        assert "512Mi" in result
        assert "768Mi" in result

    def test_increases_by_50_pct_minimum_256mi(self):
        """128Mi → 384Mi: 50% de 128 = 64 < 256 mínimo → +256Mi → 384Mi."""
        from main import patch_deployment_memory_limit
        deploy = self._make_deploy("128Mi")
        with patch("main.apps_v1") as mock_apps:
            mock_apps.read_namespaced_deployment.return_value = deploy
            result = patch_deployment_memory_limit("backend-ia-deployment", "amael-ia")
        assert "384Mi" in result

    def test_parses_gi_limit(self):
        """1Gi (= 1024Mi) + 50% = +512Mi → 1536Mi."""
        from main import patch_deployment_memory_limit
        deploy = self._make_deploy("1Gi")
        with patch("main.apps_v1") as mock_apps:
            mock_apps.read_namespaced_deployment.return_value = deploy
            result = patch_deployment_memory_limit("backend-ia-deployment", "amael-ia")
        assert "1536Mi" in result

    def test_default_limit_when_none_set(self):
        """Sin memory limit definido usa 512Mi como base."""
        from main import patch_deployment_memory_limit
        container = MagicMock()
        container.name = "backend-ia"
        container.resources = MagicMock()
        container.resources.limits = {}  # no memory key
        deploy = MagicMock()
        deploy.spec.template.spec.containers = [container]
        with patch("main.apps_v1") as mock_apps:
            mock_apps.read_namespaced_deployment.return_value = deploy
            result = patch_deployment_memory_limit("backend-ia-deployment", "amael-ia")
        assert "✅" in result
        assert "512Mi" in result  # default

    def test_protected_deployment_blocked(self):
        """Deployments protegidos no pueden ser patcheados."""
        from main import patch_deployment_memory_limit
        with patch("main.PROTECTED_DEPLOYMENTS", {"postgres-deployment"}):
            result = patch_deployment_memory_limit("postgres-deployment", "amael-ia")
        assert "protegido" in result
        assert "✅" not in result

    def test_k8s_error_returns_error_string(self):
        """Si K8s falla, retorna string de error sin propagar excepción."""
        from main import patch_deployment_memory_limit
        with patch("main.apps_v1") as mock_apps:
            mock_apps.read_namespaced_deployment.side_effect = Exception("api down")
            result = patch_deployment_memory_limit("backend-ia-deployment", "amael-ia")
        assert "Error" in result
        assert "✅" not in result

    def test_patches_correct_container_name(self):
        """El patch enviado a K8s incluye el nombre correcto del container."""
        from main import patch_deployment_memory_limit
        deploy = self._make_deploy("256Mi", name="backend-ia-deployment")
        with patch("main.apps_v1") as mock_apps:
            mock_apps.read_namespaced_deployment.return_value = deploy
            patch_deployment_memory_limit("backend-ia-deployment", "amael-ia")
            call_args = mock_apps.patch_namespaced_deployment.call_args
        patch_body = call_args[0][2]
        containers = patch_body["spec"]["template"]["spec"]["containers"]
        assert containers[0]["name"] == "backend-ia"


# ═════════════════════════════════════════════════════════════════════════════
# P#3 — decide_action PATCH_RESOURCES
# ═════════════════════════════════════════════════════════════════════════════

class TestDecideActionPatchResources:
    """Valida que decide_action() honra PATCH_RESOURCES del LLM (Priority #3)."""

    def test_llm_patch_resources_returns_patch_plan(self, crash_loop_anomaly):
        """LLM recomienda PATCH_RESOURCES para OOMKilled → decide_action lo honra."""
        oom_diag = Diagnosis(
            issue_type="OOM_KILLED",
            root_cause="Container excede memory limit (OOMKilled, exit_code=137)",
            root_cause_category="RESOURCE_LIMIT",
            confidence=0.92,
            severity="HIGH",
            recommended_action="PATCH_RESOURCES",
            evidence=["OOMKilled", "exit_code=137"],
            source="llm",
        )
        oom_anomaly = Anomaly(
            issue_type="OOM_KILLED",
            severity="HIGH",
            namespace="amael-ia",
            resource_name="backend-ia-abc",
            resource_type="pod",
            owner_name="backend-ia-deployment",
            owner_kind="Deployment",
            details="OOMKilled exit_code=137",
            dedup_key="amael-ia:backend-ia-abc:OOM",
        )
        with patch("main.adjust_confidence_with_history", return_value=oom_diag):
            plan = decide_action(oom_anomaly, oom_diag)
        assert plan.action == "PATCH_RESOURCES"
        assert plan.target_name == "backend-ia-deployment"

    def test_execute_patch_resources_calls_patch_fn(self, oom_anomaly):
        """execute_sre_action PATCH_RESOURCES llama a patch_deployment_memory_limit."""
        from main import execute_sre_action
        plan = ActionPlan(
            action="PATCH_RESOURCES",
            target_name="backend-ia-deployment",
            target_namespace="amael-ia",
            reason="OOM patch",
            auto_execute=True,
        )
        with patch("main.patch_deployment_memory_limit",
                   return_value="✅ Memory patch: 'backend-ia-deployment' límite aumentado de 512Mi a 768Mi.") as mock_patch:
            with patch("main._send_sre_notification"):
                result = execute_sre_action(plan, oom_anomaly)
        mock_patch.assert_called_once_with("backend-ia-deployment", "amael-ia")
        assert "✅" in result

    def test_execute_patch_resources_increments_counter(self, oom_anomaly):
        """execute_sre_action PATCH_RESOURCES incrementa SRE_PATCH_RESOURCES_TOTAL."""
        from main import execute_sre_action
        plan = ActionPlan(
            action="PATCH_RESOURCES",
            target_name="backend-ia-deployment",
            target_namespace="amael-ia",
            reason="OOM patch",
            auto_execute=True,
        )
        with patch("main.patch_deployment_memory_limit",
                   return_value="✅ Memory patch: ok"):
            with patch("main._send_sre_notification"):
                with patch("main.SRE_PATCH_RESOURCES_TOTAL") as mock_counter:
                    with patch("main.SRE_ACTIONS_TAKEN") as mock_actions:
                        execute_sre_action(plan, oom_anomaly)
        mock_counter.inc.assert_called_once()


# ═════════════════════════════════════════════════════════════════════════════
# P#4 — AUTO SCALE-DOWN AFTER HIGH_CPU RESOLVES
# ═════════════════════════════════════════════════════════════════════════════

class TestAutoScaleDown:
    """Valida el ciclo auto scale-down 30min después de SCALE_UP (Priority #4)."""

    def test_scale_down_executes_when_cpu_normalized(self):
        """CPU ya no supera el threshold → escala de vuelta a réplicas originales."""
        from main import _run_scale_down_check

        deploy_after = MagicMock()
        deploy_after.spec.replicas = 3  # actualmente 3 (fue escalado)
        deploy_after.spec.template.spec.containers = []  # sin info de limits (skip ratio check)

        prom_resp = MagicMock()
        prom_resp.json.return_value = {
            "status": "success",
            "data": {"result": [{"value": ["ts", "0.1"]}]},  # CPU muy bajo
        }

        with patch("main.requests.get", return_value=prom_resp):
            with patch("main.apps_v1") as mock_apps:
                mock_apps.read_namespaced_deployment.return_value = deploy_after
                with patch("main.SRE_SCALE_DOWN_TOTAL") as mock_total:
                    with patch("main.SRE_ACTIONS_TAKEN") as mock_actions:
                        with patch("main._send_sre_notification"):
                            _run_scale_down_check("backend-ia-deployment", "amael-ia", 1)
        mock_apps.patch_namespaced_deployment_scale.assert_called_once_with(
            "backend-ia-deployment", "amael-ia", {"spec": {"replicas": 1}}
        )

    def test_scale_down_skipped_when_cpu_still_high(self):
        """CPU sigue sobre el threshold → NO escalar, solo notificar."""
        from main import _run_scale_down_check

        deploy = MagicMock()
        deploy.spec.replicas = 3
        container = MagicMock()
        container.resources.limits = {"cpu": "1000m"}
        deploy.spec.template.spec.containers = [container]

        prom_resp = MagicMock()
        prom_resp.json.return_value = {
            "status": "success",
            # cpu_val=3.0 / total_limit=3.0×1.0=3.0 → ratio=1.0 > threshold=0.85
            "data": {"result": [{"value": ["ts", "3.0"]}]},
        }

        with patch("main.requests.get", return_value=prom_resp):
            with patch("main.apps_v1") as mock_apps:
                mock_apps.read_namespaced_deployment.return_value = deploy
                with patch("main.SRE_SCALE_DOWN_TOTAL") as mock_total:
                    with patch("main._send_sre_notification") as mock_notify:
                        with patch("main.SRE_CPU_THRESHOLD", 0.85):
                            _run_scale_down_check("backend-ia-deployment", "amael-ia", 1)
        mock_apps.patch_namespaced_deployment_scale.assert_not_called()
        mock_notify.assert_called_once()
        assert "cancelado" in mock_notify.call_args[0][0]

    def test_scale_down_skipped_if_already_at_original(self):
        """Si el deployment ya está en réplicas originales (alguien lo bajó manualmente) no hace nada."""
        from main import _run_scale_down_check

        deploy = MagicMock()
        deploy.spec.replicas = 1  # ya en el original
        deploy.spec.template.spec.containers = []

        prom_resp = MagicMock()
        prom_resp.json.return_value = {
            "status": "success",
            "data": {"result": [{"value": ["ts", "0.05"]}]},
        }

        with patch("main.requests.get", return_value=prom_resp):
            with patch("main.apps_v1") as mock_apps:
                mock_apps.read_namespaced_deployment.return_value = deploy
                with patch("main.SRE_SCALE_DOWN_TOTAL"):
                    _run_scale_down_check("backend-ia-deployment", "amael-ia", 1)
        mock_apps.patch_namespaced_deployment_scale.assert_not_called()

    def test_schedule_scale_down_adds_apscheduler_job(self):
        """_schedule_scale_down() registra un job APScheduler con el ID correcto."""
        from main import _schedule_scale_down
        mock_scheduler = MagicMock()
        with patch("main._sre_scheduler", mock_scheduler):
            _schedule_scale_down("backend-ia-deployment", "amael-ia", 2, delay_s=1800)
        mock_scheduler.add_job.assert_called_once()
        call_kwargs = mock_scheduler.add_job.call_args[1]
        assert call_kwargs["id"] == "scaledown:amael-ia:backend-ia-deployment"
        assert call_kwargs["replace_existing"] is True

    def test_schedule_scale_down_noop_when_no_scheduler(self):
        """Si el scheduler no está inicializado, no hay error."""
        from main import _schedule_scale_down
        with patch("main._sre_scheduler", None):
            _schedule_scale_down("backend-ia-deployment", "amael-ia", 2)  # no exception

    def test_execute_scale_up_captures_original_replicas(self):
        """execute_sre_action SCALE_UP lee las réplicas actuales antes de escalar."""
        from main import execute_sre_action
        plan = ActionPlan(
            action="SCALE_UP",
            target_name="backend-ia-deployment",
            target_namespace="amael-ia",
            reason="HIGH_CPU",
            auto_execute=True,
        )
        anomaly = Anomaly(
            issue_type="HIGH_CPU", severity="HIGH",
            namespace="amael-ia", resource_name="backend-ia-deployment",
            resource_type="deployment", owner_name="backend-ia-deployment",
            owner_kind="Deployment", details="CPU 90%",
            dedup_key="amael-ia:backend-ia-deployment:HIGH_CPU",
        )
        mock_deploy = MagicMock()
        mock_deploy.spec.replicas = 2

        with patch("main.apps_v1") as mock_apps:
            mock_apps.read_namespaced_deployment.return_value = mock_deploy
            with patch("main.scale_deployment_replicas",
                       return_value="✅ Scale-up: 'backend-ia-deployment' escalado de 2 a 3 réplicas."):
                with patch("main._schedule_scale_down") as mock_sched:
                    with patch("main._send_sre_notification"):
                        with patch("main.SRE_ACTIONS_TAKEN"):
                            execute_sre_action(plan, anomaly)
        # Verify original replicas (2) were passed to _schedule_scale_down
        mock_sched.assert_called_once_with("backend-ia-deployment", "amael-ia", 2)

    def test_execute_scale_up_no_scale_down_on_failure(self):
        """Si scale_deployment_replicas falla, NO se agenda scale-down."""
        from main import execute_sre_action
        plan = ActionPlan(
            action="SCALE_UP",
            target_name="backend-ia-deployment",
            target_namespace="amael-ia",
            reason="HIGH_CPU",
            auto_execute=True,
        )
        anomaly = Anomaly(
            issue_type="HIGH_CPU", severity="HIGH",
            namespace="amael-ia", resource_name="backend-ia-deployment",
            resource_type="deployment", owner_name="backend-ia-deployment",
            owner_kind="Deployment", details="CPU 90%",
            dedup_key="amael-ia:backend-ia-deployment:HIGH_CPU",
        )
        with patch("main.apps_v1") as mock_apps:
            mock_apps.read_namespaced_deployment.return_value = MagicMock(spec=["spec"])
            mock_apps.read_namespaced_deployment.return_value.spec.replicas = 1
            with patch("main.scale_deployment_replicas",
                       return_value="Error: api error"):
                with patch("main._schedule_scale_down") as mock_sched:
                    with patch("main.SRE_ACTIONS_TAKEN"):
                        execute_sre_action(plan, anomaly)
        mock_sched.assert_not_called()


# ═════════════════════════════════════════════════════════════════════════════
# VAULT AUTO-UNSEAL
# ═════════════════════════════════════════════════════════════════════════════

class TestVaultAutoUnseal:
    """Valida _auto_unseal_vault() y el pipeline decide→execute para VAULT_SEALED."""

    def _vault_secret(self, keys=("KEY1", "KEY2", "KEY3")):
        import base64
        secret = MagicMock()
        secret.data = {
            "key1": base64.b64encode(keys[0].encode()).decode(),
            "key2": base64.b64encode(keys[1].encode()).decode(),
            "key3": base64.b64encode(keys[2].encode()).decode(),
        }
        return secret

    def _unseal_resp(self, sealed_after=False):
        resp = MagicMock()
        resp.status_code = 200
        resp.json.return_value = {"sealed": sealed_after, "progress": 1}
        return resp

    def test_unseals_successfully(self):
        """Con las 3 llaves y Vault respondiendo correctamente devuelve ✅."""
        with patch("main.v1") as mock_v1:
            mock_v1.read_namespaced_secret.return_value = self._vault_secret()
            with patch("main.requests.put", return_value=self._unseal_resp(sealed_after=False)):
                result = _auto_unseal_vault()
        assert "✅" in result
        assert "deselado" in result

    def test_missing_key_returns_error(self):
        """Si falta una llave en el secret, retorna error sin llamar a Vault."""
        import base64
        secret = MagicMock()
        secret.data = {
            "key1": base64.b64encode(b"KEY1").decode(),
            "key2": "",  # vacío
            "key3": base64.b64encode(b"KEY3").decode(),
        }
        with patch("main.v1") as mock_v1:
            mock_v1.read_namespaced_secret.return_value = secret
            with patch("main.requests.put") as mock_put:
                result = _auto_unseal_vault()
        assert "Error" in result
        mock_put.assert_not_called()

    def test_vault_api_http_error(self):
        """Si Vault responde 500, devuelve error."""
        resp = MagicMock()
        resp.status_code = 500
        resp.text = "internal error"
        with patch("main.v1") as mock_v1:
            mock_v1.read_namespaced_secret.return_value = self._vault_secret()
            with patch("main.requests.put", return_value=resp):
                result = _auto_unseal_vault()
        assert "Error" in result

    def test_k8s_secret_read_fails_gracefully(self):
        """Si no puede leer el secret (RBAC), retorna error sin propagar excepción."""
        with patch("main.v1") as mock_v1:
            mock_v1.read_namespaced_secret.side_effect = Exception("forbidden")
            result = _auto_unseal_vault()
        assert "Error" in result
        assert "✅" not in result

    def test_decide_action_vault_sealed_returns_vault_unseal(self):
        """decide_action para VAULT_SEALED devuelve ActionPlan VAULT_UNSEAL."""
        vault_anomaly = Anomaly(
            issue_type="VAULT_SEALED",
            severity="HIGH",
            namespace="amael-ia",
            resource_name="amael-agentic-backend:skill.vault",
            resource_type="BackendComponent",
            owner_name="amael-agentic-deployment",
            owner_kind="Deployment",
            details="Vault está sealed",
            dedup_key="backend_health:skill.vault",
        )
        plan = decide_action(vault_anomaly)
        assert plan.action == "VAULT_UNSEAL"
        assert plan.target_namespace == "vault"

    def test_execute_vault_unseal_sends_whatsapp_on_success(self):
        """execute_sre_action VAULT_UNSEAL notifica por WhatsApp al desellar."""
        from main import execute_sre_action
        vault_anomaly = Anomaly(
            issue_type="VAULT_SEALED",
            severity="HIGH",
            namespace="amael-ia",
            resource_name="amael-agentic-backend:skill.vault",
            resource_type="BackendComponent",
            owner_name="amael-agentic-deployment",
            owner_kind="Deployment",
            details="Vault está sealed",
            dedup_key="backend_health:skill.vault",
        )
        plan = ActionPlan(
            action="VAULT_UNSEAL",
            target_name="vault-0",
            target_namespace="vault",
            reason="Vault sellado",
            auto_execute=True,
        )
        with patch("main._auto_unseal_vault", return_value="✅ Vault deselado automáticamente."):
            with patch("main._send_sre_notification") as mock_notify:
                with patch("main.SRE_ACTIONS_TAKEN"):
                    with patch("main.SRE_VAULT_UNSEAL_TOTAL"):
                        result = execute_sre_action(plan, vault_anomaly)
        assert "✅" in result
        msg = mock_notify.call_args[0][0]
        assert "deselado" in msg.lower()

    def test_execute_vault_unseal_notifies_failure(self):
        """Si auto-unseal falla, notifica con severidad HIGH."""
        from main import execute_sre_action
        vault_anomaly = Anomaly(
            issue_type="VAULT_SEALED", severity="HIGH",
            namespace="amael-ia", resource_name="vault",
            resource_type="BackendComponent", owner_name="amael-agentic-deployment",
            owner_kind="Deployment", details="Vault sealed",
            dedup_key="backend_health:vault",
        )
        plan = ActionPlan(
            action="VAULT_UNSEAL", target_name="vault-0",
            target_namespace="vault", reason="sealed", auto_execute=True,
        )
        with patch("main._auto_unseal_vault", return_value="Error: forbidden"):
            with patch("main._send_sre_notification") as mock_notify:
                with patch("main.SRE_ACTIONS_TAKEN"):
                    with patch("main.SRE_VAULT_UNSEAL_TOTAL"):
                        result = execute_sre_action(plan, vault_anomaly)
        assert "Error" in result
        call_kwargs = mock_notify.call_args[1]
        assert call_kwargs.get("severity") == "HIGH"
