# Plan de Pruebas — Agente SRE Autónomo (k8s-agent)

**Versión:** 5.0.2
**Cobertura:** Fases P0–P5 del ciclo autónomo `Observe → Detect → Diagnose → Decide → Act → Verify → Learn`
**Resultado actual:** 132 tests, 132 pasando ✅

---

## Contexto

El k8s-agent implementa un bucle autónomo de auto-reparación de Kubernetes que evoluciona en 6 fases (P0–P5). Cada fase agrega capacidades nuevas que deben validarse de forma aislada, sin depender de servicios externos (Kubernetes, Redis, PostgreSQL, Ollama, Prometheus, Qdrant).

La suite usa `pytest` con `unittest.mock` para simular todas las dependencias externas, lo que permite ejecutar los 132 tests en ~0.15 segundos sin infraestructura.

---

## Preparación del entorno

```bash
cd k8s-lab/Amael-IA/k8s-agent

# Crear virtualenv (solo la primera vez)
python3 -m venv .venv
source .venv/bin/activate
pip install pytest pytest-cov httpx fastapi requests pydantic

# En sesiones siguientes
source .venv/bin/activate
```

---

## Comandos de ejecución

```bash
# Todos los tests
pytest tests/ -v

# Con reporte de cobertura
pytest tests/ -v --cov=main --cov-report=term-missing

# Una fase específica
pytest tests/test_sre_agent.py::TestCircuitBreaker -v        # P0
pytest tests/test_sre_agent.py::TestDetectAnomalies -v       # P1
pytest tests/test_sre_agent.py::TestDeterministicDiagnosis -v # P2
pytest tests/test_sre_agent.py::TestHistoricalLearning -v    # P3-B
pytest tests/test_sre_agent.py::TestPostVerification -v      # P3-A
pytest tests/test_sre_agent.py::TestCorrelateAnomalies -v    # P4-B
pytest tests/test_sre_agent.py::TestMaintenanceWindow -v     # P4-C
pytest tests/test_sre_agent.py::TestAutoRunbook -v           # P4-D
pytest tests/test_sre_agent.py::TestObserveMetrics -v        # P4-A
pytest tests/test_sre_agent.py::TestObserveTrends -v         # P5-A
pytest tests/test_sre_agent.py::TestRolloutUndo -v           # P5-B
pytest tests/test_sre_agent.py::TestObserveSLO -v            # P5-C
pytest tests/test_sre_agent.py::TestIntegration -v           # E2E

# Un test específico
pytest tests/test_sre_agent.py::TestObserveSLO::test_slo_anomaly_when_burn_rate_above_2x -v
```

---

## Mapa de pruebas por fase

### P0 — Fundación

| Test | Función validada | Descripción |
|---|---|---|
| `TestCircuitBreaker::test_initial_state_is_closed` | `CircuitBreaker.state` | Estado inicial CLOSED |
| `TestCircuitBreaker::test_opens_after_threshold_failures` | `record_failure()` | Se abre tras N fallos consecutivos |
| `TestCircuitBreaker::test_does_not_open_below_threshold` | `record_failure()` | No se abre con N-1 fallos |
| `TestCircuitBreaker::test_transitions_to_half_open_after_timeout` | `state` property | Transición OPEN → HALF_OPEN tras timeout |
| `TestCircuitBreaker::test_success_closes_circuit_from_open` | `record_success()` | OPEN → CLOSED al primer éxito |
| `TestCircuitBreaker::test_failure_counter_resets_on_success` | `record_success()` | Contador de fallos se resetea |
| `TestCircuitBreaker::test_thread_safety` | `CircuitBreaker` | Sin race conditions con 4 hilos concurrentes |
| `TestHelpers::test_parse_two_with_comma` | `_parse_two()` | Parseo `"deploy, namespace"` |
| `TestHelpers::test_parse_two_without_comma_uses_default` | `_parse_two()` | Usa namespace por defecto sin coma |
| `TestHelpers::test_parse_two_strips_whitespace` | `_parse_two()` | Limpia espacios en tokens |
| `TestHelpers::test_guess_owner_standard_k8s_name` | `_guess_owner_from_pod_name()` | `deploy-rs-pod` → `deploy` |
| `TestHelpers::test_guess_owner_two_parts` | `_guess_owner_from_pod_name()` | `myapp-abc` → `myapp` |
| `TestHelpers::test_guess_owner_single_part` | `_guess_owner_from_pod_name()` | Nombre sin guión retorna igual |
| `TestHelpers::test_severity_rank_ordering` | `_SEVERITY_RANK` | CRITICAL > HIGH > MEDIUM > LOW |

---

### P1 — Bucle autónomo y detección estructural

| Test | Función validada | Descripción |
|---|---|---|
| `TestDetectAnomalies::test_healthy_cluster_returns_no_anomalies` | `detect_anomalies()` | Clúster sano → lista vacía |
| `TestDetectAnomalies::test_detects_crash_loop` | `detect_anomalies()` | CrashLoopBackOff → `CRASH_LOOP` CRITICAL |
| `TestDetectAnomalies::test_detects_oom_killed` | `detect_anomalies()` | OOMKilled → `OOM_KILLED` HIGH |
| `TestDetectAnomalies::test_detects_image_pull_backoff` | `detect_anomalies()` | ImagePullBackOff → `IMAGE_PULL_ERROR` |
| `TestDetectAnomalies::test_detects_err_image_pull` | `detect_anomalies()` | ErrImagePull → `IMAGE_PULL_ERROR` |
| `TestDetectAnomalies::test_detects_pod_failed` | `detect_anomalies()` | Phase=Failed → `POD_FAILED` |
| `TestDetectAnomalies::test_detects_pending_stuck_after_5_minutes` | `detect_anomalies()` | Pending >5min → `POD_PENDING_STUCK` |
| `TestDetectAnomalies::test_no_pending_anomaly_before_5_minutes` | `detect_anomalies()` | Pending <5min → sin anomalía |
| `TestDetectAnomalies::test_detects_high_restarts` | `detect_anomalies()` | ≥5 reinicios → `HIGH_RESTARTS` |
| `TestDetectAnomalies::test_no_high_restarts_below_threshold` | `detect_anomalies()` | <5 reinicios → sin anomalía |
| `TestDetectAnomalies::test_detects_node_not_ready` | `detect_anomalies()` | Nodo NotReady → `NODE_NOT_READY` CRITICAL |
| `TestDetectAnomalies::test_detects_node_memory_pressure` | `detect_anomalies()` | MemoryPressure=True → anomalía |
| `TestDetectAnomalies::test_detects_node_disk_pressure` | `detect_anomalies()` | DiskPressure=True → anomalía |
| `TestDetectAnomalies::test_dedup_key_format` | `detect_anomalies()` | Formato `ns:pod:TIPO` |
| `TestDedup::test_new_incident_is_not_duplicate` | `_is_duplicate_incident()` | Clave nueva → no duplicado |
| `TestDedup::test_marked_incident_is_duplicate` | `_mark_incident()` | Tras marcar → es duplicado |
| `TestDedup::test_redis_exists_returns_duplicate` | `_is_duplicate_incident()` | Redis exists=1 → duplicado |
| `TestDedup::test_redis_not_exists_returns_not_duplicate` | `_is_duplicate_incident()` | Redis exists=0 → no duplicado |
| `TestDedup::test_redis_mark_calls_set_with_ttl` | `_mark_incident()` | Llama `SET key 1 EX 600` |
| `TestDedup::test_memory_fallback_ttl_expires` | `_is_duplicate_incident()` | Cache en memoria expira tras TTL |

---

### P2 — Diagnóstico y guardrails

| Test | Función validada | Descripción |
|---|---|---|
| `TestDeterministicDiagnosis::test_known_issue_type[CRASH_LOOP-...]` | `_deterministic_diagnosis()` | 16 issue_types × acción+categoría correctas |
| `TestDeterministicDiagnosis::test_unknown_issue_type_defaults` | `_deterministic_diagnosis()` | Tipo desconocido → NOTIFY_HUMAN, conf=0.50 |
| `TestDeterministicDiagnosis::test_confidence_in_valid_range` | `_deterministic_diagnosis()` | Confianza siempre en [0.0, 1.0] |
| `TestRestartLimit::test_limit_not_hit_when_count_below_max` | `_check_restart_limit()` | Count < MAX → no bloqueado |
| `TestRestartLimit::test_limit_hit_when_count_equals_max` | `_check_restart_limit()` | Count == MAX → bloqueado |
| `TestRestartLimit::test_limit_not_hit_without_redis` | `_check_restart_limit()` | Sin Redis → nunca bloqueado (fail-open) |
| `TestRestartLimit::test_limit_hit_when_count_exceeds_max` | `_check_restart_limit()` | Count > MAX → bloqueado |
| `TestDecideAction::test_auto_heals_crash_loop_high_confidence` | `decide_action()` | Alta confianza + namespace ok → ROLLOUT_RESTART |
| `TestDecideAction::test_notifies_when_confidence_below_threshold` | `decide_action()` | Confianza < 0.75 → NOTIFY_HUMAN |
| `TestDecideAction::test_notifies_when_restart_limit_hit` | `decide_action()` | Límite alcanzado → NOTIFY_HUMAN |
| `TestDecideAction::test_notifies_for_image_pull_error` | `decide_action()` | IMAGE_PULL_ERROR siempre → NOTIFY_HUMAN |
| `TestDecideAction::test_notifies_for_node_anomaly` | `decide_action()` | Nodos nunca auto-heal |
| `TestDecideAction::test_notifies_for_slo_budget_burning` | `decide_action()` | SLO_BUDGET_BURNING siempre → NOTIFY_HUMAN |
| `TestDecideAction::test_notifies_for_protected_deployment` | `decide_action()` | postgres/ollama/vault protegidos |
| `TestDecideAction::test_notifies_for_foreign_namespace` | `decide_action()` | kube-system → NOTIFY_HUMAN |
| `TestDecideAction::test_notifies_for_disk_exhaustion_predicted` | `decide_action()` | Predictivo disco → NOTIFY_HUMAN |
| `TestDecideAction::test_notifies_for_pod_pending` | `decide_action()` | POD_PENDING_STUCK → NOTIFY_HUMAN |

---

### P3-A — Verificación post-acción

| Test | Función validada | Descripción |
|---|---|---|
| `TestPostVerification::test_verification_resolved_triggers_postmortem` | `_run_verification_job()` | Deployment healthy → postmortem en background |
| `TestPostVerification::test_verification_unresolved_with_recent_deploy_triggers_rollback` | `_run_verification_job()` | Unhealthy + deploy reciente → auto-rollback |
| `TestPostVerification::test_verification_unresolved_without_recent_deploy_notifies_human` | `_run_verification_job()` | Unhealthy + sin deploy reciente → notificación manual |
| `TestPostVerification::test_schedule_verification_with_no_scheduler_is_noop` | `_schedule_verification()` | Sin scheduler → no lanza excepción |
| `TestPostVerification::test_schedule_verification_adds_job` | `_schedule_verification()` | Agrega job tipo "date" con ID correcto |

---

### P3-B — Aprendizaje histórico

| Test | Función validada | Descripción |
|---|---|---|
| `TestHistoricalLearning::test_blending_formula` | `adjust_confidence_with_history()` | `0.7×llm + 0.3×hist` con precisión 0.001 |
| `TestHistoricalLearning::test_source_updated_to_include_history` | `adjust_confidence_with_history()` | `source` pasa a incluir `"history"` |
| `TestHistoricalLearning::test_no_history_returns_original` | `adjust_confidence_with_history()` | Sin historial → diagnosis sin cambios |
| `TestHistoricalLearning::test_confidence_clamped_to_one` | `adjust_confidence_with_history()` | Nunca supera 1.0 |
| `TestHistoricalLearning::test_confidence_clamped_to_zero` | `adjust_confidence_with_history()` | Nunca baja de 0.0 |

---

### P4-A — Monitoreo proactivo (Prometheus)

| Test | Función validada | Descripción |
|---|---|---|
| `TestObserveMetrics::test_no_anomalies_when_no_results` | `observe_metrics()` | Sin resultados Prometheus → lista vacía |
| `TestObserveMetrics::test_detects_high_cpu` | `observe_metrics()` | CPU > 85% → `HIGH_CPU` MEDIUM |
| `TestObserveMetrics::test_detects_high_memory` | `observe_metrics()` | Memoria > 85% → `HIGH_MEMORY` HIGH |
| `TestObserveMetrics::test_detects_high_error_rate` | `observe_metrics()` | 5xx > 1% → `HIGH_ERROR_RATE` |
| `TestObserveMetrics::test_returns_empty_on_prometheus_error` | `observe_metrics()` | Excepción de red → lista vacía (no crash) |

---

### P4-B — Correlación de anomalías

| Test | Función validada | Descripción |
|---|---|---|
| `TestCorrelateAnomalies::test_single_anomaly_passes_through` | `correlate_anomalies()` | 1 anomalía → pasa sin cambios |
| `TestCorrelateAnomalies::test_multi_pod_same_deployment_collapsed` | `correlate_anomalies()` | 3 pods del mismo deploy → 1 anomalía con "3 pods" |
| `TestCorrelateAnomalies::test_different_deployments_not_collapsed` | `correlate_anomalies()` | Distintos deploys → no se agrupan |
| `TestCorrelateAnomalies::test_different_issue_types_not_collapsed` | `correlate_anomalies()` | Mismo deploy, distinto tipo → no se agrupan |
| `TestCorrelateAnomalies::test_worst_severity_preserved` | `correlate_anomalies()` | Grupo hereda la peor severidad |
| `TestCorrelateAnomalies::test_non_deployment_resources_pass_through` | `correlate_anomalies()` | Nodos/endpoints no se agrupan |
| `TestCorrelateAnomalies::test_correlated_dedup_key_has_multi_suffix` | `correlate_anomalies()` | `dedup_key` termina en `:multi` |
| `TestCorrelateAnomalies::test_empty_list_returns_empty` | `correlate_anomalies()` | Lista vacía → lista vacía |

---

### P4-C — Ventanas de mantenimiento

| Test | Función validada | Descripción |
|---|---|---|
| `TestMaintenanceWindow::test_maintenance_active_with_redis_key` | `_is_maintenance_active()` | Redis key presente → activo |
| `TestMaintenanceWindow::test_maintenance_inactive_without_redis_key` | `_is_maintenance_active()` | Redis key ausente → inactivo |
| `TestMaintenanceWindow::test_maintenance_inactive_without_redis` | `_is_maintenance_active()` | Sin Redis → inactivo (fail-safe) |
| `TestMaintenanceWindow::test_activate_maintenance_sets_redis_with_ttl` | `_activate_maintenance()` | `SET key 1 EX <seg>` con duración correcta |
| `TestMaintenanceWindow::test_activate_maintenance_clamps_max_duration` | `_activate_maintenance()` | Máximo 480 min (8h) |
| `TestMaintenanceWindow::test_activate_maintenance_defaults_to_60_minutes` | `_activate_maintenance()` | Input vacío → 60 min |
| `TestMaintenanceWindow::test_deactivate_maintenance_deletes_redis_key` | `_deactivate_maintenance()` | Elimina la clave de Redis |
| `TestMaintenanceWindow::test_activate_without_redis_returns_error` | `_activate_maintenance()` | Sin Redis → mensaje de error |

---

### P4-D — Auto-generación de runbooks

| Test | Función validada | Descripción |
|---|---|---|
| `TestAutoRunbook::test_skips_if_qdrant_not_available` | `_maybe_save_runbook_entry()` | Sin Qdrant → no crash |
| `TestAutoRunbook::test_skips_if_source_is_deterministic` | `_maybe_save_runbook_entry()` | Fuente determinística → no guarda |
| `TestAutoRunbook::test_skips_if_confidence_below_0_80` | `_maybe_save_runbook_entry()` | Confianza < 0.80 → no guarda |
| `TestAutoRunbook::test_skips_if_existing_runbook_covers_issue` | `_maybe_save_runbook_entry()` | Runbook existente con score alto → no duplica |
| `TestAutoRunbook::test_saves_runbook_when_no_existing_coverage` | `_maybe_save_runbook_entry()` | LLM conf ≥ 0.80, sin cobertura → guarda en Qdrant |
| `TestAutoRunbook::test_skips_if_embedding_fails` | `_maybe_save_runbook_entry()` | Embedding vacío → no guarda |

---

### P5-A — Alertas predictivas (tendencias)

| Test | Función validada | Descripción |
|---|---|---|
| `TestObserveTrends::test_no_trends_when_all_queries_empty` | `observe_trends()` | Sin resultados → lista vacía |
| `TestObserveTrends::test_detects_disk_exhaustion_predicted` | `observe_trends()` | `predict_linear` negativo → `DISK_EXHAUSTION_PREDICTED` |
| `TestObserveTrends::test_detects_memory_leak_predicted` | `observe_trends()` | `deriv` mem > 1MB/s → `MEMORY_LEAK_PREDICTED` |
| `TestObserveTrends::test_detects_error_rate_escalating` | `observe_trends()` | `deriv` 5xx > 0.001 → `ERROR_RATE_ESCALATING` |
| `TestObserveTrends::test_returns_empty_on_exception` | `observe_trends()` | Excepción Prometheus → lista vacía |

---

### P5-B — Rollback automático

| Test | Función validada | Descripción |
|---|---|---|
| `TestRolloutUndo::test_rollback_protected_deployment_blocked` | `rollout_undo_deployment()` | postgres/ollama/vault → bloqueados |
| `TestRolloutUndo::test_rollback_at_revision_1_returns_message` | `rollout_undo_deployment()` | Rev=1 → mensaje "no hay revisión anterior" |
| `TestRolloutUndo::test_rollback_success_returns_checkmark` | `rollout_undo_deployment()` | RS previo encontrado → patch y ✅ |
| `TestRolloutUndo::test_rollback_when_rs_not_found` | `rollout_undo_deployment()` | Sin RS anterior → mensaje de error |
| `TestRolloutUndo::test_rollback_k8s_exception_returns_error` | `rollout_undo_deployment()` | Excepción K8s → mensaje de error (no crash) |

---

### P5-C — SLO / Error budget

| Test | Función validada | Descripción |
|---|---|---|
| `TestObserveSLO::test_no_slo_anomaly_when_burn_rate_below_2x` | `observe_slo()` | burn_rate=1.5× → sin anomalía |
| `TestObserveSLO::test_slo_anomaly_when_burn_rate_above_2x` | `observe_slo()` | burn_rate=3.0× → `SLO_BUDGET_BURNING` |
| `TestObserveSLO::test_slo_critical_severity_at_5x_burn_rate` | `observe_slo()` | burn_rate ≥ 5× → CRITICAL |
| `TestObserveSLO::test_slo_high_severity_at_3x_burn_rate` | `observe_slo()` | burn_rate 2–5× → HIGH |
| `TestObserveSLO::test_no_anomalies_when_no_targets` | `observe_slo()` | Sin targets configurados → lista vacía |
| `TestObserveSLO::test_slo_dedup_key_format` | `observe_slo()` | Formato `amael-ia:<handler>:SLO_BURN` |
| `TestSLOTargetLoading::test_loads_valid_json` | `load_slo_targets()` | JSON válido → lista de targets cargada |
| `TestSLOTargetLoading::test_handles_invalid_json_gracefully` | `load_slo_targets()` | JSON inválido → lista vacía (no crash) |
| `TestSLOTargetLoading::test_handles_empty_env_var` | `load_slo_targets()` | `"[]"` → lista vacía |

---

### E2E — Flujos de integración

| Test | Flujo validado |
|---|---|
| `test_full_pipeline_crash_loop_triggers_restart` | CrashLoopBackOff → detect → diagnose → decide → ROLLOUT_RESTART |
| `test_full_pipeline_image_pull_notifies_human` | ImagePullBackOff → detect → decide → NOTIFY_HUMAN |
| `test_full_pipeline_multiple_pods_correlated` | 3 pods → detect (3 anomalías) → correlate → 1 anomalía |
| `test_maintenance_window_skips_loop_iteration` | Mantenimiento activo → loop no llama `observe_cluster` |
| `test_circuit_breaker_skips_loop_when_open` | CB abierto → `sre_autonomous_loop` no ejecuta iteración |
| `test_dedup_prevents_double_processing` | Misma anomalía marcada → segunda llamada retorna duplicado |
| `test_health_check_endpoint` | `GET /health` → 200 `{"status": "ok"}` |
| `test_sre_loop_status_endpoint` | `GET /api/sre/loop/status` → 200 con `loop_enabled` y `circuit_breaker` |
| `test_metrics_endpoint_not_found_in_test_env` | `GET /metrics` → 200 o 404 (sin 500) |

---

## Arquitectura de mocking

Todos los módulos pesados se stubbean en `conftest.py` **antes** de que `main.py` se importe, evitando conexiones reales a servicios externos:

| Dependencia | Técnica | Detalle |
|---|---|---|
| `kubernetes` | `sys.modules` stub | `v1`, `apps_v1`, `coord_v1` son `MagicMock` |
| `langchain` / `langchain_ollama` | `sys.modules` stub | Toda la cadena de imports LLM mockeada |
| `redis` | `sys.modules` stub | `Redis.from_url()` → `MagicMock` que no conecta |
| `psycopg2` | `sys.modules` stub | Pool y cursor simulados |
| `prometheus_client` | `sys.modules` stub | Counter/Gauge/Histogram → `MagicMock` |
| `qdrant_client` | `sys.modules` stub | `QdrantClient` → `MagicMock` |
| `apscheduler` | `sys.modules` stub | Scheduler → `MagicMock` |
| `tracing.py` | Módulo local stub | `instrument_app` → función vacía |
| `opentelemetry` | `sys.modules` stub | Toda la cadena OTel mockeada |
| Prometheus HTTP | `patch("main.requests.get")` | Respuesta JSON simulada por test |
| WhatsApp Bridge | `patch("main.requests.post")` | No envía mensajes reales |
| K8s API calls | `patch("main.apps_v1.*")` | Respuestas simuladas por test |

---

## Notas de comportamiento documentadas por los tests

- **`HIGH_CPU` es severidad `MEDIUM`**, no HIGH. CPU pressure es menos urgente que memoria (no trigger OOMKill inmediato).
- **Confianza determinística de `CRASH_LOOP` es 0.70**, por debajo del threshold 0.75. Para que `decide_action` retorne `ROLLOUT_RESTART`, el diagnóstico debe venir del LLM (confianza ≥ 0.75) o del ajuste histórico.
- **`_parse_two` strips solo el string completo**, no cada token individual tras el split. No procesa comillas internas en tokens.
- **`CircuitBreaker.is_open()` puede retornar False inmediatamente** si `recovery_timeout=0`, porque la propiedad `.state` dispara la transición a `HALF_OPEN` en la misma llamada.
- **`/metrics` no existe en tests** porque `prometheus_fastapi_instrumentator` está mockeado y no registra el endpoint real.
