# Suite de Pruebas — Agente SRE Autónomo (k8s-agent)

Cobertura completa de las fases **P0–P5** del ciclo autónomo:
`Observe → Detect → Diagnose → Decide → Act → Verify → Learn`

---

## Estructura

```
tests/
├── conftest.py          # Fixtures compartidos + stubs de módulos pesados
├── test_sre_agent.py    # Tests unitarios e integración (P0–P5)
└── README.md            # Este documento
```

---

## Ejecución

> En Ubuntu/Debian con Python 3.12+ el entorno del sistema es "externally managed"
> (PEP 668). Se requiere un **virtualenv** para instalar paquetes de prueba.

### 1. Crear el virtualenv (una sola vez)

```bash
cd k8s-lab/Amael-IA/k8s-agent
python3 -m venv .venv
source .venv/bin/activate
pip install pytest pytest-cov httpx fastapi requests pydantic
```

> **httpx** es requerido por el `TestClient` de FastAPI.
> **No** se necesita instalar `kubernetes`, `langchain` ni servicios externos —
> todos están mockeados en `conftest.py`.

### 2. Activar el virtualenv en sesiones siguientes

```bash
cd k8s-lab/Amael-IA/k8s-agent
source .venv/bin/activate
```

### 3. Ejecutar todos los tests

```bash
pytest tests/ -v
```

### Ejecutar con reporte de cobertura

```bash
pytest tests/ -v --cov=main --cov-report=term-missing
```

### Ejecutar una clase específica

```bash
pytest tests/test_sre_agent.py::TestCircuitBreaker -v
pytest tests/test_sre_agent.py::TestDetectAnomalies -v
pytest tests/test_sre_agent.py::TestIntegration -v
```

### Ejecutar un test específico

```bash
pytest tests/test_sre_agent.py::TestObserveSLO::test_slo_anomaly_when_burn_rate_above_2x -v
```

---

## Clases de prueba por fase

| Clase | Fase | Funciones validadas |
|---|---|---|
| `TestCircuitBreaker` | P0 | `CircuitBreaker.record_failure/success`, transiciones de estado, thread-safety |
| `TestHelpers` | P0 | `_parse_two`, `_guess_owner_from_pod_name`, `_SEVERITY_RANK` |
| `TestDetectAnomalies` | P1 | `detect_anomalies` — 14 anomaly types desde `ClusterSnapshot` |
| `TestDedup` | P1 | `_is_duplicate_incident`, `_mark_incident` — Redis y fallback en memoria |
| `TestDeterministicDiagnosis` | P2 | `_deterministic_diagnosis` — los 16 issue_types mapeados |
| `TestRestartLimit` | P2 | `_check_restart_limit` — guardrail de reinicios máximos |
| `TestCorrelateAnomalies` | P4-B | `correlate_anomalies` — agrupación multi-pod |
| `TestHistoricalLearning` | P3-B | `adjust_confidence_with_history` — fórmula 70/30 |
| `TestDecideAction` | P2+P3 | `decide_action` — todos los guardrails y ramas |
| `TestMaintenanceWindow` | P4-C | `_is_maintenance_active`, `_activate_maintenance`, `_deactivate_maintenance` |
| `TestObserveMetrics` | P4-A | `observe_metrics` — HIGH_CPU, HIGH_MEMORY, HIGH_ERROR_RATE |
| `TestObserveTrends` | P5-A | `observe_trends` — DISK_EXHAUSTION_PREDICTED, MEMORY_LEAK_PREDICTED, ERROR_RATE_ESCALATING |
| `TestObserveSLO` | P5-C | `observe_slo` — burn rate ≥ 2×, severidad CRITICAL/HIGH |
| `TestSLOTargetLoading` | P5-C | `load_slo_targets` — JSON válido, inválido y vacío |
| `TestRolloutUndo` | P5-B | `rollout_undo_deployment` — rollback, protección, revisión no encontrada |
| `TestAutoRunbook` | P4-D | `_maybe_save_runbook_entry` — condiciones de guardado y omisión |
| `TestPostVerification` | P3-A | `_run_verification_job`, `_schedule_verification` — resolved/unresolved/rollback |
| `TestIntegration` | E2E | Flujos completos + endpoints `/health`, `/api/sre/loop/status`, `/metrics` |

---

## Estrategia de mocking

Todos los servicios externos se mockean en `conftest.py` y dentro de cada test con `unittest.mock.patch`:

| Dependencia | Mock |
|---|---|
| `kubernetes` | `MagicMock` completo — `v1`, `apps_v1`, `coord_v1` |
| `redis` | `MagicMock` con `exists`, `set`, `get`, `delete`, `pipeline` |
| `PostgreSQL` (psycopg2) | Pool + cursor simulados con `fetchall`, `fetchone` |
| `Prometheus` | `requests.get` mockeado con respuesta JSON `{"status":"success"}` |
| `Ollama / LLM` | `langchain_ollama.OllamaLLM` mockeado |
| `Qdrant` | `_qdrant_client` mockeado con `query_points`, `upsert` |
| `WhatsApp Bridge` | `requests.post` mockeado |
| `APScheduler` | `_sre_scheduler` mockeado |
| `tracing.py` | Stub que deshabilita OpenTelemetry |

---

## Casos de prueba destacados

### Ciclo completo crash loop → restart
```python
# TestIntegration::test_full_pipeline_crash_loop_triggers_restart
# Valida: detect_anomalies → _deterministic_diagnosis → decide_action → ROLLOUT_RESTART
```

### SLO burn rate 2×
```python
# TestObserveSLO::test_slo_anomaly_when_burn_rate_above_2x
# Valida: observe_slo() detecta burn_rate=3.0× y genera Anomaly(SLO_BUDGET_BURNING, HIGH)
```

### Rollback automático post-verificación
```python
# TestPostVerification::test_verification_unresolved_with_recent_deploy_triggers_rollback
# Valida: _run_verification_job → unhealthy + recent deploy → rollout_undo_deployment
```

### Correlación 3 pods → 1 anomalía
```python
# TestIntegration::test_full_pipeline_multiple_pods_correlated
# Valida: 3 anomalías CRASH_LOOP del mismo deploy → correlate_anomalies → 1 anomalía
```

### Circuit breaker previene ejecución
```python
# TestIntegration::test_circuit_breaker_skips_loop_when_open
# Valida: sre_autonomous_loop() no llama _run_loop_iteration cuando CB está OPEN
```

---

## Convenciones

- Un test por comportamiento, no por función.
- Nombre: `test_<acción>_<condición>_<resultado_esperado>`.
- Los tests no tienen efectos secundarios — usan `patch` como context manager o decorador.
- Los fixtures de `conftest.py` proveen objetos de datos reutilizables (`Anomaly`, `Diagnosis`, `ClusterSnapshot`).

---

## Agregar nuevos tests

1. Identifica la función o flujo a probar.
2. Agrega el caso a la clase existente correspondiente a su fase, o crea una nueva clase si es una fase nueva.
3. Usa `patch` para aislar dependencias externas.
4. Verifica el fixture en `conftest.py` — si el dato de prueba es reutilizable, agrégalo allí.
