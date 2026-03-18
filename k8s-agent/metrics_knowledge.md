# Prometheus Metrics — Amael-IA Cluster

## Métricas HTTP del Backend (http_requests_total)

La métrica correcta es `http_requests_total` con labels: handler, method, status, namespace, pod.

### Solicitudes recibidas en la última hora por endpoint:
```
sum(increase(http_requests_total{namespace="amael-ia"}[1h])) by (handler)
```

### Solicitudes al chat en la última hora:
```
sum(increase(http_requests_total{namespace="amael-ia", handler="/api/chat"}[1h]))
```

### Solicitudes totales al backend en los últimos 5 minutos (rate):
```
sum(rate(http_requests_total{namespace="amael-ia"}[5m])) by (handler)
```

### Errores HTTP (status 5xx) en la última hora:
```
sum(increase(http_requests_total{namespace="amael-ia", status="5xx"}[1h])) by (handler)
```

### Latencia promedio del /api/chat (p99):
```
histogram_quantile(0.99, sum(rate(http_request_duration_seconds_bucket{namespace="amael-ia", handler="/api/chat"}[5m])) by (le))
```

## Métricas del Agente Amael-IA (prefijo amael_)

### Latencia del planner (p99):
```
histogram_quantile(0.99, rate(amael_planner_latency_seconds_bucket[5m]))
```

### Pasos del plan ejecutados por tipo:
```
sum(increase(amael_executor_steps_total[1h])) by (step_type)
```

### Errores del executor por tipo de paso:
```
sum(increase(amael_executor_errors_total[1h])) by (step_type)
```

### Tamaño promedio de los planes generados:
```
histogram_quantile(0.5, rate(amael_planner_plan_size_steps_bucket[1h]))
```

### Truncados de contexto (señal de prompts muy largos):
```
sum(increase(amael_executor_context_truncations_total[1h]))
```

### Decisiones del supervisor (ACCEPT vs REPLAN):
```
sum(increase(amael_supervisor_decisions_total[1h])) by (decision)
```

### Score promedio del supervisor:
```
histogram_quantile(0.5, rate(amael_supervisor_quality_score_bucket[1h]))
```

## Métricas de CPU y RAM por pod (namespace amael-ia)

### CPU usage por pod:
```
sum(rate(container_cpu_usage_seconds_total{namespace="amael-ia", container!=""}[5m])) by (pod)
```

### RAM usage por pod:
```
sum(container_memory_working_set_bytes{namespace="amael-ia", container!=""}) by (pod)
```

## Endpoints disponibles (handlers)
- /api/chat          → conversaciones del usuario
- /api/k8s-agent     → llamadas al agente K8s
- /api/sre/query     → consultas RAG del k8s-agent
- /api/auth/login    → inicio de sesión OAuth
- /api/auth/callback → callback OAuth Google
- /metrics           → scraping de Prometheus (ignorar en conteos de usuario)
- /api/ingest        → ingesta de documentos

---

## Diagnóstico de Dashboards Grafana sin datos

### IMPORTANTE: comportamiento esperado tras reinicio de backend-ia

Los dashboards 1 (LLM & HTTP), 2 (Pipeline), 5 (Supervisor), 6 (Seguridad) y
7 (Service Map) usan métricas **activity-driven**: solo se generan cuando hay
conversaciones activas en /api/chat. Tras un reinicio de backend-ia, los contadores
se resetean a 0 y los dashboards quedan vacíos hasta que haya nuevo tráfico.

Los dashboards 3 (RAG), 4 (Infra) y 8 (SRE) siempre tienen datos porque usan
métricas de fondo (GPU, CPU, loop SRE).

### Paso 1 — Verificar si Prometheus scrapeea backend-ia:
```
up{namespace="amael-ia", pod=~"backend-ia.*"}
```
Resultado esperado: 1. Si es 0, el pod no está disponible o le faltan anotaciones de scraping.

### Paso 2 — Verificar si existen métricas del agente:
```
amael_planner_latency_seconds_count
```
```
amael_executor_steps_total
```
```
amael_supervisor_decisions_total
```
```
amael_tool_calls_total
```
Si todos retornan 0 o vacío → backend-ia reinició recientemente y no ha habido tráfico.
**Solución: enviar un mensaje de prueba al chat para generar métricas.**

### Paso 3 — Verificar tráfico activo al chat:
```
rate(http_requests_total{namespace="amael-ia", handler="/api/chat"}[5m])
```
Si es 0 → no hay tráfico reciente, los dashboards de rate() estarán vacíos.

### Paso 4 — Verificar integridad del ConfigMap de dashboards:
```
# Usar Listar_Pods para verificar el sidecar de Grafana en namespace observability
# Pod: kube-prometheus-stack-grafana-*
# Container: grafana-sc-dashboard (sidecar que recarga los dashboards)
```

### Mapa dashboard → métrica crítica de diagnóstico:

| Dashboard | Query de diagnóstico | Fuente |
|---|---|---|
| 1 LLM & HTTP | `amael_planner_latency_seconds_count` | backend-ia/agents/metrics.py |
| 2 Pipeline | `amael_executor_steps_total` | backend-ia/agents/metrics.py |
| 5 Supervisor | `amael_supervisor_decisions_total` | backend-ia/agents/metrics.py |
| 6 Seguridad | `amael_security_rate_limited_total` | backend-ia/main.py |
| 7 Service Map | `amael_tool_calls_total` | backend-ia/main.py |
| 8 SRE Autónomo | `amael_sre_loop_runs_total` | k8s-agent/main.py |

### Regenerar el ConfigMap de dashboards (si está corrupto):
```bash
cd GitOps-Infra/observability
python3 generate_dashboards_cm.py  # o ejecutar el script inline del CLAUDE.md
kubectl apply -f k8s/04-custom-dashboards-cm.yaml
# El sidecar de Grafana recarga automáticamente en ~30 segundos
```
