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
