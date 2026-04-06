# Runbook: OOMKilled

## Descripción
OOMKilled (Out Of Memory Killed) ocurre cuando un contenedor supera su `resources.limits.memory`. El kernel de Linux mata el proceso con SIGKILL y Kubernetes registra el motivo como OOMKilled.

## Síntomas
- `kubectl describe pod` muestra: `Last State: Terminated (OOMKilled)`
- `exit_code: 137` (128 + SIGKILL)
- El pod se reinicia y eventualmente entra en CrashLoopBackOff si la causa persiste.

## Causas en Amael-IA

### 1. Carga inusual de datos (productivity-service)
La integración con Gmail/Calendar procesa un volumen grande de emails o eventos.
- Síntoma en logs: `Processing X emails`, `Memory allocation failed`
- Memory limit actual: 512Mi
- Remediación: aumentar limit a 1Gi y añadir paginación en la API de Google.

### 2. Modelo LLM en memoria (amael-agentic-backend, k8s-agent)
Carga de contexto muy grande satura la memoria del proceso Python.
- Síntoma: log de la conversación era extremadamente larga
- Memory limit amael-agentic-backend: verificar con `kubectl get pod <pod> -o jsonpath='{.spec.containers[].resources}'`
- Remediación: rollout restart + revisar MAX_CONTEXT_CHARS en ConfigMap.
- Si el OOM ocurre durante inferencia LLM: el contexto enviado a Ollama era demasiado grande.
  Verificar `LLM_MAX_TOKENS` en el ConfigMap del backend.

### 3. GPU RTX 5070 y Ollama — OOM en VRAM vs RAM
Ollama usa la GPU RTX 5070 para inferencia. Si el modelo no cabe en VRAM (12GB para RTX 5070),
Ollama hace fallback a CPU y puede saturar RAM del sistema.
- qwen2.5:14b requiere ~9GB VRAM — cabe en RTX 5070.
- Si se intenta cargar dos modelos simultáneamente, el segundo causa OOM en VRAM.
- Diagnóstico GPU:
  ```
  kubectl describe node lab-home | grep -A10 "Allocated resources"
  kubectl exec deploy/ollama-deployment -n amael-ia -- ollama ps
  ```
- Si Ollama está usando CPU fallback: `nvidia-smi` en el nodo mostrará 0% GPU utilization durante inferencia.
- Remediación: asegurarse de que solo `qwen2.5:14b` + `nomic-embed-text` estén cargados.
  `nomic-embed-text` es pequeño (~274MB) y coexiste bien con qwen2.5:14b en la RTX 5070.

### 4. Memory leak acumulativo
El pod lleva muchas horas corriendo y la memoria creció gradualmente.
- Diagnóstico: ver métricas `container_memory_working_set_bytes` en Prometheus.
- Remediación: rollout restart (fix temporal) + investigar el leak.

### 5. Qdrant o Redis procesando datos masivos
- Remediación: rollout restart del pod afectado.

## Pasos de diagnóstico
```
1. kubectl describe pod <pod> -n amael-ia         # confirmar OOMKilled
2. kubectl logs <pod> --previous -n amael-ia      # qué estaba haciendo antes de morir
3. Prometheus query: max_over_time(container_memory_working_set_bytes{pod=~"<pod>.*"}[1h])
4. Comparar con limits: kubectl get pod <pod> -o jsonpath='{.spec.containers[].resources}'
5. kubectl describe node lab-home | grep -A10 "Allocated resources"   # ver uso real del nodo
6. kubectl exec deploy/ollama-deployment -n amael-ia -- ollama ps     # modelos cargados en GPU
```

## Remediación automática
- Rollout restart inmediato (libera memoria, pod arranca fresco).
- El fix permanente requiere: aumentar limit O reducir uso de memoria en código.
- Si el pod vuelve a OOMKilled en <15min, escalar a humano (aumentar limit manualmente).
- EXCEPCIÓN: si es Ollama el afectado, usar `kubectl delete pod -l app=ollama -n amael-ia`
  en lugar de rollout restart (evita conflicto de GPU con pod en Terminating).

## PromQL útiles
```promql
# Uso de memoria actual vs limit
sum(container_memory_working_set_bytes{namespace="amael-ia"}) by (pod)

# Tendencia de memoria en la última hora
rate(container_memory_working_set_bytes{namespace="amael-ia"}[5m])

# Detectar OOMKilled en los últimos 30 minutos
kube_pod_container_status_last_terminated_reason{reason="OOMKilled", namespace="amael-ia"}

# GPU utilization (requiere DCGM exporter)
DCGM_FI_DEV_GPU_UTIL{namespace="amael-ia"}
```

## Dashboard Grafana
- `amael-infra` (uid: `amael-infra`): panel "Memory Usage by Pod" — muestra uso vs límite.
- `amael-backend` (uid: `amael-backend`): golden signals — ver si hay spike de requests antes del OOM.
