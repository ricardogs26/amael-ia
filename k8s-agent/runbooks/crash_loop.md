# Runbook: CrashLoopBackOff

## Descripción
Un pod en CrashLoopBackOff indica que el contenedor arranca, falla, y Kubernetes lo reinicia repetidamente con backoff exponencial (10s, 20s, 40s, ..., 5min).

## Causas más comunes

### 1. Dependencia caída (causa #1 en Amael-IA)
El servicio intenta conectar a PostgreSQL, Redis, Qdrant u Ollama en el arranque y falla.
- Síntoma en logs: `connection refused`, `dial tcp`, `ECONNREFUSED`
- Acción: verificar que postgres-service, redis-service, qdrant-service estén Running.
- Remediación: rollout restart del servicio afectado DESPUÉS de que la dependencia esté healthy.

### 2. Ollama no disponible o sin GPU (crítico para amael-agentic-backend)
`amael-agentic-backend` depende de Ollama para inferencia LLM y embeddings. Si Ollama no está Ready,
el backend puede fallar en el arranque al hacer warm-up del LLM.
- Verificar estado de Ollama:
  ```
  kubectl get pod -n amael-ia -l app=ollama
  kubectl logs deployment/ollama-deployment -n amael-ia --tail=30
  ```
- Ollama requiere la GPU RTX 5070 exclusivamente. Si hay otro pod ocupando el GPU, Ollama quedará Pending.
  ```
  kubectl describe node lab-home | grep -A5 nvidia
  kubectl get pods -A -o json | grep -i "nvidia.com/gpu"
  ```
- Si Ollama crasheó: NO usar `rollout restart` (deja el pod nuevo Pending con GPU busy).
  Usar en cambio: `kubectl delete pod -l app=ollama -n amael-ia`
- Verificar que `nomic-embed-text` esté cargado (necesario para Qdrant RAG):
  ```
  kubectl exec deploy/ollama-deployment -n amael-ia -- ollama list
  ```
  Si falta: `kubectl exec deploy/ollama-deployment -n amael-ia -- ollama pull nomic-embed-text`
- Si Ollama no está disponible, el k8s-agent usará diagnóstico determinístico como fallback (sin LLM).

### 3. Variable de entorno faltante o incorrecta
El proceso falla al arrancar porque no encuentra una env var obligatoria.
- Síntoma en logs: `ValueError: missing required env`, `KeyError`, `raise ValueError`
- Acción: revisar el ConfigMap y los Secrets del deployment con `kubectl describe pod`.
- Remediación: corregir el ConfigMap/Secret y hacer rollout restart.

### 4. Error de código (bug en arranque)
El proceso lanza una excepción durante la inicialización.
- Síntoma en logs: `Traceback (most recent call last)`, `ImportError`, `ModuleNotFoundError`
- Acción: revisar los logs completos con `kubectl logs pod --previous`.
- Remediación: corregir el código, hacer build y deploy de nueva imagen.

### 5. OOMKilled en arranque
El pod es terminado por exceder el memory limit antes de estar Ready.
- Síntoma: `last_state.terminated.reason = OOMKilled`
- Acción: ver el runbook oom_killed.md
- Remediación: aumentar memory limit o reducir uso de memoria en arranque.

### 6. Liveness probe muy agresiva
El pod arranca lento y la liveness probe lo mata antes de estar listo.
- Síntoma en events: `Liveness probe failed`, `Back-off restarting failed container`
- Acción: revisar `initialDelaySeconds` en el deployment.
- Remediación: aumentar `initialDelaySeconds` o `failureThreshold`.

## Pasos de diagnóstico
```
1. kubectl logs <pod> --previous -n amael-ia         # logs del container anterior
2. kubectl describe pod <pod> -n amael-ia             # events y last state
3. kubectl get pods -n amael-ia                       # ver si dependencias están Running
4. kubectl get pod -n amael-ia -l app=ollama          # verificar Ollama específicamente
5. kubectl describe node lab-home | grep -A5 nvidia   # estado GPU RTX 5070
```

## Verificación en Grafana
- Dashboard `amael-backend` (uid: `amael-backend`): golden signals del backend, latencia, errores.
- Dashboard `amael-infra` (uid: `amael-infra`): estado GPU, memoria, CPU por pod.
- PromQL para confirmar CrashLoop:
  ```promql
  kube_pod_container_status_restarts_total{namespace="amael-ia"} > 3
  ```
- PromQL para ver si Ollama está respondiendo:
  ```promql
  up{job="ollama", namespace="amael-ia"}
  ```

## Cascada de fallos típica en Amael-IA
Si Ollama falla → amael-agentic-backend pierde LLM → respuestas degradadas (fallback determinístico).
Si PostgreSQL falla → amael-agentic-backend crashea (sin DB no puede arrancar).
Si Redis falla → SRE loop se detiene (dedup falla) + sesiones de chat perdidas.
Si llm-adapter falla → k8s-agent pierde diagnóstico LLM (usa fallback determinístico).

## Remediación automática
- Si causa raíz es dependencia caída: esperar que dependencia sane, luego rollout restart.
- Si causa es código: NO hacer rollout restart (no ayuda). Escalar a humano.
- Máximo 3 reinicios automáticos en 15 minutos. Tras eso, escalar a humano.

## Servicios afectados en Amael-IA (frecuencia histórica)
- `amael-agentic-backend`: frecuente, típicamente por PostgreSQL, Redis o Ollama no ready
- `productivity-service-deployment`: Vault sealed o sin credenciales OAuth
- `k8s-agent-deployment`: raro, típicamente por Ollama no disponible o llm-adapter caído
- `ollama-deployment`: GPU conflict o DiskPressure (modelos ~9GB por modelo)
