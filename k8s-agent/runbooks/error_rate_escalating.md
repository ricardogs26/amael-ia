# Runbook: ERROR_RATE_ESCALATING / HIGH_ERROR_RATE

## Descripción
Tasa de errores 5xx en el backend está escalando de forma sostenida (detectado por `deriv()` en Prometheus).
Puede indicar degradación progresiva, dependency failure, o un reinicio reciente del pod.

## Patrones conocidos

### Patrón 1: DCGM / GPU dependency (más frecuente en este cluster)
- **Síntoma**: ERROR_RATE_ESCALATING o HIGH_ERROR_RATE en `/api/chat`, `/api/conversations`
- **Causa raíz**: `gpu-metrics-dcgm-exporter` en namespace `observability` tiene un CrashLoop
  o el driver NVIDIA falla al iniciar. El backend (`amael-agentic-backend`) depende de Ollama
  que a su vez depende de la GPU. Si DCGM no reporta la GPU, Ollama puede quedar degradado.
- **Verificación**:
  ```
  kubectl get pods -n observability -l app.kubernetes.io/name=dcgm-exporter
  kubectl logs -n observability daemonset/gpu-metrics-dcgm-exporter --tail=30
  ```
- **Acción**: ROLLOUT_RESTART del backend resuelve el error rate mientras DCGM se recupera.
  Si persiste >10min después del restart, verificar Ollama y la GPU directamente.

### Patrón 2: Reinicio del pod durante deploy
- **Síntoma**: HIGH_ERROR_RATE 100% durante 1-2 minutos, luego se normaliza
- **Causa**: El pod nuevo aún no está Ready y hay requests entrando
- **Acción**: Monitorear 5min — si se normaliza solo, no requiere intervención.
  El SRE agent tiene dedup de 10min para este tipo.

### Patrón 3: Memory pressure / OOMKill en el backend
- **Síntoma**: ERROR_RATE_ESCALATING + HIGH_MEMORY en el mismo pod
- **Causa**: El proceso de Python empieza a swap/fallar por falta de memoria
- **Acción**: PATCH_RESOURCES para aumentar memory limit, luego ROLLOUT_RESTART

## Acción recomendada (auto-heal)
1. Verificar que `amael-agentic-deployment` está en estado Running
2. Ejecutar ROLLOUT_RESTART: `kubectl rollout restart deployment/amael-agentic-deployment -n amael-ia`
3. Si no se resuelve en 5min: escalar 1 réplica adicional (SCALE_UP)
4. Si tasa de error > 50% sostenida por >15min: NOTIFY_HUMAN

## Métricas Prometheus relevantes
```promql
# Error rate actual por endpoint
sum by (handler)(rate(amael_http_requests_total{namespace="amael-ia",status_code=~"5.."}[5m]))
/ sum by (handler)(rate(amael_http_requests_total{namespace="amael-ia"}[5m]))

# Tendencia de escalada
deriv(sum(rate(amael_http_requests_total{namespace="amael-ia",status_code=~"5.."}[5m]))[15m:1m])

# Estado de DCGM
up{job="gpu-metrics-dcgm-exporter"}
```

## Historial
- 2026-03-27: Primera ocurrencia documentada. LLM diagnosticó CrashLoopBackOff por DCGM.
  Confianza 85%. Causa DEPENDENCY. Acción: ROLLOUT_RESTART.
