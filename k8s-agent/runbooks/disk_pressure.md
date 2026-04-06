# Runbook: DiskPressure en Nodo

## Descripción
El nodo reporta DiskPressure=True cuando el espacio disponible en disco cae por debajo del umbral configurado (por defecto 10% o 1.5Gi libres). Kubernetes puede comenzar a evict pods para liberar espacio.

## Síntomas
- `kubectl describe node lab-home` muestra `DiskPressure: True`
- Los pods pueden ser evicted con reason `Evicted`
- Nuevas imágenes no pueden ser descargadas (ImagePullBackOff)
- Pods en estado Pending sin motivo aparente de recursos

## Principales consumidores de disco en Amael-IA

### 1. Modelos de Ollama (mayor consumidor — hasta 30GB+)
Los modelos LLM son el principal riesgo de DiskPressure en este clúster.
- `qwen2.5:14b` ≈ 9GB
- `nomic-embed-text` ≈ 274MB
- Listar modelos cargados: `kubectl exec deploy/ollama-deployment -n amael-ia -- ollama list`
- Ver espacio usado por Ollama:
  ```bash
  sudo du -sh /var/snap/microk8s/common/default-storage/amael-ia-ollama*
  ```
- NO borrar `qwen2.5:14b` ni `nomic-embed-text` — son los modelos activos del sistema.
- Si hay modelos obsoletos (glm4, llama3, etc.):
  `kubectl exec deploy/ollama-deployment -n amael-ia -- ollama rm <modelo>`

### 2. Imágenes de Docker/containerd acumuladas
Las imágenes antiguas de builds anteriores no se limpian automáticamente.
- Diagnóstico: `sudo du -sh /var/lib/containerd/`
- Ver imágenes sin uso: `sudo crictl images`
- Remediación: `sudo crictl rmi --prune`
- Imágenes del registry privado (registry.richardx.dev) se acumulan con cada rebuild.
  Limpiar versiones antiguas de: mirofish-backend, mirofish-frontend, amael-agentic-backend, k8s-agent, etc.

### 3. Logs de pods acumulados
Los pods con muchos reinicios (CrashLoop) generan logs grandes.
- Path: `/var/log/pods/`
- Ver los más grandes: `sudo du -sh /var/log/pods/* | sort -rh | head -10`
- Remediación: `sudo find /var/log/pods -name "*.log" -mtime +3 -delete`

### 4. PVCs de bases de datos crecientes
PostgreSQL, Qdrant y MinIO crecen con el uso normal del sistema.
- Ver tamaño actual: `kubectl get pvc -n amael-ia`
- MinIO almacena backups de documentos ingestados — puede crecer rápido con muchos PDFs.
- Qdrant almacena vectores por usuario — crece con cada documento ingestado.
- PostgreSQL almacena historial de chats, incidentes SRE y postmortems.
- Diagnóstico detallado:
  ```bash
  kubectl exec deploy/postgres -n amael-ia -- psql -U postgres -c "\l+"
  kubectl exec deploy/minio -n amael-ia -- mc du local/
  ```

### 5. Directorio de uploads de MiroFish / servicios
Si MiroFish o el backend tienen uploads sin limpiar:
- `kubectl exec deploy/mirofish-backend -n mirofish -- du -sh /app/uploads/`
- Los uploads de MiroFish son temporales — se pueden limpiar PVCs si el namespace es de prueba.

### 6. Snapshots de MicroK8s
- Path: `/var/snap/microk8s/`
- Remediación: `sudo snap remove microk8s --revision=<old>` para limpiar snapshots antiguos.

## Script de limpieza de emergencia
```bash
# 1. Ver estado general del disco
sudo df -h

# 2. Ver top consumidores
sudo du -sh /var/snap/microk8s/common/* 2>/dev/null | sort -rh | head -10
sudo du -sh /var/log/pods/* 2>/dev/null | sort -rh | head -10
sudo du -sh /var/lib/containerd/* 2>/dev/null | sort -rh | head -5

# 3. Limpiar imágenes no usadas (seguro)
sudo crictl rmi --prune

# 4. Limpiar logs de pods antiguos (>3 días)
sudo find /var/log/pods -name "*.log" -mtime +3 -delete

# 5. Ver modelos Ollama cargados
kubectl exec deploy/ollama-deployment -n amael-ia -- ollama list
```

## Umbrales de alerta recomendados
| Umbral | Acción |
|--------|--------|
| >70% disco | Alerta temprana — revisar y limpiar imágenes viejas |
| >80% disco | Alerta alta — limpiar logs + imágenes + revisar Ollama |
| >90% disco | Crítico — DiskPressure inminente, limpiar inmediatamente |

PromQL para monitorear:
```promql
# Uso de disco del nodo lab-home
(node_filesystem_size_bytes{mountpoint="/"} - node_filesystem_free_bytes{mountpoint="/"})
/ node_filesystem_size_bytes{mountpoint="/"} * 100
```

## Acción del agente SRE
DiskPressure siempre requiere notificación al humano.
El agente no puede ejecutar comandos en el nodo (no tiene acceso SSH).
Sí puede ejecutar `kubectl exec` para limpiar uploads de aplicaciones o borrar modelos Ollama obsoletos.
Umbrales configurados para alertas tempranas en `sre-agent-policy` ConfigMap.
