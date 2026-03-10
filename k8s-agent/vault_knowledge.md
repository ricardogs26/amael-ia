# Vault Knowledge Base — Amael-IA Cluster

## Despliegue
- HashiCorp Vault 1.21.2, Helm chart vault-0.32.0
- Namespace: `vault`, StatefulSet: `vault-0` (1 réplica)
- Storage backend: Raft integrado, PVC 1Gi (storageClass: vault-storage), datos en /vault/data
- TLS interno: DESHABILITADO (comunicación interna K8s)
- UI habilitada, Ingress en ingressClass `public`
- Vault Agent Injector habilitado (MutatingWebhook activo, pero no se usa sidecar actualmente)

## Estado de Sellado (Sealed)
- Vault arranca SELLADO tras reinicio del pod. Requiere desellado manual con 3 de 5 claves Shamir.
- Cuando está sellado: pods que usan Vault fallan con error 503/auth failure, pod vault-0 en estado Not Ready (readinessProbe falla con exit code 2).
- Claves de unseal y root token guardados en: /home/richardx/k8s-lab/vault.root (NUNCA en git).
- Root Token: <REDACTED_ROOT_TOKEN>

## Dessellar Vault
```bash
kubectl port-forward -n vault svc/vault 8200:8200 &
export VAULT_ADDR="http://localhost:8200"
vault operator unseal <KEY_1>   # necesita 3 de 5
vault operator unseal <KEY_2>
vault operator unseal <KEY_3>
vault status  # verificar Sealed: false
```

## Autenticación Kubernetes (JWT Flow)
- Método habilitado: `auth/kubernetes` en Vault
- Configurado con: CA cert del clúster MicroK8s, host K8s API, issuer=https://kubernetes.default.svc.cluster.local
- Cada pod presenta su JWT (/var/run/secrets/kubernetes.io/serviceaccount/token) → Vault valida con K8s TokenReview API → emite Vault Token (TTL 1h)

## Secrets Engine
- KV v2 montado en path `secret/`
- Soporta historial de versiones, soft delete, metadata separada

## Secrets Almacenados Actualmente

### Google OAuth Tokens (por usuario)
- Path: `secret/data/amael/google-tokens/{email_sanitizado}`
- Sanitización: `@` → `_at_`, `.` → `_dot_`
- Ejemplo: richard@gmail.com → `secret/data/amael/google-tokens/richard_at_gmail_dot_com`
- Contenido: token, refresh_token, token_uri, client_id, client_secret, scopes (calendar + gmail.readonly), expiry
- Servicio que los usa: `productivity-service` (namespace amael-ia)
- Metadata path: `secret/metadata/amael/google-tokens/*`

## Política: amael-productivity-policy
```
path "secret/data/amael/google-tokens/*"     → create, read, update, delete
path "secret/metadata/amael/google-tokens/*" → list, delete
```

## Rol Kubernetes: amael-productivity
- ServiceAccount permitido: `productivity-service-sa`
- Namespace permitido: `amael-ia`
- Política asignada: `amael-productivity-policy`
- TTL token: 1 hora

## ServiceAccounts
- `productivity-service-sa` (amael-ia): usado por productivity-service-deployment para autenticarse en Vault
- `vault` (vault): SA del servidor Vault, ClusterRoleBinding a system:auth-delegator
- `vault-agent-injector` (vault): SA del injector webhook

## Servicios K8s del namespace vault
- `vault` (ClusterIP :8200/:8201) → endpoint general para pods del clúster
- `vault-active` (ClusterIP) → solo nodo activo HA
- `vault-internal` (Headless) → comunicación Raft interna
- `vault-ui` (ClusterIP :8200) → UI web
- DNS interno: `vault.vault.svc.cluster.local:8200`

## Código Python (hvac) — productivity-service
- Archivo: `productivity-service/app/services/auth_service.py`
- Lee JWT desde /var/run/secrets/kubernetes.io/serviceaccount/token
- Autentica con rol `amael-productivity` vía `client.auth.kubernetes.login()`
- Lee/escribe con `client.secrets.kv.v2.read_secret_version()` y `create_or_update_secret()`
- Config en: `productivity-service/app/core/config.py` (vault_addr, vault_role)
- VAULT_ADDR: http://vault.vault.svc.cluster.local:8200
- VAULT_ROLE: amael-productivity

## Scripts de configuración
- `GitOps-Infra/vault/06-vault_k8s_auth_setup.sh` → Habilita y configura auth/kubernetes
- `GitOps-Infra/vault/07-create_secret_and_role.sh` → Secreto/rol de prueba (mi-app)
- `GitOps-Infra/vault/08-productivity-vault-setup.sh` → Política y rol para productivity-service

## Comandos Operacionales
```bash
# Ver estado
kubectl exec -n vault vault-0 -- vault status

# Listar secretos
vault kv list secret/amael/google-tokens

# Leer secreto de un usuario
vault kv get secret/amael/google-tokens/richard_at_gmail_dot_com

# Borrar secreto de un usuario (soft delete)
vault kv delete secret/amael/google-tokens/{email_sanitizado}

# Borrar permanentemente
vault kv metadata delete secret/amael/google-tokens/{email_sanitizado}

# Listar políticas
vault policy list

# Listar roles kubernetes
vault list auth/kubernetes/role

# Ver rol específico
vault read auth/kubernetes/role/amael-productivity

# Verificar auth de productivity-service
kubectl exec -n amael-ia deployment/productivity-service-deployment -- \
  sh -c 'curl -s -X POST http://vault.vault.svc.cluster.local:8200/v1/auth/kubernetes/login \
  -d "{\"role\":\"amael-productivity\",\"jwt\":\"$(cat /var/run/secrets/kubernetes.io/serviceaccount/token)\"}"'
```

## Troubleshooting Común
- **Error 503 / Vault is sealed**: Vault reinició, dessellar con 3 claves del archivo vault.root
- **permission denied**: Política no cubre el path, o el SA del pod no coincide con el rol
- **productivity-service no autentica**: Verificar que usa serviceAccountName: productivity-service-sa
- **InvalidPath al leer token de usuario**: Primera vez del usuario → OK, inicia flujo OAuth
- **vault-0 Not Ready**: Está sellado, readinessProbe falla con exit 2 → dessellar
