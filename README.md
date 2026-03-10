![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Kubernetes](https://img.shields.io/badge/kubernetes-%23326ce5.svg?logo=kubernetes&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=flat&logo=fastapi)
![LangChain](https://img.shields.io/badge/LangChain-1C3C3C?style=flat&logo=langchain)

# Amael IA 🧠🤖

> **Amael IA** es una plataforma avanzada de Inteligencia Artificial Autónoma y Multi-Agente enfocada en la asistencia conversacional y la administración automatizada de infraestructuras (DevOps).

Desplegada completamente sobre Kubernetes, Amael IA utiliza una arquitectura de orquestación basada en **LangGraph** siguiendo el patrón **Planner → Grouper → Batch Executor → Supervisor**. Esto le permite descomponer tareas complejas, ejecutar herramientas en paralelo y auto-corregirse mediante una capa de retroalimentación de calidad.

---

## ✨ Características Principales

*   💬 **Interfaz Conversacional:** Acceso mediante un frontend web moderno en **Streamlit** y conectividad nativa vía **WhatsApp** (`whatsapp-bridge` v1.2.4), con historiales y RAG **aislados por usuario**.
*   🔒 **Seguridad & Hardening:** Autenticación **Google OAuth**, encriptado con Vault, validación de prompts anti-inyección, rate limiting mediante Redis y sanitización de outputs.
*   🧠 **RAG Multiusuario:** Ingesta de documentos con vectorización en **Qdrant**. Cada usuario cuenta con memoria contextual aislada.
*   🛠️ **DevOps Autónomo (K8s SRE Agent):** Administra el clúster en tiempo real. Lista pods, revisa logs, consulta PromQL/Grafana y ejecuta acciones correctivas (`Eliminar_Pod`).
*   📅 **Productividad Integrada:** Automatización de agenda mediante integración con **Google Calendar** y **Gmail API** (`productivity-service`).
*   📊 **Observabilidad Full-Stack:** Monitoreo con **Prometheus, Grafana y Tempo**. Incluye un **Service Map** en tiempo real y dashboards especializados de seguridad y performance.

---

## 🏗️ Arquitectura de Microservicios

Amael IA orquestado por **Kubernetes (MicroK8s)** con imágenes en registro privado `registry.richardx.dev`.

### 🧠 Capa de Inferencia (Single NVIDIA RTX 5070)
*   **LLM Principal:** `qwen2.5:14b` (alojado en Ollama).
*   **Embeddings:** `nomic-embed-text` (alojado en Ollama) - 768 dim.
*   **Voz (TTS):** `CosyVoice-300M` (alojado en `cosyvoice-service`).

### Componentes Core:

| Servicio | Versión | Descripción |
|---------|---------|-------------|
| `backend-ia` | `2.11.0` | Orquestador LangGraph, FastAPI. |
| `k8s-agent` | `1.6.0` | SRE Expert, automatización K8s + Vault. |
| `productivity-service` | `1.2.0` | Integración Google Workspace. |
| `frontend-ia` | `2.0.0` | Streamlit Rich-UI, system-token theming. |
| `whatsapp-bridge` | `1.2.4` | Puppeteer bridge con reintentos y timeout extendido. |
| `llm-adapter` | `1.0.0` | Proxy OpenAI-compatible hacia Ollama. |

---

### Diagrama de Flujo y Conectividad

```mermaid
graph TD
    subgraph "Interfaces"
        UI[Frontend Streamlit]
        WA[WhatsApp App]
    end

    subgraph "Core API Layer"
        BE[Backend IA - FastAPI]
        DB[(PostgreSQL / Redis)]
        VS[(Qdrant Vector DB)]
        VA[(HashiCorp Vault)]
    end

    subgraph "Agent Orchestrator (LangGraph)"
        BE --> PLAN[Planner]
        PLAN --> GRP[Grouper]
        GRP --> EXEC[Batch Executor]
        EXEC --> SUP[Supervisor]
        SUP -- REPLAN --> PLAN
    end

    subgraph "Expert Agents"
        K8S[K8s Agent - SRE]
        PROD[Productivity - Google]
    end

    subgraph "Inference Providers"
        OL[Ollama - qwen2.5:14b]
        TF[TF Serving]
        CV[CosyVoice - TTS]
    end

    %% Connectivity
    UI & WA <--> BE
    BE <--> DB & VS & VA
    BE <--> OL & TF
    EXEC --> K8S & PROD
    K8S -->|Action| KUBE[K8s API]
    K8S -->|Metrics| PROM[Prometheus / Grafana]
    PROD -->|Sync| GAPI[Google API]
```

---

## 🚀 Despliegue (Manual CI/CD)

```bash
# 1. Build & Push
docker build -t registry.richardx.dev/<service>:<tag> ./<service>/
docker push registry.richardx.dev/<service>:<tag>

# 2. Deploy
kubectl apply -f k8s/<manifest>.yaml -n amael-ia
kubectl rollout restart deployment <service> -n amael-ia
```

## 🔐 Seguridad y Privacidad
*   **Vault Integration:** Tokens de Google OAuth se almacenan cifrados por usuario.
*   **RBAC strico:** El agente de K8s está restringido al namespace `amael-ia`.
*   **Output Sanitization:** Redacción automática de tokens `hvs.*`, JWTs y passwords en las respuestas del bot.
