// ── Catálogo de agentes (fuente de verdad) ───────────────────────────────────
// Archivo de datos — excluido de CPD en sonar-project.properties

export type AgentType = 'orchestrator' | 'pipeline' | 'direct' | 'autonomous'

export interface AgentDef {
  key: string
  name: string
  role: string
  subtitle: string
  color: string
  icon: string
  description: string
  type: AgentType
  module: string
  capabilities: string[]
}

export const AGENT_CATALOG: AgentDef[] = [
  {
    key: 'amael',
    name: 'Amael',
    role: 'Orchestrator',
    subtitle: 'Cerebro central del sistema',
    color: '#6366f1',
    icon: '✦',
    description: 'Coordina el pipeline LangGraph completo. Compila el grafo, inyecta herramientas por request y dirige el flujo Planner → Executor → Supervisor.',
    type: 'orchestrator',
    module: 'orchestration/workflow_engine.py',
    capabilities: ['LangGraph', 'Graph caching', 'Per-request tool injection', 'Intent routing'],
  },
  {
    key: 'sariel',
    name: 'Sariel',
    role: 'Planner',
    subtitle: 'Generación de planes JSON',
    color: '#8b5cf6',
    icon: '◈',
    description: 'Recibe la pregunta del usuario y genera un plan JSON de pasos (máx 8). Cada paso tiene tipo, descripción y dependencias.',
    type: 'pipeline',
    module: 'agents/planner/agent.py',
    capabilities: ['Plan JSON', 'MAX_PLAN_STEPS=8', 'Grouper pipeline', 'Step sequencing'],
  },
  {
    key: 'executor',
    name: 'Executor',
    role: 'Batch Executor',
    subtitle: 'Ejecución paralela de pasos',
    color: '#06b6d4',
    icon: '⬡',
    description: 'Ejecuta batches generados por el Grouper. Pasos de herramientas en paralelo (ThreadPoolExecutor), pasos REASONING de forma secuencial.',
    type: 'pipeline',
    module: 'agents/executor/agent.py',
    capabilities: ['Parallel tool execution', 'Sequential REASONING', 'ThreadPoolExecutor', 'Step handlers'],
  },
  {
    key: 'remiel',
    name: 'Remiel',
    role: 'Supervisor',
    subtitle: 'Control de calidad 0–10',
    color: '#f59e0b',
    icon: '◎',
    description: 'Evalúa la respuesta del Executor con score 0–10. Si el score < 6, emite REPLAN (máx 1 retry). Si ≥ 6, acepta y finaliza.',
    type: 'pipeline',
    module: 'agents/supervisor/agent.py',
    capabilities: ['Quality scoring', 'REPLAN logic', 'Max 1 retry', 'ACCEPT/REPLAN decision'],
  },
  {
    key: 'sandalphon',
    name: 'Sandalphon',
    role: 'Researcher',
    subtitle: 'RAG + búsqueda web',
    color: '#10b981',
    icon: '◉',
    description: 'Recupera documentos de Qdrant (por usuario) con filtro semántico por nombre de archivo. Reranking via cosine similarity. Fallback a DuckDuckGo.',
    type: 'direct',
    module: 'agents/researcher/agent.py',
    capabilities: ['Qdrant RAG', 'Filename filter', 'Cosine reranking', 'DuckDuckGo fallback'],
  },
  {
    key: 'haniel',
    name: 'Haniel',
    role: 'Productivity',
    subtitle: 'Google Calendar & Gmail',
    color: '#3b82f6',
    icon: '◇',
    description: 'Gestiona eventos de Google Calendar y Gmail. Lee OAuth tokens de Vault (secret/data/amael/google-tokens/*) en cada request.',
    type: 'direct',
    module: 'agents/productivity/agent.py',
    capabilities: ['Google Calendar', 'Gmail integration', 'Vault OAuth tokens', 'Event management'],
  },
  {
    key: 'raphael',
    name: 'Raphael',
    role: 'SRE Agent',
    subtitle: 'Loop autónomo cada 60s',
    color: '#ef4444',
    icon: '⬢',
    description: 'Ciclo Observe → Detect → Diagnose → Decide → Act → Report cada 60s vía APScheduler. Maneja CrashLoops, OOM, SLO burn rates y tendencias predictivas.',
    type: 'autonomous',
    module: 'agents/sre/agent.py',
    capabilities: ['60s loop', 'Auto-heal', 'Predictive trends', 'SLO tracking', 'LLM postmortems'],
  },
  {
    key: 'gabriel',
    name: 'Gabriel',
    role: 'Dev Agent',
    subtitle: 'Desarrollo autónomo en GitHub',
    color: '#22d3ee',
    icon: '◐',
    description: 'Desarrolla features de forma autónoma: analiza el request, genera código, crea ramas, hace commits y abre Pull Requests en GitHub.',
    type: 'direct',
    module: 'agents/dev/agent.py',
    capabilities: ['GitHub PRs', 'Feature branches', 'Code generation', 'Autonomous commits'],
  },
  {
    key: 'camael',
    name: 'Camael',
    role: 'DevOps Agent',
    subtitle: 'CI/CD y operaciones K8s',
    color: '#f97316',
    icon: '⬟',
    description: 'Gestiona pipelines CI/CD, configuraciones de Kubernetes y operaciones de entrega. Coordina despliegues, rollbacks y automatización de infraestructura.',
    type: 'direct',
    module: 'agents/devops/agent.py',
    capabilities: ['CI/CD', 'K8s ops', 'Deployments', 'Rollback'],
  },
  {
    key: 'uriel',
    name: 'Uriel',
    role: 'Arch Agent',
    subtitle: 'Diseño de sistemas y ADRs',
    color: '#a78bfa',
    icon: '◫',
    description: 'Diseña arquitecturas de software, genera ADRs y recomienda patrones de diseño. Evalúa trade-offs técnicos y escalabilidad del sistema.',
    type: 'direct',
    module: 'agents/arch/agent.py',
    capabilities: ['System design', 'ADRs', 'Design patterns', 'Trade-off analysis'],
  },
  {
    key: 'raziel',
    name: 'Raziel',
    role: 'CTO Agent',
    subtitle: 'Estrategia tecnológica',
    color: '#e11d48',
    icon: '✧',
    description: 'Define estrategia tecnológica, roadmap y decisiones ejecutivas de arquitectura. Alinea la visión técnica con los objetivos de negocio.',
    type: 'direct',
    module: 'agents/cto/agent.py',
    capabilities: ['Tech strategy', 'Roadmap', 'Exec decisions', 'Tech investment'],
  },
  {
    key: 'zaphkiel',
    name: 'Zaphkiel',
    role: 'Memory Agent',
    subtitle: 'Memoria episódica por usuario',
    color: '#0ea5e9',
    icon: '◌',
    description: 'Almacena y recupera contexto a largo plazo por usuario: hechos, metas y episodios previos. Enriquece cada request con memoria persistente en PostgreSQL.',
    type: 'direct',
    module: 'agents/memory_agent/agent.py',
    capabilities: ['Long-term memory', 'User facts', 'Goals tracking', 'Context enrichment'],
  },
  {
    key: 'jophiel',
    name: 'Jophiel',
    role: 'Coder Agent',
    subtitle: 'Generación y refactoring de código',
    color: '#34d399',
    icon: '◑',
    description: 'Especializado en generación, refactoring y análisis de código en memoria. Opera sobre el workspace sin necesitar acceso a GitHub — complementa a Gabriel.',
    type: 'direct',
    module: 'agents/coder/agent.py',
    capabilities: ['Code generation', 'Refactoring', 'Code analysis', 'In-memory ops'],
  },
]

export interface StepTypeDef {
  type: string
  target: string
  parallel: boolean
  color: string
  desc: string
}

export const STEP_TYPES: StepTypeDef[] = [
  { type: 'K8S_TOOL',        target: 'k8s-agent:8002',         parallel: true,  color: '#ef4444', desc: 'Kubernetes ops, diagnósticos, kubectl, Vault' },
  { type: 'RAG_RETRIEVAL',   target: 'Qdrant + Ollama',         parallel: true,  color: '#10b981', desc: 'Búsqueda semántica en documentos del usuario' },
  { type: 'PRODUCTIVITY_TOOL', target: 'productivity-service:8001', parallel: true, color: '#3b82f6', desc: 'Google Calendar / Gmail via OAuth' },
  { type: 'WEB_SEARCH',      target: 'DuckDuckGo',              parallel: true,  color: '#8b5cf6', desc: 'Búsqueda en internet en tiempo real' },
  { type: 'REASONING',       target: 'Ollama (qwen2.5:14b)',    parallel: false, color: '#f59e0b', desc: 'Reflexión LLM — siempre secuencial, con detección de idioma' },
]
