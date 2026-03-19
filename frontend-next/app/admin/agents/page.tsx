'use client'
import { useCallback, useEffect, useRef, useState } from 'react'
import { useRouter } from 'next/navigation'

// ── Constantes ──────────────────────────────────────────────────────────────
const BASE = 'https://amael-ia.richardx.dev/api'

// ── Tipos ───────────────────────────────────────────────────────────────────
type Tab = 'agents' | 'architecture' | 'sre' | 'system'

interface RegisteredAgent {
  name: string
  role: string
  version: string
  capabilities: string[]
  required_skills: string[]
  required_tools: string[]
}

interface ComponentHealth {
  name: string
  healthy: boolean
  latency_ms: number
  detail: string
}

interface ReadyResponse {
  status: 'ok' | 'degraded' | 'unavailable'
  version: string
  uptime_s: number
  components: Record<string, ComponentHealth>
}

interface SreLoopStatus {
  loop_enabled: boolean
  loop_interval_seconds?: number
  last_run_at?: string | null
  last_run_result?: string
  anomalies_in_last_run?: number
  actions_in_last_run?: number
  circuit_breaker_state: 'CLOSED' | 'OPEN' | 'HALF_OPEN'
  maintenance_active: boolean
  maintenance_expires?: string | null
  is_leader?: boolean
  leader_pod?: string
  slo_count?: number
}

interface SreIncident {
  id?: number
  severity: 'HIGH' | 'MEDIUM' | 'LOW'
  issue_type: string
  pod_name: string
  namespace?: string
  action?: string
  created_at: string
}

interface SloStatus {
  handler: string
  target_availability: number
  window_hours: number
  current_burn_rate?: number
  status: 'HEALTHY' | 'WARNING' | 'CRITICAL' | string
}

interface LearningStats {
  issue_type: string
  action: string
  success_rate: number
  sample_count: number
}

// ── Catálogo de agentes (fuente de verdad) ──────────────────────────────────
const AGENT_CATALOG = [
  {
    key: 'amael',
    name: 'Amael',
    role: 'Orchestrator',
    subtitle: 'Cerebro central del sistema',
    color: '#6366f1',
    icon: '✦',
    description: 'Coordina el pipeline LangGraph completo. Compila el grafo, inyecta herramientas por request y dirige el flujo Planner → Executor → Supervisor.',
    type: 'orchestrator' as const,
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
    type: 'pipeline' as const,
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
    type: 'pipeline' as const,
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
    type: 'pipeline' as const,
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
    type: 'direct' as const,
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
    type: 'direct' as const,
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
    type: 'autonomous' as const,
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
    type: 'direct' as const,
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
    type: 'direct' as const,
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
    type: 'direct' as const,
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
    type: 'direct' as const,
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
    type: 'direct' as const,
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
    type: 'direct' as const,
    module: 'agents/coder/agent.py',
    capabilities: ['Code generation', 'Refactoring', 'Code analysis', 'In-memory ops'],
  },
]

// ── Step types del pipeline ──────────────────────────────────────────────────
const STEP_TYPES = [
  { type: 'K8S_TOOL', target: 'k8s-agent:8002', parallel: true, color: '#ef4444', desc: 'Kubernetes ops, diagnósticos, kubectl, Vault' },
  { type: 'RAG_RETRIEVAL', target: 'Qdrant + Ollama', parallel: true, color: '#10b981', desc: 'Búsqueda semántica en documentos del usuario' },
  { type: 'PRODUCTIVITY_TOOL', target: 'productivity-service:8001', parallel: true, color: '#3b82f6', desc: 'Google Calendar / Gmail via OAuth' },
  { type: 'WEB_SEARCH', target: 'DuckDuckGo', parallel: true, color: '#8b5cf6', desc: 'Búsqueda en internet en tiempo real' },
  { type: 'REASONING', target: 'Ollama (qwen2.5:14b)', parallel: false, color: '#f59e0b', desc: 'Reflexión LLM — siempre secuencial, con detección de idioma' },
]

// ── Helpers ──────────────────────────────────────────────────────────────────
function fmtUptime(s: number) {
  if (s < 60) return `${Math.round(s)}s`
  if (s < 3600) return `${Math.floor(s / 60)}m ${Math.round(s % 60)}s`
  return `${Math.floor(s / 3600)}h ${Math.floor((s % 3600) / 60)}m`
}

function fmtTs(iso: string) {
  try { return new Date(iso).toLocaleString('es-MX', { dateStyle: 'short', timeStyle: 'short' }) }
  catch { return iso }
}

function severityColor(s: string) {
  if (s === 'HIGH') return '#ef4444'
  if (s === 'MEDIUM') return '#f59e0b'
  return '#94a3b8'
}

function sloStatusColor(s: string) {
  if (s === 'HEALTHY') return '#22c55e'
  if (s === 'WARNING') return '#f59e0b'
  if (s === 'CRITICAL') return '#ef4444'
  return '#94a3b8'
}

function cbColor(s: string) {
  if (s === 'CLOSED') return '#22c55e'
  if (s === 'HALF_OPEN') return '#f59e0b'
  return '#ef4444'
}

// ── Componente principal ─────────────────────────────────────────────────────
export default function AgentDashboard() {
  const router = useRouter()
  const [tab, setTab] = useState<Tab>('agents')
  const [token, setToken] = useState('')
  const [lastUpdated, setLastUpdated] = useState<Date | null>(null)
  const [loading, setLoading] = useState(false)

  // Data state
  const [registeredAgents, setRegisteredAgents] = useState<RegisteredAgent[]>([])
  const [readyData, setReadyData] = useState<ReadyResponse | null>(null)
  const [sreLoop, setSreLoop] = useState<SreLoopStatus | null>(null)
  const [incidents, setIncidents] = useState<SreIncident[]>([])
  const [sloList, setSloList] = useState<SloStatus[]>([])
  const [learning, setLearning] = useState<LearningStats[]>([])

  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null)

  useEffect(() => {
    const t = localStorage.getItem('amael-token') || ''
    setToken(t)
    if (!t) { router.push('/'); return }
  }, [router])

  const fetchAll = useCallback(async (tok: string) => {
    if (!tok) return
    setLoading(true)
    const h = { Authorization: `Bearer ${tok}` }
    try {
      const [readyRes, agentsRes, sreRes, incRes, sloRes, learnRes] = await Promise.allSettled([
        fetch(`${BASE.replace('/api', '')}/ready`).then(r => r.json()),
        fetch(`${BASE}/agent/list`, { headers: h }).then(r => r.json()),
        fetch(`${BASE}/sre/loop/status`, { headers: h }).then(r => r.json()),
        fetch(`${BASE}/sre/incidents?limit=8`, { headers: h }).then(r => r.json()),
        fetch(`${BASE}/sre/slo/status`, { headers: h }).then(r => r.json()),
        fetch(`${BASE}/sre/learning/stats?days=7`, { headers: h }).then(r => r.json()),
      ])
      if (readyRes.status === 'fulfilled') setReadyData(readyRes.value)
      if (agentsRes.status === 'fulfilled' && Array.isArray(agentsRes.value)) setRegisteredAgents(agentsRes.value)
      if (sreRes.status === 'fulfilled') setSreLoop(sreRes.value)
      if (incRes.status === 'fulfilled' && Array.isArray(incRes.value)) setIncidents(incRes.value)
      if (sloRes.status === 'fulfilled' && Array.isArray(sloRes.value)) setSloList(sloRes.value)
      if (learnRes.status === 'fulfilled' && Array.isArray(learnRes.value)) setLearning(learnRes.value)
      setLastUpdated(new Date())
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => {
    if (!token) return
    fetchAll(token)
    pollRef.current = setInterval(() => fetchAll(token), 30_000)
    return () => { if (pollRef.current) clearInterval(pollRef.current) }
  }, [token, fetchAll])

  // ── Derivar estado live de agentes ────────────────────────────────────────
  function getAgentLiveStatus(key: string): { status: string; color: string; detail: string } {
    if (key === 'raphael' && sreLoop) {
      if (!sreLoop.loop_enabled) return { status: 'Detenido', color: '#ef4444', detail: 'Loop no activo' }
      if (sreLoop.circuit_breaker_state === 'OPEN') return { status: 'Circuit Breaker', color: '#ef4444', detail: 'CB abierto' }
      if (sreLoop.maintenance_active) return { status: 'Mantenimiento', color: '#f59e0b', detail: 'Ventana activa' }
      const lastRun = sreLoop.last_run_result
      if (lastRun === 'pending') return { status: 'Iniciando', color: '#f59e0b', detail: 'Primer ciclo pendiente' }
      return { status: 'Activo', color: '#22c55e', detail: `Loop cada ${sreLoop.loop_interval_seconds ?? 60}s` }
    }
    if (readyData?.status === 'unavailable') return { status: 'Degradado', color: '#ef4444', detail: 'Sistema no disponible' }
    const reg = registeredAgents.find(a => a.name.toLowerCase() === key || a.name.toLowerCase().includes(key))
    if (reg) return { status: 'Registrado', color: '#22c55e', detail: `v${reg.version}` }
    if (key === 'amael' && readyData) return { status: 'Activo', color: '#22c55e', detail: `v${readyData.version}` }
    if (key === 'executor' && registeredAgents.length > 0) return { status: 'En pipeline', color: '#6366f1', detail: 'LangGraph activo' }
    if (registeredAgents.length > 0) return { status: 'En pipeline', color: '#6366f1', detail: 'LangGraph activo' }
    return { status: '—', color: '#475569', detail: 'Sin datos' }
  }

  const tabBtn = (active: boolean): React.CSSProperties => ({
    padding: '12px 20px', border: 'none', background: 'none', cursor: 'pointer', fontSize: 14, fontWeight: 500,
    color: active ? 'var(--primary)' : 'var(--text-secondary)',
    borderBottom: active ? '2px solid var(--primary)' : '2px solid transparent',
    transition: 'all .15s',
  })
  const pill = (color: string): React.CSSProperties => ({ fontSize: 11, padding: '2px 8px', borderRadius: 20, background: color + '22', color, fontWeight: 600, whiteSpace: 'nowrap' as const })
  const dot = (color: string): React.CSSProperties => ({ width: 8, height: 8, borderRadius: '50%', background: color, flexShrink: 0 })

  const s: Record<string, React.CSSProperties> = {
    root: { minHeight: '100vh', background: 'var(--bg-base)', color: 'var(--text-primary)', fontFamily: 'Inter, sans-serif' },
    header: { background: 'var(--bg-surface)', borderBottom: '1px solid var(--border)', padding: '16px 24px', display: 'flex', alignItems: 'center', gap: 16, flexWrap: 'wrap' as const },
    backBtn: { background: 'none', border: '1px solid var(--border)', color: 'var(--text-secondary)', padding: '6px 14px', borderRadius: 8, cursor: 'pointer', fontSize: 13, display: 'flex', alignItems: 'center', gap: 6 },
    title: { fontSize: 20, fontWeight: 700, flex: 1 },
    badge: { fontSize: 11, padding: '3px 8px', borderRadius: 20, background: 'var(--primary-subtle)', color: 'var(--primary)', fontWeight: 600 },
    refreshBtn: { background: 'var(--primary)', color: '#fff', border: 'none', padding: '6px 14px', borderRadius: 8, cursor: 'pointer', fontSize: 13, display: 'flex', alignItems: 'center', gap: 6 },
    tabs: { display: 'flex', gap: 0, borderBottom: '1px solid var(--border)', background: 'var(--bg-surface)', padding: '0 24px' },
    body: { padding: '24px', maxWidth: 1200, margin: '0 auto' },
    grid: { display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(320px, 1fr))', gap: 16 },
    card: { background: 'var(--bg-surface)', border: '1px solid var(--border)', borderRadius: 12, padding: 20, transition: 'border-color .2s' },
    row: { display: 'flex', alignItems: 'center', gap: 10, marginBottom: 8 },
    mono: { fontFamily: 'monospace', fontSize: 12, color: 'var(--text-secondary)' },
    label: { fontSize: 11, color: 'var(--text-disabled)', textTransform: 'uppercase' as const, letterSpacing: '.06em', marginBottom: 4 },
  }

  // ── TAB: AGENTES ────────────────────────────────────────────────────────────
  function TabAgents() {
    return (
      <div>
        <div style={{ marginBottom: 20 }}>
          <h2 style={{ fontSize: 16, fontWeight: 600, marginBottom: 4 }}>Agentes del sistema</h2>
          <p style={{ color: 'var(--text-secondary)', fontSize: 13 }}>
            {registeredAgents.length > 0
              ? `${registeredAgents.length} agentes en el registry + Amael (orchestrator) — ${AGENT_CATALOG.length} agentes en total`
              : 'Pipeline LangGraph: Sariel → Grouper → Executor → Remiel — con agentes directos por intent'}
          </p>
        </div>
        <div style={s.grid}>
          {AGENT_CATALOG.map(agent => {
            const live = getAgentLiveStatus(agent.key)
            return (
              <div key={agent.key} style={{ ...s.card, borderColor: agent.color + '44' }}>
                {/* Header del card */}
                <div style={s.row}>
                  <div style={{
                    width: 42, height: 42, borderRadius: 10, background: agent.color + '22',
                    display: 'flex', alignItems: 'center', justifyContent: 'center',
                    fontSize: 20, color: agent.color, flexShrink: 0,
                  }}>{agent.icon}</div>
                  <div style={{ flex: 1 }}>
                    <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                      <span style={{ fontWeight: 700, fontSize: 16 }}>{agent.name}</span>
                      <span style={pill(agent.color)}>{agent.role}</span>
                    </div>
                    <div style={{ fontSize: 12, color: 'var(--text-secondary)' }}>{agent.subtitle}</div>
                  </div>
                </div>

                {/* Tipo de agente */}
                <div style={{ display: 'flex', gap: 8, marginBottom: 10 }}>
                  {agent.type === 'orchestrator' && <span style={pill('#6366f1')}>Orchestrator</span>}
                  {agent.type === 'pipeline' && <span style={pill('#8b5cf6')}>Pipeline LangGraph</span>}
                  {agent.type === 'direct' && <span style={pill('#06b6d4')}>Agente directo</span>}
                  {agent.type === 'autonomous' && <span style={pill('#ef4444')}>Autónomo</span>}
                  <span style={{ ...pill(live.color), marginLeft: 'auto' }}>
                    <span style={{ ...dot(live.color), display: 'inline-block', marginRight: 4 }} />
                    {live.status}
                  </span>
                </div>

                {/* Descripción */}
                <p style={{ fontSize: 13, color: 'var(--text-secondary)', lineHeight: 1.6, marginBottom: 12, overflow: 'hidden', display: '-webkit-box', WebkitLineClamp: 3, WebkitBoxOrient: 'vertical' as const }}>
                  {agent.description}
                </p>

                {/* Capacidades */}
                <div>
                  <div style={s.label}>Capacidades</div>
                  <div style={{ display: 'flex', flexWrap: 'wrap' as const, gap: 4 }}>
                    {agent.capabilities.map(c => (
                      <span key={c} style={{ fontSize: 11, padding: '2px 7px', borderRadius: 4, background: 'var(--bg-elevated)', color: 'var(--text-secondary)', border: '1px solid var(--border)' }}>
                        {c}
                      </span>
                    ))}
                  </div>
                </div>

                {/* Módulo */}
                <div style={{ marginTop: 12, ...s.mono }}>{agent.module}</div>
              </div>
            )
          })}
        </div>

        {/* Agentes registrados en el registry (datos live) */}
        {registeredAgents.length > 0 && (
          <div style={{ ...s.card, marginTop: 24 }}>
            <h3 style={{ fontSize: 14, fontWeight: 600, marginBottom: 4 }}>
              Registry en vivo — {registeredAgents.length} agentes registrados
            </h3>
            <p style={{ fontSize: 12, color: 'var(--text-disabled)', marginBottom: 14 }}>
              Amael (orchestrator) no aparece aquí — gestiona el grafo LangGraph directamente, sin registrarse como agente.
            </p>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(280px, 1fr))', gap: 10 }}>
              {registeredAgents.map(ag => (
                <div key={ag.name} style={{ background: 'var(--bg-elevated)', borderRadius: 8, padding: '10px 14px' }}>
                  <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 4 }}>
                    <div style={{ width: 7, height: 7, borderRadius: '50%', background: '#22c55e', flexShrink: 0 }} />
                    <span style={{ fontWeight: 600, fontSize: 13, textTransform: 'capitalize' }}>{ag.name}</span>
                    <span style={{ fontSize: 11, color: 'var(--text-disabled)' }}>v{ag.version}</span>
                  </div>
                  <div style={{ marginBottom: 4, overflow: 'hidden' }}>
                    <span style={{ ...pill('#6366f1'), display: '-webkit-box', WebkitLineClamp: 2, WebkitBoxOrient: 'vertical' as const, overflow: 'hidden', whiteSpace: 'normal' as const, lineHeight: 1.4 }}>{ag.role}</span>
                  </div>
                  {ag.capabilities.length > 0 && (
                    <div style={{ display: 'flex', flexWrap: 'wrap' as const, gap: 3, marginTop: 4 }}>
                      {ag.capabilities.slice(0, 4).map(c => (
                        <span key={c} style={{ fontSize: 10, padding: '1px 6px', borderRadius: 4, background: 'var(--bg-surface)', color: 'var(--text-secondary)', border: '1px solid var(--border)' }}>{c}</span>
                      ))}
                    </div>
                  )}
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Tabla de intents → agente */}
        <div style={{ ...s.card, marginTop: 24 }}>
          <h3 style={{ fontSize: 14, fontWeight: 600, marginBottom: 14 }}>Routing de intents → agente</h3>
          <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
            <thead>
              <tr>
                {['Intent', 'Modo', 'Agente', 'Descripción'].map(h => (
                  <th key={h} style={{ textAlign: 'left', padding: '8px 12px', borderBottom: '1px solid var(--border)', color: 'var(--text-secondary)', fontSize: 11, fontWeight: 600, textTransform: 'uppercase' }}>{h}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {[
                { intent: 'sre', mode: 'Directo', agent: 'Raphael', desc: 'Diagnóstico K8s, incidentes, SLO' },
                { intent: 'productivity', mode: 'Directo', agent: 'Haniel', desc: 'Calendario, Gmail, agenda' },
                { intent: 'research', mode: 'Directo', agent: 'Sandalphon', desc: 'RAG sobre documentos del usuario' },
                { intent: 'general / k8s / monitoring', mode: 'Pipeline', agent: 'Amael → Sariel → Remiel', desc: 'Flujo LangGraph completo' },
              ].map((row, i) => (
                <tr key={i} style={{ borderBottom: '1px solid var(--border)' }}>
                  <td style={{ padding: '10px 12px' }}><code style={{ fontSize: 12, color: 'var(--primary)' }}>{row.intent}</code></td>
                  <td style={{ padding: '10px 12px', color: 'var(--text-secondary)' }}>{row.mode}</td>
                  <td style={{ padding: '10px 12px', fontWeight: 500 }}>{row.agent}</td>
                  <td style={{ padding: '10px 12px', color: 'var(--text-secondary)' }}>{row.desc}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    )
  }

  // ── TAB: ARQUITECTURA ────────────────────────────────────────────────────────
  function TabArchitecture() {
    return (
      <div>
        <h2 style={{ fontSize: 16, fontWeight: 600, marginBottom: 4 }}>Arquitectura del sistema</h2>
        <p style={{ color: 'var(--text-secondary)', fontSize: 13, marginBottom: 24 }}>
          Pipeline LangGraph multi-agente con routing por intent y loop SRE autónomo
        </p>

        {/* Pipeline visual */}
        <div style={{ ...s.card, marginBottom: 20 }}>
          <h3 style={{ fontSize: 14, fontWeight: 600, marginBottom: 16 }}>Pipeline principal (intent: general)</h3>
          <div style={{ overflowX: 'auto', paddingBottom: 8 }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: 0, minWidth: 640 }}>
              {/* User */}
              <PipeNode label="Usuario" icon="👤" color="#94a3b8" sub="Pregunta" />
              <Arrow />
              {/* Amael */}
              <PipeNode label="Amael" icon="✦" color="#6366f1" sub="Orchestrator" />
              <Arrow />
              {/* Sariel */}
              <PipeNode label="Sariel" icon="◈" color="#8b5cf6" sub="Planner" badge="JSON Plan" />
              <Arrow />
              {/* Grouper */}
              <PipeNode label="Grouper" icon="⬡" color="#06b6d4" sub="Batching" badge="Paralelo" />
              <Arrow />
              {/* Executor */}
              <PipeNode label="Executor" icon="⬡" color="#06b6d4" sub="Batch Executor" />
              <Arrow />
              {/* Remiel */}
              <PipeNode label="Remiel" icon="◎" color="#f59e0b" sub="Supervisor" badge="Score 0–10" />
            </div>
            {/* REPLAN arrow */}
            <div style={{ display: 'flex', justifyContent: 'flex-end', marginTop: 4, paddingRight: 8 }}>
              <div style={{ fontSize: 11, color: '#f59e0b', display: 'flex', alignItems: 'center', gap: 4 }}>
                <span>↩</span>
                <span>REPLAN si score &lt; 6 (máx 1 retry) → vuelve a Sariel</span>
              </div>
            </div>
          </div>
        </div>

        {/* Direct agents */}
        <div style={{ ...s.card, marginBottom: 20 }}>
          <h3 style={{ fontSize: 14, fontWeight: 600, marginBottom: 16 }}>Agentes directos (routing por intent)</h3>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(260px, 1fr))', gap: 12 }}>
            {[
              { intent: 'sre', agent: 'Raphael', color: '#ef4444', icon: '⬢', flow: 'User → Amael → Raphael' },
              { intent: 'productivity', agent: 'Haniel', color: '#3b82f6', icon: '◇', flow: 'User → Amael → Haniel' },
              { intent: 'research', agent: 'Sandalphon', color: '#10b981', icon: '◉', flow: 'User → Amael → Sandalphon' },
            ].map(item => (
              <div key={item.agent} style={{ background: 'var(--bg-elevated)', borderRadius: 8, padding: 14, border: `1px solid ${item.color}33` }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 6 }}>
                  <span style={{ fontSize: 18, color: item.color }}>{item.icon}</span>
                  <span style={{ fontWeight: 600 }}>{item.agent}</span>
                  <span style={pill(item.color)}>{item.intent}</span>
                </div>
                <div style={{ ...s.mono, color: 'var(--text-disabled)' }}>{item.flow}</div>
              </div>
            ))}
          </div>
        </div>

        {/* Step types */}
        <div style={{ ...s.card, marginBottom: 20 }}>
          <h3 style={{ fontSize: 14, fontWeight: 600, marginBottom: 14 }}>Tipos de pasos del Executor</h3>
          <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
            <thead>
              <tr>
                {['Tipo', 'Target', 'Paralelo', 'Descripción'].map(h => (
                  <th key={h} style={{ textAlign: 'left', padding: '8px 12px', borderBottom: '1px solid var(--border)', color: 'var(--text-secondary)', fontSize: 11, fontWeight: 600, textTransform: 'uppercase' }}>{h}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {STEP_TYPES.map((st, i) => (
                <tr key={i} style={{ borderBottom: '1px solid var(--border)' }}>
                  <td style={{ padding: '10px 12px' }}><span style={pill(st.color)}>{st.type}</span></td>
                  <td style={{ padding: '10px 12px', ...s.mono }}>{st.target}</td>
                  <td style={{ padding: '10px 12px' }}>
                    <span style={{ color: st.parallel ? '#22c55e' : '#f59e0b', fontWeight: 600, fontSize: 12 }}>
                      {st.parallel ? '✓ Sí' : '✗ No'}
                    </span>
                  </td>
                  <td style={{ padding: '10px 12px', color: 'var(--text-secondary)' }}>{st.desc}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>

        {/* Infraestructura */}
        <div style={s.card}>
          <h3 style={{ fontSize: 14, fontWeight: 600, marginBottom: 14 }}>Infraestructura de soporte</h3>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(200px, 1fr))', gap: 10 }}>
            {[
              { name: 'Ollama', detail: 'qwen2.5:14b + nomic-embed-text', color: '#6366f1', icon: '🧠' },
              { name: 'PostgreSQL', detail: 'Chat history, incidentes, users', color: '#3b82f6', icon: '🗄️' },
              { name: 'Redis', detail: 'Rate limit, dedup, sesiones', color: '#ef4444', icon: '⚡' },
              { name: 'Qdrant', detail: 'RAG por usuario + SRE runbooks', color: '#10b981', icon: '🔍' },
              { name: 'MinIO', detail: 'Backup documentos (amael-uploads)', color: '#f59e0b', icon: '📦' },
              { name: 'Vault', detail: 'OAuth tokens Google por usuario', color: '#8b5cf6', icon: '🔐' },
              { name: 'k8s-agent', detail: 'Kubernetes + SRE tools :8002', color: '#06b6d4', icon: '⚙️' },
              { name: 'productivity-service', detail: 'Calendar/Gmail proxy :8001', color: '#3b82f6', icon: '📅' },
            ].map(item => (
              <div key={item.name} style={{ background: 'var(--bg-elevated)', borderRadius: 8, padding: 12, display: 'flex', gap: 10, alignItems: 'flex-start' }}>
                <span style={{ fontSize: 20 }}>{item.icon}</span>
                <div>
                  <div style={{ fontWeight: 600, fontSize: 13 }}>{item.name}</div>
                  <div style={{ fontSize: 12, color: 'var(--text-secondary)', marginTop: 2 }}>{item.detail}</div>
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>
    )
  }

  // ── TAB: RAPHAEL SRE ──────────────────────────────────────────────────────
  function TabSRE() {
    return (
      <div>
        <div style={{ display: 'flex', alignItems: 'center', gap: 12, marginBottom: 20 }}>
          <span style={{ fontSize: 28, color: '#ef4444' }}>⬢</span>
          <div>
            <h2 style={{ fontSize: 16, fontWeight: 600 }}>Raphael — SRE Autónomo</h2>
            <p style={{ color: 'var(--text-secondary)', fontSize: 13 }}>Loop Observe → Detect → Diagnose → Decide → Act → Report (60s)</p>
          </div>
        </div>

        {/* Loop status */}
        {sreLoop ? (
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(180px, 1fr))', gap: 12, marginBottom: 20 }}>
            <StatCard label="Loop" value={sreLoop.loop_enabled ? (sreLoop.last_run_result === 'pending' ? 'Iniciando' : 'Corriendo') : 'Detenido'} color={sreLoop.loop_enabled ? (sreLoop.last_run_result === 'pending' ? '#f59e0b' : '#22c55e') : '#ef4444'} />
            <StatCard label="Circuit Breaker" value={sreLoop.circuit_breaker_state} color={cbColor(sreLoop.circuit_breaker_state)} />
            <StatCard label="Mantenimiento" value={sreLoop.maintenance_active ? 'Activo' : 'Inactivo'} color={sreLoop.maintenance_active ? '#f59e0b' : '#22c55e'} />
            <StatCard label="Último resultado" value={sreLoop.last_run_result ?? '—'} color="#6366f1" />
            {sreLoop.anomalies_in_last_run !== undefined && <StatCard label="Anomalías (último run)" value={String(sreLoop.anomalies_in_last_run)} color={sreLoop.anomalies_in_last_run > 0 ? '#f59e0b' : '#22c55e'} />}
            {sreLoop.actions_in_last_run !== undefined && <StatCard label="Acciones (último run)" value={String(sreLoop.actions_in_last_run)} color={sreLoop.actions_in_last_run > 0 ? '#ef4444' : '#94a3b8'} />}
          </div>
        ) : (
          <div style={{ ...s.card, marginBottom: 20, color: 'var(--text-disabled)', fontSize: 13 }}>
            Cargando estado del loop SRE…
          </div>
        )}

        {/* SLO status */}
        {sloList.length > 0 && (
          <div style={{ ...s.card, marginBottom: 20 }}>
            <h3 style={{ fontSize: 14, fontWeight: 600, marginBottom: 14 }}>SLO Targets</h3>
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
              <thead>
                <tr>
                  {['Endpoint', 'Target', 'Ventana', 'Burn Rate', 'Estado'].map(h => (
                    <th key={h} style={{ textAlign: 'left', padding: '8px 12px', borderBottom: '1px solid var(--border)', color: 'var(--text-secondary)', fontSize: 11, fontWeight: 600, textTransform: 'uppercase' }}>{h}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {sloList.map((slo, i) => (
                  <tr key={i} style={{ borderBottom: '1px solid var(--border)' }}>
                    <td style={{ padding: '10px 12px' }}><code style={{ fontSize: 12, color: 'var(--primary)' }}>{slo.handler}</code></td>
                    <td style={{ padding: '10px 12px', fontWeight: 600 }}>{(slo.target_availability * 100).toFixed(1)}%</td>
                    <td style={{ padding: '10px 12px', color: 'var(--text-secondary)' }}>{slo.window_hours}h</td>
                    <td style={{ padding: '10px 12px' }}>{slo.current_burn_rate !== undefined ? slo.current_burn_rate.toFixed(4) : '—'}</td>
                    <td style={{ padding: '10px 12px' }}><span style={pill(sloStatusColor(slo.status))}>{slo.status}</span></td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}

        {/* Incidents */}
        <div style={{ ...s.card, marginBottom: 20 }}>
          <h3 style={{ fontSize: 14, fontWeight: 600, marginBottom: 14 }}>Incidentes recientes</h3>
          {incidents.length === 0 ? (
            <div style={{ color: 'var(--text-disabled)', fontSize: 13 }}>Sin incidentes registrados</div>
          ) : (
            <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
              {incidents.map((inc, i) => (
                <div key={i} style={{ background: 'var(--bg-elevated)', borderRadius: 8, padding: '10px 14px', display: 'flex', alignItems: 'center', gap: 12, flexWrap: 'wrap' as const }}>
                  <span style={pill(severityColor(inc.severity))}>{inc.severity}</span>
                  <span style={{ fontSize: 12, fontWeight: 600, color: 'var(--primary)' }}>{inc.issue_type}</span>
                  <span style={{ fontSize: 13 }}>{inc.pod_name}</span>
                  {inc.action && <span style={{ fontSize: 12, color: 'var(--text-secondary)' }}>→ {inc.action}</span>}
                  <span style={{ fontSize: 11, color: 'var(--text-disabled)', marginLeft: 'auto' }}>{fmtTs(inc.created_at)}</span>
                </div>
              ))}
            </div>
          )}
        </div>

        {/* Learning stats */}
        {learning.length > 0 && (
          <div style={s.card}>
            <h3 style={{ fontSize: 14, fontWeight: 600, marginBottom: 14 }}>Estadísticas de aprendizaje (7 días)</h3>
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
              <thead>
                <tr>
                  {['Tipo de anomalía', 'Acción', 'Tasa de éxito', 'Muestras'].map(h => (
                    <th key={h} style={{ textAlign: 'left', padding: '8px 12px', borderBottom: '1px solid var(--border)', color: 'var(--text-secondary)', fontSize: 11, fontWeight: 600, textTransform: 'uppercase' }}>{h}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {learning.map((row, i) => (
                  <tr key={i} style={{ borderBottom: '1px solid var(--border)' }}>
                    <td style={{ padding: '10px 12px' }}><span style={pill('#6366f1')}>{row.issue_type}</span></td>
                    <td style={{ padding: '10px 12px', color: 'var(--text-secondary)' }}>{row.action}</td>
                    <td style={{ padding: '10px 12px' }}>
                      <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                        <div style={{ flex: 1, height: 6, background: 'var(--bg-elevated)', borderRadius: 3, overflow: 'hidden', maxWidth: 80 }}>
                          <div style={{ width: `${row.success_rate * 100}%`, height: '100%', background: row.success_rate > 0.7 ? '#22c55e' : row.success_rate > 0.4 ? '#f59e0b' : '#ef4444', borderRadius: 3 }} />
                        </div>
                        <span style={{ fontWeight: 600, minWidth: 36 }}>{(row.success_rate * 100).toFixed(0)}%</span>
                      </div>
                    </td>
                    <td style={{ padding: '10px 12px', color: 'var(--text-secondary)' }}>{row.sample_count}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>
    )
  }

  // ── TAB: SISTEMA ─────────────────────────────────────────────────────────────
  function TabSystem() {
    const core = readyData?.components
    const coreKeys = core ? Object.keys(core).filter(k => !k.startsWith('skill.') && !k.startsWith('tool.')) : []
    const skillKeys = core ? Object.keys(core).filter(k => k.startsWith('skill.')) : []
    const toolKeys = core ? Object.keys(core).filter(k => k.startsWith('tool.')) : []

    return (
      <div>
        <h2 style={{ fontSize: 16, fontWeight: 600, marginBottom: 4 }}>Salud del sistema</h2>
        <p style={{ color: 'var(--text-secondary)', fontSize: 13, marginBottom: 20 }}>
          Estado de componentes vía <code style={{ fontSize: 12 }}>/ready</code>
        </p>

        {/* Summary */}
        {readyData && (
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(180px, 1fr))', gap: 12, marginBottom: 20 }}>
            <StatCard label="Estado global" value={readyData.status.toUpperCase()} color={readyData.status === 'ok' ? '#22c55e' : readyData.status === 'degraded' ? '#f59e0b' : '#ef4444'} />
            <StatCard label="Versión" value={readyData.version} color="#6366f1" />
            <StatCard label="Uptime" value={fmtUptime(readyData.uptime_s)} color="#06b6d4" />
            <StatCard label="Componentes OK" value={`${Object.values(readyData.components).filter(c => c.healthy).length} / ${Object.keys(readyData.components).length}`} color="#22c55e" />
          </div>
        )}

        {/* Core components */}
        {readyData && coreKeys.length > 0 && (
          <div style={{ ...s.card, marginBottom: 16 }}>
            <h3 style={{ fontSize: 14, fontWeight: 600, marginBottom: 14 }}>Componentes core</h3>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(220px, 1fr))', gap: 10 }}>
              {coreKeys.map(k => <ComponentCard key={k} comp={readyData.components[k]} />)}
            </div>
          </div>
        )}

        {/* Skills */}
        {readyData && skillKeys.length > 0 && (
          <div style={{ ...s.card, marginBottom: 16 }}>
            <h3 style={{ fontSize: 14, fontWeight: 600, marginBottom: 14 }}>Skills registradas</h3>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(220px, 1fr))', gap: 10 }}>
              {skillKeys.map(k => <ComponentCard key={k} comp={readyData.components[k]} />)}
            </div>
          </div>
        )}

        {/* Tools */}
        {readyData && toolKeys.length > 0 && (
          <div style={s.card}>
            <h3 style={{ fontSize: 14, fontWeight: 600, marginBottom: 14 }}>Tools registradas</h3>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(220px, 1fr))', gap: 10 }}>
              {toolKeys.map(k => <ComponentCard key={k} comp={readyData.components[k]} />)}
            </div>
          </div>
        )}

        {!readyData && (
          <div style={{ ...s.card, color: 'var(--text-disabled)', fontSize: 13 }}>
            Cargando datos del sistema…
          </div>
        )}
      </div>
    )
  }

  // ── Subcomponentes ───────────────────────────────────────────────────────────
  function PipeNode({ label, icon, color, sub, badge }: { label: string; icon: string; color: string; sub: string; badge?: string }) {
    return (
      <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 4 }}>
        <div style={{
          width: 52, height: 52, borderRadius: 12, background: color + '22',
          border: `2px solid ${color}66`, display: 'flex', alignItems: 'center',
          justifyContent: 'center', fontSize: 18, color,
        }}>{icon}</div>
        <div style={{ fontSize: 12, fontWeight: 700, color: 'var(--text-primary)' }}>{label}</div>
        <div style={{ fontSize: 10, color: 'var(--text-disabled)' }}>{sub}</div>
        {badge && <div style={{ fontSize: 9, padding: '1px 5px', borderRadius: 10, background: color + '22', color, fontWeight: 600 }}>{badge}</div>}
      </div>
    )
  }

  function Arrow() {
    return <div style={{ color: 'var(--text-disabled)', fontSize: 18, padding: '0 4px', alignSelf: 'center', marginBottom: 14 }}>→</div>
  }

  function StatCard({ label, value, color }: { label: string; value: string; color: string }) {
    return (
      <div style={{ background: 'var(--bg-surface)', border: '1px solid var(--border)', borderRadius: 10, padding: '14px 16px' }}>
        <div style={{ fontSize: 11, color: 'var(--text-disabled)', textTransform: 'uppercase', letterSpacing: '.06em', marginBottom: 6 }}>{label}</div>
        <div style={{ fontSize: 18, fontWeight: 700, color }}>{value}</div>
      </div>
    )
  }

  function ComponentCard({ comp }: { comp: ComponentHealth }) {
    const displayName = comp.name.replace(/^(skill|tool)\./, '')
    return (
      <div style={{ background: 'var(--bg-elevated)', borderRadius: 8, padding: '10px 14px', display: 'flex', gap: 10, alignItems: 'center', border: `1px solid ${comp.healthy ? '#22c55e22' : '#ef444422'}` }}>
        <div style={{ width: 8, height: 8, borderRadius: '50%', background: comp.healthy ? '#22c55e' : '#ef4444', flexShrink: 0 }} />
        <div style={{ flex: 1, minWidth: 0 }}>
          <div style={{ fontWeight: 600, fontSize: 13 }}>{displayName}</div>
          {comp.latency_ms > 0 && <div style={{ fontSize: 11, color: 'var(--text-disabled)' }}>{comp.latency_ms.toFixed(1)}ms</div>}
          {comp.detail && <div style={{ fontSize: 11, color: 'var(--text-secondary)', marginTop: 1, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{comp.detail}</div>}
        </div>
      </div>
    )
  }

  // ── Render ────────────────────────────────────────────────────────────────
  const TABS: { id: Tab; label: string }[] = [
    { id: 'agents', label: 'Agentes' },
    { id: 'architecture', label: 'Arquitectura' },
    { id: 'sre', label: '⬢ Raphael (SRE)' },
    { id: 'system', label: 'Sistema' },
  ]

  return (
    <div style={s.root}>
      {/* Header */}
      <div style={s.header}>
        <button style={s.backBtn} onClick={() => router.push('/')}>← Chat</button>
        <div style={s.title}>Dashboard Amael</div>
        <span style={s.badge}>Admin</span>
        {lastUpdated && (
          <span style={{ fontSize: 12, color: 'var(--text-disabled)' }}>
            Actualizado: {lastUpdated.toLocaleTimeString('es-MX')}
          </span>
        )}
        <button style={{ ...s.refreshBtn, opacity: loading ? 0.6 : 1 }} onClick={() => fetchAll(token)} disabled={loading}>
          {loading ? '…' : '↻'} Refrescar
        </button>
      </div>

      {/* Tabs */}
      <div style={s.tabs}>
        {TABS.map(t => (
          <button key={t.id} style={tabBtn(tab === t.id)} onClick={() => setTab(t.id)}>
            {t.label}
          </button>
        ))}
      </div>

      {/* Body */}
      <div style={s.body}>
        {tab === 'agents' && <TabAgents />}
        {tab === 'architecture' && <TabArchitecture />}
        {tab === 'sre' && <TabSRE />}
        {tab === 'system' && <TabSystem />}
      </div>
    </div>
  )
}
