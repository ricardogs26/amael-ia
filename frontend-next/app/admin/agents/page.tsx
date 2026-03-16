'use client'
import { useEffect, useRef, useState } from 'react'
import { useRouter } from 'next/navigation'

// ── Tipos ──────────────────────────────────────────────────────────────────────
type AgentId = 'cto' | 'dev' | 'arch' | 'sre'
type Status  = 'Active' | 'Thinking' | 'Idle'
type Tab     = 'overview' | 'interactions' | 'chat' | 'pipeline'

interface Agent { id: AgentId; name: string; role: string; color: string; tasks: number; status: Status }
interface Interaction { id: number; from: AgentId; to: AgentId; msg: string; ts: string }
interface ChatMsg     { id: number; role: 'user' | 'assistant'; agent: AgentId; text: string }

// ── Datos estáticos ────────────────────────────────────────────────────────────
const AGENTS: Agent[] = [
  { id: 'cto',  name: 'CTO Agent',      role: 'Estrategia & Arquitectura',  color: '#7F77DD', tasks: 12, status: 'Active'   },
  { id: 'dev',  name: 'Dev Agent',       role: 'Desarrollo & Pull Requests', color: '#1D9E75', tasks: 8,  status: 'Thinking' },
  { id: 'arch', name: 'Architect Agent', role: 'Diseño & API Contracts',     color: '#D85A30', tasks: 5,  status: 'Active'   },
  { id: 'sre',  name: 'SRE Agent',       role: 'Observabilidad & Alertas',   color: '#185FA5', tasks: 3,  status: 'Idle'     },
]

const INIT_INTERACTIONS: Interaction[] = [
  { id: 1, from: 'cto',  to: 'arch', msg: 'Revisar diseño hexagonal architecture para módulo de pagos', ts: '14:32' },
  { id: 2, from: 'dev',  to: 'cto',  msg: 'PR #47 listo — cambios en rate limiting y sliding window',   ts: '14:28' },
  { id: 3, from: 'sre',  to: 'dev',  msg: 'Latencia p99 subió a 820ms en /api/chat, investigar',        ts: '14:21' },
  { id: 4, from: 'arch', to: 'dev',  msg: 'Contratos API v2 aprobados, proceder con implementación',     ts: '14:15' },
  { id: 5, from: 'cto',  to: 'dev',  msg: 'Sprint 12 kick-off: priorizar módulo de embeddings RAG',      ts: '14:05' },
]

const RAND_INTERACTIONS: Omit<Interaction, 'id' | 'ts'>[] = [
  { from: 'cto',  to: 'sre',  msg: 'Revisar SLO para /api/k8s-agent antes del release' },
  { from: 'dev',  to: 'arch', msg: 'Propongo mover validación al domain layer (DDD)' },
  { from: 'sre',  to: 'cto',  msg: 'Error budget 99.5% consumido al 72% — dentro de rango' },
  { from: 'arch', to: 'cto',  msg: 'ADR-012: adoptar event sourcing para audit trail' },
  { from: 'dev',  to: 'sre',  msg: 'Deploy v1.5.5 en staging, validar health checks' },
]

const AGENT_RESPONSES: Record<AgentId, string[]> = {
  cto: [
    'Para escalar Amael-IA recomiendo migrar a event-driven con Redis Streams. A partir de 10k RPM necesitaremos particionado y consumer groups dedicados por tipo de agente.',
    'La deuda técnica más crítica es el idioma de respuesta del LLM. Recomiendo cambiar a llama3.1:70b para mejor seguimiento de instrucciones en español. El trade-off en latencia se compensa con calidad.',
    'Para Q2 propongo: (1) LangGraph 0.2, (2) agent memory persistente en Qdrant con TTL configurable, (3) dashboard de BI integrado en el frontend para métricas de negocio.',
  ],
  dev: [
    'El PR #47 implementa rate limiting por sliding window en Redis.\n`ZADD rate:{uid} {ts} {ts}` + `ZCOUNT` — O(log n) vs O(n) del contador simple. Listo para merge después de code review del CTO.',
    'Nuevo helper unificado en rag_retriever.py:\n`def get_user_collection(email):\n    return email.replace("@","_at_").replace(".","_dot_")`\nEsto elimina lógica duplicada en 3 archivos.',
    'Buildando imagen v1.5.6 con fix de concurrencia en ThreadPoolExecutor. El bug era un race condition en tools_map — ahora cada request crea su propio scope. ETA: 10min.',
  ],
  arch: [
    'Propongo ADR-012: Hexagonal Architecture para el módulo de agentes. Puertos: AgentPort, ToolPort, StoragePort. Adaptadores: LangGraphAdapter, HttpToolAdapter, PostgresAdapter.',
    'El contrato API v2 para /api/chat incluye routing_hint: string | null para que el cliente sugiera intent. Reduce latencia del router ~40ms al saltarse el LLM fallback.',
    'Revisando el diagrama de secuencia: el flujo User→Orchestrator→Bus→Agent→Supervisor tiene 4 saltos de red. Podemos colapsar Bus→Agent para intents de baja complejidad.',
  ],
  sre: [
    'Métricas actuales — p50: 420ms, p95: 780ms, p99: 1.2s. Error budget /api/chat: 72% consumido (SLO 99.5%). Dentro del rango, pero recomiendo revisar si Ollama GPU tiene throttling.',
    '`amael_sre_loop_runs_total{result="ok_clean"}` = 847 últimas 24h. 0 restarts automáticos, 0 escalaciones. Circuit breaker cerrado — sistema estable.',
    'Alerta predictiva: predict_linear proyecta OOM en backend-ia en ~2h. Recomiendo rollout restart preventivo: `kubectl rollout restart deployment/backend-ia -n amael-ia`.',
  ],
}

const PIPELINE_STEPS = [
  { n: 1, title: 'Input del usuario',          desc: 'La solicitud llega al orquestador via REST o WhatsApp Bridge. Se valida, sanitiza (max 4000 chars) y clasifica por intent.' },
  { n: 2, title: 'Router & clasificación',     desc: 'AgentRouter analiza el intent con keyword matching (conf 0.9) y fallback LLM. Genera RoutingDecision con agentes destino y confianza.' },
  { n: 3, title: 'Dispatch directo o pipeline', desc: 'AgentDispatcher elige la ruta: intents simples (cto/dev/arch/sre/productivity) van directo al agente especializado. Intents complejos (general/k8s) pasan al pipeline LangGraph completo.' },
  { n: 4, title: 'Procesamiento por agente',   desc: 'Agentes ejecutan pipeline LangGraph: planner → grouper → batch_executor → supervisor. Herramientas: RAG, k8s, productivity, web.' },
  { n: 5, title: 'Respuesta & escalación',     desc: 'El supervisor evalúa calidad (0-10). Score < 6 activa REPLAN (max 1 retry). Agentes pueden escalar a otros via message bus.' },
  { n: 6, title: 'Agregación & entrega',       desc: 'El orquestador agrega respuestas, aplica sanitize_output() para redactar tokens, y entrega al usuario via JSON o SSE streaming.' },
]

const RESPONSIBILITIES = [
  { agent: 'CTO',  color: '#7F77DD', items: ['Estrategia técnica', 'Decisiones de arquitectura', 'Roadmap y priorización'] },
  { agent: 'DEV',  color: '#1D9E75', items: ['Implementación de features', 'Code reviews', 'CI/CD y testing'] },
  { agent: 'ARCH', color: '#D85A30', items: ['Diseño de sistemas', 'API contracts (ADRs)', 'Documentación técnica'] },
  { agent: 'SRE',  color: '#185FA5', items: ['Observabilidad y SLOs', 'Alertas e incidents', 'Auto-healing loop'] },
]

const STACK_ITEMS = [
  { key: 'LLM',          val: 'qwen2.5:14b (Ollama)'         },
  { key: 'Embeddings',   val: 'nomic-embed-text'              },
  { key: 'Orquestador',  val: 'LangGraph + FastAPI'           },
  { key: 'Vector DB',    val: 'Qdrant (per-user collections)' },
  { key: 'Storage',      val: 'PostgreSQL + Redis + MinIO'    },
  { key: 'API Gateway',  val: 'Nginx + Kong (LLM adapter)'    },
  { key: 'Observabilidad', val: 'Prometheus + Grafana + Tempo' },
  { key: 'Infra',        val: 'MicroK8s · RTX 5070 GPU'      },
]

const BACKEND = 'https://amael-ia.richardx.dev/api'

// ── Helpers ────────────────────────────────────────────────────────────────────
const now = () => new Date().toLocaleTimeString('es-MX', { hour: '2-digit', minute: '2-digit' })
const agentByID = (id: AgentId) => AGENTS.find(a => a.id === id)!
const rnd       = (min: number, max: number) => Math.floor(Math.random() * (max - min + 1)) + min

function sreLoopToStatuses(data: Record<string, unknown>): Record<AgentId, Status> {
  const cb      = data.circuit_breaker_state as string
  const maint   = data.maintenance_active   as boolean
  const enabled = data.loop_enabled          as boolean
  const result  = data.last_run_result       as string
  const actions = (data.actions_in_last_run  as number) ?? 0

  const sre: Status =
    !enabled || maint        ? 'Idle'     :
    cb === 'OPEN'            ? 'Thinking' :
    result === 'error'       ? 'Thinking' : 'Active'

  const dev: Status  = actions > 0 ? 'Thinking' : 'Active'

  return { cto: 'Active', dev, arch: 'Active', sre }
}

// ── Componente principal ───────────────────────────────────────────────────────
export default function AgentsPage() {
  const router = useRouter()

  const [activeTab,      setActiveTab]      = useState<Tab>('overview')
  const [selectedAgent,  setSelectedAgent]  = useState<number>(0)
  const [chatAgent,      setChatAgent]      = useState<AgentId>('cto')
  const [messages,       setMessages]       = useState<ChatMsg[]>([
    { id: 0, role: 'assistant', agent: 'cto', text: '¡Hola! Soy el CTO Agent de Amael-IA. ¿En qué puedo ayudarte hoy? Puedo orientarte en estrategia técnica, arquitectura de la plataforma o roadmap del proyecto.' },
  ])
  const [interactions,   setInteractions]   = useState<Interaction[]>(INIT_INTERACTIONS)
  const [chatInput,      setChatInput]      = useState('')
  const [thinking,       setThinking]       = useState(false)
  const [agentStatuses,  setAgentStatuses]  = useState<Record<AgentId, Status>>({
    cto: 'Active', dev: 'Thinking', arch: 'Active', sre: 'Idle',
  })

  const messagesEndRef    = useRef<HTMLDivElement>(null)
  const interactionEndRef = useRef<HTMLDivElement>(null)
  const interactionIdRef  = useRef(INIT_INTERACTIONS.length + 1)
  const chatTimeoutRef    = useRef<ReturnType<typeof setTimeout> | null>(null)

  // Polling de estado real del SRE loop (cada 30s)
  useEffect(() => {
    const fetchStatus = () => {
      fetch(`${BACKEND}/sre/loop/status`)
        .then(r => r.ok ? r.json() : Promise.reject(r.status))
        .then(data => {
          const next = sreLoopToStatuses(data)
          setAgentStatuses(prev => {
            const changed = (Object.keys(next) as AgentId[]).some(k => prev[k] !== next[k])
            return changed ? next : prev
          })
        })
        .catch(() => {
          setAgentStatuses(prev => {
            const idle: Record<AgentId, Status> = { cto: 'Idle', dev: 'Idle', arch: 'Idle', sre: 'Idle' }
            const changed = (Object.keys(idle) as AgentId[]).some(k => prev[k] !== idle[k])
            return changed ? idle : prev
          })
        })
    }
    fetchStatus()
    const iv = setInterval(fetchStatus, 30_000)
    return () => clearInterval(iv)
  }, [])

  // Cleanup chat timeout on unmount
  useEffect(() => () => { if (chatTimeoutRef.current) clearTimeout(chatTimeoutRef.current) }, [])

  // Auto-scroll chat
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages, thinking])

  // Auto-scroll interactions
  useEffect(() => {
    interactionEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [interactions])

  // ── Handlers ──────────────────────────────────────────────────────────────
  const sendMessage = () => {
    const text = chatInput.trim()
    if (!text || thinking) return
    setChatInput('')

    const userMsg: ChatMsg = { id: Date.now(), role: 'user', agent: chatAgent, text }
    setMessages(prev => [...prev, userMsg])
    setThinking(true)

    chatTimeoutRef.current = setTimeout(() => {
      const pool = AGENT_RESPONSES[chatAgent]
      const reply = pool[rnd(0, pool.length - 1)]
      setMessages(prev => [...prev, { id: Date.now() + 1, role: 'assistant', agent: chatAgent, text: reply }])
      setThinking(false)
    }, rnd(1500, 2500))
  }

  const simulateInteraction = () => {
    const template = RAND_INTERACTIONS[rnd(0, RAND_INTERACTIONS.length - 1)]
    const newItem: Interaction = { ...template, id: interactionIdRef.current++, ts: now() }
    setInteractions(prev => [newItem, ...prev].slice(0, 20))
  }

  const switchChatAgent = (id: AgentId) => {
    setChatAgent(id)
    const ag = agentByID(id)
    setMessages([{
      id: Date.now(),
      role: 'assistant',
      agent: id,
      text: `Hola, soy el ${ag.name}. Mi especialidad es ${ag.role.toLowerCase()}. ¿Cómo puedo ayudarte?`,
    }])
  }

  // ── Estilos base ──────────────────────────────────────────────────────────
  const card: React.CSSProperties = {
    background: 'var(--bg-surface)',
    border: '1px solid var(--border)',
    borderRadius: '12px',
    padding: '16px',
  }

  const tabBtn = (t: Tab): React.CSSProperties => ({
    background: activeTab === t ? 'var(--primary-subtle)' : 'none',
    border: `1px solid ${activeTab === t ? 'var(--primary)' : 'transparent'}`,
    color: activeTab === t ? 'var(--primary)' : 'var(--text-secondary)',
    borderRadius: '8px',
    padding: '8px 18px',
    fontSize: '13px',
    fontWeight: activeTab === t ? 600 : 400,
    cursor: 'pointer',
    transition: 'all .15s',
  })

  const metricCard = (label: string, value: string, sub: string, color: string) => (
    <div key={label} style={{ ...card, flex: 1, minWidth: '130px' }}>
      <div style={{ fontSize: '11px', color: 'var(--text-secondary)', textTransform: 'uppercase', letterSpacing: '0.06em', marginBottom: '6px' }}>{label}</div>
      <div style={{ fontSize: '28px', fontWeight: 700, color, fontFamily: 'monospace' }}>{value}</div>
      <div style={{ fontSize: '11px', color: 'var(--text-secondary)', marginTop: '4px' }}>{sub}</div>
    </div>
  )

  // ── Render ────────────────────────────────────────────────────────────────
  return (
    <div style={{ minHeight: '100dvh', background: 'var(--bg-base)', color: 'var(--text-primary)', fontFamily: 'Inter, system-ui, sans-serif' }}>

      {/* ── Top header bar ─────────────────────────────────────────────────── */}
      <div style={{
        display: 'flex', alignItems: 'center', gap: '16px',
        padding: '0 24px', height: '56px',
        background: 'var(--bg-surface)', borderBottom: '1px solid var(--border)',
        position: 'sticky', top: 0, zIndex: 50,
      }}>
        <button
          onClick={() => router.push('/')}
          style={{
            background: 'none', border: 'none', cursor: 'pointer',
            color: 'var(--text-secondary)', fontSize: '13px', display: 'flex',
            alignItems: 'center', gap: '6px', padding: '6px 10px',
            borderRadius: '6px', transition: 'color .15s',
          }}
          onMouseEnter={e => (e.currentTarget.style.color = 'var(--text-primary)')}
          onMouseLeave={e => (e.currentTarget.style.color = 'var(--text-secondary)')}
        >
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
            <path d="M19 12H5M12 5l-7 7 7 7" />
          </svg>
          Volver
        </button>

        <div style={{ width: '1px', height: '20px', background: 'var(--border)' }} />

        <div style={{
          width: '26px', height: '26px', borderRadius: '6px',
          background: 'var(--primary)', display: 'flex', alignItems: 'center',
          justifyContent: 'center', fontSize: '13px', fontWeight: 700, color: '#fff',
        }}>A</div>

        <span style={{ fontSize: '15px', fontWeight: 700, letterSpacing: '-0.3px' }}>
          Agent Organization
        </span>

        <div style={{ display: 'flex', alignItems: 'center', gap: '6px', marginLeft: '4px' }}>
          <span style={{
            width: '8px', height: '8px', borderRadius: '50%', background: '#1D9E75',
            display: 'inline-block',
            boxShadow: '0 0 0 0 rgba(29,158,117,0.4)',
            animation: 'pulse-green 2s infinite',
          }} />
          <span style={{ fontSize: '12px', color: '#1D9E75', fontWeight: 500 }}>4 agents online</span>
        </div>

        <div style={{ marginLeft: 'auto', display: 'flex', alignItems: 'center', gap: '8px' }}>
          <code style={{
            fontSize: '11px', color: 'var(--text-secondary)',
            background: 'var(--bg-elevated)', padding: '3px 8px', borderRadius: '4px',
          }}>
            branch: develop
          </code>
        </div>

        <style>{`
          @keyframes pulse-green {
            0%   { box-shadow: 0 0 0 0 rgba(29,158,117,0.6); }
            70%  { box-shadow: 0 0 0 6px rgba(29,158,117,0); }
            100% { box-shadow: 0 0 0 0 rgba(29,158,117,0); }
          }
          @keyframes agent-pulse {
            0%,100% { box-shadow: 0 0 0 0 rgba(127,119,221,0.5); }
            50%      { box-shadow: 0 0 0 5px rgba(127,119,221,0); }
          }
          @keyframes bounce-dot {
            0%,80%,100% { transform: scale(0); opacity: 0.4; }
            40%          { transform: scale(1); opacity: 1; }
          }
        `}</style>
      </div>

      {/* ── Dos columnas ───────────────────────────────────────────────────── */}
      <div style={{ display: 'flex', height: 'calc(100dvh - 56px)' }}>

        {/* ── Panel izquierdo ─────────────────────────────────────────────── */}
        <div style={{
          width: '280px', minWidth: '280px',
          background: 'var(--sidebar-bg)', borderRight: '1px solid var(--border)',
          overflowY: 'auto', padding: '16px 12px', display: 'flex', flexDirection: 'column', gap: '8px',
        }}>

          {/* Agent cards */}
          {AGENTS.map((ag, i) => (
            <div
              key={ag.id}
              onClick={() => setSelectedAgent(i)}
              style={{
                ...card,
                padding: '14px',
                cursor: 'pointer',
                borderLeft: selectedAgent === i ? `3px solid ${ag.color}` : '1px solid var(--border)',
                paddingLeft: selectedAgent === i ? '13px' : '14px',
                transition: 'all .15s',
                background: selectedAgent === i ? 'var(--bg-elevated)' : 'var(--bg-surface)',
              }}
            >
              <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
                {/* Avatar */}
                <div style={{
                  width: '36px', height: '36px', borderRadius: '9px', flexShrink: 0,
                  background: ag.color, display: 'flex', alignItems: 'center',
                  justifyContent: 'center', fontSize: '11px', fontWeight: 700, color: '#fff',
                  boxShadow: agentStatuses[ag.id] === 'Active' ? `0 0 0 0 ${ag.color}80` : 'none',
                  animation: agentStatuses[ag.id] === 'Active' ? 'agent-pulse 2.5s infinite' : 'none',
                }}>
                  {ag.id.toUpperCase()}
                </div>
                <div style={{ flex: 1, minWidth: 0 }}>
                  <div style={{ fontSize: '13px', fontWeight: 600, marginBottom: '2px' }}>{ag.name}</div>
                  <div style={{ fontSize: '11px', color: 'var(--text-secondary)', whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>{ag.role}</div>
                </div>
              </div>
              <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginTop: '10px' }}>
                {/* Status badge */}
                <span style={{
                  fontSize: '11px', fontWeight: 500, padding: '2px 8px', borderRadius: '99px',
                  background: agentStatuses[ag.id] === 'Active'   ? 'rgba(29,158,117,.18)'  :
                              agentStatuses[ag.id] === 'Thinking' ? 'rgba(127,119,221,.18)' : 'rgba(148,163,184,.12)',
                  color:      agentStatuses[ag.id] === 'Active'   ? '#1D9E75' :
                              agentStatuses[ag.id] === 'Thinking' ? '#7F77DD'  : 'var(--text-secondary)',
                }}>
                  {agentStatuses[ag.id] === 'Active' && '● '}{agentStatuses[ag.id]}
                </span>
                <span style={{ fontSize: '11px', color: 'var(--text-secondary)' }}>{ag.tasks} tareas</span>
              </div>
            </div>
          ))}

          {/* Proyecto section */}
          <div style={{ ...card, padding: '14px', marginTop: '8px' }}>
            <div style={{ fontSize: '11px', fontWeight: 600, color: 'var(--text-disabled)', textTransform: 'uppercase', letterSpacing: '0.06em', marginBottom: '10px' }}>Proyecto</div>
            <div style={{ fontSize: '13px', fontWeight: 600, marginBottom: '2px' }}>Amael-AgenticIA</div>
            <div style={{ fontSize: '11px', color: 'var(--text-secondary)', marginBottom: '10px', fontFamily: 'monospace' }}>branch: develop</div>
            <div style={{ display: 'flex', flexWrap: 'wrap', gap: '4px' }}>
              {['Python', 'FastAPI', 'Claude API', 'GitHub'].map(t => (
                <span key={t} style={{
                  fontSize: '10px', padding: '2px 7px', borderRadius: '4px',
                  background: 'var(--bg-elevated)', border: '1px solid var(--border)',
                  color: 'var(--text-secondary)',
                }}>{t}</span>
              ))}
            </div>
          </div>

          {/* Dispatcher routes */}
          <div style={{ ...card, padding: '14px' }}>
            <div style={{ fontSize: '11px', fontWeight: 600, color: 'var(--text-disabled)', textTransform: 'uppercase', letterSpacing: '0.06em', marginBottom: '10px' }}>Dispatcher</div>

            <div style={{ marginBottom: '10px' }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '6px', marginBottom: '5px' }}>
                <span style={{ width: '8px', height: '8px', borderRadius: '50%', background: '#1D9E75', flexShrink: 0 }} />
                <span style={{ fontSize: '11px', fontWeight: 600, color: '#1D9E75' }}>Direct</span>
                <span style={{ fontSize: '10px', color: 'var(--text-secondary)', marginLeft: 'auto' }}>5 intents</span>
              </div>
              {['cto', 'dev', 'arch', 'sre', 'productivity'].map(i => (
                <div key={i} style={{ fontSize: '10px', color: 'var(--text-secondary)', paddingLeft: '14px', marginBottom: '1px' }}>→ {i}</div>
              ))}
            </div>

            <div style={{ borderTop: '1px solid var(--border)', paddingTop: '10px' }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '6px', marginBottom: '5px' }}>
                <span style={{ width: '8px', height: '8px', borderRadius: '50%', background: '#7F77DD', flexShrink: 0 }} />
                <span style={{ fontSize: '11px', fontWeight: 600, color: '#7F77DD' }}>Pipeline LangGraph</span>
                <span style={{ fontSize: '10px', color: 'var(--text-secondary)', marginLeft: 'auto' }}>5 intents</span>
              </div>
              {['general', 'kubernetes', 'monitoring', 'qa', 'memory'].map(i => (
                <div key={i} style={{ fontSize: '10px', color: 'var(--text-secondary)', paddingLeft: '14px', marginBottom: '1px' }}>→ {i}</div>
              ))}
            </div>
          </div>
        </div>

        {/* ── Área principal ──────────────────────────────────────────────── */}
        <div style={{ flex: 1, overflowY: 'auto', padding: '24px' }}>

          {/* Tabs */}
          <div style={{ display: 'flex', gap: '6px', marginBottom: '24px', flexWrap: 'wrap' }}>
            {(['overview', 'interactions', 'chat', 'pipeline'] as Tab[]).map(t => (
              <button key={t} onClick={() => setActiveTab(t)} style={tabBtn(t)}>
                {t === 'overview'      ? '📊 Overview'      :
                 t === 'interactions'  ? '🔀 Interacciones' :
                 t === 'chat'          ? '💬 Chat'          : '⚙️ Pipeline'}
              </button>
            ))}
          </div>

          {/* ── Tab: Overview ──────────────────────────────────────────────── */}
          {activeTab === 'overview' && (
            <div style={{ display: 'flex', flexDirection: 'column', gap: '24px' }}>

              {/* Métricas */}
              <div style={{ display: 'flex', gap: '12px', flexWrap: 'wrap' }}>
                {metricCard('Tareas activas',  '28',    'en este momento',   '#7F77DD')}
                {metricCard('Completadas',      '1,247', 'última semana',     '#1D9E75')}
                {metricCard('Tokens usados',    '2.4M',  'en 24h',            'var(--text-primary)')}
                {metricCard('Latencia media',   '620ms', 'p50 end-to-end',    '#D85A30')}
              </div>

              {/* Architecture SVG */}
              <div style={{ ...card }}>
                <div style={{ fontSize: '13px', fontWeight: 600, marginBottom: '16px', color: 'var(--text-secondary)' }}>Arquitectura del sistema</div>
                <svg viewBox="0 0 700 300" style={{ width: '100%', maxWidth: '700px', overflow: 'visible' }}>
                  <defs>
                    <marker id="arr"  markerWidth="8" markerHeight="8" refX="6" refY="3" orient="auto"><path d="M0,0 L0,6 L8,3 z" fill="#6366f1" /></marker>
                    <marker id="arrG" markerWidth="8" markerHeight="8" refX="6" refY="3" orient="auto"><path d="M0,0 L0,6 L8,3 z" fill="#1D9E75" /></marker>
                    <marker id="arrP" markerWidth="8" markerHeight="8" refX="6" refY="3" orient="auto"><path d="M0,0 L0,6 L8,3 z" fill="#7F77DD" /></marker>
                    <marker id="arrS" markerWidth="8" markerHeight="8" refX="6" refY="3" orient="auto"><path d="M0,0 L0,6 L8,3 z" fill="#94a3b8" /></marker>
                  </defs>

                  {/* Usuario */}
                  <rect x="290" y="8" width="120" height="38" rx="8" fill="#1a2236" stroke="#6366f1" strokeWidth="1.5" />
                  <text x="350" y="23" textAnchor="middle" fill="#f1f5f9" fontSize="11" fontWeight="600">Usuario</text>
                  <text x="350" y="38" textAnchor="middle" fill="#94a3b8" fontSize="9">REST · WhatsApp Bridge</text>

                  {/* User → Router */}
                  <line x1="350" y1="46" x2="350" y2="62" stroke="#6366f1" strokeWidth="1.5" markerEnd="url(#arr)" />

                  {/* AgentRouter */}
                  <rect x="175" y="62" width="350" height="38" rx="8" fill="#1a2236" stroke="#6366f1" strokeWidth="1.5" />
                  <text x="350" y="77" textAnchor="middle" fill="#f1f5f9" fontSize="11" fontWeight="600">AgentRouter</text>
                  <text x="350" y="92" textAnchor="middle" fill="#94a3b8" fontSize="9">keyword match (conf 0.9) · LLM fallback → RoutingDecision</text>

                  {/* Router → Dispatcher */}
                  <line x1="350" y1="100" x2="350" y2="116" stroke="#6366f1" strokeWidth="1.5" markerEnd="url(#arr)" />

                  {/* AgentDispatcher */}
                  <rect x="220" y="116" width="260" height="38" rx="8" fill="#1a2236" stroke="#6366f1" strokeWidth="1.5" />
                  <text x="350" y="131" textAnchor="middle" fill="#f1f5f9" fontSize="11" fontWeight="600">AgentDispatcher</text>
                  <text x="350" y="146" textAnchor="middle" fill="#94a3b8" fontSize="9">elige ruta según intent</text>

                  {/* Fork left: Dispatcher → Direct */}
                  <polyline points="220,135 115,135 115,168" fill="none" stroke="#1D9E75" strokeWidth="1.5" markerEnd="url(#arrG)" />
                  <rect x="62" y="126" width="46" height="16" rx="4" fill="rgba(29,158,117,0.15)" />
                  <text x="85" y="137" textAnchor="middle" fill="#1D9E75" fontSize="9" fontWeight="700">Direct</text>

                  {/* Fork right: Dispatcher → Pipeline */}
                  <polyline points="480,135 580,135 580,168" fill="none" stroke="#7F77DD" strokeWidth="1.5" markerEnd="url(#arrP)" />
                  <rect x="550" y="126" width="58" height="16" rx="4" fill="rgba(127,119,221,0.15)" />
                  <text x="579" y="137" textAnchor="middle" fill="#7F77DD" fontSize="9" fontWeight="700">Pipeline</text>

                  {/* Direct dispatch box */}
                  <rect x="15" y="168" width="200" height="84" rx="8" fill="#1a2236" stroke="#1D9E75" strokeWidth="1.5" />
                  <text x="115" y="183" textAnchor="middle" fill="#1D9E75" fontSize="10" fontWeight="700">Direct dispatch</text>
                  <text x="115" y="197" textAnchor="middle" fill="#94a3b8" fontSize="9">cto · dev · arch</text>
                  <text x="115" y="210" textAnchor="middle" fill="#94a3b8" fontSize="9">sre · productivity</text>
                  <line x1="30" y1="218" x2="200" y2="218" stroke="#1D9E75" strokeWidth="0.5" strokeOpacity="0.4" />
                  <text x="115" y="231" textAnchor="middle" fill="#1D9E75" fontSize="8">agent.run(task)</text>
                  <text x="115" y="243" textAnchor="middle" fill="#94a3b8" fontSize="8">sin LangGraph</text>

                  {/* LangGraph pipeline box */}
                  <rect x="465" y="168" width="220" height="84" rx="8" fill="#1a2236" stroke="#7F77DD" strokeWidth="1.5" />
                  <text x="575" y="183" textAnchor="middle" fill="#7F77DD" fontSize="10" fontWeight="700">LangGraph pipeline</text>
                  <text x="575" y="197" textAnchor="middle" fill="#94a3b8" fontSize="9">planner → grouper</text>
                  <text x="575" y="210" textAnchor="middle" fill="#94a3b8" fontSize="9">executor → supervisor</text>
                  <line x1="480" y1="218" x2="670" y2="218" stroke="#7F77DD" strokeWidth="0.5" strokeOpacity="0.4" />
                  <text x="575" y="231" textAnchor="middle" fill="#7F77DD" fontSize="8">run_workflow()</text>
                  <text x="575" y="243" textAnchor="middle" fill="#94a3b8" fontSize="8">general · k8s · monitoring</text>

                  {/* Storage row */}
                  <line x1="115" y1="252" x2="115" y2="268" stroke="#94a3b8" strokeWidth="1" strokeDasharray="3,2" markerEnd="url(#arrS)" />
                  <line x1="575" y1="252" x2="575" y2="268" stroke="#94a3b8" strokeWidth="1" strokeDasharray="3,2" markerEnd="url(#arrS)" />
                  <rect x="80" y="268" width="540" height="26" rx="8" fill="#111827" stroke="#334155" strokeWidth="1" />
                  <text x="350" y="284" textAnchor="middle" fill="#64748b" fontSize="9">Qdrant · PostgreSQL · Redis · MinIO · Ollama (qwen2.5:14b)</text>
                </svg>
              </div>

              {/* Responsabilidades + Stack */}
              <div style={{ display: 'flex', gap: '16px', flexWrap: 'wrap' }}>
                <div style={{ ...card, flex: 1, minWidth: '280px' }}>
                  <div style={{ fontSize: '13px', fontWeight: 600, marginBottom: '14px', color: 'var(--text-secondary)' }}>Responsabilidades por agente</div>
                  {RESPONSIBILITIES.map(r => (
                    <div key={r.agent} style={{ marginBottom: '12px' }}>
                      <div style={{ fontSize: '12px', fontWeight: 600, marginBottom: '4px', color: r.color }}>{r.agent}</div>
                      {r.items.map(item => (
                        <div key={item} style={{ display: 'flex', alignItems: 'center', gap: '6px', fontSize: '12px', color: 'var(--text-secondary)', marginBottom: '2px' }}>
                          <span style={{ width: '6px', height: '6px', borderRadius: '50%', background: r.color, flexShrink: 0 }} />
                          {item}
                        </div>
                      ))}
                    </div>
                  ))}
                </div>

                <div style={{ ...card, flex: 1, minWidth: '240px' }}>
                  <div style={{ fontSize: '13px', fontWeight: 600, marginBottom: '14px', color: 'var(--text-secondary)' }}>Stack de implementación</div>
                  {STACK_ITEMS.map(s => (
                    <div key={s.key} style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', padding: '8px 0', borderBottom: '1px solid var(--border)' }}>
                      <span style={{ fontSize: '12px', color: 'var(--text-secondary)' }}>{s.key}</span>
                      <code style={{ fontSize: '11px', color: 'var(--primary)', background: 'var(--primary-subtle)', padding: '2px 7px', borderRadius: '4px' }}>{s.val}</code>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          )}

          {/* ── Tab: Interacciones ─────────────────────────────────────────── */}
          {activeTab === 'interactions' && (
            <div style={{ display: 'flex', flexDirection: 'column', gap: '24px' }}>

              {/* Métricas */}
              <div style={{ display: 'flex', gap: '12px', flexWrap: 'wrap' }}>
                {metricCard('Mensajes totales', '3,241', 'histórico',      'var(--text-primary)')}
                {metricCard('Inter-agente',      '2,187', '67% del total',  '#7F77DD')}
                {metricCard('User → Agente',     '1,054', '33% del total',  '#1D9E75')}
                {metricCard('Escalaciones',       '48',    'last 24h',       '#D85A30')}
              </div>

              <div style={{ display: 'flex', gap: '16px', flexWrap: 'wrap' }}>
                {/* Feed de interacciones */}
                <div style={{ ...card, flex: 1, minWidth: '300px' }}>
                  <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '14px' }}>
                    <span style={{ fontSize: '13px', fontWeight: 600, color: 'var(--text-secondary)' }}>Feed de mensajes</span>
                    <button
                      onClick={simulateInteraction}
                      style={{
                        background: 'var(--primary-subtle)', border: '1px solid var(--primary)',
                        color: 'var(--primary)', borderRadius: '6px', padding: '5px 12px',
                        fontSize: '12px', cursor: 'pointer', fontWeight: 500,
                      }}
                    >
                      + Simular evento
                    </button>
                  </div>
                  <div style={{ maxHeight: '320px', overflowY: 'auto', display: 'flex', flexDirection: 'column', gap: '8px' }}>
                    {interactions.map(item => {
                      const fromAg = agentByID(item.from)
                      const toAg   = agentByID(item.to)
                      return (
                        <div key={item.id} style={{
                          display: 'flex', gap: '10px', alignItems: 'flex-start',
                          padding: '10px', borderRadius: '8px', background: 'var(--bg-elevated)',
                          animation: 'fadeUp .25s ease-out',
                        }}>
                          <div style={{
                            width: '30px', height: '30px', borderRadius: '7px', flexShrink: 0,
                            background: fromAg.color, display: 'flex', alignItems: 'center',
                            justifyContent: 'center', fontSize: '9px', fontWeight: 700, color: '#fff',
                          }}>
                            {fromAg.id.toUpperCase()}
                          </div>
                          <div style={{ flex: 1, minWidth: 0 }}>
                            <div style={{ fontSize: '11px', fontWeight: 600, marginBottom: '2px' }}>
                              <span style={{ color: fromAg.color }}>{fromAg.name}</span>
                              <span style={{ color: 'var(--text-disabled)' }}> → </span>
                              <span style={{ color: toAg.color }}>{toAg.name}</span>
                            </div>
                            <div style={{ fontSize: '12px', color: 'var(--text-secondary)', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                              {item.msg}
                            </div>
                          </div>
                          <span style={{ fontSize: '10px', color: 'var(--text-disabled)', flexShrink: 0 }}>{item.ts}</span>
                        </div>
                      )
                    })}
                    <div ref={interactionEndRef} />
                  </div>
                </div>

                {/* Grafo SVG */}
                <div style={{ ...card, minWidth: '260px', flex: '0 0 300px' }}>
                  <div style={{ fontSize: '13px', fontWeight: 600, marginBottom: '14px', color: 'var(--text-secondary)' }}>Grafo de interacciones</div>
                  <svg viewBox="0 0 280 280" style={{ width: '100%' }}>
                    {/* Edges con stroke proporcional al volumen */}
                    <line x1="140" y1="60"  x2="80"  y2="140" stroke="#7F77DD" strokeWidth="2.5" opacity="0.6" />
                    <line x1="140" y1="60"  x2="200" y2="140" stroke="#D85A30" strokeWidth="2"   opacity="0.5" />
                    <line x1="80"  y1="140" x2="140" y2="200" stroke="#1D9E75" strokeWidth="3"   opacity="0.6" />
                    <line x1="200" y1="140" x2="140" y2="200" stroke="#185FA5" strokeWidth="1.5" opacity="0.5" />
                    <line x1="80"  y1="140" x2="200" y2="140" stroke="#6366f1" strokeWidth="1"   opacity="0.35" strokeDasharray="4,3" />
                    {/* Orchestrator center */}
                    <circle cx="140" cy="140" r="28" fill="#1a2236" stroke="#6366f1" strokeWidth="1.5" />
                    <text x="140" y="136" textAnchor="middle" fill="#f1f5f9" fontSize="9" fontWeight="600">Orches-</text>
                    <text x="140" y="148" textAnchor="middle" fill="#f1f5f9" fontSize="9" fontWeight="600">trator</text>
                    {/* User node top */}
                    <circle cx="140" cy="24" r="18" fill="#1a2236" stroke="#6366f1" strokeWidth="1.5" />
                    <text x="140" y="28" textAnchor="middle" fill="#94a3b8" fontSize="9">User</text>
                    <line x1="140" y1="42" x2="140" y2="112" stroke="#6366f1" strokeWidth="1.5" strokeDasharray="4,3" />
                    {/* Agent nodes */}
                    {[
                      { cx: 48,  cy: 80,  id: 'CTO',  color: '#7F77DD' },
                      { cx: 48,  cy: 200, id: 'DEV',  color: '#1D9E75' },
                      { cx: 232, cy: 80,  id: 'ARCH', color: '#D85A30' },
                      { cx: 232, cy: 200, id: 'SRE',  color: '#185FA5' },
                    ].map(n => (
                      <g key={n.id}>
                        <circle cx={n.cx} cy={n.cy} r="22" fill="#1a2236" stroke={n.color} strokeWidth="1.5" />
                        <text x={n.cx} y={n.cy + 4} textAnchor="middle" fill={n.color} fontSize="10" fontWeight="700">{n.id}</text>
                        <line x1={n.cx < 140 ? n.cx + 22 : n.cx - 22} y1={n.cy} x2={n.cx < 140 ? 112 : 168} y2="140" stroke={n.color} strokeWidth="1.5" opacity="0.5" />
                      </g>
                    ))}
                    {/* Edge labels */}
                    <text x="88"  y="100" fill="#7F77DD" fontSize="8" opacity="0.8">87 msg</text>
                    <text x="168" y="100" fill="#D85A30" fontSize="8" opacity="0.8">54 msg</text>
                    <text x="78"  y="182" fill="#1D9E75" fontSize="8" opacity="0.8">102 msg</text>
                    <text x="166" y="182" fill="#185FA5" fontSize="8" opacity="0.8">38 msg</text>
                  </svg>
                </div>
              </div>
            </div>
          )}

          {/* ── Tab: Chat ──────────────────────────────────────────────────── */}
          {activeTab === 'chat' && (
            <div style={{ display: 'flex', flexDirection: 'column', gap: '16px' }}>

              {/* Agent selector chips */}
              <div style={{ display: 'flex', gap: '8px', flexWrap: 'wrap' }}>
                {AGENTS.map(ag => (
                  <button
                    key={ag.id}
                    onClick={() => switchChatAgent(ag.id)}
                    style={{
                      display: 'flex', alignItems: 'center', gap: '8px',
                      padding: '7px 14px', borderRadius: '99px', cursor: 'pointer',
                      background: chatAgent === ag.id ? ag.color : 'var(--bg-surface)',
                      border: `1.5px solid ${chatAgent === ag.id ? ag.color : 'var(--border)'}`,
                      color: chatAgent === ag.id ? '#fff' : 'var(--text-secondary)',
                      fontSize: '12px', fontWeight: chatAgent === ag.id ? 600 : 400,
                      transition: 'all .15s',
                    }}
                  >
                    <span style={{
                      width: '20px', height: '20px', borderRadius: '50%',
                      background: chatAgent === ag.id ? 'rgba(255,255,255,0.25)' : ag.color,
                      display: 'flex', alignItems: 'center', justifyContent: 'center',
                      fontSize: '8px', fontWeight: 700, color: '#fff',
                    }}>
                      {ag.id.toUpperCase().slice(0,3)}
                    </span>
                    {ag.name}
                  </button>
                ))}
              </div>

              {/* Chat area */}
              <div style={{ ...card, display: 'flex', flexDirection: 'column', height: '480px' }}>
                <div style={{ flex: 1, overflowY: 'auto', padding: '8px 4px', display: 'flex', flexDirection: 'column', gap: '12px' }}>
                  {messages.map(msg => {
                    const ag = agentByID(msg.agent)
                    return (
                      <div key={msg.id} style={{
                        display: 'flex',
                        flexDirection: msg.role === 'user' ? 'row-reverse' : 'row',
                        gap: '10px', alignItems: 'flex-end',
                      }}>
                        {msg.role === 'assistant' && (
                          <div style={{
                            width: '32px', height: '32px', borderRadius: '8px', flexShrink: 0,
                            background: ag.color, display: 'flex', alignItems: 'center',
                            justifyContent: 'center', fontSize: '9px', fontWeight: 700, color: '#fff',
                          }}>
                            {ag.id.toUpperCase()}
                          </div>
                        )}
                        <div style={{
                          maxWidth: '72%',
                          background:   msg.role === 'user' ? '#7F77DD'          : 'var(--bg-elevated)',
                          color:        msg.role === 'user' ? '#fff'              : 'var(--text-primary)',
                          padding: '10px 14px', borderRadius: msg.role === 'user' ? '16px 4px 16px 16px' : '4px 16px 16px 16px',
                          fontSize: '13px', lineHeight: '1.5',
                          whiteSpace: 'pre-wrap', wordBreak: 'break-word',
                        }}>
                          {msg.text}
                        </div>
                      </div>
                    )
                  })}

                  {/* Thinking animation */}
                  {thinking && (
                    <div style={{ display: 'flex', gap: '10px', alignItems: 'flex-end' }}>
                      <div style={{
                        width: '32px', height: '32px', borderRadius: '8px', flexShrink: 0,
                        background: agentByID(chatAgent).color,
                        display: 'flex', alignItems: 'center', justifyContent: 'center',
                        fontSize: '9px', fontWeight: 700, color: '#fff',
                      }}>
                        {chatAgent.toUpperCase()}
                      </div>
                      <div style={{ background: 'var(--bg-elevated)', padding: '12px 16px', borderRadius: '4px 16px 16px 16px', display: 'flex', gap: '4px', alignItems: 'center' }}>
                        {[0, 1, 2].map(i => (
                          <span key={i} style={{
                            width: '6px', height: '6px', borderRadius: '50%',
                            background: 'var(--text-secondary)', display: 'block',
                            animation: `bounce-dot 1.2s ${i * 0.2}s infinite ease-in-out`,
                          }} />
                        ))}
                      </div>
                    </div>
                  )}
                  <div ref={messagesEndRef} />
                </div>

                {/* Input bar */}
                <div style={{ display: 'flex', gap: '8px', paddingTop: '12px', borderTop: '1px solid var(--border)', marginTop: '8px' }}>
                  <input
                    value={chatInput}
                    onChange={e => setChatInput(e.target.value)}
                    onKeyDown={e => { if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); sendMessage() } }}
                    placeholder={`Mensaje al ${agentByID(chatAgent).name}…`}
                    disabled={thinking}
                    style={{
                      flex: 1, background: 'var(--input-bg)', border: '1px solid var(--border)',
                      borderRadius: '8px', padding: '10px 14px', fontSize: '13px',
                      color: 'var(--text-primary)', outline: 'none',
                    }}
                  />
                  <button
                    onClick={sendMessage}
                    disabled={!chatInput.trim() || thinking}
                    style={{
                      background: 'var(--primary)', border: 'none', borderRadius: '8px',
                      padding: '10px 18px', color: '#fff', fontSize: '13px', fontWeight: 600,
                      cursor: !chatInput.trim() || thinking ? 'default' : 'pointer',
                      opacity: !chatInput.trim() || thinking ? 0.5 : 1,
                      transition: 'opacity .15s',
                    }}
                  >
                    Enviar
                  </button>
                </div>
              </div>
            </div>
          )}

          {/* ── Tab: Pipeline ──────────────────────────────────────────────── */}
          {activeTab === 'pipeline' && (
            <div style={{ display: 'flex', flexDirection: 'column', gap: '24px' }}>

              {/* Pipeline steps */}
              <div style={{ ...card }}>
                <div style={{ fontSize: '13px', fontWeight: 600, marginBottom: '16px', color: 'var(--text-secondary)' }}>Flujo de procesamiento</div>
                <div style={{ display: 'flex', flexDirection: 'column', gap: '0' }}>
                  {PIPELINE_STEPS.map((step, i) => (
                    <div key={step.n} style={{ display: 'flex', gap: '16px', alignItems: 'flex-start' }}>
                      {/* Step number + connector */}
                      <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', flexShrink: 0 }}>
                        <div style={{
                          width: '32px', height: '32px', borderRadius: '50%',
                          background: 'var(--primary)', display: 'flex', alignItems: 'center',
                          justifyContent: 'center', fontSize: '12px', fontWeight: 700, color: '#fff',
                        }}>{step.n}</div>
                        {i < PIPELINE_STEPS.length - 1 && (
                          <div style={{ width: '2px', height: '32px', background: 'var(--border)', margin: '4px 0' }} />
                        )}
                      </div>
                      <div style={{ paddingBottom: i < PIPELINE_STEPS.length - 1 ? '8px' : 0, paddingTop: '4px' }}>
                        <div style={{ fontSize: '13px', fontWeight: 600, marginBottom: '4px' }}>{step.title}</div>
                        <div style={{ fontSize: '12px', color: 'var(--text-secondary)', lineHeight: '1.5' }}>{step.desc}</div>
                      </div>
                    </div>
                  ))}
                </div>
              </div>

              {/* Syntax-highlighted code */}
              <div style={{ ...card }}>
                <div style={{ fontSize: '13px', fontWeight: 600, marginBottom: '16px', color: 'var(--text-secondary)' }}>Skeleton de implementación</div>
                <pre style={{
                  margin: 0, padding: '20px', borderRadius: '8px',
                  background: '#0d1117', border: '1px solid var(--border)',
                  fontSize: '12px', lineHeight: '1.7', overflowX: 'auto',
                  fontFamily: '"JetBrains Mono", "Fira Code", Consolas, monospace',
                }}>
                  <code dangerouslySetInnerHTML={{ __html: `<span style="color:#7F77DD">from</span> <span style="color:#f1f5f9">langchain_core.messages</span> <span style="color:#7F77DD">import</span> <span style="color:#f1f5f9">SystemMessage, HumanMessage</span>\n<span style="color:#7F77DD">from</span> <span style="color:#f1f5f9">langgraph.graph</span> <span style="color:#7F77DD">import</span> <span style="color:#f1f5f9">StateGraph</span>\n\n<span style="color:#94a3b8"># ── CTOAgent ─────────────────────────────────────────</span>\n<span style="color:#7F77DD">class</span> <span style="color:#1D9E75">CTOAgent</span>:\n    <span style="color:#7F77DD">def</span> <span style="color:#f1f5f9">__init__</span>(<span style="color:#D85A30">self</span>, llm, tools_map):\n        <span style="color:#D85A30">self</span>.llm       = llm\n        <span style="color:#D85A30">self</span>.tools_map = tools_map\n        <span style="color:#D85A30">self</span>.graph     = <span style="color:#D85A30">self</span>._build_graph()\n\n    <span style="color:#7F77DD">def</span> <span style="color:#f1f5f9">_build_graph</span>(<span style="color:#D85A30">self</span>) -> <span style="color:#1D9E75">StateGraph</span>:\n        graph = <span style="color:#1D9E75">StateGraph</span>(<span style="color:#f1f5f9">AgentState</span>)\n        graph.add_node(<span style="color:#D85A30">"planner"</span>,    <span style="color:#D85A30">self</span>.plan_step)\n        graph.add_node(<span style="color:#D85A30">"executor"</span>,   <span style="color:#D85A30">self</span>.execute_step)\n        graph.add_node(<span style="color:#D85A30">"supervisor"</span>, <span style="color:#D85A30">self</span>.supervise_step)\n        graph.set_entry_point(<span style="color:#D85A30">"planner"</span>)\n        <span style="color:#7F77DD">return</span> graph.compile()\n\n    <span style="color:#7F77DD">async def</span> <span style="color:#f1f5f9">run</span>(<span style="color:#D85A30">self</span>, question: <span style="color:#1D9E75">str</span>) -> <span style="color:#1D9E75">str</span>:\n        result = <span style="color:#7F77DD">await</span> <span style="color:#D85A30">self</span>.graph.ainvoke(\n            {<span style="color:#D85A30">"question"</span>: question, <span style="color:#D85A30">"tools_map"</span>: <span style="color:#D85A30">self</span>.tools_map}\n        )\n        <span style="color:#7F77DD">return</span> result[<span style="color:#D85A30">"final_answer"</span>]\n\n<span style="color:#94a3b8"># ── MessageBus ───────────────────────────────────────</span>\n<span style="color:#7F77DD">class</span> <span style="color:#1D9E75">MessageBus</span>:\n    <span style="color:#7F77DD">def</span> <span style="color:#f1f5f9">__init__</span>(<span style="color:#D85A30">self</span>, redis_client):\n        <span style="color:#D85A30">self</span>.redis    = redis_client\n        <span style="color:#D85A30">self</span>.stream   = <span style="color:#D85A30">"amael:agent:bus"</span>\n        <span style="color:#D85A30">self</span>.agents   = {}  <span style="color:#94a3b8"># agent_id → CTOAgent | DevAgent | ...</span>\n\n    <span style="color:#7F77DD">async def</span> <span style="color:#f1f5f9">publish</span>(<span style="color:#D85A30">self</span>, task: <span style="color:#1D9E75">dict</span>) -> <span style="color:#1D9E75">str</span>:\n        msg_id = <span style="color:#7F77DD">await</span> <span style="color:#D85A30">self</span>.redis.xadd(<span style="color:#D85A30">self</span>.stream, task)\n        <span style="color:#7F77DD">return</span> msg_id\n\n    <span style="color:#7F77DD">async def</span> <span style="color:#f1f5f9">consume</span>(<span style="color:#D85A30">self</span>, agent_id: <span style="color:#1D9E75">str</span>):\n        <span style="color:#7F77DD">async for</span> msg <span style="color:#7F77DD">in</span> <span style="color:#D85A30">self</span>.redis.xread({<span style="color:#D85A30">self</span>.stream: <span style="color:#D85A30">"$"</span>}, block=<span style="color:#185FA5">0</span>):\n            agent = <span style="color:#D85A30">self</span>.agents[agent_id]\n            response = <span style="color:#7F77DD">await</span> agent.run(msg[<span style="color:#D85A30">"question"</span>])\n            <span style="color:#7F77DD">await</span> <span style="color:#D85A30">self</span>.redis.xadd(<span style="color:#D85A30">f"amael:response:{agent_id}"</span>, {<span style="color:#D85A30">"answer"</span>: response})` }} />
                </pre>
              </div>
            </div>
          )}

        </div>
      </div>
    </div>
  )
}
