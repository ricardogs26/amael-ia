'use client'
import { useEffect, useRef, useState, useCallback } from 'react'
import Message from './Message'
import Sidebar from './Sidebar'

const BACKEND = 'https://amael-ia.richardx.dev/api'

interface Msg {
  role: 'user' | 'assistant'
  content: string
  ts: string
}

interface Conv {
  id: number
  title: string
  last_active_at: string
}

interface Props {
  token: string
  userName: string
  userPicture: string
  onLogout: () => void
}

// ── Status banner shown while agent runs ─────────────────────────────────────
function StatusBanner({ msg }: { msg: string }) {
  return (
    <div style={{
      display: 'flex', alignItems: 'center', gap: '10px',
      padding: '4px 0', color: 'var(--text-secondary)', fontSize: '13px',
    }}>
      <div style={{
        width: '28px', height: '28px', minWidth: '28px', borderRadius: '7px',
        background: 'var(--primary)', display: 'flex', alignItems: 'center',
        justifyContent: 'center', fontSize: '12px', fontWeight: 700, color: '#fff',
      }}>A</div>
      <div style={{ display: 'flex', gap: '3px', alignItems: 'center' }}>
        {[0,1,2].map(i => (
          <span key={i} className={`dot-${i+1}`} style={{
            display: 'block', width: '5px', height: '5px', borderRadius: '50%',
            background: 'var(--primary)',
          }} />
        ))}
      </div>
      <span style={{ color: 'var(--text-secondary)' }}>{msg}</span>
    </div>
  )
}

function EmptyState({ name }: { name: string }) {
  const chips = [
    '🔍 Consultas sobre Kubernetes',
    '📅 Organizar mi agenda',
    '📄 Analizar documentos',
    '📊 Generar gráficos',
  ]
  return (
    <div className="fade-up" style={{
      display: 'flex', flexDirection: 'column', alignItems: 'center',
      justifyContent: 'center', height: '100%', textAlign: 'center', padding: '0 20px',
    }}>
      <div style={{
        width: '56px', height: '56px', background: 'var(--primary-subtle)',
        borderRadius: '14px', display: 'flex', alignItems: 'center',
        justifyContent: 'center', marginBottom: '20px', fontSize: '26px',
      }}>◆</div>
      <h2 style={{ fontSize: '22px', fontWeight: 600, color: 'var(--text-primary)', marginBottom: '8px', letterSpacing: '-0.3px' }}>
        Hola{name ? `, ${name}` : ''}
      </h2>
      <p style={{ fontSize: '15px', color: 'var(--text-secondary)', maxWidth: '340px', lineHeight: 1.6, marginBottom: '28px' }}>
        ¿En qué puedo ayudarte hoy?
      </p>
      <div style={{ display: 'flex', gap: '8px', flexWrap: 'wrap', justifyContent: 'center' }}>
        {chips.map(c => (
          <span key={c} style={{
            background: 'var(--bg-elevated)', border: '1px solid var(--border)',
            borderRadius: '20px', padding: '7px 14px', fontSize: '13px',
            color: 'var(--text-secondary)',
          }}>{c}</span>
        ))}
      </div>
    </div>
  )
}

// Paperclip icon
function IcoPaperclip() {
  return (
    <svg width="17" height="17" viewBox="0 0 24 24" fill="none"
      stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <path d="M21.44 11.05l-9.19 9.19a6 6 0 0 1-8.49-8.49l9.19-9.19a4 4 0 0 1 5.66 5.66L9.41 17.41a2 2 0 0 1-2.83-2.83l8.49-8.48" />
    </svg>
  )
}

// Hamburger icon for mobile header
function IcoHamburger() {
  return (
    <svg width="20" height="20" viewBox="0 0 24 24" fill="none"
      stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <path d="M3 6h18M3 12h18M3 18h18" />
    </svg>
  )
}

// ── Profile Modal ─────────────────────────────────────────────────────────────
function ProfileModal({ token, onClose }: { token: string; onClose: () => void }) {
  const [name,        setName]        = useState('')
  const [role,        setRole]        = useState('')
  const [institution, setInstitution] = useState('')
  const [timezone,    setTimezone]    = useState('America/Mexico_City')
  const [saving,      setSaving]      = useState(false)
  const [saved,       setSaved]       = useState(false)

  useEffect(() => {
    fetch(`${BACKEND}/memory/profile`, { headers: { Authorization: `Bearer ${token}` } })
      .then(r => r.ok ? r.json() : null)
      .then(d => {
        if (!d) return
        setName(d.display_name || '')
        setTimezone(d.timezone || 'America/Mexico_City')
        setRole(d.preferences?.role || '')
        setInstitution(d.preferences?.institution || '')
      })
      .catch(() => {})
  }, [token])

  const save = async () => {
    setSaving(true)
    await fetch(`${BACKEND}/memory/profile`, {
      method: 'PATCH',
      headers: { Authorization: `Bearer ${token}`, 'Content-Type': 'application/json' },
      body: JSON.stringify({ display_name: name, timezone, preferences: { role, institution } }),
    })
    setSaving(false)
    setSaved(true)
    setTimeout(() => { setSaved(false); onClose() }, 1200)
  }

  const fieldStyle: React.CSSProperties = {
    width: '100%', background: 'var(--input-bg)', border: '1px solid var(--border)',
    borderRadius: '8px', padding: '10px 14px', fontSize: '14px',
    color: 'var(--text-primary)', outline: 'none', boxSizing: 'border-box',
    fontFamily: 'Inter, sans-serif',
  }
  const labelStyle: React.CSSProperties = {
    fontSize: '12px', fontWeight: 600, color: 'var(--text-secondary)',
    textTransform: 'uppercase', letterSpacing: '0.05em', marginBottom: '6px', display: 'block',
  }

  return (
    <div style={{
      position: 'fixed', inset: 0, zIndex: 500,
      background: 'rgba(0,0,0,0.6)', backdropFilter: 'blur(6px)',
      display: 'flex', alignItems: 'center', justifyContent: 'center', padding: '20px',
    }} onClick={e => { if (e.target === e.currentTarget) onClose() }}>
      <div style={{
        background: 'var(--bg-surface)', border: '1px solid var(--border)',
        borderRadius: '16px', padding: '28px', width: '100%', maxWidth: '460px',
        boxShadow: '0 24px 64px rgba(0,0,0,0.4)',
      }}>
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '24px' }}>
          <div>
            <h2 style={{ fontSize: '18px', fontWeight: 700, color: 'var(--text-primary)', margin: 0 }}>Mi perfil</h2>
            <p style={{ fontSize: '13px', color: 'var(--text-secondary)', margin: '4px 0 0' }}>
              Esta información personaliza tus documentos y el briefing diario
            </p>
          </div>
          <button onClick={onClose} style={{
            background: 'none', border: 'none', cursor: 'pointer',
            color: 'var(--text-disabled)', fontSize: '20px', lineHeight: 1, padding: '4px',
          }}>✕</button>
        </div>

        <div style={{ display: 'flex', flexDirection: 'column', gap: '16px' }}>
          <div>
            <label style={labelStyle}>Nombre completo</label>
            <input value={name} onChange={e => setName(e.target.value)} placeholder="Ej. Ricardo Guzmán"
              style={fieldStyle}
              onFocus={e => (e.currentTarget.style.borderColor = 'var(--border-focus)')}
              onBlur={e => (e.currentTarget.style.borderColor = 'var(--border)')} />
          </div>
          <div>
            <label style={labelStyle}>Cargo</label>
            <input value={role} onChange={e => setRole(e.target.value)}
              placeholder="Ej. Subdirector de Infraestructura Digital"
              style={fieldStyle}
              onFocus={e => (e.currentTarget.style.borderColor = 'var(--border-focus)')}
              onBlur={e => (e.currentTarget.style.borderColor = 'var(--border)')} />
          </div>
          <div>
            <label style={labelStyle}>Institución</label>
            <input value={institution} onChange={e => setInstitution(e.target.value)}
              placeholder="Ej. Secretaría de Hacienda"
              style={fieldStyle}
              onFocus={e => (e.currentTarget.style.borderColor = 'var(--border-focus)')}
              onBlur={e => (e.currentTarget.style.borderColor = 'var(--border)')} />
          </div>
          <div>
            <label style={labelStyle}>Zona horaria</label>
            <select value={timezone} onChange={e => setTimezone(e.target.value)} style={fieldStyle}>
              <option value="America/Mexico_City">Ciudad de México (CST/CDT)</option>
              <option value="America/Monterrey">Monterrey (CST/CDT)</option>
              <option value="America/Tijuana">Tijuana (PST/PDT)</option>
              <option value="America/Cancun">Cancún (EST)</option>
            </select>
          </div>
        </div>

        <div style={{ display: 'flex', gap: '10px', marginTop: '24px' }}>
          <button onClick={onClose} style={{
            flex: 1, padding: '11px', borderRadius: '8px',
            background: 'none', border: '1px solid var(--border)',
            color: 'var(--text-secondary)', cursor: 'pointer', fontSize: '14px',
          }}>Cancelar</button>
          <button onClick={save} disabled={saving} style={{
            flex: 2, padding: '11px', borderRadius: '8px',
            background: saved ? '#22c55e' : 'var(--primary)',
            border: 'none', color: '#fff', cursor: saving ? 'wait' : 'pointer',
            fontSize: '14px', fontWeight: 600, transition: 'background .2s',
          }}>
            {saved ? '✓ Guardado' : saving ? 'Guardando…' : 'Guardar perfil'}
          </button>
        </div>
      </div>
    </div>
  )
}

export default function Chat({ token, userName, userPicture, onLogout }: Props) {
  const [messages,        setMessages]        = useState<Msg[]>([])
  const [conversations,   setConversations]   = useState<Conv[]>([])
  const [convId,          setConvId]          = useState<number | null>(null)
  const [input,           setInput]           = useState('')
  const [loading,         setLoading]         = useState(false)
  const [statusMsg,       setStatusMsg]       = useState('🧠 Analizando consulta…')
  const [streamingContent,setStreamingContent]= useState('')
  const [feedback,        setFeedback]        = useState<Record<number, 'positive' | 'negative'>>({})
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false)
  const [isMobile, setIsMobile] = useState(false)
  const [showProfile, setShowProfile] = useState(false)
  const [selectedFile, setSelectedFile] = useState<File | null>(null)
  const [uploadStatus, setUploadStatus] = useState<'idle' | 'uploading' | 'done' | 'error'>('idle')
  const fileInputRef = useRef<HTMLInputElement>(null)

  const bottomRef  = useRef<HTMLDivElement>(null)
  const inputRef   = useRef<HTMLTextAreaElement>(null)
  const abortRef   = useRef<AbortController | null>(null)

  const headers = useCallback(
    () => ({ Authorization: `Bearer ${token}` }),
    [token]
  )

  // ── Mobile detection ─────────────────────────────────────────────────────────
  useEffect(() => {
    const check = () => {
      const mobile = window.innerWidth < 768
      setIsMobile(mobile)
      if (mobile) setSidebarCollapsed(true)
    }
    check()
    window.addEventListener('resize', check)
    return () => window.removeEventListener('resize', check)
  }, [])

  // ── Load conversations on mount ─────────────────────────────────────────────
  useEffect(() => {
    fetch(`${BACKEND}/conversations`, { headers: headers() })
      .then(r => r.json())
      .then(d => {
        const convs: Conv[] = d.conversations || []
        setConversations(convs)
        if (convs.length > 0) loadConversation(convs[0].id)
        else                   createConversation()
      })
      .catch(() => createConversation())
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  // ── Scroll to bottom on new content ─────────────────────────────────────────
  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages, loading, streamingContent])

  // ── Conversation helpers ─────────────────────────────────────────────────────
  const loadConversation = (id: number) => {
    if (window.innerWidth < 768) setSidebarCollapsed(true)
    fetch(`${BACKEND}/conversations/${id}/messages`, { headers: headers() })
      .then(r => r.json())
      .then(d => {
        setMessages(d.messages || [])
        setConvId(id)
        setFeedback({})
        setStreamingContent('')
      })
  }

  const createConversation = () => {
    if (window.innerWidth < 768) setSidebarCollapsed(true)
    fetch(`${BACKEND}/conversations`, {
      method: 'POST',
      headers: { ...headers(), 'Content-Type': 'application/json' },
      body: JSON.stringify({ title: 'Nueva conversación' }),
    })
      .then(r => r.json())
      .then(d => {
        const c: Conv = { id: d.id, title: d.title, last_active_at: new Date().toISOString() }
        setConversations(prev => [c, ...prev])
        setConvId(d.id)
        setMessages([])
        setFeedback({})
        setStreamingContent('')
      })
      .catch(() => { setConvId(null); setMessages([]) })
  }

  const renameConversation = (id: number, title: string) => {
    setConversations(prev => prev.map(c => c.id === id ? { ...c, title } : c))
    fetch(`${BACKEND}/conversations/${id}`, {
      method: 'PATCH',
      headers: { ...headers(), 'Content-Type': 'application/json' },
      body: JSON.stringify({ title }),
    }).catch(() => {})
  }

  const deleteConversation = (id: number) => {
    setConversations(prev => {
      const remaining = prev.filter(c => c.id !== id)
      if (convId === id) {
        if (remaining.length > 0) loadConversation(remaining[0].id)
        else createConversation()
      }
      return remaining
    })
    fetch(`${BACKEND}/conversations/${id}`, {
      method: 'DELETE',
      headers: headers(),
    }).catch(() => {})
  }

  const sendFeedback = (idx: number, sentiment: 'positive' | 'negative') => {
    setFeedback(prev => ({ ...prev, [idx]: sentiment }))
    fetch(`${BACKEND}/feedback`, {
      method: 'POST',
      headers: { ...headers(), 'Content-Type': 'application/json' },
      body: JSON.stringify({ conversation_id: convId, message_index: idx, sentiment }),
    }).catch(() => {})
  }

  // ── File upload helper ───────────────────────────────────────────────────────
  const uploadFile = async (file: File): Promise<{ summary: string; filename: string } | null> => {
    setUploadStatus('uploading')
    const formData = new FormData()
    formData.append('file', file)
    try {
      const res = await fetch(`${BACKEND}/ingest`, {
        method: 'POST',
        headers: headers(),
        body: formData,
      })
      if (!res.ok) {
        const err = await res.json().catch(() => ({ detail: `HTTP ${res.status}` }))
        throw new Error(err.detail)
      }
      const data = await res.json()
      setUploadStatus('done')
      return { summary: data.summary, filename: data.filename }
    } catch (e) {
      setUploadStatus('error')
      return null
    }
  }

  // ── Main send with streaming SSE ─────────────────────────────────────────────
  const send = async () => {
    const prompt = input.trim()
    if ((!prompt && !selectedFile) || loading) return

    abortRef.current?.abort()
    const abort = new AbortController()
    abortRef.current = abort

    setInput('')
    const ts = new Date().toLocaleTimeString('es', { hour: '2-digit', minute: '2-digit' })

    // If file attached, upload first
    let docContext = ''
    if (selectedFile) {
      const fileToUpload = selectedFile
      setSelectedFile(null)
      const uploaded = await uploadFile(fileToUpload)
      if (uploaded) {
        docContext = `[Documento subido: ${uploaded.filename}]\nResumen: ${uploaded.summary}`
        const docMsg: Msg = {
          role: 'assistant',
          content: `📄 **${uploaded.filename}** indexado correctamente.\n\n${uploaded.summary}`,
          ts,
        }
        setMessages(prev => [...prev, docMsg])
      } else {
        const errMsg: Msg = { role: 'assistant', content: '⚠️ Error al subir el documento.', ts }
        setMessages(prev => [...prev, errMsg])
        setLoading(false)
        setUploadStatus('idle')
        return
      }
      setUploadStatus('idle')
      if (!prompt) { setLoading(false); return }
    }

    const effectivePrompt = docContext ? `${prompt}\n\n${docContext}` : prompt
    const userMsg: Msg = { role: 'user', content: prompt, ts }
    setMessages(prev => [...prev, userMsg])
    setLoading(true)
    setStatusMsg('🧠 Analizando consulta…')
    setStreamingContent('')

    const currentMessages = [...messages, userMsg]

    try {
      const res = await fetch(`${BACKEND}/chat/stream`, {
        method:  'POST',
        headers: { ...headers(), 'Content-Type': 'application/json' },
        body:    JSON.stringify({ prompt: effectivePrompt, history: currentMessages, conversation_id: convId }),
        signal:  abort.signal,
      })

      if (!res.ok || !res.body) {
        const err = await res.json().catch(() => ({ detail: `HTTP ${res.status}` }))
        throw new Error(err.detail)
      }

      const reader  = res.body.getReader()
      const decoder = new TextDecoder()
      let   buffer  = ''
      let   accumulated = ''

      while (true) {
        const { done, value } = await reader.read()
        if (done) break

        buffer += decoder.decode(value, { stream: true })
        const parts = buffer.split('\n\n')
        buffer = parts.pop() ?? ''

        for (const part of parts) {
          if (!part.startsWith('data: ')) continue
          const raw = part.slice(6).trim()
          if (!raw) continue

          let event: Record<string, string>
          try { event = JSON.parse(raw) } catch { continue }

          if (event.type === 'status') {
            setStatusMsg(event.msg)
          } else if (event.type === 'token') {
            accumulated += event.content
            setStreamingContent(accumulated)
          } else if (event.type === 'error') {
            throw new Error(event.msg)
          } else if (event.type === 'done') {
            const replyTs = new Date().toLocaleTimeString('es', { hour: '2-digit', minute: '2-digit' })
            const reply: Msg = { role: 'assistant', content: accumulated, ts: replyTs }
            setMessages(prev => [...prev, reply])
            setStreamingContent('')

            if (convId && currentMessages.length === 1) {
              const title = prompt.slice(0, 50) + (prompt.length > 50 ? '…' : '')
              setConversations(prev => prev.map(c =>
                c.id === convId && c.title === 'Nueva conversación' ? { ...c, title } : c
              ))
            }
          }
        }
      }
    } catch (e: unknown) {
      if ((e as Error).name === 'AbortError') return
      const errMsg: Msg = {
        role: 'assistant',
        content: `⚠️ ${e instanceof Error ? e.message : 'Error al contactar el backend.'}`,
        ts: new Date().toLocaleTimeString('es', { hour: '2-digit', minute: '2-digit' }),
      }
      setMessages(prev => [...prev, errMsg])
      setStreamingContent('')
    } finally {
      setLoading(false)
      setStreamingContent('')
      inputRef.current?.focus()
    }
  }

  const handleKey = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); send() }
  }

  const stopGeneration = () => {
    abortRef.current?.abort()
    if (streamingContent) {
      const ts = new Date().toLocaleTimeString('es', { hour: '2-digit', minute: '2-digit' })
      setMessages(prev => [...prev, { role: 'assistant', content: streamingContent, ts }])
    }
    setLoading(false)
    setStreamingContent('')
  }

  const firstName = userName.split(' ')[0]

  return (
    <div className="chat-root" style={{ display: 'flex', background: 'var(--bg-base)', overflow: 'hidden' }}>

      {/* Mobile sidebar backdrop */}
      {isMobile && !sidebarCollapsed && (
        <div className="sidebar-backdrop" onClick={() => setSidebarCollapsed(true)} />
      )}

      {/* Sidebar */}
      <Sidebar
        user={userName ? { name: userName, picture: userPicture } : null}
        conversations={conversations}
        activeId={convId}
        collapsed={sidebarCollapsed}
        isMobile={isMobile}
        onToggle={() => setSidebarCollapsed(p => !p)}
        onSelect={loadConversation}
        onNew={createConversation}
        onRename={renameConversation}
        onDelete={deleteConversation}
        onLogout={onLogout}
        onProfile={() => setShowProfile(true)}
      />

      {/* Profile modal */}
      {showProfile && (
        <ProfileModal token={token} onClose={() => setShowProfile(false)} />
      )}

      {/* Main chat column */}
      <div style={{ flex: 1, display: 'flex', flexDirection: 'column', overflow: 'hidden', minWidth: 0 }}>

        {/* Mobile-only header */}
        <div className="mobile-header" style={{
          display: 'none',
          alignItems: 'center', gap: '10px',
          padding: '0 12px', height: '52px',
          borderBottom: '1px solid var(--border)',
          flexShrink: 0, background: 'var(--bg-base)',
        }}>
          <button
            onClick={() => setSidebarCollapsed(false)}
            style={{
              background: 'none', border: 'none', cursor: 'pointer',
              color: 'var(--text-secondary)',
              width: '44px', height: '44px',
              display: 'flex', alignItems: 'center', justifyContent: 'center',
              borderRadius: '10px', flexShrink: 0,
            }}
            aria-label="Abrir menú"
          >
            <IcoHamburger />
          </button>
          <span style={{ fontSize: '16px', fontWeight: 700, color: 'var(--text-primary)', letterSpacing: '-0.3px' }}>
            Amael
          </span>
        </div>

        {/* Messages area */}
        <div style={{ flex: 1, overflowY: 'auto', padding: '32px 0 24px' }}>
          <div style={{ maxWidth: '720px', margin: '0 auto', padding: isMobile ? '0 16px' : '0 24px' }}>

            {messages.length === 0 && !loading && !streamingContent
              ? <EmptyState name={firstName} />
              : <>
                  {messages.map((m, i) => (
                    <Message
                      key={i}
                      role={m.role}
                      content={m.content}
                      ts={m.ts}
                      feedback={m.role === 'assistant' ? (feedback[i] ?? null) : undefined}
                      onFeedback={m.role === 'assistant' ? s => sendFeedback(i, s) : undefined}
                    />
                  ))}

                  {streamingContent && (
                    <Message
                      role="assistant"
                      content={streamingContent}
                      isStreaming
                    />
                  )}

                  {loading && !streamingContent && (
                    <StatusBanner msg={statusMsg} />
                  )}
                </>
            }
            <div ref={bottomRef} />
          </div>
        </div>

        {/* Input bar */}
        <div style={{
          padding: '12px 24px',
          paddingBottom: 'max(20px, calc(12px + env(safe-area-inset-bottom)))',
          background: 'linear-gradient(to top, var(--bg-base) 80%, transparent)',
        }}>
          <div style={{ maxWidth: '720px', margin: '0 auto', position: 'relative' }}>

            {/* Hidden file input */}
            <input
              ref={fileInputRef}
              type="file"
              accept=".pdf,.txt,.docx,.md"
              style={{ display: 'none' }}
              onChange={e => {
                const f = e.target.files?.[0] ?? null
                setSelectedFile(f)
                setUploadStatus('idle')
                e.target.value = ''
              }}
            />

            {/* Selected file badge */}
            {selectedFile && (
              <div style={{
                display: 'flex', alignItems: 'center', gap: '8px',
                marginBottom: '8px', padding: '6px 12px',
                background: 'var(--primary-subtle)', border: '1px solid rgba(99,102,241,0.25)',
                borderRadius: '8px', fontSize: '13px', color: 'var(--text-secondary)',
              }}>
                <span style={{ fontSize: '15px' }}>📄</span>
                <span style={{ flex: 1, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                  {selectedFile.name}
                </span>
                <button
                  onClick={() => setSelectedFile(null)}
                  style={{
                    background: 'none', border: 'none', cursor: 'pointer',
                    color: 'var(--text-disabled)', fontSize: '16px', lineHeight: 1,
                    padding: '0 2px',
                  }}
                >×</button>
              </div>
            )}

            {/* Clip button (left of textarea) */}
            <button
              onClick={() => fileInputRef.current?.click()}
              disabled={loading}
              title="Subir documento (PDF, TXT, DOCX)"
              style={{
                position: 'absolute', left: '9px', bottom: '9px',
                width: '44px', height: '44px', borderRadius: '12px',
                background: 'none', border: 'none', cursor: loading ? 'not-allowed' : 'pointer',
                display: 'flex', alignItems: 'center', justifyContent: 'center',
                color: selectedFile ? 'var(--primary)' : 'var(--text-disabled)',
                transition: 'color .15s',
              }}
            >
              <IcoPaperclip />
            </button>

            <textarea
              ref={inputRef}
              value={input}
              onChange={e => setInput(e.target.value)}
              onKeyDown={handleKey}
              placeholder="Pregunta algo a Amael… (Enter para enviar, Shift+Enter nueva línea)"
              rows={1}
              disabled={loading}
              enterKeyHint="send"
              inputMode="text"
              style={{
                width: '100%', background: 'var(--input-bg)',
                border: `1px solid ${input ? 'var(--border-focus)' : 'var(--border)'}`,
                borderRadius: '14px', padding: '14px 62px 14px 56px',
                color: 'var(--text-primary)', fontSize: '16px', resize: 'none',
                outline: 'none', lineHeight: '1.5', fontFamily: 'inherit',
                boxShadow: input ? '0 0 0 3px var(--primary-subtle)' : 'none',
                transition: 'border-color .15s, box-shadow .15s',
                maxHeight: '160px', overflowY: 'auto',
              }}
              onInput={e => {
                const t = e.currentTarget
                t.style.height = 'auto'
                t.style.height = Math.min(t.scrollHeight, 160) + 'px'
              }}
            />

            {/* Send / Stop button — 44×44px for touch */}
            {loading ? (
              <button onClick={stopGeneration} style={{
                position: 'absolute', right: '9px', bottom: '9px',
                width: '44px', height: '44px', borderRadius: '12px',
                background: 'var(--error)', border: 'none', cursor: 'pointer',
                display: 'flex', alignItems: 'center', justifyContent: 'center',
                color: '#fff', fontSize: '14px', fontWeight: 700,
              }} title="Detener generación">■</button>
            ) : (
              <button onClick={send} disabled={!input.trim()} style={{
                position: 'absolute', right: '9px', bottom: '9px',
                width: '44px', height: '44px', borderRadius: '12px',
                background: input.trim() ? 'var(--primary)' : 'var(--bg-elevated)',
                border: 'none', cursor: input.trim() ? 'pointer' : 'not-allowed',
                display: 'flex', alignItems: 'center', justifyContent: 'center',
                color: '#fff', fontSize: '17px', transition: 'background .15s',
              }}>↑</button>
            )}
          </div>

          {/* Hint */}
          <p style={{ textAlign: 'center', fontSize: '11px', color: 'var(--text-disabled)', marginTop: '8px' }}>
            {loading
              ? streamingContent
                ? `${streamingContent.split(' ').filter(Boolean).length} palabras generadas…`
                : statusMsg
              : 'Amael puede cometer errores — verifica información importante.'
            }
          </p>
        </div>
      </div>
    </div>
  )
}
