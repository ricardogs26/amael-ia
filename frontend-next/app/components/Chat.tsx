'use client'
import { useEffect, useRef, useState, useCallback } from 'react'
import { useRouter } from 'next/navigation'
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
  calendarNotif?: 'connected' | 'error' | null
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

// ── Image Modal ─────────────────────────────────────────────────────────────
function ImageModal({ src, onClose }: { src: string; onClose: () => void }) {
  useEffect(() => {
    const handleEsc = (e: KeyboardEvent) => { if (e.key === 'Escape') onClose() }
    window.addEventListener('keydown', handleEsc)
    return () => window.removeEventListener('keydown', handleEsc)
  }, [onClose])

  return (
    <div 
      onClick={onClose}
      style={{
        position: 'fixed', inset: 0, zIndex: 9999,
        background: 'rgba(0,0,0,0.85)', backdropFilter: 'blur(8px)',
        display: 'flex', alignItems: 'center', justifyContent: 'center', padding: '20px',
        cursor: 'zoom-out',
        animation: 'fadeIn 0.2s ease-out'
      }}
    >
      <div style={{ position: 'relative', maxWidth: '95vw', maxHeight: '95vh', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
        <img 
          src={src} 
          alt="Expanded" 
          style={{ 
            maxWidth: '100%', 
            maxHeight: '95vh', 
            borderRadius: '8px', 
            boxShadow: '0 20px 50px rgba(0,0,0,0.5)',
            cursor: 'default'
          }} 
          onClick={e => e.stopPropagation()}
        />
        <button 
          onClick={onClose}
          style={{
            position: 'absolute', top: '-40px', right: 0,
            background: 'none', border: 'none', color: '#fff',
            fontSize: '24px', cursor: 'pointer', padding: '10px',
            opacity: 0.8, transition: 'opacity 0.2s'
          }}
          onMouseEnter={e => e.currentTarget.style.opacity = '1'}
          onMouseLeave={e => e.currentTarget.style.opacity = '0.8'}
        >
          ✕
        </button>
      </div>
      <style jsx>{`
        @keyframes fadeIn {
          from { opacity: 0; }
          to { opacity: 1; }
        }
      `}</style>
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
  const [name,           setName]           = useState('')
  const [role,           setRole]           = useState('')
  const [institution,    setInstitution]    = useState('')
  const [timezone,       setTimezone]       = useState('America/Mexico_City')
  const [saving,         setSaving]         = useState(false)
  const [saved,          setSaved]          = useState(false)
  const [calConnected,   setCalConnected]   = useState<boolean | null>(null)

  // WhatsApp Personal
  const [wpStatus,       setWpStatus]       = useState<string | null>(null)   // null=loading | 'ready' | 'awaiting_qr' | 'disconnected' | 'unreachable'
  const [wpPhone,        setWpPhone]        = useState<string | null>(null)
  const [wpQR,           setWpQR]           = useState<string | null>(null)
  const [wpAutoReply,       setWpAutoReply]       = useState(true)
  const [wpAiAssist,        setWpAiAssist]        = useState(true)
  const [wpQuietEnabled,    setWpQuietEnabled]    = useState(true)
  const [wpQuietStart,      setWpQuietStart]      = useState(22)
  const [wpQuietEnd,        setWpQuietEnd]        = useState(8)
  const [wpScope,           setWpScope]           = useState('all')
  const [wpAllowedContacts, setWpAllowedContacts] = useState<string[]>([])
  const [wpContactInput,    setWpContactInput]    = useState('')
  const [wpOfflineMsg,      setWpOfflineMsg]      = useState('')
  const [wpActiveDays,      setWpActiveDays]      = useState([1,2,3,4,5])
  const [wpSettingsOpen, setWpSettingsOpen] = useState(false)
  const [wpSaving,       setWpSaving]       = useState(false)
  const [wpQrPolling,    setWpQrPolling]    = useState<ReturnType<typeof setInterval> | null>(null)

  const h = { Authorization: `Bearer ${token}` }

  const loadWpStatus = async () => {
    try {
      const r = await fetch(`${BACKEND}/whatsapp-personal/status`, { headers: h })
      if (!r.ok) { setWpStatus('unreachable'); return }
      const d = await r.json()
      setWpStatus(d.status)
      setWpPhone(d.phone || null)
      if (d.settings) {
        setWpAutoReply(d.settings.auto_reply ?? true)
        setWpAiAssist(d.settings.ai_assist ?? true)
        setWpQuietEnabled(d.settings.quiet_enabled ?? true)
        setWpQuietStart(d.settings.quiet_start ?? 22)
        setWpQuietEnd(d.settings.quiet_end ?? 8)
        setWpScope(d.settings.reply_scope ?? 'all')
        setWpAllowedContacts(d.settings.allowed_contacts ?? [])
        setWpOfflineMsg(d.settings.offline_msg ?? '')
        setWpActiveDays(d.settings.active_days ?? [1,2,3,4,5])
      }
    } catch { setWpStatus('unreachable') }
  }

  const loadWpQR = async () => {
    try {
      const r = await fetch(`${BACKEND}/whatsapp-personal/qr`, { headers: h })
      if (!r.ok) return
      const d = await r.json()
      setWpStatus(d.status)
      if (d.qr) setWpQR(d.qr)
      else if (d.status === 'ready') { setWpQR(null); setWpPhone(d.phone); stopQrPolling() }
    } catch {}
  }

  const stopQrPolling = () => {
    if (wpQrPolling) { clearInterval(wpQrPolling); setWpQrPolling(null) }
  }

  const startQrPolling = () => {
    stopQrPolling()
    const id = setInterval(loadWpQR, 5000)
    setWpQrPolling(id)
  }

  const handleWpDisconnect = async () => {
    if (!confirm('¿Desconectar WhatsApp personal?')) return
    await fetch(`${BACKEND}/whatsapp-personal/disconnect`, { method: 'POST', headers: h })
    setWpStatus('disconnected'); setWpPhone(null); setWpQR(null)
  }

  const saveWpSettings = async () => {
    setWpSaving(true)
    await fetch(`${BACKEND}/whatsapp-personal/settings`, {
      method: 'PATCH',
      headers: { ...h, 'Content-Type': 'application/json' },
      body: JSON.stringify({
        auto_reply:       wpAutoReply,
        ai_assist:        wpAiAssist,
        quiet_enabled:    wpQuietEnabled,
        quiet_start:      wpQuietStart,
        quiet_end:        wpQuietEnd,
        active_days:      wpActiveDays,
        reply_scope:      wpScope,
        allowed_contacts: wpAllowedContacts,
        offline_msg:      wpOfflineMsg || null,
      }),
    }).catch(() => {})
    setWpSaving(false)
  }

  const toggleDay = (d: number) =>
    setWpActiveDays(prev => prev.includes(d) ? prev.filter(x => x !== d) : [...prev, d].sort())

  useEffect(() => {
    fetch(`${BACKEND}/memory/profile`, { headers: h })
      .then(r => r.ok ? r.json() : null)
      .then(d => {
        if (!d?.profile) return
        setName(d.profile.display_name || '')
        setTimezone(d.profile.timezone || 'America/Mexico_City')
        setRole(d.profile.preferences?.role || '')
        setInstitution(d.profile.preferences?.institution || '')
      })
      .catch(() => {})
    fetch(`${BACKEND}/auth/calendar/status`, { headers: h })
      .then(r => r.ok ? r.json() : { connected: false })
      .then(d => setCalConnected(d.connected))
      .catch(() => setCalConnected(false))
    loadWpStatus()
    return () => stopQrPolling()
  // eslint-disable-next-line react-hooks/exhaustive-deps
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

        {/* Google Calendar connection */}
        <div style={{
          marginTop: '20px', padding: '14px 16px',
          background: 'var(--bg-elevated)', borderRadius: '10px',
          border: '1px solid var(--border)',
          display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: '12px',
        }}>
          <div>
            <div style={{ fontSize: '13px', fontWeight: 600, color: 'var(--text-primary)', display: 'flex', alignItems: 'center', gap: '6px' }}>
              <span>📅</span> Google Calendar
            </div>
            <div style={{ fontSize: '12px', color: 'var(--text-secondary)', marginTop: '2px' }}>
              {calConnected === null ? 'Verificando…' : calConnected ? '✅ Conectado — el briefing usa tu agenda real' : '⚠️ No conectado — el briefing no puede leer tu agenda'}
            </div>
          </div>
          {calConnected === false && (
            <a href={`${BACKEND}/auth/calendar?token=${token}`}
              style={{
                background: '#4285F4', color: '#fff', border: 'none',
                borderRadius: '7px', padding: '7px 14px', fontSize: '12px',
                fontWeight: 600, cursor: 'pointer', textDecoration: 'none',
                whiteSpace: 'nowrap', fontFamily: 'Inter, sans-serif',
              }}>
              Conectar
            </a>
          )}
          {calConnected === true && (
            <a href={`${BACKEND}/auth/calendar?token=${token}`}
              style={{
                background: 'none', color: 'var(--text-secondary)', border: '1px solid var(--border)',
                borderRadius: '7px', padding: '7px 14px', fontSize: '12px',
                cursor: 'pointer', textDecoration: 'none', whiteSpace: 'nowrap',
                fontFamily: 'Inter, sans-serif',
              }}>
              Reconectar
            </a>
          )}
        </div>

        {/* ── WhatsApp Personal ─────────────────────────────────────────── */}
        <div style={{
          marginTop: '16px', borderRadius: '12px',
          border: '1px solid var(--border)', overflow: 'hidden',
        }}>
          {/* Header row */}
          <div style={{
            padding: '14px 16px', background: 'var(--bg-elevated)',
            display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: '12px',
          }}>
            <div>
              <div style={{ fontSize: '13px', fontWeight: 600, color: 'var(--text-primary)', display: 'flex', alignItems: 'center', gap: '6px' }}>
                <span>💬</span> WhatsApp Personal
              </div>
              <div style={{ fontSize: '12px', color: 'var(--text-secondary)', marginTop: '2px' }}>
                {wpStatus === null        && 'Verificando…'}
                {wpStatus === 'ready'     && `✅ ${wpPhone || 'Conectado'} — ${wpAutoReply ? (wpAiAssist ? 'respondiendo con IA' : 'respuestas sin IA') : 'pausado'}`}
                {wpStatus === 'awaiting_qr' && '📱 Escanea el código QR con tu WhatsApp'}
                {wpStatus === 'initializing' && '⏳ Iniciando servicio…'}
                {wpStatus === 'disconnected' && '⚠️ Desconectado — escanea el QR para reconectar'}
                {wpStatus === 'unreachable'  && '🔴 Servicio no disponible'}
                {wpStatus?.startsWith('loading') && `⏳ Cargando… ${wpStatus.split(':')[1] || ''}%`}
              </div>
            </div>
            <div style={{ display: 'flex', gap: '8px', alignItems: 'center', flexShrink: 0 }}>
              {/* Master on/off — visible siempre que esté conectado */}
              {wpStatus === 'ready' && (
                <button
                  title={wpAutoReply ? 'Auto-respuesta activa — clic para pausar' : 'Auto-respuesta pausada — clic para activar'}
                  onClick={async () => {
                    const next = !wpAutoReply
                    setWpAutoReply(next)
                    await fetch(`${BACKEND}/whatsapp-personal/settings`, {
                      method: 'PATCH',
                      headers: { ...h, 'Content-Type': 'application/json' },
                      body: JSON.stringify({ auto_reply: next }),
                    }).catch(() => {})
                  }}
                  style={{
                    width: '48px', height: '26px', borderRadius: '13px', border: 'none',
                    cursor: 'pointer', position: 'relative', transition: 'background .2s',
                    background: wpAutoReply ? '#25D366' : 'var(--border)', flexShrink: 0,
                  }}>
                  <span style={{
                    position: 'absolute', top: '4px', left: wpAutoReply ? '24px' : '4px',
                    width: '18px', height: '18px', borderRadius: '50%', background: '#fff',
                    transition: 'left .2s', display: 'block',
                  }}/>
                </button>
              )}
              {wpStatus === 'ready' && (
                <button onClick={() => setWpSettingsOpen(o => !o)} style={{
                  background: 'none', border: '1px solid var(--border)', borderRadius: '7px',
                  padding: '6px 12px', fontSize: '12px', color: 'var(--text-secondary)',
                  cursor: 'pointer', fontFamily: 'Inter, sans-serif',
                }}>⚙ Reglas</button>
              )}
              {wpStatus === 'ready' && (
                <button onClick={handleWpDisconnect} style={{
                  background: 'none', border: '1px solid #ef4444', borderRadius: '7px',
                  padding: '6px 12px', fontSize: '12px', color: '#ef4444',
                  cursor: 'pointer', fontFamily: 'Inter, sans-serif',
                }}>Desconectar</button>
              )}
              {(wpStatus === 'disconnected' || wpStatus === 'awaiting_qr' || wpStatus === 'unreachable') && (
                <button onClick={() => { loadWpQR(); startQrPolling() }} style={{
                  background: '#25D366', color: '#fff', border: 'none',
                  borderRadius: '7px', padding: '7px 14px', fontSize: '12px',
                  fontWeight: 600, cursor: 'pointer', fontFamily: 'Inter, sans-serif',
                }}>Conectar QR</button>
              )}
            </div>
          </div>

          {/* QR Panel */}
          {wpStatus === 'awaiting_qr' && wpQR && (
            <div style={{
              padding: '20px', display: 'flex', flexDirection: 'column',
              alignItems: 'center', gap: '12px', background: 'var(--bg-surface)',
            }}>
              <img
                src={`https://api.qrserver.com/v1/create-qr-code/?size=220x220&data=${encodeURIComponent(wpQR)}`}
                alt="QR WhatsApp" width={220} height={220}
                style={{ borderRadius: '8px', border: '3px solid #25D366' }}
              />
              <div style={{ fontSize: '12px', color: 'var(--text-secondary)', textAlign: 'center' }}>
                Abre WhatsApp → Dispositivos vinculados → Vincular dispositivo
              </div>
              <div style={{ fontSize: '11px', color: 'var(--text-disabled)' }}>
                Se actualiza automáticamente cada 30s
              </div>
            </div>
          )}

          {/* Settings Panel */}
          {wpStatus === 'ready' && wpSettingsOpen && (
            <div style={{ padding: '16px', background: 'var(--bg-surface)', borderTop: '1px solid var(--border)' }}>
              <div style={{ display: 'flex', flexDirection: 'column', gap: '14px' }}>

                {/* Auto-reply toggle */}
                <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                  <div>
                    <div style={{ fontSize: '13px', fontWeight: 600, color: 'var(--text-primary)' }}>Auto-respuesta</div>
                    <div style={{ fontSize: '11px', color: 'var(--text-secondary)' }}>Responder mensajes automáticamente</div>
                  </div>
                  <button onClick={() => setWpAutoReply(v => !v)} style={{
                    width: '44px', height: '24px', borderRadius: '12px', border: 'none', cursor: 'pointer',
                    background: wpAutoReply ? '#25D366' : 'var(--border)', position: 'relative', transition: 'background .2s',
                  }}>
                    <span style={{
                      position: 'absolute', top: '3px', left: wpAutoReply ? '22px' : '3px',
                      width: '18px', height: '18px', borderRadius: '50%', background: '#fff',
                      transition: 'left .2s', display: 'block',
                    }}/>
                  </button>
                </div>

                {/* AI Assist toggle */}
                <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                  <div>
                    <div style={{ fontSize: '13px', fontWeight: 600, color: 'var(--text-primary)' }}>Asistencia IA</div>
                    <div style={{ fontSize: '11px', color: 'var(--text-secondary)' }}>Generar respuestas con el agente</div>
                  </div>
                  <button onClick={() => setWpAiAssist(v => !v)} style={{
                    width: '44px', height: '24px', borderRadius: '12px', border: 'none', cursor: 'pointer',
                    background: wpAiAssist ? 'var(--primary)' : 'var(--border)', position: 'relative', transition: 'background .2s',
                  }}>
                    <span style={{
                      position: 'absolute', top: '3px', left: wpAiAssist ? '22px' : '3px',
                      width: '18px', height: '18px', borderRadius: '50%', background: '#fff',
                      transition: 'left .2s', display: 'block',
                    }}/>
                  </button>
                </div>

                {/* Scope */}
                <div>
                  <label style={{ fontSize: '11px', fontWeight: 600, color: 'var(--text-secondary)', textTransform: 'uppercase', letterSpacing: '0.05em', display: 'block', marginBottom: '6px' }}>
                    Responder a
                  </label>
                  <select value={wpScope} onChange={e => setWpScope(e.target.value)} style={{
                    width: '100%', background: 'var(--input-bg)', border: '1px solid var(--border)',
                    borderRadius: '8px', padding: '8px 12px', fontSize: '13px',
                    color: 'var(--text-primary)', outline: 'none', fontFamily: 'Inter, sans-serif',
                  }}>
                    <option value="all">Todos los chats</option>
                    <option value="contacts_only">Solo contactos guardados</option>
                    <option value="no_groups">Solo mensajes directos (sin grupos)</option>
                    <option value="custom">Personalizado (lista de contactos)</option>
                  </select>
                </div>

                {/* Custom contacts list */}
                {wpScope === 'custom' && (
                  <div>
                    <label style={{ fontSize: '11px', fontWeight: 600, color: 'var(--text-secondary)', textTransform: 'uppercase', letterSpacing: '0.05em', display: 'block', marginBottom: '6px' }}>
                      Contactos autorizados
                    </label>
                    <div style={{ display: 'flex', gap: '6px', marginBottom: '8px' }}>
                      <input
                        value={wpContactInput}
                        onChange={e => setWpContactInput(e.target.value)}
                        onKeyDown={e => {
                          if (e.key === 'Enter') {
                            const num = wpContactInput.trim().replace(/\D/g, '')
                            if (num && !wpAllowedContacts.includes(num)) {
                              setWpAllowedContacts(prev => [...prev, num])
                            }
                            setWpContactInput('')
                          }
                        }}
                        placeholder="Ej. 5219993437008"
                        style={{
                          flex: 1, background: 'var(--input-bg)', border: '1px solid var(--border)',
                          borderRadius: '8px', padding: '8px 12px', fontSize: '13px',
                          color: 'var(--text-primary)', outline: 'none', fontFamily: 'Inter, sans-serif',
                        }}
                      />
                      <button
                        onClick={() => {
                          const num = wpContactInput.trim().replace(/\D/g, '')
                          if (num && !wpAllowedContacts.includes(num)) {
                            setWpAllowedContacts(prev => [...prev, num])
                          }
                          setWpContactInput('')
                        }}
                        style={{
                          padding: '8px 14px', borderRadius: '8px', background: 'var(--primary)',
                          border: 'none', color: '#fff', cursor: 'pointer', fontSize: '18px',
                          fontWeight: 300, lineHeight: 1,
                        }}>+</button>
                    </div>
                    {wpAllowedContacts.length === 0 && (
                      <div style={{ fontSize: '12px', color: 'var(--text-disabled)', fontStyle: 'italic' }}>
                        Sin contactos — el servicio no responderá a nadie
                      </div>
                    )}
                    <div style={{ display: 'flex', flexWrap: 'wrap', gap: '6px' }}>
                      {wpAllowedContacts.map(num => (
                        <div key={num} style={{
                          display: 'flex', alignItems: 'center', gap: '6px',
                          background: 'var(--bg-elevated)', border: '1px solid var(--border)',
                          borderRadius: '20px', padding: '4px 10px 4px 12px',
                          fontSize: '12px', color: 'var(--text-primary)',
                        }}>
                          <span>+{num}</span>
                          <button onClick={() => setWpAllowedContacts(prev => prev.filter(n => n !== num))} style={{
                            background: 'none', border: 'none', cursor: 'pointer',
                            color: 'var(--text-disabled)', fontSize: '14px', lineHeight: 1,
                            padding: '0', display: 'flex', alignItems: 'center',
                          }}>×</button>
                        </div>
                      ))}
                    </div>
                  </div>
                )}

                {/* Quiet hours */}
                <div>
                  <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '8px' }}>
                    <label style={{ fontSize: '11px', fontWeight: 600, color: 'var(--text-secondary)', textTransform: 'uppercase', letterSpacing: '0.05em' }}>
                      Horario silencioso
                    </label>
                    <button onClick={() => setWpQuietEnabled(v => !v)} style={{
                      width: '36px', height: '20px', borderRadius: '10px', border: 'none',
                      cursor: 'pointer', position: 'relative', transition: 'background .2s', flexShrink: 0,
                      background: wpQuietEnabled ? 'var(--primary)' : 'var(--border)',
                    }}>
                      <span style={{
                        position: 'absolute', top: '3px', left: wpQuietEnabled ? '18px' : '3px',
                        width: '14px', height: '14px', borderRadius: '50%', background: '#fff',
                        transition: 'left .2s', display: 'block',
                      }}/>
                    </button>
                  </div>
                  <div style={{ opacity: wpQuietEnabled ? 1 : 0.4, pointerEvents: wpQuietEnabled ? 'auto' : 'none', display: 'flex', gap: '8px', alignItems: 'center' }}>
                    <span style={{ fontSize: '12px', color: 'var(--text-secondary)' }}>Desde</span>
                    <select value={wpQuietStart} onChange={e => setWpQuietStart(Number(e.target.value))} style={{
                      flex: 1, background: 'var(--input-bg)', border: '1px solid var(--border)',
                      borderRadius: '6px', padding: '6px 8px', fontSize: '13px',
                      color: 'var(--text-primary)', fontFamily: 'Inter, sans-serif',
                    }}>
                      {Array.from({length: 24}, (_, i) => (
                        <option key={i} value={i}>{String(i).padStart(2,'0')}:00</option>
                      ))}
                    </select>
                    <span style={{ fontSize: '12px', color: 'var(--text-secondary)' }}>Hasta</span>
                    <select value={wpQuietEnd} onChange={e => setWpQuietEnd(Number(e.target.value))} style={{
                      flex: 1, background: 'var(--input-bg)', border: '1px solid var(--border)',
                      borderRadius: '6px', padding: '6px 8px', fontSize: '13px',
                      color: 'var(--text-primary)', fontFamily: 'Inter, sans-serif',
                    }}>
                      {Array.from({length: 24}, (_, i) => (
                        <option key={i} value={i}>{String(i).padStart(2,'0')}:00</option>
                      ))}
                    </select>
                  </div>
                </div>

                {/* Active days */}
                <div>
                  <label style={{ fontSize: '11px', fontWeight: 600, color: 'var(--text-secondary)', textTransform: 'uppercase', letterSpacing: '0.05em', display: 'block', marginBottom: '6px' }}>
                    Días activos
                  </label>
                  <div style={{ display: 'flex', gap: '6px', flexWrap: 'wrap' }}>
                    {[['L',1],['M',2],['X',3],['J',4],['V',5],['S',6],['D',7]].map(([label, num]) => (
                      <button key={num} onClick={() => toggleDay(num as number)} style={{
                        width: '32px', height: '32px', borderRadius: '50%', border: 'none',
                        background: wpActiveDays.includes(num as number) ? 'var(--primary)' : 'var(--border)',
                        color: wpActiveDays.includes(num as number) ? '#fff' : 'var(--text-secondary)',
                        cursor: 'pointer', fontSize: '12px', fontWeight: 600,
                        fontFamily: 'Inter, sans-serif',
                      }}>{label}</button>
                    ))}
                  </div>
                </div>

                {/* Offline message */}
                <div>
                  <label style={{ fontSize: '11px', fontWeight: 600, color: 'var(--text-secondary)', textTransform: 'uppercase', letterSpacing: '0.05em', display: 'block', marginBottom: '6px' }}>
                    Mensaje fuera de horario
                  </label>
                  <input value={wpOfflineMsg} onChange={e => setWpOfflineMsg(e.target.value)}
                    placeholder="Ej. Estoy fuera de horario, te respondo mañana."
                    style={{
                      width: '100%', background: 'var(--input-bg)', border: '1px solid var(--border)',
                      borderRadius: '8px', padding: '9px 12px', fontSize: '13px',
                      color: 'var(--text-primary)', outline: 'none', boxSizing: 'border-box',
                      fontFamily: 'Inter, sans-serif',
                    }}
                  />
                  <div style={{ fontSize: '11px', color: 'var(--text-disabled)', marginTop: '4px' }}>
                    Vacío = no responder durante horario silencioso
                  </div>
                </div>

                <button onClick={saveWpSettings} disabled={wpSaving} style={{
                  padding: '9px', borderRadius: '8px', background: 'var(--primary)',
                  border: 'none', color: '#fff', cursor: wpSaving ? 'wait' : 'pointer',
                  fontSize: '13px', fontWeight: 600,
                }}>
                  {wpSaving ? 'Guardando…' : 'Guardar reglas'}
                </button>
              </div>
            </div>
          )}
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

// ── Admin Panel ───────────────────────────────────────────────────────────────
interface AdminUser {
  user_id: string
  display_name: string | null
  role: string
  status: string
  identities: { type: string; value: string }[]
}

function AdminPanel({ token, onClose }: { token: string; onClose: () => void }) {
  const [users,          setUsers]          = useState<AdminUser[]>([])
  const [loading,        setLoading]        = useState(true)
  const [allowRequests,  setAllowRequests]  = useState(false)
  const [newEmail,       setNewEmail]       = useState('')
  const [newName,        setNewName]        = useState('')
  const [newPhone,       setNewPhone]       = useState('')
  const [newRole,        setNewRole]        = useState('user')
  const [adding,         setAdding]         = useState(false)
  const [editingPhone,   setEditingPhone]   = useState<string | null>(null)
  const [phoneInput,     setPhoneInput]     = useState('')
  const [editingName,    setEditingName]    = useState<string | null>(null)
  const [nameInput,      setNameInput]      = useState('')
  const [confirmDelete,  setConfirmDelete]  = useState<string | null>(null)

  const h = { Authorization: `Bearer ${token}`, 'Content-Type': 'application/json' }

  const load = () => {
    setLoading(true)
    Promise.all([
      fetch(`${BACKEND}/admin/users`, { headers: h }).then(r => r.json()),
      fetch(`${BACKEND}/admin/settings`, { headers: h }).then(r => r.json()),
    ]).then(([u, s]) => {
      setUsers(u.users || [])
      setAllowRequests(s.allow_access_requests ?? false)
    }).finally(() => setLoading(false))
  }

  useEffect(() => { load() }, [])

  const addUser = async () => {
    if (!newEmail.trim()) return
    setAdding(true)
    await fetch(`${BACKEND}/admin/users`, {
      method: 'POST', headers: h,
      body: JSON.stringify({ email: newEmail.trim(), display_name: newName.trim() || undefined, role: newRole, phone: newPhone.trim() || undefined }),
    })
    setNewEmail(''); setNewName(''); setNewPhone(''); setNewRole('user')
    setAdding(false)
    load()
  }

  const toggleStatus = async (uid: string, currentStatus: string) => {
    const newStatus = currentStatus === 'active' ? 'inactive' : 'active'
    await fetch(`${BACKEND}/admin/users/${encodeURIComponent(uid)}`, {
      method: 'PATCH', headers: h, body: JSON.stringify({ status: newStatus }),
    })
    load()
  }

  const changeRole = async (uid: string, role: string) => {
    await fetch(`${BACKEND}/admin/users/${encodeURIComponent(uid)}`, {
      method: 'PATCH', headers: h, body: JSON.stringify({ role }),
    })
    load()
  }

  const savePhone = async (uid: string) => {
    if (!phoneInput.trim()) return
    await fetch(`${BACKEND}/admin/users/${encodeURIComponent(uid)}/identity`, {
      method: 'POST', headers: h,
      body: JSON.stringify({ identity_type: 'whatsapp', identity_value: phoneInput.trim() }),
    })
    setEditingPhone(null); setPhoneInput(''); load()
  }

  const removePhone = async (uid: string, value: string) => {
    await fetch(`${BACKEND}/admin/users/${encodeURIComponent(uid)}/identity/${encodeURIComponent(value)}`, {
      method: 'DELETE', headers: h,
    })
    load()
  }

  const saveName = async (uid: string) => {
    await fetch(`${BACKEND}/admin/users/${encodeURIComponent(uid)}`, {
      method: 'PATCH', headers: h, body: JSON.stringify({ display_name: nameInput.trim() }),
    })
    setEditingName(null); setNameInput(''); load()
  }

  const deleteUser = async (uid: string) => {
    await fetch(`${BACKEND}/admin/users/${encodeURIComponent(uid)}`, { method: 'DELETE', headers: h })
    setConfirmDelete(null); load()
  }

  const toggleRequests = async () => {
    const val = !allowRequests
    setAllowRequests(val)
    await fetch(`${BACKEND}/admin/settings`, {
      method: 'PATCH', headers: h, body: JSON.stringify({ allow_access_requests: val }),
    })
  }

  const inputSt: React.CSSProperties = {
    background: 'var(--input-bg)', border: '1px solid var(--border)',
    borderRadius: '8px', padding: '9px 12px', fontSize: '13px',
    color: 'var(--text-primary)', outline: 'none', fontFamily: 'Inter, sans-serif',
  }
  const badgeSt = (role: string): React.CSSProperties => ({
    fontSize: '11px', fontWeight: 700, borderRadius: '5px', padding: '2px 8px',
    background: role === 'admin' ? 'rgba(99,102,241,0.15)' : 'var(--bg-elevated)',
    color: role === 'admin' ? 'var(--primary)' : 'var(--text-secondary)',
    textTransform: 'uppercase', letterSpacing: '0.04em',
  })

  return (
    <div style={{
      position: 'fixed', inset: 0, zIndex: 500,
      background: 'rgba(0,0,0,0.6)', backdropFilter: 'blur(6px)',
      display: 'flex', alignItems: 'center', justifyContent: 'center', padding: '20px',
    }} onClick={e => { if (e.target === e.currentTarget) onClose() }}>
      <div style={{
        background: 'var(--bg-surface)', border: '1px solid var(--border)',
        borderRadius: '16px', width: '100%', maxWidth: '600px', maxHeight: '85vh',
        boxShadow: '0 24px 64px rgba(0,0,0,0.4)', display: 'flex', flexDirection: 'column',
      }}>
        {/* Header */}
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', padding: '24px 28px 16px' }}>
          <div>
            <h2 style={{ fontSize: '18px', fontWeight: 700, color: 'var(--text-primary)', margin: 0 }}>Panel de Administración</h2>
            <p style={{ fontSize: '13px', color: 'var(--text-secondary)', margin: '4px 0 0' }}>Gestión de usuarios e identidades</p>
          </div>
          <button onClick={onClose} style={{ background: 'none', border: 'none', cursor: 'pointer', color: 'var(--text-disabled)', fontSize: '20px' }}>✕</button>
        </div>

        <div style={{ flex: 1, overflowY: 'auto', padding: '0 28px 24px', display: 'flex', flexDirection: 'column', gap: '20px' }}>

          {/* Add user */}
          <div style={{ background: 'var(--bg-elevated)', borderRadius: '12px', padding: '16px' }}>
            <div style={{ fontSize: '12px', fontWeight: 600, color: 'var(--text-secondary)', textTransform: 'uppercase', letterSpacing: '0.05em', marginBottom: '12px' }}>Agregar usuario</div>
            <div style={{ display: 'flex', gap: '8px', flexWrap: 'wrap' }}>
              <input value={newName} onChange={e => setNewName(e.target.value)} placeholder="Nombre completo" style={{ ...inputSt, flex: '2 1 160px' }} />
              <input value={newEmail} onChange={e => setNewEmail(e.target.value)} placeholder="email@dominio.com" style={{ ...inputSt, flex: '2 1 180px' }} />
              <input value={newPhone} onChange={e => setNewPhone(e.target.value)} placeholder="WhatsApp (ej. 521999...)" style={{ ...inputSt, flex: '1 1 140px' }} />
              <select value={newRole} onChange={e => setNewRole(e.target.value)} style={{ ...inputSt, flex: '0 0 auto' }}>
                <option value="user">Usuario</option>
                <option value="admin">Admin</option>
                <option value="readonly">Solo lectura</option>
              </select>
              <button onClick={addUser} disabled={adding || !newEmail.trim()} style={{
                background: 'var(--primary)', color: '#fff', border: 'none', borderRadius: '8px',
                padding: '9px 18px', fontSize: '13px', fontWeight: 600, cursor: adding ? 'wait' : 'pointer',
                flex: '0 0 auto', opacity: newEmail.trim() ? 1 : 0.5,
              }}>
                {adding ? '…' : '+ Agregar'}
              </button>
            </div>
          </div>

          {/* Users list */}
          <div>
            <div style={{ fontSize: '12px', fontWeight: 600, color: 'var(--text-secondary)', textTransform: 'uppercase', letterSpacing: '0.05em', marginBottom: '8px' }}>
              Usuarios ({users.length})
            </div>
            {loading ? (
              <p style={{ fontSize: '13px', color: 'var(--text-disabled)', textAlign: 'center', padding: '20px 0' }}>Cargando…</p>
            ) : users.map(u => {
              const phone = u.identities.find(i => i.type === 'whatsapp')
              const isEditingPhoneRow = editingPhone === u.user_id
              const isEditingNameRow  = editingName  === u.user_id
              const isConfirmingDelete = confirmDelete === u.user_id
              return (
                <div key={u.user_id} style={{
                  borderBottom: '1px solid var(--border)', padding: '12px 0',
                  display: 'flex', flexDirection: 'column', gap: '8px',
                }}>
                  {/* Nombre + email */}
                  <div style={{ display: 'flex', alignItems: 'center', gap: '8px', flexWrap: 'wrap' }}>
                    <div style={{ flex: 1, minWidth: 0 }}>
                      {isEditingNameRow ? (
                        <div style={{ display: 'flex', gap: '6px', alignItems: 'center' }}>
                          <input value={nameInput} onChange={e => setNameInput(e.target.value)} autoFocus
                            style={{ ...inputSt, fontSize: '13px', padding: '4px 8px', flex: 1 }}
                            onKeyDown={e => { if (e.key === 'Enter') saveName(u.user_id); if (e.key === 'Escape') setEditingName(null) }} />
                          <button onClick={() => saveName(u.user_id)} style={{ fontSize: '12px', color: 'var(--primary)', background: 'none', border: 'none', cursor: 'pointer' }}>✓</button>
                          <button onClick={() => setEditingName(null)} style={{ fontSize: '12px', color: 'var(--text-disabled)', background: 'none', border: 'none', cursor: 'pointer' }}>✕</button>
                        </div>
                      ) : (
                        <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
                          <div style={{ fontSize: '13px', fontWeight: 500, color: u.status === 'active' ? 'var(--text-primary)' : 'var(--text-disabled)', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                            {u.display_name || u.user_id}
                          </div>
                          <button onClick={() => { setEditingName(u.user_id); setNameInput(u.display_name || '') }}
                            title="Editar nombre" style={{ background: 'none', border: 'none', cursor: 'pointer', color: 'var(--text-disabled)', fontSize: '11px', padding: '0 2px', flexShrink: 0 }}>✎</button>
                        </div>
                      )}
                      {u.display_name && (
                        <div style={{ fontSize: '11px', color: 'var(--text-disabled)', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{u.user_id}</div>
                      )}
                    </div>
                    <span style={badgeSt(u.role)}>{u.role}</span>
                  </div>

                  {/* Acciones: rol, activar/desactivar, eliminar */}
                  <div style={{ display: 'flex', alignItems: 'center', gap: '8px', flexWrap: 'wrap' }}>
                    <select value={u.role} onChange={e => changeRole(u.user_id, e.target.value)}
                      style={{ ...inputSt, fontSize: '12px', padding: '4px 8px' }}>
                      <option value="user">Usuario</option>
                      <option value="admin">Admin</option>
                      <option value="readonly">Solo lectura</option>
                    </select>
                    <button onClick={() => toggleStatus(u.user_id, u.status)} style={{
                      fontSize: '12px', padding: '4px 10px', borderRadius: '6px', border: '1px solid var(--border)',
                      background: 'none', cursor: 'pointer',
                      color: u.status === 'active' ? '#f59e0b' : '#22c55e',
                    }}>
                      {u.status === 'active' ? 'Desactivar' : 'Activar'}
                    </button>
                    {isConfirmingDelete ? (
                      <>
                        <span style={{ fontSize: '12px', color: 'var(--error)' }}>¿Eliminar permanentemente?</span>
                        <button onClick={() => deleteUser(u.user_id)} style={{ fontSize: '12px', color: '#fff', background: 'var(--error)', border: 'none', borderRadius: '5px', padding: '3px 10px', cursor: 'pointer' }}>Sí, eliminar</button>
                        <button onClick={() => setConfirmDelete(null)} style={{ fontSize: '12px', color: 'var(--text-secondary)', background: 'none', border: '1px solid var(--border)', borderRadius: '5px', padding: '3px 10px', cursor: 'pointer' }}>Cancelar</button>
                      </>
                    ) : (
                      <button onClick={() => setConfirmDelete(u.user_id)} style={{
                        fontSize: '12px', padding: '4px 10px', borderRadius: '6px',
                        border: '1px solid var(--border)', background: 'none',
                        cursor: 'pointer', color: 'var(--error)',
                      }}>Eliminar</button>
                    )}
                  </div>
                  {/* WhatsApp identity */}
                  <div style={{ display: 'flex', alignItems: 'center', gap: '8px', flexWrap: 'wrap' }}>
                    <span style={{ fontSize: '11px', color: 'var(--text-disabled)' }}>📱 WhatsApp:</span>
                    {phone ? (
                      <>
                        <span style={{ fontSize: '12px', color: 'var(--text-secondary)', background: 'var(--bg-elevated)', padding: '2px 8px', borderRadius: '5px' }}>
                          {phone.value}
                        </span>
                        <button onClick={() => removePhone(u.user_id, phone.value)} style={{ fontSize: '11px', color: 'var(--error)', background: 'none', border: 'none', cursor: 'pointer' }}>✕ quitar</button>
                      </>
                    ) : isEditingPhoneRow ? (
                      <>
                        <input value={phoneInput} onChange={e => setPhoneInput(e.target.value)}
                          placeholder="521999..." autoFocus
                          style={{ ...inputSt, fontSize: '12px', padding: '4px 8px', width: '140px' }}
                          onKeyDown={e => { if (e.key === 'Enter') savePhone(u.user_id); if (e.key === 'Escape') setEditingPhone(null) }} />
                        <button onClick={() => savePhone(u.user_id)} style={{ fontSize: '12px', color: 'var(--primary)', background: 'none', border: 'none', cursor: 'pointer' }}>✓ guardar</button>
                        <button onClick={() => setEditingPhone(null)} style={{ fontSize: '12px', color: 'var(--text-disabled)', background: 'none', border: 'none', cursor: 'pointer' }}>✕</button>
                      </>
                    ) : (
                      <button onClick={() => { setEditingPhone(u.user_id); setPhoneInput('') }} style={{ fontSize: '12px', color: 'var(--primary)', background: 'none', border: 'none', cursor: 'pointer' }}>+ vincular número</button>
                    )}
                  </div>
                </div>
              )
            })}
          </div>

          {/* Settings */}
          <div style={{ background: 'var(--bg-elevated)', borderRadius: '12px', padding: '16px' }}>
            <div style={{ fontSize: '12px', fontWeight: 600, color: 'var(--text-secondary)', textTransform: 'uppercase', letterSpacing: '0.05em', marginBottom: '12px' }}>Configuración de acceso</div>
            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
              <div>
                <div style={{ fontSize: '13px', fontWeight: 500, color: 'var(--text-primary)' }}>Solicitudes de acceso por WhatsApp</div>
                <div style={{ fontSize: '12px', color: 'var(--text-secondary)', marginTop: '2px' }}>
                  Permite que números desconocidos soliciten acceso desde WhatsApp
                </div>
              </div>
              <button onClick={toggleRequests} style={{
                width: '44px', height: '24px', borderRadius: '12px', border: 'none', cursor: 'pointer',
                background: allowRequests ? 'var(--primary)' : 'var(--bg-base)',
                position: 'relative', transition: 'background .2s', flexShrink: 0,
                boxShadow: 'inset 0 0 0 1px var(--border)',
              }}>
                <span style={{
                  position: 'absolute', top: '3px', width: '18px', height: '18px',
                  borderRadius: '50%', background: '#fff', transition: 'left .2s',
                  left: allowRequests ? '23px' : '3px',
                }} />
              </button>
            </div>
          </div>

        </div>
      </div>
    </div>
  )
}

// ─────────────────────────────────────────────────────────────────────────────

export default function Chat({ token, userName, userPicture, onLogout, calendarNotif }: Props) {
  const router = useRouter()
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
  const [showAdmin,   setShowAdmin]   = useState(false)
  const [userRole,    setUserRole]    = useState<string>('user')
  const [calendarToast, setCalendarToast] = useState<'connected' | 'error' | null>(calendarNotif ?? null)
  const [selectedFile, setSelectedFile] = useState<File | null>(null)
  const [uploadStatus, setUploadStatus] = useState<'idle' | 'uploading' | 'done' | 'error'>('idle')
  const [expandedImage, setExpandedImage] = useState<string | null>(null)
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

  // ── Fetch user role on mount ─────────────────────────────────────────────────
  useEffect(() => {
    fetch(`${BACKEND}/memory/profile`, { headers: headers() })
      .then(r => r.ok ? r.json() : null)
      .then(d => { if (d?.profile?.role) setUserRole(d.profile.role) })
      .catch(() => {})
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  // ── Load conversations on mount ─────────────────────────────────────────────
  useEffect(() => {
    fetch(`${BACKEND}/conversations`, { headers: headers() })
      .then(r => r.ok ? r.json() : Promise.reject('503'))
      .then(d => {
        const convs: Conv[] = d?.conversations || []
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
      .then(r => r.ok ? r.json() : Promise.reject('Error loading messages'))
      .then(d => {
        setMessages(d?.messages || [])
        setConvId(id)
        setFeedback({})
        setStreamingContent('')
      })
      .catch(err => console.error(err))
  }

  const createConversation = () => {
    if (window.innerWidth < 768) setSidebarCollapsed(true)
    fetch(`${BACKEND}/conversations`, {
      method: 'POST',
      headers: { ...headers(), 'Content-Type': 'application/json' },
      body: JSON.stringify({ title: 'Nueva conversación' }),
    })
      .then(r => r.ok ? r.json() : Promise.reject('Error creating conversation'))
      .then(d => {
        if (!d?.id) throw new Error('Invalid response')
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
        onAdmin={() => setShowAdmin(true)}
        onAgents={() => router.push('/admin/agents')}
        isAdmin={userRole === 'admin'}
      />

      {/* Profile modal */}
      {showProfile && (
        <ProfileModal token={token} onClose={() => setShowProfile(false)} />
      )}

      {/* Admin panel */}
      {showAdmin && (
        <AdminPanel token={token} onClose={() => setShowAdmin(false)} />
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

        {/* Calendar OAuth toast */}
        {calendarToast && (
          <div style={{
            display: 'flex', alignItems: 'center', justifyContent: 'space-between',
            padding: '10px 20px', flexShrink: 0,
            background: calendarToast === 'connected' ? 'rgba(34,197,94,0.12)' : 'rgba(239,68,68,0.12)',
            borderBottom: `1px solid ${calendarToast === 'connected' ? 'rgba(34,197,94,0.3)' : 'rgba(239,68,68,0.3)'}`,
          }}>
            <span style={{ fontSize: '13px', color: calendarToast === 'connected' ? '#22c55e' : '#ef4444' }}>
              {calendarToast === 'connected'
                ? '✅ Google Calendar conectado correctamente. El briefing diario usará tu agenda real.'
                : '⚠️ No se pudo conectar Google Calendar. Intenta nuevamente desde Mi Perfil.'}
            </span>
            <button onClick={() => setCalendarToast(null)} style={{
              background: 'none', border: 'none', cursor: 'pointer',
              color: 'var(--text-disabled)', fontSize: '16px', lineHeight: 1, padding: '0 4px',
            }}>✕</button>
          </div>
        )}

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
                      onImageClick={setExpandedImage}
                    />
                  ))}

                  {streamingContent && (
                    <Message
                      role="assistant"
                      content={streamingContent}
                      isStreaming
                      onImageClick={setExpandedImage}
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
      {expandedImage && (
        <ImageModal src={expandedImage} onClose={() => setExpandedImage(null)} />
      )}
    </div>
  )
}
