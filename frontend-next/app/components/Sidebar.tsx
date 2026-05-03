'use client'
import { useState, useRef, useEffect } from 'react'
import { useTheme } from './ThemeProvider'

interface Conversation {
  id: number
  title: string
  last_active_at: string
}

interface Props {
  user: { name: string; picture: string } | null
  conversations: Conversation[]
  activeId: number | null
  collapsed: boolean
  isMobile?: boolean
  onToggle: () => void
  onSelect: (id: number) => void
  onNew: () => void
  onRename: (id: number, title: string) => void
  onDelete: (id: number) => void
  onLogout: () => void
  onProfile?: () => void
  onAdmin?: () => void
  onAgents?: () => void
  isAdmin?: boolean
  activeAgent?: string
  onAgentChange?: (agent: string) => void
}

// ── Agent definitions ─────────────────────────────────────────────────────────
const AGENTS = [
  { id: 'amael',   label: 'Amael',   short: 'A', color: '#6366f1', desc: 'Orquestador' },
  { id: 'raphael', label: 'Raphael', short: 'R', color: '#10b981', desc: 'SRE' },
  { id: 'camael',  label: 'Camael',  short: 'C', color: '#f59e0b', desc: 'DevOps' },
]

// ── Minimal icon set ──────────────────────────────────────────────────────────
const Ico = ({ d, size = 16 }: { d: string; size?: number }) => (
  <svg width={size} height={size} viewBox="0 0 24 24" fill="none"
    stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <path d={d} />
  </svg>
)

const IcoMenu    = () => <Ico d="M3 6h18M3 12h18M3 18h18" />
const IcoNewChat = () => (
  <svg width="16" height="16" viewBox="0 0 24 24" fill="none"
    stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <path d="M11 4H4a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h14a2 2 0 0 0 2-2v-7" />
    <path d="M18.5 2.5a2.121 2.121 0 0 1 3 3L12 15l-4 1 1-4 9.5-9.5z" />
  </svg>
)
const IcoSun  = () => <Ico d="M12 1v2M12 21v2M4.22 4.22l1.42 1.42M18.36 18.36l1.42 1.42M1 12h2M21 12h2M4.22 19.78l1.42-1.42M18.36 5.64l1.42-1.42" />
const IcoMoon = () => <Ico d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z" />
const IcoOut     = () => <Ico d="M9 21H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h4M16 17l5-5-5-5M21 12H9" />
const IcoProfile = () => <Ico d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2M12 11a4 4 0 1 0 0-8 4 4 0 0 0 0 8z" />
const IcoAdmin   = () => <Ico d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z" />
const IcoAgents  = () => (
  <svg width="16" height="16" viewBox="0 0 24 24" fill="none"
    stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <circle cx="9" cy="7" r="3" />
    <path d="M3 21v-2a4 4 0 0 1 4-4h4a4 4 0 0 1 4 4v2" />
    <path d="M16 3.13a4 4 0 0 1 0 7.75" />
    <path d="M21 21v-2a4 4 0 0 0-3-3.85" />
  </svg>
)
const IcoDots = () => (
  <svg width="14" height="14" viewBox="0 0 24 24" fill="currentColor">
    <circle cx="5" cy="12" r="2" /><circle cx="12" cy="12" r="2" /><circle cx="19" cy="12" r="2" />
  </svg>
)

// ── Icon button ───────────────────────────────────────────────────────────────
function IconBtn({
  title, icon, onClick, danger = false, collapsed, label,
}: {
  title: string
  icon: React.ReactNode
  onClick: () => void
  danger?: boolean
  collapsed: boolean
  label?: string
}) {
  return (
    <button
      title={title}
      onClick={onClick}
      style={{
        width: '100%', display: 'flex', alignItems: 'center',
        justifyContent: collapsed ? 'center' : 'flex-start',
        gap: '10px', background: 'none', border: 'none', cursor: 'pointer',
        color: danger ? 'var(--error)' : 'var(--text-secondary)',
        fontSize: '13px', padding: collapsed ? '10px 0' : '9px 10px',
        borderRadius: '7px', transition: 'background .15s, color .15s',
        minHeight: '44px',
      }}
      onMouseEnter={e => {
        e.currentTarget.style.background = 'var(--bg-elevated)'
        e.currentTarget.style.color = danger ? 'var(--error)' : 'var(--text-primary)'
      }}
      onMouseLeave={e => {
        e.currentTarget.style.background = 'none'
        e.currentTarget.style.color = danger ? 'var(--error)' : 'var(--text-secondary)'
      }}
    >
      <span style={{ display: 'flex', alignItems: 'center', flexShrink: 0 }}>{icon}</span>
      {!collapsed && label && <span>{label}</span>}
    </button>
  )
}

export default function Sidebar({
  user, conversations, activeId, collapsed, isMobile = false, onToggle,
  onSelect, onNew, onRename, onDelete, onLogout, onProfile, onAdmin, onAgents, isAdmin,
  activeAgent = 'amael', onAgentChange,
}: Props) {
  const { theme, toggle } = useTheme()
  const [menuOpenId,  setMenuOpenId]  = useState<number | null>(null)
  const [renamingId,  setRenamingId]  = useState<number | null>(null)
  const [renameValue, setRenameValue] = useState('')
  const renameRef = useRef<HTMLInputElement>(null)

  useEffect(() => {
    if (renamingId !== null) setTimeout(() => renameRef.current?.focus(), 0)
  }, [renamingId])

  useEffect(() => {
    if (menuOpenId === null) return
    const close = (e: MouseEvent) => {
      const target = e.target as Element
      if (!target.closest('[data-menu]')) setMenuOpenId(null)
    }
    document.addEventListener('mousedown', close)
    return () => document.removeEventListener('mousedown', close)
  }, [menuOpenId])

  const startRename = (c: Conversation) => {
    setMenuOpenId(null)
    setRenamingId(c.id)
    setRenameValue(c.title)
  }

  const commitRename = () => {
    if (renamingId !== null && renameValue.trim()) {
      onRename(renamingId, renameValue.trim())
    }
    setRenamingId(null)
  }

  const menuItems = (c: Conversation) => [
    { label: 'Renombrar', onClick: () => startRename(c) },
    { label: 'Agregar a proyecto', onClick: () => setMenuOpenId(null), disabled: true },
    { label: 'Eliminar', onClick: () => { onDelete(c.id); setMenuOpenId(null) }, danger: true },
  ]

  // On desktop: 60px when collapsed, 260px when expanded
  // On mobile: CSS controls via .sidebar-root (fixed, 280px), hidden by transform
  const w = collapsed ? '60px' : '260px'

  return (
    <div
      className={isMobile ? `sidebar-root${collapsed ? ' mobile-hidden' : ''}` : ''}
      style={{
        width: isMobile ? '280px' : w,
        minWidth: isMobile ? '280px' : w,
        height: '100dvh',
        background: 'var(--sidebar-bg)', borderRight: '1px solid var(--border)',
        display: 'flex', flexDirection: 'column', overflow: 'hidden',
        transition: 'width .2s, min-width .2s', flexShrink: 0,
      }}
    >

      {/* ── Header ─────────────────────────────────────────────────────────── */}
      <div style={{
        display: 'flex', alignItems: 'center',
        justifyContent: (!isMobile && collapsed) ? 'center' : 'space-between',
        padding: '14px 12px', gap: '8px',
        borderBottom: (!isMobile && collapsed) ? 'none' : '1px solid var(--border)',
        flexShrink: 0,
      }}>
        {(isMobile || !collapsed) && (
          <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
            <div style={{
              width: '26px', height: '26px', borderRadius: '6px',
              background: 'var(--primary)', display: 'flex', alignItems: 'center',
              justifyContent: 'center', fontSize: '13px', fontWeight: 700, color: '#fff',
              flexShrink: 0,
            }}>A</div>
            <span style={{ fontSize: '17px', fontWeight: 700, color: 'var(--text-primary)', letterSpacing: '-0.3px' }}>
              Amael
            </span>
          </div>
        )}
        <button
          onClick={onToggle}
          title={collapsed ? 'Expandir' : 'Colapsar'}
          style={{
            background: 'none', border: 'none', cursor: 'pointer',
            color: 'var(--text-disabled)', padding: '4px', borderRadius: '6px',
            display: 'flex', alignItems: 'center', justifyContent: 'center',
            transition: 'color .15s', minWidth: '32px', minHeight: '32px',
          }}
          onMouseEnter={e => (e.currentTarget.style.color = 'var(--text-primary)')}
          onMouseLeave={e => (e.currentTarget.style.color = 'var(--text-disabled)')}
        >
          <IcoMenu />
        </button>
      </div>

      {/* ── Nueva conversación ─────────────────────────────────────────────── */}
      <div style={{ padding: (!isMobile && collapsed) ? '10px 8px' : '10px 10px 0', flexShrink: 0 }}>
        <button
          onClick={onNew}
          title="Nueva conversación"
          style={{
            width: '100%', display: 'flex', alignItems: 'center',
            justifyContent: (!isMobile && collapsed) ? 'center' : 'flex-start',
            gap: '10px', background: 'none', border: 'none', cursor: 'pointer',
            color: 'var(--text-secondary)', fontSize: '13px', fontWeight: 500,
            padding: (!isMobile && collapsed) ? '10px 0' : '9px 10px',
            borderRadius: '7px', transition: 'background .15s, color .15s',
            minHeight: '44px',
          }}
          onMouseEnter={e => {
            e.currentTarget.style.background = 'var(--bg-elevated)'
            e.currentTarget.style.color = 'var(--text-primary)'
          }}
          onMouseLeave={e => {
            e.currentTarget.style.background = 'none'
            e.currentTarget.style.color = 'var(--text-secondary)'
          }}
        >
          <span style={{ display: 'flex', alignItems: 'center', flexShrink: 0 }}><IcoNewChat /></span>
          {(isMobile || !collapsed) && <span>Nueva conversación</span>}
        </button>
      </div>

      {/* ── Agent selector ────────────────────────────────────────────────── */}
      {onAgentChange && (
        <div style={{
          padding: (!isMobile && collapsed) ? '8px 6px' : '10px 10px 0',
          flexShrink: 0,
        }}>
          {(!isMobile && collapsed) ? (
            // Collapsed: show only dot/letter for active agent
            <div style={{ display: 'flex', flexDirection: 'column', gap: '4px', alignItems: 'center' }}>
              {AGENTS.map(a => (
                <button
                  key={a.id}
                  title={`${a.label} — ${a.desc}`}
                  onClick={() => onAgentChange(a.id)}
                  style={{
                    width: '36px', height: '36px', borderRadius: '8px', border: 'none',
                    cursor: 'pointer', fontSize: '12px', fontWeight: 700, color: '#fff',
                    background: activeAgent === a.id ? a.color : 'var(--bg-elevated)',
                    transition: 'background .15s',
                    display: 'flex', alignItems: 'center', justifyContent: 'center',
                  }}
                  onMouseEnter={e => {
                    if (activeAgent !== a.id) e.currentTarget.style.background = 'var(--border)'
                  }}
                  onMouseLeave={e => {
                    if (activeAgent !== a.id) e.currentTarget.style.background = 'var(--bg-elevated)'
                  }}
                >
                  <span style={{ color: activeAgent === a.id ? '#fff' : 'var(--text-secondary)' }}>
                    {a.short}
                  </span>
                </button>
              ))}
            </div>
          ) : (
            // Expanded: full agent tabs
            <div>
              <div style={{
                fontSize: '11px', fontWeight: 600, color: 'var(--text-disabled)',
                textTransform: 'uppercase', letterSpacing: '0.06em',
                padding: '4px 2px 8px',
              }}>Agente</div>
              <div style={{ display: 'flex', gap: '6px' }}>
                {AGENTS.map(a => (
                  <button
                    key={a.id}
                    onClick={() => onAgentChange(a.id)}
                    title={a.desc}
                    style={{
                      flex: 1, padding: '7px 4px', borderRadius: '8px', border: 'none',
                      cursor: 'pointer', fontSize: '12px', fontWeight: 600,
                      background: activeAgent === a.id ? a.color : 'var(--bg-elevated)',
                      color: activeAgent === a.id ? '#fff' : 'var(--text-secondary)',
                      transition: 'background .15s, color .15s',
                      display: 'flex', flexDirection: 'column', alignItems: 'center', gap: '2px',
                    }}
                    onMouseEnter={e => {
                      if (activeAgent !== a.id) {
                        e.currentTarget.style.background = 'var(--border)'
                        e.currentTarget.style.color = 'var(--text-primary)'
                      }
                    }}
                    onMouseLeave={e => {
                      if (activeAgent !== a.id) {
                        e.currentTarget.style.background = 'var(--bg-elevated)'
                        e.currentTarget.style.color = 'var(--text-secondary)'
                      }
                    }}
                  >
                    <span style={{ fontSize: '14px', fontWeight: 700 }}>{a.short}</span>
                    <span style={{ fontSize: '10px', opacity: 0.9 }}>{a.label}</span>
                  </button>
                ))}
              </div>
            </div>
          )}
        </div>
      )}

      {/* ── Conversations list ─────────────────────────────────────────────── */}
      {(isMobile || !collapsed) && (
        <>
          <div style={{
            fontSize: '11px', fontWeight: 600, color: 'var(--text-disabled)',
            textTransform: 'uppercase', letterSpacing: '0.06em',
            padding: '16px 12px 4px', flexShrink: 0,
          }}>Historial</div>

          <div style={{ flex: 1, overflowY: 'auto', padding: '2px 6px' }}>
            {conversations.map(c => {
              const isActive = activeId === c.id
              const isMenuOpen = menuOpenId === c.id
              const isRenaming = renamingId === c.id

              return (
                <div key={c.id} style={{ position: 'relative', marginBottom: '1px' }}>
                  {isRenaming ? (
                    <input
                      ref={renameRef}
                      value={renameValue}
                      onChange={e => setRenameValue(e.target.value)}
                      onBlur={commitRename}
                      onKeyDown={e => {
                        if (e.key === 'Enter') commitRename()
                        if (e.key === 'Escape') setRenamingId(null)
                      }}
                      style={{
                        width: '100%', background: 'var(--input-bg)',
                        border: '1px solid var(--border-focus)',
                        borderRadius: '6px', padding: '7px 10px', fontSize: '16px',
                        color: 'var(--text-primary)', outline: 'none',
                        boxSizing: 'border-box',
                      }}
                    />
                  ) : (
                    <button
                      onClick={() => onSelect(c.id)}
                      style={{
                        width: '100%', textAlign: 'left',
                        background: isActive ? 'var(--primary-subtle)' : 'none',
                        border: 'none', borderRadius: '6px',
                        padding: '9px 32px 9px 10px', fontSize: '13px',
                        color: isActive ? 'var(--primary)' : 'var(--text-primary)',
                        fontWeight: isActive ? 500 : 400,
                        cursor: 'pointer', display: 'block',
                        whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis',
                        transition: 'background .1s', minHeight: '40px',
                      }}
                      onMouseEnter={e => {
                        if (!isActive) e.currentTarget.style.background = 'var(--bg-elevated)'
                        const btn = e.currentTarget.nextElementSibling as HTMLElement | null
                        if (btn) btn.style.opacity = '1'
                      }}
                      onMouseLeave={e => {
                        if (!isActive) e.currentTarget.style.background = 'none'
                        const btn = e.currentTarget.nextElementSibling as HTMLElement | null
                        if (btn && !isMenuOpen) btn.style.opacity = '0'
                      }}
                    >
                      {c.title.length > 30 ? c.title.slice(0, 30) + '…' : c.title}
                    </button>
                  )}

                  {/* 3-dot button */}
                  {!isRenaming && (
                    <button
                      data-menu
                      onClick={e => {
                        e.stopPropagation()
                        setMenuOpenId(prev => prev === c.id ? null : c.id)
                      }}
                      style={{
                        position: 'absolute', right: '2px', top: '50%',
                        transform: 'translateY(-50%)',
                        background: 'none', border: 'none', cursor: 'pointer',
                        color: 'var(--text-secondary)', padding: '4px 6px',
                        borderRadius: '4px', opacity: isMenuOpen ? 1 : 0,
                        transition: 'opacity .1s, background .1s',
                        display: 'flex', alignItems: 'center', minHeight: '32px',
                      }}
                      title="Opciones"
                      onMouseEnter={e => {
                        e.currentTarget.style.background = 'var(--border)'
                        e.currentTarget.style.opacity = '1'
                      }}
                      onMouseLeave={e => {
                        e.currentTarget.style.background = 'none'
                        if (!isMenuOpen) e.currentTarget.style.opacity = '0'
                      }}
                    >
                      <IcoDots />
                    </button>
                  )}

                  {/* Dropdown menu */}
                  {isMenuOpen && (
                    <div
                      data-menu
                      style={{
                        position: 'absolute', right: 0, top: '100%', zIndex: 200,
                        background: 'var(--bg-surface)', border: '1px solid var(--border)',
                        borderRadius: '9px',
                        boxShadow: '0 6px 24px rgba(0,0,0,0.25)',
                        minWidth: '170px', overflow: 'hidden', marginTop: '2px',
                      }}
                    >
                      {menuItems(c).map(item => (
                        <button
                          key={item.label}
                          onClick={item.onClick}
                          disabled={item.disabled}
                          style={{
                            width: '100%', textAlign: 'left', background: 'none',
                            border: 'none', padding: '11px 14px', fontSize: '13px',
                            color: item.disabled
                              ? 'var(--text-disabled)'
                              : item.danger ? 'var(--error)' : 'var(--text-primary)',
                            cursor: item.disabled ? 'default' : 'pointer',
                            display: 'block', transition: 'background .1s',
                            minHeight: '44px',
                          }}
                          onMouseEnter={e => {
                            if (!item.disabled) e.currentTarget.style.background = 'var(--bg-elevated)'
                          }}
                          onMouseLeave={e => e.currentTarget.style.background = 'none'}
                        >
                          {item.label}
                        </button>
                      ))}
                    </div>
                  )}
                </div>
              )
            })}
          </div>
        </>
      )}

      {/* Spacer when collapsed on desktop */}
      {!isMobile && collapsed && <div style={{ flex: 1 }} />}

      {/* ── Bottom controls ─────────────────────────────────────────────────── */}
      <div style={{
        padding: (!isMobile && collapsed) ? '10px 8px' : '8px 10px',
        borderTop: '1px solid var(--border)', flexShrink: 0,
        paddingBottom: 'max(8px, calc(8px + env(safe-area-inset-bottom)))',
      }}>
        {/* Profile */}
        {user && (
          <div style={{
            display: 'flex', alignItems: 'center',
            justifyContent: (!isMobile && collapsed) ? 'center' : 'flex-start',
            gap: '10px', marginBottom: '4px',
            padding: (!isMobile && collapsed) ? '4px 0' : '6px 10px',
          }}>
            {user.picture ? (
              <img src={user.picture} alt="" style={{
                width: '28px', height: '28px', borderRadius: '50%',
                objectFit: 'cover', border: '1.5px solid var(--border)', flexShrink: 0,
              }} />
            ) : (
              <div style={{
                width: '28px', height: '28px', borderRadius: '50%',
                background: 'var(--primary-subtle)', border: '1.5px solid var(--border)',
                display: 'flex', alignItems: 'center', justifyContent: 'center',
                fontSize: '12px', fontWeight: 600, color: 'var(--primary)', flexShrink: 0,
              }}>{user.name[0].toUpperCase()}</div>
            )}
            {(isMobile || !collapsed) && (
              <span style={{
                fontSize: '13px', fontWeight: 500, color: 'var(--text-primary)',
                whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis',
              }}>{user.name.split(' ')[0]}</span>
            )}
          </div>
        )}

        {isAdmin && onAdmin && (
          <IconBtn
            title="Panel de administración"
            icon={<IcoAdmin />}
            label="Admin"
            onClick={onAdmin}
            collapsed={!isMobile && collapsed}
          />
        )}
        {isAdmin && onAgents && (
          <IconBtn
            title="Organización de agentes"
            icon={<IcoAgents />}
            label="Agentes"
            onClick={onAgents}
            collapsed={!isMobile && collapsed}
          />
        )}
        {onProfile && (
          <IconBtn
            title="Mi perfil"
            icon={<IcoProfile />}
            label="Mi perfil"
            onClick={onProfile}
            collapsed={!isMobile && collapsed}
          />
        )}
        <IconBtn
          title={theme === 'dark' ? 'Modo claro' : 'Modo oscuro'}
          icon={theme === 'dark' ? <IcoSun /> : <IcoMoon />}
          label={theme === 'dark' ? 'Modo claro' : 'Modo oscuro'}
          onClick={toggle}
          collapsed={!isMobile && collapsed}
        />
        <IconBtn
          title="Cerrar sesión"
          icon={<IcoOut />}
          label="Cerrar sesión"
          onClick={onLogout}
          danger
          collapsed={!isMobile && collapsed}
        />
      </div>
    </div>
  )
}
