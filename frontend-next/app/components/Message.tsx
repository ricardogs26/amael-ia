'use client'
import { useState } from 'react'

interface Props {
  role: 'user' | 'assistant'
  content: string
  ts?: string
  isStreaming?: boolean
  feedback?: 'positive' | 'negative' | null
  onFeedback?: (sentiment: 'positive' | 'negative') => void
}

// ── Content block types ───────────────────────────────────────────────────────
type Block =
  | { kind: 'text';       text: string }
  | { kind: 'code';       lang: string; code: string }
  | { kind: 'image-b64'; data: string }
  | { kind: 'image-url'; url:  string }

const TERMINAL_LANGS = new Set(['bash', 'sh', 'zsh', 'shell', 'kubectl', 'terminal', 'console', ''])

/** Split text into text/code blocks on ``` fences */
function splitCodeFences(text: string): Block[] {
  const out: Block[] = []
  const re = /```([^\n`]*)\n([\s\S]*?)```/g
  let last = 0; let m: RegExpExecArray | null
  while ((m = re.exec(text)) !== null) {
    if (m.index > last) out.push({ kind: 'text', text: text.slice(last, m.index) })
    out.push({ kind: 'code', lang: m[1].trim().toLowerCase(), code: m[2].trimEnd() })
    last = m.index + m[0].length
  }
  if (last < text.length) out.push({ kind: 'text', text: text.slice(last) })
  return out
}

/** Split raw assistant message into typed blocks */
function parseContent(raw: string): Block[] {
  const out: Block[] = []
  const mediaRe = /\[MEDIA:([A-Za-z0-9+/=\s]+?)\]/g
  const chartRe  = /(?:!\[.*?\]\()?(https:\/\/quickchart\.io\/chart[^\s)\]]*)\)?/g

  // 1. Extract [MEDIA:...] tags
  let tmp = ''; let last = 0; let m: RegExpExecArray | null
  mediaRe.lastIndex = 0
  while ((m = mediaRe.exec(raw)) !== null) {
    tmp += raw.slice(last, m.index)
    if (tmp) { out.push(...splitCodeFences(tmp)); tmp = '' }
    out.push({ kind: 'image-b64', data: m[1].replace(/\s/g, '') })
    last = m.index + m[0].length
  }
  let remaining = tmp + raw.slice(last)

  // 2. Extract QuickChart URLs
  last = 0; chartRe.lastIndex = 0
  while ((m = chartRe.exec(remaining)) !== null) {
    const before = remaining.slice(last, m.index)
    if (before) out.push(...splitCodeFences(before))
    out.push({ kind: 'image-url', url: m[1].replace(/ /g, '%20') })
    last = m.index + m[0].length
    if (remaining[last] === ')') last++
  }
  const tail = remaining.slice(last)
  if (tail) out.push(...splitCodeFences(tail))

  return out.filter(b => !(b.kind === 'text' && !b.text.trim()))
}

// ── CodeBlock — terminal-style for bash/sh/kubectl, editor-style for yaml/json ─
function CodeBlock({ lang, code }: { lang: string; code: string }) {
  const [copied, setCopied] = useState(false)
  const isTerminal = TERMINAL_LANGS.has(lang)

  const copy = () => {
    navigator.clipboard.writeText(code)
    setCopied(true)
    setTimeout(() => setCopied(false), 2000)
  }

  const label = lang || 'terminal'

  return (
    <div style={{
      borderRadius: '10px',
      overflow: 'hidden',
      margin: '12px 0',
      border: '1px solid var(--border)',
      fontFamily: "'JetBrains Mono','Fira Code','Cascadia Code','Consolas',monospace",
    }}>
      {/* Header bar */}
      <div style={{
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between',
        padding: '8px 14px',
        background: isTerminal ? '#0d1117' : 'var(--bg-elevated)',
        borderBottom: '1px solid var(--border)',
        userSelect: 'none',
      }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
          {isTerminal ? (
            <>
              <span style={{ fontSize: '10px', color: '#ef4444' }}>●</span>
              <span style={{ fontSize: '10px', color: '#f59e0b' }}>●</span>
              <span style={{ fontSize: '10px', color: '#22c55e' }}>●</span>
              <span style={{
                marginLeft: '8px', fontSize: '12px',
                color: 'var(--text-disabled)', fontFamily: 'inherit',
              }}>~/terminal</span>
            </>
          ) : (
            <span style={{
              fontSize: '12px', color: 'var(--text-disabled)',
              fontFamily: 'inherit', textTransform: 'uppercase', letterSpacing: '0.05em',
            }}>{label}</span>
          )}
        </div>
        <button
          onClick={copy}
          style={{
            background: copied ? 'rgba(34,197,94,0.12)' : 'rgba(255,255,255,0.06)',
            border: '1px solid var(--border)',
            borderRadius: '6px',
            color: copied ? '#22c55e' : 'var(--text-secondary)',
            cursor: 'pointer',
            fontSize: '12px',
            padding: '3px 10px',
            transition: 'all .15s',
            fontFamily: 'Inter, sans-serif',
            display: 'flex', alignItems: 'center', gap: '5px',
          }}
          onMouseEnter={e => { if (!copied) e.currentTarget.style.background = 'rgba(255,255,255,0.1)' }}
          onMouseLeave={e => { if (!copied) e.currentTarget.style.background = 'rgba(255,255,255,0.06)' }}
        >
          {copied ? '✓ Copiado' : '⧉ Copiar'}
        </button>
      </div>

      {/* Code content */}
      <pre style={{
        margin: 0,
        padding: '16px 18px',
        background: isTerminal ? '#0a0d14' : '#0f1623',
        overflowX: 'auto',
        fontSize: '13px',
        lineHeight: '1.65',
        color: isTerminal ? '#a8ff78' : '#e2e8f0',
        whiteSpace: 'pre-wrap',
        overflowWrap: 'break-word',
      }}>
        <code style={{ fontFamily: 'inherit', background: 'none', color: 'inherit', padding: 0 }}>
          {code}
        </code>
      </pre>
    </div>
  )
}

// ── MarkdownContent — plain text + inline markdown (no code fences, handled above) ──
function MarkdownContent({ text }: { text: string }) {
  return (
    <div
      className="prose-amael"
      dangerouslySetInnerHTML={{
        __html: text
          .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
          // inline code
          .replace(/`([^`]+)`/g, '<code>$1</code>')
          // bold
          .replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>')
          // italic
          .replace(/\*(.+?)\*/g, '<em>$1</em>')
          // headers
          .replace(/^### (.+)$/gm, '<h3>$1</h3>')
          .replace(/^## (.+)$/gm, '<h2>$1</h2>')
          .replace(/^# (.+)$/gm, '<h1>$1</h1>')
          // hr
          .replace(/^---$/gm, '<hr>')
          // blockquote
          .replace(/^> (.+)$/gm, '<blockquote><p>$1</p></blockquote>')
          // unordered list
          .replace(/^\s*[-*] (.+)$/gm, '<li>$1</li>')
          .replace(/(<li>[\s\S]+?<\/li>)/g, '<ul>$1</ul>')
          // paragraphs
          .replace(/\n{2,}/g, '</p><p>')
          .replace(/^(?!<[hup\/<])(.+)/gm, m => m.startsWith('<') ? m : `<p>${m}</p>`)
      }}
    />
  )
}

// ── Main Message component ────────────────────────────────────────────────────
export default function Message({ role, content, ts, isStreaming, feedback, onFeedback }: Props) {
  const [copied, setCopied] = useState(false)
  const [hovered, setHovered] = useState(false)
  const isUser = role === 'user'

  const copy = () => {
    navigator.clipboard.writeText(content)
    setCopied(true)
    setTimeout(() => setCopied(false), 2000)
  }

  const renderBlocks = () => {
    if (isStreaming) return <><MarkdownContent text={content} /><span className="cursor" /></>
    return parseContent(content).map((block, i) => {
      if (block.kind === 'image-b64') {
        return (
          <img key={i} src={`data:image/png;base64,${block.data}`} alt="Grafana screenshot"
            style={{ maxWidth: '100%', borderRadius: '8px', margin: '8px 0', display: 'block' }} />
        )
      }
      if (block.kind === 'image-url') {
        return (
          <img key={i} src={block.url} alt="Chart"
            style={{ maxWidth: '100%', borderRadius: '8px', margin: '8px 0', display: 'block' }} />
        )
      }
      if (block.kind === 'code') {
        return <CodeBlock key={i} lang={block.lang} code={block.code} />
      }
      return <MarkdownContent key={i} text={block.text} />
    })
  }

  return (
    <div className="slide-in" style={{ marginBottom: '2px' }}>
      <div
        style={{
          display: 'flex',
          flexDirection: isUser ? 'row-reverse' : 'row',
          gap: '10px',
          alignItems: 'flex-start',
          position: 'relative',
        }}
        onMouseEnter={() => setHovered(true)}
        onMouseLeave={() => setHovered(false)}
      >
        {/* Avatar — only for assistant */}
        {!isUser && (
          <div style={{
            width: '28px', height: '28px', minWidth: '28px', borderRadius: '7px',
            background: 'var(--primary)', display: 'flex', alignItems: 'center',
            justifyContent: 'center', fontSize: '12px', fontWeight: 700, color: '#fff',
            marginTop: '2px', flexShrink: 0,
          }}>A</div>
        )}

        {/* Bubble / content */}
        <div style={isUser ? {
          background: 'var(--user-bubble-bg)',
          border: '1px solid var(--user-bubble-border)',
          borderRadius: '18px 18px 4px 18px',
          padding: '11px 16px',
          maxWidth: '82%',
          marginLeft: 'auto',
          fontSize: '15px',
          lineHeight: '1.6',
          color: 'var(--text-primary)',
        } : {
          flex: 1,
          fontSize: '15px',
          lineHeight: '1.7',
          color: 'var(--text-primary)',
          padding: '0 4px',
        }}>
          {isUser
            ? <p style={{ margin: 0, color: 'var(--text-primary)' }}>{content}</p>
            : renderBlocks()
          }
        </div>

        {/* Copy button — assistant only, visible on hover */}
        {!isUser && !isStreaming && (
          <button
            onClick={copy}
            style={{
              position: 'absolute', top: 0, right: 0,
              background: 'none', border: 'none', cursor: 'pointer',
              color: copied ? 'var(--primary)' : 'var(--text-disabled)',
              fontSize: '14px', padding: '4px 8px', borderRadius: '6px',
              opacity: hovered ? 1 : 0, transition: 'opacity .15s, color .15s',
            }}
            title="Copiar respuesta"
          >
            {copied ? '✓' : '⧉'}
          </button>
        )}
      </div>

      {/* Timestamp */}
      {ts && (
        <div style={{
          fontSize: '11px', color: 'var(--text-disabled)',
          marginTop: '4px', marginBottom: '4px',
          textAlign: isUser ? 'right' : 'left',
          paddingLeft: isUser ? 0 : '38px',
        }}>{ts}</div>
      )}

      {/* Feedback — assistant only */}
      {!isUser && !isStreaming && (
        <div style={{ paddingLeft: '38px', marginBottom: '8px', display: 'flex', gap: '4px' }}>
          {feedback ? (
            <span style={{ fontSize: '13px', color: 'var(--text-disabled)' }}>
              {feedback === 'positive' ? '👍' : '👎'} Gracias por tu feedback
            </span>
          ) : (
            <>
              {(['positive', 'negative'] as const).map(s => (
                <button
                  key={s}
                  onClick={() => onFeedback?.(s)}
                  style={{
                    background: 'none', border: 'none', cursor: 'pointer',
                    fontSize: '16px', minWidth: '44px', minHeight: '44px',
                    padding: '0 6px', borderRadius: '8px',
                    display: 'flex', alignItems: 'center', justifyContent: 'center',
                    opacity: 0.5, transition: 'opacity .15s',
                  }}
                  onMouseEnter={e => (e.currentTarget.style.opacity = '1')}
                  onMouseLeave={e => (e.currentTarget.style.opacity = '0.5')}
                  title={s === 'positive' ? 'Útil' : 'No útil'}
                >{s === 'positive' ? '👍' : '👎'}</button>
              ))}
            </>
          )}
        </div>
      )}
    </div>
  )
}
