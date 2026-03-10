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

// ── Content block types produced by parseContent() ───────────────────────────
type Block =
  | { kind: 'text';        text: string }
  | { kind: 'image-b64';  data: string }
  | { kind: 'image-url';  url:  string }

/** Split a raw assistant message into text / image blocks. */
function parseContent(raw: string): Block[] {
  const blocks: Block[] = []

  // Handle [MEDIA:base64...] tags (Grafana screenshots, etc.)
  const mediaRe = /\[MEDIA:([A-Za-z0-9+/=\s]+?)\]/g
  // Handle QuickChart URLs (markdown image or bare URL)
  const chartRe = /(?:!\[.*?\]\()?(https:\/\/quickchart\.io\/chart[^\s)\]]*)\)?/g

  let remaining = raw

  // First extract [MEDIA:...] blocks
  let tmp = ''
  let last = 0
  let m: RegExpExecArray | null
  mediaRe.lastIndex = 0
  while ((m = mediaRe.exec(raw)) !== null) {
    tmp += raw.slice(last, m.index)
    if (tmp) { blocks.push({ kind: 'text', text: tmp }); tmp = '' }
    blocks.push({ kind: 'image-b64', data: m[1].replace(/\s/g, '') })
    last = m.index + m[0].length
  }
  remaining = raw.slice(last)

  // Then extract QuickChart URLs from the remaining text
  last = 0
  chartRe.lastIndex = 0
  while ((m = chartRe.exec(remaining)) !== null) {
    const before = remaining.slice(last, m.index)
    if (before) blocks.push({ kind: 'text', text: before })
    blocks.push({ kind: 'image-url', url: m[1].replace(/ /g, '%20') })
    last = m.index + m[0].length
    // skip trailing ) if it was a markdown image
    if (remaining[last] === ')') last++
  }
  const tail = remaining.slice(last)
  if (tail) blocks.push({ kind: 'text', text: tail })

  return blocks.filter(b => !(b.kind === 'text' && !b.text.trim()))
}

function MarkdownContent({ text }: { text: string }) {
  return (
    <div
      className="prose-amael"
      dangerouslySetInnerHTML={{
        __html: text
          .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
          // code blocks first
          .replace(/```[\w]*\n?([\s\S]*?)```/g, '<pre><code>$1</code></pre>')
          // inline code
          .replace(/`([^`]+)`/g, '<code>$1</code>')
          // bold
          .replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>')
          // headers
          .replace(/^### (.+)$/gm, '<h3>$1</h3>')
          .replace(/^## (.+)$/gm, '<h2>$1</h2>')
          .replace(/^# (.+)$/gm, '<h1>$1</h1>')
          // hr
          .replace(/^---$/gm, '<hr>')
          // blockquote
          .replace(/^> (.+)$/gm, '<blockquote><p>$1</p></blockquote>')
          // unordered list items
          .replace(/^\s*[-*] (.+)$/gm, '<li>$1</li>')
          .replace(/(<li>[\s\S]+?<\/li>)/g, '<ul>$1</ul>')
          // paragraphs (double newline)
          .replace(/\n{2,}/g, '</p><p>')
          .replace(/^(?!<[hup\/<])(.+)/gm, (m) => m.startsWith('<') ? m : `<p>${m}</p>`)
      }}
    />
  )
}

export default function Message({ role, content, ts, isStreaming, feedback, onFeedback }: Props) {
  const [copied, setCopied] = useState(false)
  const [hovered, setHovered] = useState(false)
  const isUser = role === 'user'

  const copy = () => {
    navigator.clipboard.writeText(content)
    setCopied(true)
    setTimeout(() => setCopied(false), 2000)
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
            : isStreaming
              ? <><MarkdownContent text={content} /><span className="cursor" /></>
              : <>
                  {parseContent(content).map((block, i) => {
                    if (block.kind === 'image-b64') {
                      return (
                        <img
                          key={i}
                          src={`data:image/png;base64,${block.data}`}
                          alt="Grafana screenshot"
                          style={{ maxWidth: '100%', borderRadius: '8px', margin: '8px 0', display: 'block' }}
                        />
                      )
                    }
                    if (block.kind === 'image-url') {
                      return (
                        <img
                          key={i}
                          src={block.url}
                          alt="Chart"
                          style={{ maxWidth: '100%', borderRadius: '8px', margin: '8px 0', display: 'block' }}
                        />
                      )
                    }
                    return <MarkdownContent key={i} text={block.text} />
                  })}
                </>
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
