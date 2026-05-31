import { useEffect, useRef, useState } from 'react'
import { publicApi } from './publicApi'
import './public.css'

const SendIcon = () => (
  <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"
       strokeLinecap="round" strokeLinejoin="round"><path d="M22 2 11 13" /><path d="M22 2 15 22l-4-9-9-4 20-7z" /></svg>
)

// Typewriter reveal for the answer text (only when the host enables streaming).
function AnswerText({ text, animate }) {
  const [shown, setShown] = useState(animate ? '' : text)
  useEffect(() => {
    if (!animate) { setShown(text); return }
    let i = 0
    setShown('')
    const id = setInterval(() => {
      i += Math.max(1, Math.round(text.length / 240))  // ~3s regardless of length
      setShown(text.slice(0, i))
      if (i >= text.length) clearInterval(id)
    }, 16)
    return () => clearInterval(id)
  }, [text, animate])
  const done = shown.length >= text.length
  return <span className="txt">{shown}{!done && <span className="pub-caret" />}</span>
}

export default function PublicApp() {
  const [meta, setMeta] = useState(undefined)   // undefined=loading, null=error
  const [unlocked, setUnlocked] = useState(false)
  const [code, setCode] = useState('')
  const [mode, setMode] = useState(null)
  const [input, setInput] = useState('')
  const [thread, setThread] = useState([])      // {q, a?, loading, error?}
  const [busy, setBusy] = useState(false)
  const threadEnd = useRef(null)
  const taRef = useRef(null)

  useEffect(() => {
    publicApi.meta()
      .then((m) => { setMeta(m); setMode(m.default_mode); document.title = m.name || 'Assistant' })
      .catch(() => setMeta(null))
  }, [])

  useEffect(() => { threadEnd.current?.scrollIntoView({ behavior: 'smooth' }) }, [thread, busy])

  const accentStyle = meta?.accent ? { '--accent': meta.accent } : undefined

  if (meta === undefined) return <div className="pub-root"><div className="pub-state">Loading…</div></div>
  if (meta === null || !meta.published) {
    return (
      <div className="pub-root">
        <div className="pub-state">This assistant isn’t available right now.</div>
      </div>
    )
  }

  // access gate
  if (meta.access_required && !unlocked) {
    return (
      <div className="pub-root" style={accentStyle}>
        <div className="pub-gate">
          <h2>{meta.name}</h2>
          <p>Enter the access code to continue.</p>
          <input
            type="password" value={code} placeholder="Access code"
            onChange={(e) => setCode(e.target.value)}
            onKeyDown={(e) => { if (e.key === 'Enter' && code) setUnlocked(true) }}
          />
          <button className="primary" style={{ width: '100%' }} disabled={!code} onClick={() => setUnlocked(true)}>
            Enter
          </button>
        </div>
      </div>
    )
  }

  const ask = async (q) => {
    const question = (q ?? input).trim()
    if (!question || busy) return
    setInput('')
    setBusy(true)
    const idx = thread.length
    setThread((t) => [...t, { q: question, loading: true }])
    try {
      const r = await publicApi.query(question, mode, code)
      setThread((t) => t.map((m, i) => i === idx ? { q: question, a: r.answer, animate: meta.streaming } : m))
    } catch (e) {
      if (e.status === 401) { setUnlocked(false); setThread((t) => t.slice(0, idx)) }
      else setThread((t) => t.map((m, i) => i === idx ? { q: question, error: e.message } : m))
    } finally {
      setBusy(false)
      taRef.current?.focus()
    }
  }

  const onKey = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); ask() }
  }

  return (
    <div className="pub-root" style={accentStyle}>
      <div className="pub-wrap">
        <header className="pub-head">
          <div className="pub-badge">{(meta.name || 'A').charAt(0).toUpperCase()}</div>
          <h1 className="pub-name">{meta.name}</h1>
          {meta.welcome && <div className="pub-welcome">{meta.welcome}</div>}

          {thread.length === 0 && meta.suggested?.length > 0 && (
            <div className="pub-suggest">
              {meta.suggested.map((s) => (
                <span key={s} className="pub-chip" onClick={() => ask(s)}>{s}</span>
              ))}
            </div>
          )}
        </header>

        <div className="pub-thread">
          {thread.map((m, i) => (
            <div key={i}>
              <div className="pub-msg q">{m.q}</div>
              {m.loading && <div className="pub-msg a"><span className="pub-thinking">Thinking</span></div>}
              {m.error && <div className="pub-msg a refused"><span className="txt">⚠ {m.error}</span></div>}
              {m.a && (
                <div className={`pub-msg a ${m.a.refused ? 'refused' : ''}`}>
                  <AnswerText
                    text={m.a.refused ? (m.a.refusal_reason || 'I don’t have enough information to answer that.') : m.a.text}
                    animate={!!m.animate && !m.a.refused}
                  />
                  {(!!m.a.citations?.length || m.a.confidence != null) && (
                    <div className="pub-meta-row">
                      {m.a.citations?.map((c, j) => (
                        <span key={j} className="pub-cite"><b>{c.source}</b>{c.page != null ? ` · p${c.page}` : ''}</span>
                      ))}
                      {m.a.confidence != null && (
                        <span className="pub-conf">confidence {Number(m.a.confidence).toFixed(2)}</span>
                      )}
                    </div>
                  )}
                </div>
              )}
            </div>
          ))}
          <div ref={threadEnd} />
        </div>

        {meta.modes?.length > 1 && (
          <div className="pub-modes">
            {meta.modes.map((mo) => (
              <span key={mo} className={`pub-mode ${mode === mo ? 'on' : ''}`} onClick={() => setMode(mo)}>
                {mo}
              </span>
            ))}
          </div>
        )}

        <div className="pub-askbar">
          <textarea
            ref={taRef} rows={1} value={input} placeholder={meta.title || 'Ask a question…'}
            onChange={(e) => setInput(e.target.value)} onKeyDown={onKey} disabled={busy}
          />
          <button className="pub-send" onClick={() => ask()} disabled={busy || !input.trim()} aria-label="Send">
            <SendIcon />
          </button>
        </div>

        <div className="pub-foot">
          {meta.footer ? meta.footer : <>Powered by <a href="https://github.com/" target="_blank" rel="noreferrer">ragzero</a></>}
        </div>
      </div>
    </div>
  )
}
