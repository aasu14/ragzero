import { useEffect, useState, useRef } from 'react'
import { api } from '../api'

const MODE_INFO = {
  simple: {
    label: 'Simple',
    description: 'Standard retrieval → constrained generation',
    color: '#6ea8fe',
  },
  graph: {
    label: 'Graph',
    description: 'Vector finds meaning · graph finds connections',
    color: '#a78bfa',
  },
  agentic: {
    label: 'Agentic',
    description: 'Plans, searches, reflects, then answers',
    color: '#fb923c',
  },
  multilingual: {
    label: 'Multilingual',
    description: 'Ask in any language, answer in any language',
    color: '#4ade80',
  },
}

function ModeSelector({ mode, multilingual, onChange }) {
  return (
    <div className="card" style={{ padding: 12 }}>
      <div className="row" style={{ marginBottom: 0, gap: 8 }}>
        {Object.entries(MODE_INFO).map(([id, info]) => {
          const active = (id === 'multilingual' && multilingual) ||
                         (id !== 'multilingual' && mode === id)
          return (
            <div
              key={id}
              onClick={() => onChange(id)}
              style={{
                flex: 1,
                cursor: 'pointer',
                padding: '8px 12px',
                borderRadius: 6,
                border: `1px solid ${active ? info.color : 'var(--border)'}`,
                background: active ? `${info.color}15` : 'var(--bg-elev-2)',
                transition: 'all .12s',
              }}
            >
              <div style={{
                fontSize: 13,
                fontWeight: 500,
                color: active ? info.color : 'var(--text)',
                marginBottom: 2,
              }}>
                {info.label}
              </div>
              <div className="faint" style={{ fontSize: 10, lineHeight: 1.3 }}>
                {info.description}
              </div>
            </div>
          )
        })}
      </div>
    </div>
  )
}

const EVENT_KIND_STYLE = {
  info: { icon: 'i', color: 'var(--text-faint)' },
  thought: { icon: '◆', color: '#fb923c' },
  tool_call: { icon: '→', color: '#6ea8fe' },
  tool_result: { icon: '←', color: '#6ea8fe' },
  retrieval: { icon: '⌕', color: '#a78bfa' },
  translation: { icon: '⇄', color: '#4ade80' },
  graph_hop: { icon: '⤍', color: '#a78bfa' },
  generation: { icon: '✎', color: '#facc15' },
  final: { icon: '✓', color: 'var(--success)' },
  error: { icon: '!', color: 'var(--danger)' },
}

function LiveTrace({ events, isStreaming }) {
  const containerRef = useRef()
  useEffect(() => {
    if (containerRef.current) {
      containerRef.current.scrollTop = containerRef.current.scrollHeight
    }
  }, [events])

  if (!events || events.length === 0) {
    return <div className="empty">No events yet.</div>
  }

  return (
    <div className="trace" ref={containerRef} style={{ maxHeight: 380 }}>
      {events.map((e, i) => {
        const style = EVENT_KIND_STYLE[e.kind] || EVENT_KIND_STYLE.info
        const isLast = i === events.length - 1
        return (
          <div key={i} className="trace-event" style={{
            opacity: isStreaming && isLast ? 0.8 : 1,
          }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
              <span style={{
                color: style.color,
                fontWeight: 600,
                minWidth: 18,
                textAlign: 'center',
              }}>{style.icon}</span>
              <span className="stage" style={{ color: style.color }}>{e.kind}</span>
              <span style={{ color: 'var(--text)', fontSize: 11 }}>{e.label}</span>
            </div>
            {e.data && Object.keys(e.data).length > 0 && e.kind !== 'final' && (
              <pre>{JSON.stringify(e.data, null, 2)}</pre>
            )}
          </div>
        )
      })}
      {isStreaming && (
        <div className="trace-event" style={{
          color: 'var(--text-faint)', fontStyle: 'italic',
        }}>
          <span style={{ marginRight: 8 }}>...</span>processing
        </div>
      )}
    </div>
  )
}

function AnswerView({ events, finalAnswer }) {
  if (!finalAnswer) return null
  const a = finalAnswer
  return (
    <div className="card">
      <div className="card-title">
        <span>Answer</span>
        <span className="meta">
          {a.refused ? (
            <span className="status refused"><span className="indicator" />refused</span>
          ) : (
            <span className="status ok">
              <span className="indicator" />
              confidence {Number(a.confidence).toFixed(3)}
            </span>
          )}
        </span>
      </div>

      {a.refused ? (
        <div className="answer-box refused">
          Refused: {a.refusal_reason || 'insufficient evidence'}
        </div>
      ) : (
        <div className="answer-box">{a.text}</div>
      )}

      {a.citations && a.citations.length > 0 && (
        <div className="citations">
          <div className="dim" style={{ fontSize: 11, marginBottom: 6 }}>
            Citations ({a.citations.length})
          </div>
          {a.citations.map((c) => (
            <div key={c.chunk_id} className="citation">
              <span className="src">{c.source}</span>
              {c.page != null && <span className="faint">page {c.page}</span>}
              <span className="id">{c.chunk_id}</span>
            </div>
          ))}
        </div>
      )}
    </div>
  )
}

export default function QueryPage({ session, onChange, notify }) {
  const [query, setQuery] = useState('')
  const [strategy, setStrategy] = useState(null)
  const [events, setEvents] = useState([])
  const [finalAnswer, setFinalAnswer] = useState(null)
  const [streaming, setStreaming] = useState(false)
  const [outputLang, setOutputLang] = useState('auto')
  const cancelRef = useRef()

  const loadStrategy = async () => {
    try {
      const s = await api.getStrategy()
      setStrategy(s)
      setOutputLang(s.output_language)
    } catch (e) {
      notify('err', e.message)
    }
  }
  useEffect(() => { loadStrategy() }, [])

  const updateMode = async (modeId) => {
    const cfg = {
      mode: modeId === 'multilingual' ? 'simple' : modeId,
      multilingual_enabled: modeId === 'multilingual',
      output_language: outputLang,
      agent_max_iterations: strategy?.agent_max_iterations || 3,
      graph_max_depth: strategy?.graph_max_depth || 2,
    }
    try {
      await api.setStrategy(cfg)
      await loadStrategy()
    } catch (e) {
      notify('err', e.message)
    }
  }

  const updateLanguage = async (lang) => {
    setOutputLang(lang)
    if (!strategy) return
    await api.setStrategy({
      mode: strategy.mode,
      multilingual_enabled: strategy.multilingual_enabled,
      output_language: lang,
      agent_max_iterations: strategy.agent_max_iterations,
      graph_max_depth: strategy.graph_max_depth,
    })
    await loadStrategy()
  }

  const submit = () => {
    if (!query.trim() || streaming) return
    setEvents([])
    setFinalAnswer(null)
    setStreaming(true)
    cancelRef.current = api.queryStream(
      query,
      (ev) => {
        setEvents((prev) => [...prev, ev])
        if (ev.kind === 'final' && ev.data?.answer) {
          setFinalAnswer(ev.data.answer)
        }
      },
      () => {
        setStreaming(false)
        onChange()
      },
      (err) => {
        notify('err', err)
        setStreaming(false)
      },
    )
  }

  const cancel = () => {
    if (cancelRef.current) cancelRef.current()
    setStreaming(false)
  }

  const handleKey = (e) => {
    if (e.key === 'Enter' && (e.metaKey || e.ctrlKey)) submit()
  }

  const isMultilingual = strategy?.multilingual_enabled
  const currentModeId = isMultilingual ? 'multilingual' : strategy?.mode || 'simple'

  return (
    <>
      <div className="page-header">
        <h1>Query</h1>
        <div className="subtitle">
          Ask a question. Press <span className="kbd">Cmd/Ctrl+Enter</span> to submit.
          Agent reasoning streams live below.
        </div>
      </div>

      {strategy && (
        <ModeSelector
          mode={strategy.mode}
          multilingual={isMultilingual}
          onChange={updateMode}
        />
      )}

      {isMultilingual && strategy && (
        <div className="card" style={{ padding: 12 }}>
          <div className="row" style={{ marginBottom: 0 }}>
            <div>
              <label>Output language</label>
              <select value={outputLang} onChange={(e) => updateLanguage(e.target.value)}>
                {Object.entries(strategy.languages).map(([code, name]) => (
                  <option key={code} value={code}>{name} ({code})</option>
                ))}
              </select>
            </div>
            <div style={{ alignSelf: 'center' }}>
              <span className="faint" style={{ fontSize: 12 }}>
                "auto" returns the answer in the same language as the query.
              </span>
            </div>
          </div>
        </div>
      )}

      <div className="card">
        <div className="field">
          <textarea
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            onKeyDown={handleKey}
            rows={3}
            placeholder="Ask anything..."
            disabled={streaming}
          />
        </div>
        <div className="row">
          <div className="faint" style={{ fontSize: 12 }}>
            {session?.n_sources ? `${session.n_sources} sources indexed` : 'No sources — load data first'}
            {currentModeId === 'graph' && ` · graph mode active`}
          </div>
          <div className="actions">
            {streaming && <button onClick={cancel}>Cancel</button>}
            <button
              className="primary"
              onClick={submit}
              disabled={streaming || !query || !session?.n_sources}
            >
              {streaming ? 'Thinking...' : 'Ask'}
            </button>
          </div>
        </div>
      </div>

      {(events.length > 0 || streaming) && (
        <div className="card">
          <div className="card-title">
            <span>Agent activity</span>
            <span className="meta">
              {events.length} event{events.length !== 1 ? 's' : ''}
              {streaming ? ' · live' : ''}
            </span>
          </div>
          <LiveTrace events={events} isStreaming={streaming} />
        </div>
      )}

      <AnswerView events={events} finalAnswer={finalAnswer} />
    </>
  )
}
