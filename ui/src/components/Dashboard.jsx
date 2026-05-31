/* Landing page modeled after H2O Driverless AI's experiment dashboard:
   sparse stat tiles, a circular gauge for pipeline confidence, top-sources
   ranking (the "variable importance" analogue), and a sparkline of recent
   query confidence (the "iteration data" analogue). */
import { useEffect, useState } from 'react'
import { api } from '../api'

export default function Dashboard({ session, providers, onNavigate, onQuickAction, notify }) {
  const [history, setHistory] = useState([])

  useEffect(() => {
    api.history().then((r) => setHistory(r.history || [])).catch(() => {})
  }, [session?.session_id, session?.history_count])

  if (!session) return <div className="empty">Loading session…</div>

  // ---- Pipeline-status derived metrics for the gauge + tiles
  const sources = session.n_sources || 0
  const chunks = session.n_chunks || 0
  const built = !!session.pipeline_built

  // Average confidence across recent non-refused answers
  const confs = history
    .map((h) => h.answer?.confidence)
    .filter((c) => typeof c === 'number')
  const avgConf = confs.length
    ? confs.reduce((a, b) => a + b, 0) / confs.length
    : 0
  const refusedRate = history.length
    ? history.filter((h) => h.answer?.refused).length / history.length
    : 0

  // Provider readiness signal (mock LLM + hash embedder = not configured)
  const usingMock = session.llm?.provider === 'mock'
  const usingHash = session.embedder?.provider === 'hash'
  const realProviders = !usingMock && !usingHash

  // ---- Top sources from history (the "variable importance" analogue)
  const sourceCounts = {}
  for (const h of history) {
    for (const c of h.answer?.citations || []) {
      sourceCounts[c.source] = (sourceCounts[c.source] || 0) + 1
    }
  }
  const topSources = Object.entries(sourceCounts)
    .sort((a, b) => b[1] - a[1])
    .slice(0, 12)
  const maxCount = topSources.length ? topSources[0][1] : 1

  // ---- Confidence sparkline data (last 30)
  const sparkPoints = history.slice(-30).map((h) => h.answer?.confidence || 0)
  const maxSpark = Math.max(0.001, ...sparkPoints)

  return (
    <>
      <div className="page-header">
        <h1>Experiment · session overview</h1>
        <div className="subtitle">
          Live view of the 10-stage hallucination-gated RAG pipeline. Every query passes
          through ingest → chunk → hybrid-retrieve → confidence → constrained-generate
          → citation-gate → fallback → cache → trace.
        </div>
      </div>

      {/* ---- Getting started: a guided 1-2-3 until the session is set up ---- */}
      {!(realProviders && sources > 0) && (
        <div className="card gs">
          <div className="card-title">
            <span>Getting started</span>
            <span className="meta">three steps to your first answer</span>
          </div>
          <div className="steps">
            <Step
              n={1}
              done={realProviders}
              active={!realProviders}
              title="Connect a model"
              desc={realProviders
                ? `${session.llm.provider} · ${session.embedder.provider}`
                : 'Pick an LLM + embedder'}
              onClick={() => onQuickAction('providers')}
            />
            <Step
              n={2}
              done={sources > 0}
              active={realProviders && sources === 0}
              title="Add data"
              desc={sources > 0
                ? `${sources} source${sources === 1 ? '' : 's'} · ${chunks} chunks`
                : 'Upload docs, files or URLs'}
              onClick={() => onQuickAction('data')}
            />
            <Step
              n={3}
              done={history.length > 0}
              active={realProviders && sources > 0 && history.length === 0}
              title="Ask a question"
              desc={history.length > 0 ? `${history.length} asked` : 'Query your knowledge base'}
              onClick={() => onQuickAction('query')}
            />
          </div>
        </div>
      )}

      {/* ---- TOP ROW: stats ---- */}
      <div className="stat-grid">
        <div className="stat-tile">
          <div className="label">Sources</div>
          <div className="value">{sources.toLocaleString()}</div>
        </div>
        <div className="stat-tile">
          <div className="label">Chunks indexed</div>
          <div className="value">{chunks.toLocaleString()}</div>
        </div>
        <div className="stat-tile">
          <div className="label">Queries</div>
          <div className="value">{history.length.toLocaleString()}</div>
        </div>
        <div className="stat-tile">
          <div className="label">Avg confidence</div>
          <div className="value accent">{avgConf ? avgConf.toFixed(3) : '——'}</div>
        </div>
        <div className="stat-tile">
          <div className="label">Refused</div>
          <div className="value">
            {history.length ? (refusedRate * 100).toFixed(0) : '—'}
            <span className="unit">%</span>
          </div>
        </div>
        <div className="stat-tile">
          <div className="label">Pipeline</div>
          <div className={`value small ${built ? 'accent' : ''}`}>
            {built ? 'BUILT' : 'PENDING'}
          </div>
          <div className="sub">{built ? 'ready' : 'builds on first query'}</div>
        </div>
      </div>

      {/* ---- MIDDLE ROW: 3-column ---- */}
      <div className="grid-3">
        {/* Setup / providers card — like H2O's "EXPERIMENT SETUP" */}
        <div className="card">
          <div className="card-title">
            <span>Setup</span>
            <a onClick={() => onNavigate('providers')} style={{ cursor: 'pointer' }}>
              configure ›
            </a>
          </div>
          <KeyVal label="LLM" value={session.llm?.provider} accent={!usingMock} />
          <KeyVal label="Embedder" value={session.embedder?.provider} accent={!usingHash} />
          <KeyVal label="Vector store" value={session.vector_store?.provider} accent />
          <KeyVal
            label="Embedding model"
            value={session.embedder?.settings?.model || '—'}
            mono
          />
          <KeyVal
            label="Index target"
            value={
              session.vector_store?.settings?.collection
              || session.vector_store?.settings?.index_name
              || session.vector_store?.settings?.class_name
              || session.vector_store?.settings?.table
              || '—'
            }
            mono
          />
          {!realProviders && (
            <div style={{
              marginTop: 12, padding: 8,
              borderLeft: '2px solid var(--warn)',
              fontFamily: 'var(--mono)', fontSize: 10,
              color: 'var(--warn)', letterSpacing: '0.08em',
            }}>
              MOCK PROVIDERS ACTIVE — answers will not use a real LLM.
              Configure providers to run for real.
            </div>
          )}
        </div>

        {/* Centre: confidence gauge */}
        <div className="card" style={{ display: 'flex', flexDirection: 'column' }}>
          <div className="card-title">
            <span>Aggregate confidence</span>
            <span className="meta">{history.length} queries</span>
          </div>
          <div style={{ display: 'flex', justifyContent: 'center', flex: 1, alignItems: 'center' }}>
            <Gauge value={avgConf} max={1} label={avgConf ? avgConf.toFixed(3) : '——'} unit="AVG" />
          </div>
          <div style={{ display: 'flex', justifyContent: 'space-between', marginTop: 12 }}>
            <Spark values={sparkPoints} max={maxSpark} />
          </div>
          <div className="faint" style={{
            fontFamily: 'var(--mono)', fontSize: 10,
            letterSpacing: '0.1em', marginTop: 6, textAlign: 'center',
          }}>
            CONFIDENCE OVER LAST {sparkPoints.length} QUERIES
          </div>
        </div>

        {/* Right: training/index settings — H2O's "TRAINING SETTINGS" analogue */}
        <div className="card">
          <div className="card-title">
            <span>Index settings</span>
            <a onClick={() => onNavigate('config')} style={{ cursor: 'pointer' }}>
              tune ›
            </a>
          </div>
          <KeyVal label="Hybrid mode" value="dense + bm25" mono />
          <KeyVal label="Fusion" value="reciprocal-rank" mono />
          <KeyVal label="Citation gate" value="enabled" mono accent />
          <KeyVal label="Fallback gate" value="enabled" mono accent />
          <KeyVal label="Cache" value="in-memory" mono />
        </div>
      </div>

      {/* ---- BOTTOM ROW: rankings + recent history ---- */}
      <div className="grid-2">
        <div className="card">
          <div className="card-title">
            <span>Most-cited sources</span>
            <span className="meta">across {history.length} answers</span>
          </div>
          {topSources.length === 0 ? (
            <div className="empty">No queries yet · ask something to populate.</div>
          ) : (
            <div className="rank-list">
              {topSources.map(([src, n]) => (
                <div key={src} className="rank-row">
                  <div className="bar-track">
                    <div className="bar-fill" style={{ width: `${(n / maxCount) * 100}%` }} />
                  </div>
                  <div className="val">{n.toString().padStart(2, '0')}</div>
                  <div className="label" style={{ gridColumn: '1 / -1', marginTop: 2 }}>{src}</div>
                </div>
              ))}
            </div>
          )}
        </div>

        <div className="card">
          <div className="card-title">
            <span>Recent queries</span>
            <a onClick={() => onNavigate('history')} style={{ cursor: 'pointer' }}>
              all ›
            </a>
          </div>
          {history.length === 0 ? (
            <div className="empty">No queries yet.</div>
          ) : (
            <div className="source-list">
              {history.slice(-6).reverse().map((h, i) => (
                <div key={i} className="source-item">
                  <div style={{ minWidth: 0, flex: 1 }}>
                    <div className="name" style={{ overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                      {h.query}
                    </div>
                    <div className="meta">
                      {h.mode || 'simple'} · {h.answer?.citations?.length || 0} citations
                    </div>
                  </div>
                  <div>
                    {h.answer?.refused ? (
                      <span className="status refused">
                        <span className="indicator" />refused
                      </span>
                    ) : (
                      <span className="status ok">
                        <span className="indicator" />
                        {h.answer?.confidence != null ? h.answer.confidence.toFixed(2) : '—'}
                      </span>
                    )}
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
      </div>
    </>
  )
}


function Step({ n, done, active, title, desc, onClick }) {
  return (
    <div
      className={`step ${done ? 'done' : ''} ${active ? 'active' : ''}`}
      onClick={onClick}
      role="button"
      tabIndex={0}
      onKeyDown={(e) => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); onClick() } }}
    >
      <div className="badge">{done ? '✓' : n}</div>
      <div style={{ minWidth: 0 }}>
        <div className="t">{title}</div>
        <div className="d">{desc}</div>
      </div>
      {active && <span className="arrow">›</span>}
    </div>
  )
}


function KeyVal({ label, value, mono, accent }) {
  return (
    <div style={{ display: 'flex', justifyContent: 'space-between', padding: '4px 0', gap: 8 }}>
      <span style={{
        fontSize: 10, letterSpacing: '0.12em', textTransform: 'uppercase',
        color: 'var(--text-faint)',
      }}>{label}</span>
      <span style={{
        fontFamily: mono ? 'var(--mono)' : undefined,
        fontSize: 11,
        color: accent ? 'var(--accent)' : 'var(--text)',
      }}>
        {value || '—'}
      </span>
    </div>
  )
}


function Gauge({ value, max = 1, label, unit }) {
  // SVG ring gauge — minimal H2O-style dial.
  const radius = 52
  const stroke = 6
  const circumference = 2 * Math.PI * radius
  const pct = Math.max(0, Math.min(1, value / max))
  const dashoffset = circumference * (1 - pct)
  // Tick marks like the H2O gauge — 32 segments around the ring.
  const ticks = Array.from({ length: 32 }, (_, i) => i)
  return (
    <div className="gauge">
      <svg width="124" height="124" viewBox="-62 -62 124 124">
        {/* tick ring */}
        {ticks.map((i) => {
          const a = (i / ticks.length) * 2 * Math.PI
          const r1 = radius + 4
          const r2 = radius + 8
          const x1 = Math.cos(a) * r1
          const y1 = Math.sin(a) * r1
          const x2 = Math.cos(a) * r2
          const y2 = Math.sin(a) * r2
          const lit = i / ticks.length < pct
          return (
            <line
              key={i} x1={x1} y1={y1} x2={x2} y2={y2}
              stroke={lit ? 'var(--accent)' : 'var(--border-hi)'}
              strokeWidth="1.4"
            />
          )
        })}
        <circle r={radius} className="gauge-track" strokeWidth={stroke} />
        <circle
          r={radius}
          className="gauge-fill"
          strokeWidth={stroke}
          strokeDasharray={circumference}
          strokeDashoffset={dashoffset}
          strokeLinecap="butt"
        />
      </svg>
      <div className="gauge-label">{label}</div>
      <div className="gauge-unit">{unit}</div>
    </div>
  )
}


function Spark({ values, max }) {
  if (!values || values.length === 0) {
    return <div className="faint mono" style={{ fontSize: 10 }}>no data</div>
  }
  return (
    <div className="spark" style={{ width: '100%' }}>
      {values.map((v, i) => (
        <div
          key={i}
          className={`spark-bar ${v === 0 ? 'dim' : ''}`}
          style={{ height: `${Math.max(2, (v / max) * 28)}px` }}
        />
      ))}
    </div>
  )
}
