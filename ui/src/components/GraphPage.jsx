import { useEffect, useRef, useState } from 'react'
import { api } from '../api'
import JobProgress from './JobProgress'

/**
 * Lightweight SVG graph viz. Force-directed layout computed in JS — no D3
 * dependency. For graphs > a few hundred nodes, swap in cytoscape or vis.js.
 */
function GraphViz({ nodes, edges, width = 720, height = 420 }) {
  const [positions, setPositions] = useState({})
  const animRef = useRef()

  useEffect(() => {
    if (!nodes || nodes.length === 0) {
      setPositions({})
      return
    }
    // Initialize positions in a circle
    const init = {}
    const cx = width / 2
    const cy = height / 2
    const r = Math.min(width, height) / 3
    nodes.forEach((n, i) => {
      const angle = (2 * Math.PI * i) / nodes.length
      init[n.id] = {
        x: cx + r * Math.cos(angle),
        y: cy + r * Math.sin(angle),
        vx: 0, vy: 0,
      }
    })

    // Iterations of a simple force model
    let iterations = 0
    const maxIter = 150

    const step = () => {
      const pos = { ...init }
      // Repulsion between all nodes
      const idsList = Object.keys(pos)
      for (let i = 0; i < idsList.length; i++) {
        for (let j = i + 1; j < idsList.length; j++) {
          const a = pos[idsList[i]]
          const b = pos[idsList[j]]
          const dx = b.x - a.x
          const dy = b.y - a.y
          const dist2 = dx * dx + dy * dy + 0.01
          const force = 800 / dist2
          const fx = (dx / Math.sqrt(dist2)) * force
          const fy = (dy / Math.sqrt(dist2)) * force
          a.vx -= fx; a.vy -= fy
          b.vx += fx; b.vy += fy
        }
      }
      // Spring attraction along edges
      edges.forEach((e) => {
        const a = pos[e.source]
        const b = pos[e.target]
        if (!a || !b) return
        const dx = b.x - a.x
        const dy = b.y - a.y
        const dist = Math.sqrt(dx * dx + dy * dy) + 0.01
        const force = (dist - 100) * 0.02
        const fx = (dx / dist) * force
        const fy = (dy / dist) * force
        a.vx += fx; a.vy += fy
        b.vx -= fx; b.vy -= fy
      })
      // Apply velocities with damping; clamp to canvas
      Object.values(pos).forEach((p) => {
        p.vx *= 0.85; p.vy *= 0.85
        p.x = Math.max(20, Math.min(width - 20, p.x + p.vx))
        p.y = Math.max(20, Math.min(height - 20, p.y + p.vy))
      })
      Object.assign(init, pos)
      iterations++
      if (iterations < maxIter) {
        animRef.current = requestAnimationFrame(step)
      }
      setPositions({ ...pos })
    }
    animRef.current = requestAnimationFrame(step)
    return () => cancelAnimationFrame(animRef.current)
  }, [nodes, edges, width, height])

  if (!nodes || nodes.length === 0) {
    return (
      <div className="empty">Graph is empty. Build it from the controls above.</div>
    )
  }

  const typeColor = (t) => {
    const colors = {
      PERSON: '#fb923c', ORG: '#6ea8fe', LOCATION: '#4ade80',
      CONCEPT: '#a78bfa', PRODUCT: '#facc15', EVENT: '#f472b6',
      OTHER: '#9aa0b0', UNKNOWN: '#6b7080',
    }
    return colors[t] || '#9aa0b0'
  }

  return (
    <svg width="100%" viewBox={`0 0 ${width} ${height}`} style={{
      background: 'var(--bg)',
      border: '1px solid var(--border)',
      borderRadius: 6,
    }}>
      <defs>
        <marker id="arrowhead" viewBox="0 0 10 10" refX="9" refY="5"
                markerWidth="6" markerHeight="6" orient="auto">
          <path d="M 0 0 L 10 5 L 0 10 z" fill="var(--text-faint)" />
        </marker>
      </defs>
      {edges.map((e, i) => {
        const s = positions[e.source]
        const t = positions[e.target]
        if (!s || !t) return null
        return (
          <g key={`e${i}`}>
            <line x1={s.x} y1={s.y} x2={t.x} y2={t.y}
                  stroke="var(--border-hi)" strokeWidth="1"
                  markerEnd="url(#arrowhead)" />
            <text x={(s.x + t.x) / 2} y={(s.y + t.y) / 2 - 3}
                  fill="var(--text-faint)" fontSize="9" textAnchor="middle">
              {e.type}
            </text>
          </g>
        )
      })}
      {nodes.map((n) => {
        const p = positions[n.id]
        if (!p) return null
        return (
          <g key={n.id} transform={`translate(${p.x},${p.y})`}>
            <circle r="14" fill={typeColor(n.type)} opacity="0.8"
                    stroke="var(--bg)" strokeWidth="2" />
            <text y="-20" textAnchor="middle" fill="var(--text)" fontSize="11"
                  fontWeight="500">{n.id}</text>
            <text y="32" textAnchor="middle" fill="var(--text-faint)" fontSize="9">
              {n.type}
            </text>
          </g>
        )
      })}
    </svg>
  )
}

export default function GraphPage({ session, onChange, notify }) {
  const [snapshot, setSnapshot] = useState({ nodes: [], edges: [], stats: { nodes: 0, edges: 0 } })
  const [activeJobId, setActiveJobId] = useState(null)

  const refresh = async () => {
    try {
      const s = await api.graphSnapshot()
      setSnapshot(s)
    } catch (e) {
      notify('err', e.message)
    }
  }

  useEffect(() => { refresh() }, [])

  const build = async () => {
    try {
      const { job_id } = await api.startGraphBuildJob()
      setActiveJobId(job_id)
    } catch (e) {
      notify('err', e.message)
    }
  }

  const handleJobComplete = async (result) => {
    setActiveJobId(null)
    if (result?.summary) notify('ok', result.summary)
    else notify('ok', 'Graph built')
    await refresh()
    onChange()
  }

  const handleJobError = (err) => {
    setActiveJobId(null)
    notify('err', err)
  }

  const clear = async () => {
    if (!confirm('Clear the entire knowledge graph for this session?')) return
    await api.clearGraph()
    notify('ok', 'Graph cleared')
    await refresh()
  }

  return (
    <>
      <div className="page-header">
        <h1>Knowledge graph</h1>
        <div className="subtitle">
          Extract entities and relations from your documents. Used by Graph RAG to traverse
          structured connections in addition to semantic similarity.
        </div>
      </div>

      <div className="card">
        <div className="card-title">
          <span>Controls</span>
          <span className="meta">
            <span className="provider-pill installed">
              {snapshot.stats.nodes} nodes · {snapshot.stats.edges} edges
            </span>
          </span>
        </div>
        <div className="row">
          <div className="faint" style={{ fontSize: 12 }}>
            {!session?.n_sources
              ? 'Index documents first on the Data tab.'
              : `Building extracts entities + relations from all ${session.n_sources} indexed source(s) using the configured LLM. Each chunk is one LLM call — large corpora can take a while.`}
          </div>
          <div className="actions">
            {snapshot.stats.nodes > 0 && !activeJobId && (
              <button className="danger ghost" onClick={clear}>Clear graph</button>
            )}
            <button className="primary" onClick={build} disabled={!!activeJobId || !session?.n_sources}>
              {activeJobId ? 'Building...' : (snapshot.stats.nodes > 0 ? 'Rebuild graph' : 'Build graph')}
            </button>
          </div>
        </div>
        {activeJobId && (
          <div style={{ marginTop: 12 }}>
            <JobProgress
              jobId={activeJobId}
              onComplete={handleJobComplete}
              onError={handleJobError}
            />
          </div>
        )}
      </div>

      <div className="card">
        <div className="card-title">
          <span>Visualization</span>
          <span className="meta">drag-resize coming soon</span>
        </div>
        <GraphViz nodes={snapshot.nodes} edges={snapshot.edges} />
        {snapshot.nodes.length > 50 && (
          <div className="faint" style={{ fontSize: 11, marginTop: 8 }}>
            Showing first {Math.min(500, snapshot.nodes.length)} nodes — large graphs may need a
            dedicated viz library. Consider Cytoscape.js for production use.
          </div>
        )}
      </div>

      {snapshot.edges.length > 0 && (
        <div className="card">
          <div className="card-title">Relations ({snapshot.edges.length})</div>
          <div style={{ maxHeight: 320, overflow: 'auto' }}>
            {snapshot.edges.slice(0, 100).map((e, i) => (
              <div key={i} className="row" style={{ marginBottom: 4 }}>
                <span className="mono" style={{ fontSize: 12 }}>{e.source}</span>
                <span className="faint" style={{ fontSize: 11 }}>--{e.type}--&gt;</span>
                <span className="mono" style={{ fontSize: 12 }}>{e.target}</span>
              </div>
            ))}
          </div>
        </div>
      )}
    </>
  )
}
