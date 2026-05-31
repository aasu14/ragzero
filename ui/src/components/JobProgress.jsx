import { useEffect, useRef, useState } from 'react'
import { api } from '../api'

/**
 * Live progress UI for a single job. Subscribes via SSE on mount, renders
 * a progress bar + current phase + scrolling event log.
 *
 * Props:
 *   jobId      — start streaming this job on mount
 *   onComplete — called with the job's `result` when it finishes
 *   onError    — called with the error message if the job fails
 *   compact    — render in a smaller mode for inline use
 */
export default function JobProgress({ jobId, onComplete, onError, compact = false }) {
  const [snapshot, setSnapshot] = useState({
    status: 'pending', phase: 'queued', current: 0, total: 0,
  })
  const [events, setEvents] = useState([])
  const [cancelling, setCancelling] = useState(false)
  const cancelStreamRef = useRef()
  const logRef = useRef()

  useEffect(() => {
    if (!jobId) return
    cancelStreamRef.current = api.streamJob(
      jobId,
      (event) => {
        // Heartbeats are silent
        if (event.kind === 'ping') return
        setEvents((prev) => {
          const next = [...prev, event]
          return next.length > 200 ? next.slice(-200) : next
        })
        if (event.data) {
          setSnapshot((s) => ({
            ...s,
            phase: event.data.phase || s.phase,
            current: event.data.current ?? s.current,
            total: event.data.total ?? s.total,
          }))
        }
        if (event.kind === 'done') {
          setSnapshot((s) => ({ ...s, status: 'completed' }))
        } else if (event.kind === 'error') {
          setSnapshot((s) => ({ ...s, status: 'failed' }))
        } else if (event.kind === 'terminal') {
          setSnapshot((s) => ({ ...s, status: event.data?.status || 'completed' }))
        } else if (snapshot.status === 'pending') {
          setSnapshot((s) => ({ ...s, status: 'running' }))
        }
      },
      async (terminalEvent) => {
        // Stream finished — fetch the final job state to get the result.
        try {
          const final = await api.getJob(jobId)
          setSnapshot((s) => ({ ...s, ...final }))
          if (final.status === 'completed' && onComplete) {
            onComplete(final.result)
          } else if (final.status === 'failed' && onError) {
            onError(final.error || 'job failed')
          }
        } catch (e) {
          if (onError) onError(String(e))
        }
      },
      (err) => {
        if (onError) onError(err)
      },
    )
    return () => {
      if (cancelStreamRef.current) cancelStreamRef.current()
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [jobId])

  useEffect(() => {
    if (logRef.current) {
      logRef.current.scrollTop = logRef.current.scrollHeight
    }
  }, [events])

  const cancel = async () => {
    setCancelling(true)
    try { await api.cancelJob(jobId) } catch (e) { /* ignore */ }
  }

  const pct = snapshot.total > 0
    ? Math.min(100, Math.round((snapshot.current / snapshot.total) * 100))
    : null

  const statusColor = {
    pending: 'var(--text-faint)',
    running: 'var(--accent)',
    completed: 'var(--success)',
    failed: 'var(--danger)',
    cancelled: 'var(--warn)',
  }[snapshot.status] || 'var(--text-faint)'

  const phaseLabels = {
    queued: 'Queued',
    parsing: 'Parsing files',
    initializing: 'Initializing pipeline',
    chunking: 'Chunking documents',
    embedding: 'Generating embeddings',
    extracting: 'Extracting entities & relations',
    done: 'Done',
    failed: 'Failed',
    running: 'Running',
  }

  return (
    <div style={{
      padding: compact ? 10 : 14,
      background: 'var(--bg-elev-2)',
      border: '1px solid var(--border)',
      borderRadius: 6,
      marginBottom: 10,
    }}>
      <div style={{
        display: 'flex',
        justifyContent: 'space-between',
        alignItems: 'center',
        marginBottom: 8,
      }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
          <span style={{
            width: 8, height: 8, borderRadius: '50%',
            background: statusColor,
            animation: snapshot.status === 'running' ? 'pulse 1.5s ease-in-out infinite' : 'none',
          }} />
          <span style={{ fontSize: 13, fontWeight: 500 }}>
            {phaseLabels[snapshot.phase] || snapshot.phase}
          </span>
          {snapshot.total > 0 && (
            <span className="faint" style={{ fontSize: 11 }}>
              {snapshot.current} / {snapshot.total}
            </span>
          )}
        </div>
        <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
          {pct !== null && (
            <span style={{ fontSize: 13, fontFamily: 'var(--mono)' }}>{pct}%</span>
          )}
          {snapshot.status === 'running' && !cancelling && (
            <button className="ghost" onClick={cancel} style={{ padding: '2px 10px', fontSize: 11 }}>
              Cancel
            </button>
          )}
        </div>
      </div>

      {/* Progress bar */}
      <div style={{
        height: 6,
        background: 'var(--bg)',
        borderRadius: 3,
        overflow: 'hidden',
        position: 'relative',
      }}>
        {pct !== null ? (
          <div style={{
            height: '100%',
            width: `${pct}%`,
            background: statusColor,
            transition: 'width .2s ease',
          }} />
        ) : (
          // Indeterminate: a sliding bar
          <div style={{
            height: '100%',
            width: '30%',
            background: statusColor,
            position: 'absolute',
            animation: snapshot.status === 'running' ? 'slide 1.5s ease-in-out infinite' : 'none',
          }} />
        )}
      </div>

      {/* Event log (collapsed in compact mode) */}
      {!compact && events.length > 0 && (
        <div
          ref={logRef}
          style={{
            marginTop: 10,
            maxHeight: 140,
            overflowY: 'auto',
            background: 'var(--bg)',
            borderRadius: 4,
            padding: 8,
            fontFamily: 'var(--mono)',
            fontSize: 11,
            color: 'var(--text-dim)',
            border: '1px solid var(--border)',
          }}
        >
          {events.slice(-50).map((e, i) => (
            <div key={i} style={{
              padding: '1px 0',
              color: e.kind === 'error' ? 'var(--danger)' :
                     e.kind === 'log' ? 'var(--warn)' :
                     e.kind === 'phase' ? 'var(--accent)' :
                     'var(--text-dim)',
            }}>
              {e.message}
            </div>
          ))}
        </div>
      )}

      {snapshot.status === 'failed' && snapshot.error && (
        <div style={{
          marginTop: 8,
          padding: 8,
          background: 'rgba(248, 113, 113, .08)',
          border: '1px solid var(--danger)',
          borderRadius: 4,
          color: 'var(--danger)',
          fontSize: 12,
        }}>
          {snapshot.error}
        </div>
      )}

      <style>{`
        @keyframes pulse {
          0%, 100% { opacity: 1; transform: scale(1); }
          50% { opacity: .5; transform: scale(1.3); }
        }
        @keyframes slide {
          0% { left: -30%; }
          100% { left: 100%; }
        }
      `}</style>
    </div>
  )
}
