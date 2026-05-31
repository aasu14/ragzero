import { useEffect, useState } from 'react'
import { api } from '../api'

export default function HistoryPage({ notify }) {
  const [history, setHistory] = useState([])
  const [selected, setSelected] = useState(null)

  const refresh = async () => {
    try {
      const r = await api.history()
      setHistory(r.history.slice().reverse())
    } catch (e) {
      notify('err', e.message)
    }
  }
  useEffect(() => { refresh() }, [])

  return (
    <>
      <div className="page-header">
        <h1>Query history</h1>
        <div className="subtitle">
          Recent queries this session. Click one to inspect its trace.
        </div>
      </div>

      <div className="grid-2">
        <div>
          {history.length === 0 ? (
            <div className="empty">No queries yet.</div>
          ) : (
            history.map((h, i) => (
              <div
                key={i}
                className="history-item"
                onClick={() => setSelected(h)}
                style={selected === h ? { borderColor: 'var(--accent)' } : {}}
              >
                <div className="q">{h.query}</div>
                <div className="a">
                  {h.answer.refused
                    ? <span style={{ color: 'var(--refused)' }}>[refused] {h.answer.refusal_reason}</span>
                    : h.answer.text || '(empty)'}
                </div>
                <div className="faint mono" style={{ fontSize: 10, marginTop: 4 }}>
                  {h.timestamp.split('T')[1]?.split('.')[0]} · conf {h.answer.confidence.toFixed(2)}
                </div>
              </div>
            ))
          )}
        </div>

        <div>
          {selected ? (
            <div className="card">
              <div className="card-title">
                <span>Trace</span>
                <span className="mono meta">{selected.trace_id}</span>
              </div>
              <div className="trace">
                {(selected.events || []).map((e, i) => (
                  <div key={i} className="trace-event">
                    <div>
                      <span className="ts">[{e.timestamp.split('T')[1]?.split('.')[0]}]</span>{' '}
                      <span className="stage">{e.stage}</span>
                    </div>
                    <pre>{JSON.stringify(e.data, null, 2)}</pre>
                  </div>
                ))}
              </div>
            </div>
          ) : (
            <div className="empty">Select a query to view its trace.</div>
          )}
        </div>
      </div>
    </>
  )
}
