import { useEffect, useState } from 'react'
import { api } from '../api'

function NumField({ label, value, onChange, step = 1, min, max, help }) {
  return (
    <div className="field">
      <label>{label}</label>
      <input
        type="number"
        value={value ?? ''}
        step={step}
        min={min}
        max={max}
        onChange={(e) => onChange(e.target.value === '' ? null : Number(e.target.value))}
      />
      {help && <div className="faint" style={{ fontSize: 11, marginTop: 4 }}>{help}</div>}
    </div>
  )
}

export default function ConfigPage({ session, onChange, notify }) {
  const [config, setConfig] = useState(null)
  const [draft, setDraft] = useState({ pipeline: {}, confidence: {}, fallback: {} })
  const [saving, setSaving] = useState(false)

  const refresh = async () => {
    try {
      const c = await api.getConfig()
      setConfig(c)
      setDraft({
        pipeline: { ...c.pipeline },
        confidence: { ...c.confidence },
        fallback: { ...c.fallback },
      })
    } catch (e) {
      notify('err', e.message)
    }
  }

  useEffect(() => { refresh() }, [])

  const upd = (section, key) => (val) => {
    setDraft((d) => ({ ...d, [section]: { ...d[section], [key]: val } }))
  }

  const save = async () => {
    setSaving(true)
    try {
      await api.setPipelineTuning(draft)
      notify('ok', 'Config updated')
      await refresh()
      onChange()
    } catch (e) {
      notify('err', e.message)
    } finally {
      setSaving(false)
    }
  }

  const resetToDefaults = async () => {
    await api.setPipelineTuning({})
    notify('ok', 'Reset to YAML defaults')
    await refresh()
    onChange()
  }

  if (!config) return <div className="empty">Loading config...</div>

  return (
    <>
      <div className="page-header">
        <h1>Pipeline configuration</h1>
        <div className="subtitle">
          Tune retrieval, confidence, and fallback for your session. Changes apply on the next query.
        </div>
      </div>

      <div className="grid-2">
        <div className="card">
          <div className="card-title">Retrieval & chunking</div>
          <NumField label="Retriever k (per retriever)" value={draft.pipeline.retriever_k}
            onChange={upd('pipeline', 'retriever_k')} min={1} max={500}
            help="How many candidates each retriever fetches before fusion." />
          <NumField label="Final k (after fusion)" value={draft.pipeline.final_k}
            onChange={upd('pipeline', 'final_k')} min={1} max={100} />
          <NumField label="Chunk size (chars)" value={draft.pipeline.chunk_size}
            onChange={upd('pipeline', 'chunk_size')} min={100} max={4000} step={50} />
          <NumField label="Chunk overlap (chars)" value={draft.pipeline.chunk_overlap}
            onChange={upd('pipeline', 'chunk_overlap')} min={0} max={500} step={10} />
          <NumField label="Max context chunks" value={draft.pipeline.max_context_chunks}
            onChange={upd('pipeline', 'max_context_chunks')} min={1} max={50} />
          <NumField label="Max tokens" value={draft.pipeline.max_tokens}
            onChange={upd('pipeline', 'max_tokens')} min={64} max={4096} step={32} />
        </div>

        <div className="card">
          <div className="card-title">Confidence scoring (step 4)</div>
          <NumField label="Half-life days" value={draft.confidence.half_life_days}
            onChange={upd('confidence', 'half_life_days')} min={1} max={3650} step={1}
            help="Older sources lose freshness on this curve." />
          <NumField label="Default source quality" value={draft.confidence.default_source_quality}
            onChange={upd('confidence', 'default_source_quality')} min={0} max={1} step={0.05} />
          <NumField label="Weight: freshness" value={draft.confidence.w_freshness}
            onChange={upd('confidence', 'w_freshness')} min={0} max={1} step={0.05} />
          <NumField label="Weight: source quality" value={draft.confidence.w_source}
            onChange={upd('confidence', 'w_source')} min={0} max={1} step={0.05} />
          <NumField label="Weight: retrieval consistency" value={draft.confidence.w_consistency}
            onChange={upd('confidence', 'w_consistency')} min={0} max={1} step={0.05}
            help="Weights must sum to 1.0." />
          <NumField label="Aggregate top N" value={draft.confidence.aggregate_top_n}
            onChange={upd('confidence', 'aggregate_top_n')} min={1} max={50} />
        </div>

        <div className="card">
          <div className="card-title">Hallucination fallback (step 7)</div>
          <NumField label="Min aggregate confidence" value={draft.fallback.min_aggregate_confidence}
            onChange={upd('fallback', 'min_aggregate_confidence')} min={0} max={1} step={0.05}
            help="Below this, the answer is refused." />
          <NumField label="Min chunks required" value={draft.fallback.min_chunks}
            onChange={upd('fallback', 'min_chunks')} min={0} max={20} />
          <NumField label="Min distinct citations" value={draft.fallback.require_min_citations}
            onChange={upd('fallback', 'require_min_citations')} min={0} max={20} />
          <NumField label="Min top retrieval score" value={draft.fallback.min_top_retrieval_score}
            onChange={upd('fallback', 'min_top_retrieval_score')} min={0} max={10} step={0.01}
            help="0 disables. Tune after measuring score distribution on known-irrelevant queries." />
        </div>

        <div className="card">
          <div className="card-title">Resolved config (read-only)</div>
          <pre className="mono" style={{
            background: 'var(--bg)',
            padding: 12,
            borderRadius: 6,
            border: '1px solid var(--border)',
            maxHeight: 320,
            overflow: 'auto',
            fontSize: 11,
          }}>
            {JSON.stringify(config, null, 2)}
          </pre>
        </div>
      </div>

      <div className="row">
        <button className="primary" onClick={save} disabled={saving}>
          {saving ? 'Saving...' : 'Apply changes'}
        </button>
        <div className="actions">
          <button className="ghost" onClick={resetToDefaults}>Reset to YAML defaults</button>
        </div>
      </div>
    </>
  )
}
