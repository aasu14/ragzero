import { useEffect, useState } from 'react'
import { api } from '../api'

function ProviderForm({ kind, providers, current, onSave }) {
  const [selected, setSelected] = useState(current?.provider || providers[0]?.id || '')
  const [settings, setSettings] = useState({})
  const [saving, setSaving] = useState(false)

  const spec = providers.find((p) => p.id === selected)

  useEffect(() => {
    if (!spec) return
    const init = {}
    for (const f of spec.fields) init[f.key] = f.default || ''
    // If this provider matches the current one, preserve any non-secret values
    if (current?.provider === selected) {
      for (const [k, v] of Object.entries(current.settings || {})) {
        if (v && !String(v).startsWith('••')) init[k] = v
      }
    }
    setSettings(init)
  }, [selected, current?.provider])

  const update = (key, val) => setSettings((s) => ({ ...s, [key]: val }))

  const handleSave = async () => {
    setSaving(true)
    try {
      if (kind === 'llm') await api.setLLM(selected, settings)
      else if (kind === 'embedder') await api.setEmbedder(selected, settings)
      else await api.setVectorStore(selected, settings)
      onSave()
    } finally {
      setSaving(false)
    }
  }

  const kindLabel = {
    llm: 'Language model',
    embedder: 'Embedding model',
    vector_store: 'Vector store / index',
  }[kind] || kind

  return (
    <div className="card">
      <div className="card-title">
        <span>{kindLabel}</span>
        <span className="meta">
          current: <span className="mono">{current?.provider || '—'}</span>
        </span>
      </div>

      <div className="field">
        <label>Provider</label>
        <select value={selected} onChange={(e) => setSelected(e.target.value)}>
          {providers.map((p) => (
            <option key={p.id} value={p.id} disabled={!p.installed}>
              {p.label} {!p.installed ? `(install: pip install ${p.requires_package})` : ''}
            </option>
          ))}
        </select>
        {spec?.description && (
          <div className="faint" style={{ fontSize: 11, marginTop: 4 }}>{spec.description}</div>
        )}
      </div>

      {spec?.fields.map((f) => (
        <div key={f.key} className="field">
          <label>
            {f.label} {!f.required && <span className="faint">(optional)</span>}
          </label>
          {f.type === 'select' ? (
            <select value={settings[f.key] || ''} onChange={(e) => update(f.key, e.target.value)}>
              {f.options.map((o) => (
                <option key={o} value={o}>{o}</option>
              ))}
            </select>
          ) : f.type === 'text_with_suggestions' ? (
            <>
              <input
                list={`${spec.id}-${f.key}-suggestions`}
                type="text"
                value={settings[f.key] || ''}
                onChange={(e) => update(f.key, e.target.value)}
                placeholder={f.placeholder || ''}
                autoComplete="off"
                spellCheck={false}
              />
              <datalist id={`${spec.id}-${f.key}-suggestions`}>
                {f.options.map((o) => (
                  <option key={o} value={o} />
                ))}
              </datalist>
              {f.options.length > 0 && (
                <div style={{
                  display: 'flex',
                  flexWrap: 'wrap',
                  gap: 4,
                  marginTop: 6,
                }}>
                  {f.options.slice(0, 8).map((o) => (
                    <span
                      key={o}
                      onClick={() => update(f.key, o)}
                      style={{
                        fontSize: 11,
                        padding: '2px 8px',
                        borderRadius: 10,
                        background: settings[f.key] === o ? 'var(--accent-dim)' : 'var(--bg-elev-2)',
                        border: `1px solid ${settings[f.key] === o ? 'var(--accent)' : 'var(--border)'}`,
                        color: settings[f.key] === o ? 'var(--accent)' : 'var(--text-dim)',
                        cursor: 'pointer',
                        fontFamily: 'var(--mono)',
                      }}
                    >
                      {o}
                    </span>
                  ))}
                  {f.options.length > 8 && (
                    <span style={{ fontSize: 11, color: 'var(--text-faint)' }}>
                      +{f.options.length - 8} more suggestions
                    </span>
                  )}
                </div>
              )}
            </>
          ) : (
            <input
              type={f.type === 'password' ? 'password' : 'text'}
              value={settings[f.key] || ''}
              onChange={(e) => update(f.key, e.target.value)}
              placeholder={f.placeholder || ''}
            />
          )}
          {f.help && <div className="faint" style={{ fontSize: 11, marginTop: 4 }}>{f.help}</div>}
        </div>
      ))}

      <div className="row">
        <button className="primary" onClick={handleSave} disabled={saving}>
          {saving ? 'Saving...' : 'Save'}
        </button>
      </div>
    </div>
  )
}

export default function ProvidersPage({ session, providers, onChange, notify }) {
  const [testing, setTesting] = useState(false)
  const [testResult, setTestResult] = useState(null)

  const runTest = async () => {
    setTesting(true)
    setTestResult(null)
    try {
      const r = await api.testConnections()
      setTestResult(r)
      if (r.llm?.ok && r.embedder?.ok) notify('ok', 'Both connections OK')
      else notify('err', 'One or more connections failed — see details')
    } catch (e) {
      notify('err', e.message)
    } finally {
      setTesting(false)
    }
  }

  if (!session || !providers) return <div className="empty">Loading...</div>

  return (
    <>
      <div className="page-header">
        <h1>Providers & credentials</h1>
        <div className="subtitle">
          Pick your LLM and embedding providers. API keys are kept only in this
          session and never written to disk.
        </div>
      </div>

      <div className="grid-2">
        <ProviderForm
          kind="llm"
          providers={providers.llm}
          current={session.llm}
          onSave={() => {
            onChange()
            notify('ok', 'LLM updated')
          }}
        />
        <ProviderForm
          kind="embedder"
          providers={providers.embedder}
          current={session.embedder}
          onSave={() => {
            onChange()
            notify('ok', 'Embedder updated — index will rebuild on next query')
          }}
        />
      </div>

      {providers.vector_store && providers.vector_store.length > 0 && (
        <ProviderForm
          kind="vector_store"
          providers={providers.vector_store}
          current={session.vector_store}
          onSave={() => {
            onChange()
            notify('ok', 'Vector store updated — index will rebuild on next query')
          }}
        />
      )}

      <div className="card">
        <div className="card-title">
          <span>Connection test</span>
          <button onClick={runTest} disabled={testing}>
            {testing ? 'Testing...' : 'Test connections'}
          </button>
        </div>
        {testResult && (
          <div className="row" style={{ gap: 24 }}>
            <div>
              <div className="dim" style={{ fontSize: 12, marginBottom: 4 }}>Embedder</div>
              {testResult.embedder?.ok ? (
                <span className="status ok"><span className="indicator" /> ok · dim {testResult.embedder.dim}</span>
              ) : (
                <span className="status err"><span className="indicator" /> {testResult.embedder?.error}</span>
              )}
            </div>
            <div>
              <div className="dim" style={{ fontSize: 12, marginBottom: 4 }}>LLM</div>
              {testResult.llm?.ok ? (
                <span className="status ok"><span className="indicator" /> ok · reply: {JSON.stringify(testResult.llm.response)}</span>
              ) : (
                <span className="status err"><span className="indicator" /> {testResult.llm?.error}</span>
              )}
            </div>
          </div>
        )}
        {!testResult && (
          <div className="faint" style={{ fontSize: 12 }}>
            Tests will run a 1-token generation and a 1-vector embedding to verify the
            credentials are valid.
          </div>
        )}
      </div>
    </>
  )
}
