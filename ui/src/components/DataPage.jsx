import { useEffect, useRef, useState } from 'react'
import { api } from '../api'
import JobProgress from './JobProgress'

// Extensions handled by the per-row structured ingest path. Anything else
// goes through the regular chunking pipeline.
const STRUCTURED_EXTS = new Set(['.csv', '.tsv', '.json', '.jsonl', '.ndjson'])

function isStructured(filename) {
  const ext = filename.slice(filename.lastIndexOf('.')).toLowerCase()
  return STRUCTURED_EXTS.has(ext)
}

export default function DataPage({ session, onChange, notify }) {
  const [tab, setTab] = useState('upload')
  const [sources, setSources] = useState([])
  const [activeJob, setActiveJob] = useState(null)
  const fileRef = useRef()
  const [dragging, setDragging] = useState(false)
  const [pathInput, setPathInput] = useState('')
  const [urlInput, setUrlInput] = useState('')
  const [textTitle, setTextTitle] = useState('')
  const [textContent, setTextContent] = useState('')
  const [selectedDocId, setSelectedDocId] = useState(null)
  const [detail, setDetail] = useState(null)
  const [detailLoading, setDetailLoading] = useState(false)
  const [structuredBusy, setStructuredBusy] = useState(null)  // null | {current, total, label}

  const refresh = async () => {
    try {
      const r = await api.listSources()
      setSources(r.sources)
    } catch (e) {
      notify('err', e.message)
    }
  }
  useEffect(() => { refresh() }, [session?.session_id, session?.n_sources])

  useEffect(() => {
    if (!selectedDocId) { setDetail(null); return }
    let cancelled = false
    setDetailLoading(true)
    api.sourceChunks(selectedDocId)
      .then((r) => { if (!cancelled) setDetail(r) })
      .catch((e) => {
        if (!cancelled) {
          notify('err', `Couldn't load source: ${e.message}`)
          setSelectedDocId(null)
        }
      })
      .finally(() => { if (!cancelled) setDetailLoading(false) })
    return () => { cancelled = true }
  }, [selectedDocId])

  const handleJobComplete = async (result) => {
    setActiveJob(null)
    if (result?.summary) {
      let msg = result.summary
      if (result.skipped?.length) msg += ` (skipped ${result.skipped.length})`
      notify('ok', msg)
    } else {
      notify('ok', 'Done')
    }
    await refresh()
    onChange()
  }

  const handleJobError = (err) => {
    setActiveJob(null)
    notify('err', err)
  }

  // ---- Structured (CSV / JSON / JSONL): preview, then auto-commit with
  // sensible defaults. No schema-review step is forced on the user; for
  // fine-grained control they can hit /api/ingest/structured/* directly.
  // Returns {added, error?} so the batch driver can aggregate.
  const ingestStructuredOne = async (file) => {
    try {
      const pv = await api.structuredPreview(file)
      // Default content columns: string columns if any, otherwise all columns
      // (so we never end up with zero-content rows).
      const stringCols = pv.columns.filter((c) => c.type === 'string').map((c) => c.name)
      const contentColumns = stringCols.length ? stringCols : pv.columns.map((c) => c.name)
      const fields = pv.columns.map((c) => ({
        name: c.name,
        type: c.type,
        retrievable: true,
        filterable: true,
        sortable: c.type === 'int' || c.type === 'float' || c.type === 'date',
        facetable: false,
        searchable: c.type === 'string',
      }))
      const r = await api.structuredCommit(pv.preview_id, fields, contentColumns)
      return { added: r.added, file: file.name }
    } catch (e) {
      return { error: e.message, file: file.name }
    }
  }

  // Drive a batch of structured files sequentially so progress is observable
  // and we don't blast the embedder with parallel commits.
  const ingestStructuredBatch = async (files) => {
    const results = []
    for (let i = 0; i < files.length; i++) {
      setStructuredBusy({ current: i + 1, total: files.length, label: files[i].name })
      results.push(await ingestStructuredOne(files[i]))
    }
    setStructuredBusy(null)
    const ok = results.filter((r) => !r.error)
    const failed = results.filter((r) => r.error)
    const totalRows = ok.reduce((sum, r) => sum + (r.added || 0), 0)
    if (failed.length === 0) {
      notify('ok', `Indexed ${totalRows} row(s) from ${ok.length} file${ok.length === 1 ? '' : 's'}`)
    } else if (ok.length === 0) {
      notify('err', `All ${failed.length} structured file(s) failed. First: ${failed[0].file}: ${failed[0].error}`)
    } else {
      notify('err',
        `${ok.length} file(s) indexed (${totalRows} rows); ${failed.length} failed. ` +
        `First failure: ${failed[0].file}: ${failed[0].error}`)
    }
    await refresh()
    onChange()
  }

  const startUpload = async (fileList) => {
    if (!fileList || fileList.length === 0) return
    const files = Array.from(fileList)
    // Route structured files (CSV / JSON / JSONL) through the per-row pipeline,
    // others through the chunked-document pipeline. Both paths run a batch so
    // multi-file uploads are first-class.
    const structuredFiles = files.filter((f) => isStructured(f.name))
    const docFiles = files.filter((f) => !isStructured(f.name))

    // Run document-file job first (background, with its own progress card),
    // then process structured files sequentially. Both can be in flight.
    if (docFiles.length > 0) {
      try {
        const { job_id } = await api.startUploadJob(docFiles)
        setActiveJob({
          jobId: job_id,
          kind: `upload · ${docFiles.length} file${docFiles.length === 1 ? '' : 's'}`,
        })
      } catch (e) {
        notify('err', e.message)
      }
    }
    if (structuredFiles.length > 0) {
      await ingestStructuredBatch(structuredFiles)
    }
  }

  const startPath = async () => {
    const items = pathInput.split(/[\n,]+/).map((s) => s.trim()).filter(Boolean)
    if (items.length === 0) return
    try {
      const { job_id, n } = await api.startPathJob(items)
      setActiveJob({ jobId: job_id, kind: `path · ${n} item${n === 1 ? '' : 's'}` })
      setPathInput('')
    } catch (e) {
      notify('err', e.message)
    }
  }

  const startUrl = async () => {
    const items = urlInput.split(/[\n,]+/).map((s) => s.trim()).filter(Boolean)
    if (items.length === 0) return
    try {
      const { job_id, n } = await api.startUrlJob(items)
      setActiveJob({ jobId: job_id, kind: `url · ${n} URL${n === 1 ? '' : 's'}` })
      setUrlInput('')
    } catch (e) {
      notify('err', e.message)
    }
  }

  const pathCount = pathInput.split(/[\n,]+/).map((s) => s.trim()).filter(Boolean).length
  const urlCount = urlInput.split(/[\n,]+/).map((s) => s.trim()).filter(Boolean).length

  const startText = async () => {
    if (!textContent.trim()) return
    try {
      const { job_id } = await api.startTextJob(textTitle, textContent)
      setActiveJob({ jobId: job_id, kind: 'text' })
      setTextTitle('')
      setTextContent('')
    } catch (e) {
      notify('err', e.message)
    }
  }

  const handleClear = async () => {
    if (!confirm('Clear all indexed documents for this session?')) return
    try {
      const r = await api.clearSources()
      let msg = 'Cleared'
      if (r?.cleared_store) msg += ` · purged ${r.cleared_store}`
      if (r?.cleared_store_error) msg += ` (warning: ${r.cleared_store_error})`
      notify('ok', msg)
      await refresh()
      onChange()
    } catch (e) {
      notify('err', e.message)
    }
  }

  const inputsDisabled = !!activeJob || structuredBusy

  if (selectedDocId) {
    return (
      <SourceDetailView
        detail={detail}
        loading={detailLoading}
        onBack={() => setSelectedDocId(null)}
      />
    )
  }

  return (
    <>
      <div className="page-header">
        <h1>Data</h1>
        <div className="subtitle">
          Documents (.txt, .md, .pdf) are chunked. Structured files (.csv, .json, .jsonl)
          are indexed per row — schema is inferred automatically.
        </div>
      </div>

      <IndexStatusCard session={session} />

      {activeJob && (
        <div className="card" style={{ padding: 14 }}>
          <div className="card-title">
            <span>Ingesting · {activeJob.kind}</span>
            <span className="meta mono">{activeJob.jobId}</span>
          </div>
          <JobProgress
            jobId={activeJob.jobId}
            onComplete={handleJobComplete}
            onError={handleJobError}
          />
        </div>
      )}

      {structuredBusy && (
        <div className="card" style={{ padding: 14 }}>
          <div className="card-title">
            <span>
              Indexing structured file {structuredBusy.current} of {structuredBusy.total}
            </span>
            <span className="meta mono">{structuredBusy.label}</span>
          </div>
        </div>
      )}

      <div className="card">
        <div className="tabs">
          <div className={`tab ${tab === 'upload' ? 'active' : ''}`} onClick={() => setTab('upload')}>Upload</div>
          <div className={`tab ${tab === 'path' ? 'active' : ''}`} onClick={() => setTab('path')}>Filesystem path</div>
          <div className={`tab ${tab === 'url' ? 'active' : ''}`} onClick={() => setTab('url')}>URL</div>
          <div className={`tab ${tab === 'text' ? 'active' : ''}`} onClick={() => setTab('text')}>Paste text</div>
        </div>

        {tab === 'upload' && (
          <div>
            <div
              className={`dropzone ${dragging ? 'dragging' : ''}`}
              onClick={() => !inputsDisabled && fileRef.current?.click()}
              onDragOver={(e) => { if (!inputsDisabled) { e.preventDefault(); setDragging(true) } }}
              onDragLeave={() => setDragging(false)}
              onDrop={(e) => {
                e.preventDefault()
                setDragging(false)
                if (!inputsDisabled) startUpload(e.dataTransfer.files)
              }}
              style={{ opacity: inputsDisabled ? 0.5 : 1, cursor: inputsDisabled ? 'not-allowed' : 'pointer' }}
            >
              <div style={{ fontSize: 14, marginBottom: 6 }}>
                Drop files here or click to browse
              </div>
              <div className="faint" style={{ fontSize: 11 }}>
                .txt / .md / .pdf chunked · .csv / .json / .jsonl indexed per row
              </div>
              <input
                ref={fileRef}
                type="file"
                multiple
                accept=".txt,.md,.pdf,.csv,.tsv,.json,.jsonl,.ndjson"
                style={{ display: 'none' }}
                disabled={inputsDisabled}
                onChange={(e) => startUpload(e.target.files)}
              />
            </div>
          </div>
        )}

        {tab === 'path' && (
          <div>
            <div className="field">
              <label>
                Server-side paths
                {pathCount > 0 && (
                  <span className="faint" style={{ marginLeft: 8, fontSize: 11 }}>
                    {pathCount} item{pathCount === 1 ? '' : 's'}
                  </span>
                )}
              </label>
              <textarea
                value={pathInput}
                onChange={(e) => setPathInput(e.target.value)}
                rows={4}
                placeholder={'One file or directory per line, e.g.\n/home/user/docs\n./data/report.pdf\n./reports/'}
                disabled={inputsDisabled}
                onKeyDown={(e) => {
                  if (e.key === 'Enter' && (e.metaKey || e.ctrlKey)) startPath()
                }}
                style={{ fontFamily: 'var(--mono)' }}
              />
              <div className="faint" style={{ fontSize: 11, marginTop: 4 }}>
                Directories are walked recursively. Cmd/Ctrl+Enter to submit.
              </div>
            </div>
            <button className="primary" onClick={startPath} disabled={inputsDisabled || pathCount === 0}>
              Ingest {pathCount > 0 ? `${pathCount} item${pathCount === 1 ? '' : 's'}` : ''}
            </button>
          </div>
        )}

        {tab === 'url' && (
          <div>
            <div className="field">
              <label>
                URLs
                {urlCount > 0 && (
                  <span className="faint" style={{ marginLeft: 8, fontSize: 11 }}>
                    {urlCount} URL{urlCount === 1 ? '' : 's'}
                  </span>
                )}
              </label>
              <textarea
                value={urlInput}
                onChange={(e) => setUrlInput(e.target.value)}
                rows={4}
                placeholder={'One URL per line, e.g.\nhttps://example.com/article\nhttps://example.com/news'}
                disabled={inputsDisabled}
                onKeyDown={(e) => {
                  if (e.key === 'Enter' && (e.metaKey || e.ctrlKey)) startUrl()
                }}
                style={{ fontFamily: 'var(--mono)' }}
              />
              <div className="faint" style={{ fontSize: 11, marginTop: 4 }}>
                Cmd/Ctrl+Enter to submit.
              </div>
            </div>
            <button className="primary" onClick={startUrl} disabled={inputsDisabled || urlCount === 0}>
              Fetch {urlCount > 0 ? `${urlCount} URL${urlCount === 1 ? '' : 's'}` : ''}
            </button>
          </div>
        )}

        {tab === 'text' && (
          <div>
            <div className="field">
              <label>Title</label>
              <input
                value={textTitle}
                onChange={(e) => setTextTitle(e.target.value)}
                placeholder="optional"
                disabled={inputsDisabled}
              />
            </div>
            <div className="field">
              <label>Content</label>
              <textarea
                value={textContent}
                onChange={(e) => setTextContent(e.target.value)}
                rows={8}
                placeholder="Paste text to index..."
                disabled={inputsDisabled}
              />
            </div>
            <button className="primary" onClick={startText} disabled={inputsDisabled || !textContent}>
              Add
            </button>
          </div>
        )}
      </div>

      <div className="card">
        <div className="card-title">
          <span>Indexed sources ({sources.length})</span>
          {sources.length > 0 && (
            <button className="danger ghost" onClick={handleClear}>Clear all</button>
          )}
        </div>
        {sources.length === 0 ? (
          <div className="empty">No sources indexed yet.</div>
        ) : (
          <div className="source-list">
            {sources.map((s) => (
              <div
                key={s.doc_id}
                className="source-item"
                onClick={() => setSelectedDocId(s.doc_id)}
                role="button"
                tabIndex={0}
                onKeyDown={(e) => { if (e.key === 'Enter' || e.key === ' ') setSelectedDocId(s.doc_id) }}
                style={{ cursor: 'pointer' }}
                title="View chunks + embedding details"
              >
                <div>
                  <div className="name">{s.source}</div>
                  <div className="meta">
                    {s.doc_type} · {s.char_count} chars · v{s.version}
                  </div>
                </div>
                <div className="faint mono">{s.doc_id} ›</div>
              </div>
            ))}
          </div>
        )}
      </div>
    </>
  )
}


function IndexStatusCard({ session }) {
  if (!session) return null
  const vs = session.vector_store || {}
  const em = session.embedder || {}
  return (
    <div className="card" style={{ padding: 14 }}>
      <div className="card-title">
        <span>Index status</span>
        <span className="meta">
          {session.n_sources || 0} source{(session.n_sources || 0) === 1 ? '' : 's'}
          {session.n_chunks != null && ` · ${session.n_chunks} chunk${session.n_chunks === 1 ? '' : 's'}`}
        </span>
      </div>
      <div className="row" style={{ gap: 24, flexWrap: 'wrap' }}>
        <div>
          <div className="dim" style={{ fontSize: 11, marginBottom: 2 }}>Embedding model</div>
          <span className="mono" style={{ fontSize: 13 }}>{em.provider || '—'}</span>
          {em.settings?.model && (
            <span className="faint mono" style={{ fontSize: 11, marginLeft: 6 }}>
              ({em.settings.model})
            </span>
          )}
        </div>
        <div>
          <div className="dim" style={{ fontSize: 11, marginBottom: 2 }}>Vector store</div>
          <span className="mono" style={{ fontSize: 13 }}>{vs.provider || '—'}</span>
          {(vs.settings?.collection
            || vs.settings?.index_name
            || vs.settings?.class_name
            || vs.settings?.table) && (
            <span className="faint mono" style={{ fontSize: 11, marginLeft: 6 }}>
              ({vs.settings.collection
                || vs.settings.index_name
                || vs.settings.class_name
                || vs.settings.table})
            </span>
          )}
        </div>
        <div>
          <div className="dim" style={{ fontSize: 11, marginBottom: 2 }}>Pipeline</div>
          <span className="mono" style={{ fontSize: 13 }}>
            {session.pipeline_built ? 'built' : 'not built (builds on first query)'}
          </span>
        </div>
      </div>
    </div>
  )
}


function SourceDetailView({ detail, loading, onBack }) {
  const [expanded, setExpanded] = useState(new Set())
  const toggle = (id) => {
    setExpanded((prev) => {
      const next = new Set(prev)
      if (next.has(id)) next.delete(id); else next.add(id)
      return next
    })
  }

  const StoreSettingChips = ({ settings }) => {
    const entries = Object.entries(settings || {}).filter(([, v]) => v && !String(v).startsWith('••'))
    if (entries.length === 0) return null
    return (
      <div style={{ display: 'flex', flexWrap: 'wrap', gap: 6, marginTop: 6 }}>
        {entries.map(([k, v]) => (
          <span key={k} style={{
            fontSize: 11, padding: '2px 8px', borderRadius: 10,
            background: 'var(--bg-elev-2)', border: '1px solid var(--border)',
            color: 'var(--text-dim)', fontFamily: 'var(--mono)',
          }}>
            {k}: {String(v)}
          </span>
        ))}
      </div>
    )
  }

  return (
    <>
      <div className="page-header">
        <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
          <button onClick={onBack} className="ghost">← Back to sources</button>
          <h1 style={{ margin: 0 }}>Source detail</h1>
        </div>
        <div className="subtitle">
          How this document is chunked and which embedding model + vector store back its index.
        </div>
      </div>

      {loading && <div className="empty">Loading chunks…</div>}

      {detail && (
        <>
          <div className="card" style={{ padding: 14 }}>
            <div className="card-title">
              <span style={{ wordBreak: 'break-all' }}>{detail.source}</span>
              <span className="meta mono">{detail.doc_id} · v{detail.version}</span>
            </div>
            <div className="row" style={{ gap: 24, flexWrap: 'wrap' }}>
              <div>
                <div className="dim" style={{ fontSize: 11, marginBottom: 2 }}>Type</div>
                <span className="mono" style={{ fontSize: 13 }}>{detail.doc_type}</span>
              </div>
              <div>
                <div className="dim" style={{ fontSize: 11, marginBottom: 2 }}>Document size</div>
                <span className="mono" style={{ fontSize: 13 }}>{detail.char_count.toLocaleString()} chars</span>
              </div>
              <div>
                <div className="dim" style={{ fontSize: 11, marginBottom: 2 }}>Chunks</div>
                <span className="mono" style={{ fontSize: 13 }}>
                  {detail.chunks.length} via {detail.chunker.strategy}
                </span>
                {detail.chunker.settings && Object.keys(detail.chunker.settings).length > 0 && (
                  <span className="faint mono" style={{ fontSize: 11, marginLeft: 6 }}>
                    ({Object.entries(detail.chunker.settings).map(([k, v]) => `${k}=${v}`).join(', ')})
                  </span>
                )}
              </div>
            </div>
          </div>

          <div className="grid-2">
            <div className="card" style={{ padding: 14 }}>
              <div className="card-title"><span>Embedding model</span></div>
              <div>
                <div className="dim" style={{ fontSize: 11, marginBottom: 2 }}>Provider</div>
                <span className="mono" style={{ fontSize: 13 }}>{detail.embedder.provider}</span>
              </div>
              {detail.embedder.model && (
                <div style={{ marginTop: 8 }}>
                  <div className="dim" style={{ fontSize: 11, marginBottom: 2 }}>Model</div>
                  <span className="mono" style={{ fontSize: 13 }}>{detail.embedder.model}</span>
                </div>
              )}
              {detail.embedder.dim != null && (
                <div style={{ marginTop: 8 }}>
                  <div className="dim" style={{ fontSize: 11, marginBottom: 2 }}>Vector dimension</div>
                  <span className="mono" style={{ fontSize: 13 }}>{detail.embedder.dim}</span>
                </div>
              )}
            </div>

            <div className="card" style={{ padding: 14 }}>
              <div className="card-title"><span>Vector store</span></div>
              <div>
                <div className="dim" style={{ fontSize: 11, marginBottom: 2 }}>Provider</div>
                <span className="mono" style={{ fontSize: 13 }}>
                  {detail.vector_store.provider}
                  {detail.vector_store.class_name && detail.vector_store.class_name !== detail.vector_store.provider && (
                    <span className="faint" style={{ marginLeft: 6 }}>
                      ({detail.vector_store.class_name})
                    </span>
                  )}
                </span>
              </div>
              {detail.vector_store.total_chunks_in_store != null && (
                <div style={{ marginTop: 8 }}>
                  <div className="dim" style={{ fontSize: 11, marginBottom: 2 }}>
                    Total chunks in store (all sources)
                  </div>
                  <span className="mono" style={{ fontSize: 13 }}>
                    {detail.vector_store.total_chunks_in_store.toLocaleString()}
                  </span>
                </div>
              )}
              <StoreSettingChips settings={detail.vector_store.settings} />
            </div>
          </div>

          <div className="card">
            <div className="card-title">
              <span>Chunks ({detail.chunks.length})</span>
              <span className="meta">click a row to expand</span>
            </div>
            <div className="source-list">
              {detail.chunks.map((c) => {
                const isOpen = expanded.has(c.chunk_id)
                const preview = c.text.length > 200 ? c.text.slice(0, 200) + '…' : c.text
                return (
                  <div
                    key={c.chunk_id}
                    className="source-item"
                    style={{ flexDirection: 'column', alignItems: 'stretch', cursor: 'pointer' }}
                    onClick={() => toggle(c.chunk_id)}
                  >
                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'baseline' }}>
                      <div className="mono" style={{ fontSize: 12 }}>
                        #{c.position}
                        {c.page != null && <span className="faint"> · page {c.page}</span>}
                        <span className="faint"> · {c.char_count} chars</span>
                      </div>
                      <div className="faint mono" style={{ fontSize: 11 }}>{c.chunk_id}</div>
                    </div>
                    <div style={{
                      marginTop: 6,
                      fontSize: 12,
                      lineHeight: 1.5,
                      whiteSpace: isOpen ? 'pre-wrap' : 'normal',
                      color: 'var(--text-dim)',
                    }}>
                      {isOpen ? c.text : preview}
                    </div>
                  </div>
                )
              })}
            </div>
          </div>
        </>
      )}
    </>
  )
}
