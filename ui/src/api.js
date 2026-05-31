// Centralized API client. Always include credentials (cookies) so the
// session ID round-trips.

const base = ''

// Admin token (set when the server is started with --admin-token). Persisted in
// localStorage and sent on every admin request so the server's guard lets us in.
const TOKEN_KEY = 'rag_admin_token'
export const adminToken = {
  get: () => { try { return localStorage.getItem(TOKEN_KEY) || '' } catch { return '' } },
  set: (v) => { try { localStorage.setItem(TOKEN_KEY, v) } catch {} },
  clear: () => { try { localStorage.removeItem(TOKEN_KEY) } catch {} },
}
export function authHeaders() {
  const t = adminToken.get()
  return t ? { 'X-Admin-Token': t } : {}
}

async function request(path, opts = {}) {
  const res = await fetch(base + path, {
    credentials: 'include',
    ...opts,
    headers: {
      'Content-Type': 'application/json',
      ...authHeaders(),
      ...(opts.headers || {}),
    },
  })
  if (!res.ok) {
    let detail = res.statusText
    try {
      const body = await res.json()
      detail = body.detail || JSON.stringify(body)
    } catch {}
    const err = new Error(`${res.status}: ${detail}`)
    err.status = res.status
    throw err
  }
  return res.json()
}

export const api = {
  health: () => request('/api/health'),
  providers: () => request('/api/providers'),
  session: () => request('/api/session'),
  setLLM: (provider, settings) =>
    request('/api/session/llm', {
      method: 'POST',
      body: JSON.stringify({ provider, settings }),
    }),
  setEmbedder: (provider, settings) =>
    request('/api/session/embedder', {
      method: 'POST',
      body: JSON.stringify({ provider, settings }),
    }),
  setVectorStore: (provider, settings) =>
    request('/api/session/vector_store', {
      method: 'POST',
      body: JSON.stringify({ provider, settings }),
    }),
  testConnections: () => request('/api/session/test', { method: 'POST' }),
  setPipelineTuning: (tuning) =>
    request('/api/session/pipeline', {
      method: 'POST',
      body: JSON.stringify(tuning),
    }),
  getConfig: () => request('/api/session/config'),
  resetSession: () => request('/api/session', { method: 'DELETE' }),

  ingestText: (title, content, metadata) =>
    request('/api/ingest/text', {
      method: 'POST',
      body: JSON.stringify({ title, content, metadata: metadata || {} }),
    }),
  ingestUrl: (url, title, metadata) =>
    request('/api/ingest/url', {
      method: 'POST',
      body: JSON.stringify({ url, title, metadata: metadata || {} }),
    }),
  ingestPath: (path, metadata) =>
    request('/api/ingest/path', {
      method: 'POST',
      body: JSON.stringify({ path, metadata: metadata || {} }),
    }),
  ingestUpload: async (files) => {
    const form = new FormData()
    for (const f of files) form.append('files', f)
    const res = await fetch('/api/ingest/upload', {
      method: 'POST',
      credentials: 'include',
      headers: { ...authHeaders() },
      body: form,
    })
    if (!res.ok) {
      let detail = res.statusText
      try {
        const body = await res.json()
        detail = body.detail || JSON.stringify(body)
      } catch {}
      throw new Error(`${res.status}: ${detail}`)
    }
    return res.json()
  },
  listSources: () => request('/api/sources'),
  sourceChunks: (docId) => request(`/api/sources/${encodeURIComponent(docId)}/chunks`),
  clearSources: () => request('/api/sources', { method: 'DELETE' }),

  query: (q, filters) =>
    request('/api/query', {
      method: 'POST',
      body: JSON.stringify({ query: q, filters: filters || [] }),
    }),

  // ---- Jobs (long-running tasks with progress streaming) ----
  startUploadJob: async (files, metadata) => {
    const form = new FormData()
    for (const f of files) form.append('files', f)
    if (metadata && Object.keys(metadata).length) {
      form.append('metadata', JSON.stringify(metadata))
    }
    const res = await fetch('/api/jobs/ingest/upload', {
      method: 'POST', credentials: 'include', headers: { ...authHeaders() }, body: form,
    })
    if (!res.ok) {
      let detail = res.statusText
      try { const body = await res.json(); detail = body.detail || JSON.stringify(body) } catch {}
      throw new Error(`${res.status}: ${detail}`)
    }
    return res.json()  // {job_id}
  },
  startPathJob: (pathOrPaths, metadata) => {
    const body = Array.isArray(pathOrPaths)
      ? { paths: pathOrPaths, metadata: metadata || {} }
      : { path: pathOrPaths, metadata: metadata || {} }
    return request('/api/jobs/ingest/path', {
      method: 'POST', body: JSON.stringify(body),
    })
  },
  startUrlJob: (urlOrUrls, title, metadata) => {
    const body = Array.isArray(urlOrUrls)
      ? { urls: urlOrUrls, title: title || null, metadata: metadata || {} }
      : { url: urlOrUrls, title: title || null, metadata: metadata || {} }
    return request('/api/jobs/ingest/url', {
      method: 'POST', body: JSON.stringify(body),
    })
  },
  startTextJob: (title, content, metadata) => request('/api/jobs/ingest/text', {
    method: 'POST', body: JSON.stringify({ title, content, metadata: metadata || {} }),
  }),
  startGraphBuildJob: () => request('/api/jobs/graph/build', { method: 'POST' }),

  // ---- Schema + chunker ----
  getSchema: () => request('/api/schema'),
  setSchema: (fields) => request('/api/schema', {
    method: 'POST', body: JSON.stringify({ fields }),
  }),
  listChunkers: () => request('/api/chunkers'),
  setChunker: (strategy, settings) => request('/api/session/chunker', {
    method: 'POST', body: JSON.stringify({ strategy, settings: settings || {} }),
  }),

  // ---- Structured ingest ----
  structuredPreview: async (file) => {
    const form = new FormData()
    form.append('file', file)
    const res = await fetch('/api/ingest/structured/preview', {
      method: 'POST', credentials: 'include', headers: { ...authHeaders() }, body: form,
    })
    if (!res.ok) {
      let detail = res.statusText
      try { const body = await res.json(); detail = body.detail || JSON.stringify(body) } catch {}
      throw new Error(`${res.status}: ${detail}`)
    }
    return res.json()
  },
  structuredCommit: (preview_id, fields, content_columns, extra_metadata) =>
    request('/api/ingest/structured/commit', {
      method: 'POST',
      body: JSON.stringify({
        preview_id, fields, content_columns,
        extra_metadata: extra_metadata || {},
      }),
    }),
  discardPreview: (preview_id) =>
    request(`/api/ingest/structured/preview/${preview_id}`, { method: 'DELETE' }),

  getJob: (jobId) => request(`/api/jobs/${jobId}`),
  cancelJob: (jobId) => request(`/api/jobs/${jobId}/cancel`, { method: 'POST' }),

  /**
   * Subscribe to job progress via SSE. Calls onEvent for each event;
   * resolves when the job reaches a terminal state.
   * Returns a cancel fn that aborts the stream (does NOT cancel the job).
   */
  streamJob: (jobId, onEvent, onDone, onError) => {
    const controller = new AbortController()
    fetch(`/api/jobs/${jobId}/stream`, {
      method: 'GET', credentials: 'include', headers: { ...authHeaders() }, signal: controller.signal,
    }).then(async (res) => {
      if (!res.ok) {
        const body = await res.text()
        onError(`${res.status}: ${body}`)
        return
      }
      const reader = res.body.getReader()
      const decoder = new TextDecoder()
      let buffer = ''
      while (true) {
        const { value, done } = await reader.read()
        if (done) break
        buffer += decoder.decode(value, { stream: true })
        const messages = buffer.split('\n\n')
        buffer = messages.pop()
        for (const msg of messages) {
          for (const line of msg.split('\n')) {
            if (line.startsWith('data: ')) {
              try {
                const payload = JSON.parse(line.slice(6))
                onEvent(payload)
                if (payload.kind === 'terminal') {
                  onDone(payload)
                  return
                }
              } catch (e) { /* skip malformed */ }
            }
          }
        }
      }
      onDone(null)
    }).catch((e) => {
      if (e.name !== 'AbortError') onError(String(e))
    })
    return () => controller.abort()
  },

  queryStream: (q, onEvent, onDone, onError, filters) => {
    // Server-Sent Events streaming. We use fetch + a reader because EventSource
    // doesn't support POST requests with custom bodies.
    const controller = new AbortController()
    fetch('/api/query/stream', {
      method: 'POST',
      credentials: 'include',
      headers: { 'Content-Type': 'application/json', ...authHeaders() },
      body: JSON.stringify({ query: q, filters: filters || [] }),
      signal: controller.signal,
    }).then(async (res) => {
      if (!res.ok) {
        const body = await res.text()
        onError(`${res.status}: ${body}`)
        return
      }
      const reader = res.body.getReader()
      const decoder = new TextDecoder()
      let buffer = ''
      while (true) {
        const { value, done } = await reader.read()
        if (done) break
        buffer += decoder.decode(value, { stream: true })
        // SSE messages are separated by double newlines
        const messages = buffer.split('\n\n')
        buffer = messages.pop()
        for (const msg of messages) {
          if (!msg.trim()) continue
          for (const line of msg.split('\n')) {
            if (line.startsWith('data: ')) {
              try {
                const payload = JSON.parse(line.slice(6))
                onEvent(payload)
              } catch (e) { /* malformed line, skip */ }
            }
          }
        }
      }
      onDone()
    }).catch((e) => {
      if (e.name !== 'AbortError') onError(String(e))
    })
    return () => controller.abort()
  },
  trace: (id) => request(`/api/trace/${id}`),
  history: () => request('/api/history'),
  getStrategy: () => request('/api/strategy'),
  setStrategy: (cfg) => request('/api/strategy', {
    method: 'POST', body: JSON.stringify(cfg),
  }),
  buildGraph: () => request('/api/graph/build', { method: 'POST' }),
  graphSnapshot: () => request('/api/graph/snapshot'),
  clearGraph: () => request('/api/graph', { method: 'DELETE' }),

  // ---- Published Space (public ask-only assistant) ----
  getSpace: () => request('/api/space'),
  saveSpace: (settings) => request('/api/space', {
    method: 'POST', body: JSON.stringify(settings),
  }),
  publishSpace: (settings) => request('/api/space/publish', {
    method: 'POST', body: JSON.stringify(settings),
  }),
  unpublishSpace: () => request('/api/space/unpublish', { method: 'POST' }),
  spaceHosting: () => request('/api/space/hosting'),
}
