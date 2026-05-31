// Public assistant API client — the ONLY endpoints the published page touches.
// No credentials, no admin surface.

async function post(path, body) {
  const res = await fetch(path, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  })
  if (!res.ok) {
    let detail = res.statusText
    try { const b = await res.json(); detail = b.detail || JSON.stringify(b) } catch {}
    const err = new Error(detail)
    err.status = res.status
    throw err
  }
  return res.json()
}

export const publicApi = {
  meta: async () => {
    const res = await fetch('/public/meta')
    if (!res.ok) throw new Error('Could not load assistant')
    return res.json()
  },
  query: (query, mode, accessCode) =>
    post('/public/query', { query, mode, access_code: accessCode || null }),
}
