import { useEffect, useState } from 'react'
import { api } from '../api'

const MODES = [
  ['simple', 'Simple — retrieval + answer'],
  ['graph', 'Graph — needs a built knowledge graph'],
  ['agentic', 'Agentic — plans & reflects (slower)'],
  ['multilingual', 'Multilingual — ask/answer any language'],
]

export default function PublishPage({ session, notify }) {
  const [data, setData] = useState(null)        // {space, current, admin_token_set}
  const [hosting, setHosting] = useState(null)
  const [busy, setBusy] = useState(false)

  // form state
  const [name, setName] = useState('Knowledge Assistant')
  const [title, setTitle] = useState('Ask me anything')
  const [welcome, setWelcome] = useState('')
  const [accent, setAccent] = useState('#d4ff00')
  const [footer, setFooter] = useState('')
  const [suggested, setSuggested] = useState('')
  const [modes, setModes] = useState(['simple'])
  const [defaultMode, setDefaultMode] = useState('simple')
  const [showCitations, setShowCitations] = useState(true)
  const [showConfidence, setShowConfidence] = useState(true)
  const [streaming, setStreaming] = useState(true)
  const [requireCode, setRequireCode] = useState(false)
  const [accessCode, setAccessCode] = useState('')
  const [rateLimit, setRateLimit] = useState(12)
  const [dailyCap, setDailyCap] = useState(500)

  const load = async () => {
    try {
      const d = await api.getSpace()
      setData(d)
      const sp = d.space
      if (sp) {
        setName(sp.name); setTitle(sp.title); setWelcome(sp.welcome)
        setAccent(sp.accent); setFooter(sp.footer || '')
        setSuggested((sp.suggested_questions || []).join('\n'))
        setModes(sp.allowed_modes?.length ? sp.allowed_modes : ['simple'])
        setDefaultMode(sp.default_mode || 'simple')
        setShowCitations(sp.show_citations); setShowConfidence(sp.show_confidence)
        setStreaming(sp.streaming)
        setRequireCode(sp.access_required); setAccessCode('')
        setRateLimit(sp.rate_limit_per_min); setDailyCap(sp.daily_cap)
      }
    } catch (e) { notify('err', e.message) }
    try { setHosting(await api.spaceHosting()) } catch {}
  }
  useEffect(() => { load() }, [])  // eslint-disable-line

  const buildSettings = () => {
    const s = {
      name, title, welcome, accent, footer,
      suggested_questions: suggested.split('\n').map((x) => x.trim()).filter(Boolean),
      allowed_modes: modes.length ? modes : ['simple'],
      default_mode: modes.includes(defaultMode) ? defaultMode : modes[0],
      show_citations: showCitations,
      show_confidence: showConfidence,
      streaming,
      rate_limit_per_min: Number(rateLimit) || 12,
      daily_cap: Number(dailyCap) || 500,
    }
    if (!requireCode) s.access_code = ''
    else if (accessCode) s.access_code = accessCode
    return s
  }

  const doSave = async () => {
    setBusy(true)
    try { await api.saveSpace(buildSettings()); notify('ok', 'Settings saved'); await load() }
    catch (e) { notify('err', e.message) } finally { setBusy(false) }
  }
  const doPublish = async () => {
    setBusy(true)
    try { await api.publishSpace(buildSettings()); notify('ok', 'Published! Share the public link.'); await load() }
    catch (e) { notify('err', e.message) } finally { setBusy(false) }
  }
  const doUnpublish = async () => {
    if (!confirm('Take the public assistant offline?')) return
    setBusy(true)
    try { await api.unpublishSpace(); notify('ok', 'Unpublished'); await load() }
    catch (e) { notify('err', e.message) } finally { setBusy(false) }
  }

  const toggleMode = (m) => {
    setModes((cur) => {
      const next = cur.includes(m) ? cur.filter((x) => x !== m) : [...cur, m]
      return next.length ? next : ['simple']
    })
  }

  if (!data) return <div className="empty">Loading…</div>

  const sp = data.space
  const published = sp?.published
  const slug = sp?.slug || 'assistant'
  const origin = window.location.origin
  const publicUrl = `${origin}/a/${slug}`
  const port = window.location.port || (window.location.protocol === 'https:' ? '443' : '80')
  const lanUrl = hosting?.lan_ip ? `http://${hosting.lan_ip}:${port}/a/${slug}` : null

  const noData = (data.current?.n_sources || 0) === 0
  const mock = !data.current?.real_providers

  const copy = (text) => {
    navigator.clipboard?.writeText(text).then(
      () => notify('ok', 'Copied'),
      () => notify('err', 'Copy failed'),
    )
  }

  return (
    <>
      <div className="page-header">
        <h1>Publish assistant</h1>
        <div className="subtitle">
          Turn this session into a public, ask-only page. Visitors can only ask questions and
          read answers — they never see your data, models, keys, or settings.
        </div>
      </div>

      {/* status / share */}
      <div className="card">
        <div className="card-title">
          <span>Status</span>
          <span className={`status ${published ? 'ok' : 'warn'}`}>
            <span className="indicator" />{published ? 'Live' : 'Not published'}
          </span>
        </div>
        {published ? (
          <>
            <div className="field">
              <label>Public link — share this</label>
              <div className="row" style={{ alignItems: 'center' }}>
                <input readOnly value={publicUrl} onFocus={(e) => e.target.select()} />
                <div className="actions" style={{ display: 'flex', gap: 8 }}>
                  <button onClick={() => copy(publicUrl)}>Copy</button>
                  <button onClick={() => window.open(publicUrl, '_blank')}>Preview ›</button>
                </div>
              </div>
            </div>
            <div className="row">
              <button className="danger ghost" onClick={doUnpublish} disabled={busy}>Take offline</button>
              <div className="actions">
                <button className="primary" onClick={doPublish} disabled={busy}>
                  {busy ? 'Saving…' : 'Re-publish with current data & settings'}
                </button>
              </div>
            </div>
            <div className="faint" style={{ fontSize: 11, marginTop: 8 }}>
              "Re-publish" snapshots your current providers + indexed data again. Plain "Save settings"
              below only changes the look/options, not the data.
            </div>
          </>
        ) : (
          <div className="faint" style={{ fontSize: 12 }}>
            Configure the assistant below, then publish. It will snapshot your current providers
            and indexed data so visitors query against them.
          </div>
        )}
      </div>

      {(noData || mock) && (
        <div className="card" style={{ borderLeft: '2px solid var(--warn)' }}>
          <div className="card-title"><span>Before you publish</span></div>
          {noData && <div style={{ color: 'var(--warn)', fontSize: 12, marginBottom: 4 }}>
            • No data indexed — add documents on the Data page first (the assistant needs something to answer from).
          </div>}
          {mock && <div style={{ color: 'var(--warn)', fontSize: 12 }}>
            • Mock providers are active — visitors will get placeholder answers. Connect a real model on the Providers page.
          </div>}
        </div>
      )}

      <div className="grid-2">
        {/* appearance */}
        <div className="card">
          <div className="card-title"><span>Appearance</span></div>
          <div className="field"><label>Assistant name</label>
            <input value={name} onChange={(e) => setName(e.target.value)} /></div>
          <div className="field"><label>Ask-box placeholder</label>
            <input value={title} onChange={(e) => setTitle(e.target.value)} /></div>
          <div className="field"><label>Welcome message</label>
            <textarea rows={2} value={welcome} onChange={(e) => setWelcome(e.target.value)} /></div>
          <div className="row">
            <div className="field" style={{ flex: '0 0 auto' }}><label>Accent color</label>
              <input type="color" value={accent} onChange={(e) => setAccent(e.target.value)}
                     style={{ width: 54, height: 38, padding: 2 }} /></div>
            <div className="field"><label>Footer text (optional)</label>
              <input value={footer} onChange={(e) => setFooter(e.target.value)} placeholder="e.g. Acme Support · internal use" /></div>
          </div>
          <div className="field"><label>Suggested questions (one per line)</label>
            <textarea rows={4} value={suggested} onChange={(e) => setSuggested(e.target.value)}
                      placeholder={'What is our refund policy?\nSummarize the onboarding guide'} /></div>
        </div>

        {/* behavior + access */}
        <div className="card">
          <div className="card-title"><span>Query options</span></div>
          <label>Allowed modes</label>
          <div style={{ display: 'flex', flexDirection: 'column', gap: 6, marginBottom: 12 }}>
            {MODES.map(([m, desc]) => (
              <label key={m} style={{ display: 'flex', gap: 8, textTransform: 'none', letterSpacing: 0, color: 'var(--text)', cursor: 'pointer' }}>
                <input type="checkbox" checked={modes.includes(m)} onChange={() => toggleMode(m)} />
                <span style={{ fontSize: 12 }}>{desc}</span>
              </label>
            ))}
          </div>
          <div className="field"><label>Default mode</label>
            <select value={defaultMode} onChange={(e) => setDefaultMode(e.target.value)}>
              {modes.map((m) => <option key={m} value={m}>{m}</option>)}
            </select>
          </div>
          <label>Shown in answers</label>
          <div style={{ display: 'flex', gap: 16, marginBottom: 14 }}>
            <label style={{ textTransform: 'none', letterSpacing: 0, color: 'var(--text)', display: 'flex', gap: 6 }}>
              <input type="checkbox" checked={showCitations} onChange={(e) => setShowCitations(e.target.checked)} /> Citations
            </label>
            <label style={{ textTransform: 'none', letterSpacing: 0, color: 'var(--text)', display: 'flex', gap: 6 }}>
              <input type="checkbox" checked={showConfidence} onChange={(e) => setShowConfidence(e.target.checked)} /> Confidence
            </label>
            <label style={{ textTransform: 'none', letterSpacing: 0, color: 'var(--text)', display: 'flex', gap: 6 }}>
              <input type="checkbox" checked={streaming} onChange={(e) => setStreaming(e.target.checked)} /> Typewriter
            </label>
          </div>

          <div className="card-title" style={{ marginTop: 6 }}><span>Access & limits</span></div>
          <label style={{ textTransform: 'none', letterSpacing: 0, color: 'var(--text)', display: 'flex', gap: 6, marginBottom: 8 }}>
            <input type="checkbox" checked={requireCode} onChange={(e) => setRequireCode(e.target.checked)} />
            Require an access code
          </label>
          {requireCode && (
            <div className="field">
              <input type="text" value={accessCode} onChange={(e) => setAccessCode(e.target.value)}
                     placeholder={sp?.access_required ? 'Leave blank to keep current code' : 'Set an access code'} />
            </div>
          )}
          <div className="row">
            <div className="field"><label>Rate limit / min / visitor</label>
              <input type="number" min={1} value={rateLimit} onChange={(e) => setRateLimit(e.target.value)} /></div>
            <div className="field"><label>Daily cap / visitor</label>
              <input type="number" min={1} value={dailyCap} onChange={(e) => setDailyCap(e.target.value)} /></div>
          </div>
        </div>
      </div>

      <div className="row">
        <button className="primary" onClick={doPublish} disabled={busy || noData}>
          {published ? 'Re-publish' : 'Publish assistant'}
        </button>
        <div className="actions">
          <button onClick={doSave} disabled={busy}>Save settings only</button>
        </div>
      </div>

      {/* how to host */}
      <div className="card">
        <div className="card-title"><span>How to host it for other people</span></div>
        {!data.admin_token_set && (
          <div style={{ color: 'var(--warn)', fontSize: 12, marginBottom: 10, borderLeft: '2px solid var(--warn)', paddingLeft: 8 }}>
            ⚠ No admin token set. Before exposing this beyond your own machine, restart with
            <span className="kbd"> ragzero serve --admin-token YOUR_SECRET</span> so visitors can't reach this admin console.
          </div>
        )}
        <HostStep n="1" title="Same network (LAN)"
          body={<>Start the server bound to all interfaces:&nbsp;
            <span className="kbd">ragzero serve --host 0.0.0.0 --admin-token SECRET</span>
            {lanUrl ? <> then share <a href={lanUrl} target="_blank" rel="noreferrer">{lanUrl}</a></>
                    : <> then share <span className="mono">http://&lt;your-ip&gt;:{port}/a/{slug}</span></>}</>} />
        <HostStep n="2" title="Over the internet (quick tunnel)"
          body={<>Keep the server running, then in another terminal:&nbsp;
            <span className="kbd">cloudflared tunnel --url http://localhost:{port}</span>
            &nbsp;(or <span className="kbd">ngrok http {port}</span>). Share the
            &nbsp;<span className="mono">https://…/a/{slug}</span> URL it prints.</>} />
        <HostStep n="3" title="Always-on server"
          body={<>Run it on a VPS/cloud box (Docker), put Caddy/nginx in front for HTTPS + a custom
            domain, set <span className="kbd">RAGZERO_ADMIN_TOKEN</span> and a persistent vector store so
            data survives restarts.</>} />
      </div>
    </>
  )
}

function HostStep({ n, title, body }) {
  return (
    <div style={{ display: 'flex', gap: 12, padding: '8px 0', borderBottom: '1px solid var(--border)' }}>
      <div style={{
        width: 24, height: 24, flex: '0 0 auto', borderRadius: '50%',
        border: '1px solid var(--border-hi)', color: 'var(--accent)',
        display: 'grid', placeItems: 'center', fontFamily: 'var(--mono)', fontSize: 12,
      }}>{n}</div>
      <div>
        <div style={{ fontSize: 13, marginBottom: 3 }}>{title}</div>
        <div className="faint" style={{ fontSize: 12, lineHeight: 1.6 }}>{body}</div>
      </div>
    </div>
  )
}
