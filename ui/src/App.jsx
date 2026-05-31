import { useEffect, useState, useCallback } from 'react'
import { api } from './api'
import Workspace from './components/Workspace'
import Toast from './components/Toast'
import Modal from './components/Modal'
import AdminGate from './components/AdminGate'
import ProvidersPage from './components/ProvidersPage'
import DataPage from './components/DataPage'
import QueryPage from './components/QueryPage'

// Quick-action modals — set up details inline without leaving the page.
const MODALS = {
  providers: { icon: '◈', title: 'Connect a model', subtitle: 'llm · embedder · vector store' },
  data: { icon: '⬚', title: 'Add data', subtitle: 'ingest · chunk · embed' },
  query: { icon: '⌕', title: 'Ask a question', subtitle: 'retrieve · constrain · answer' },
}

// Section order matches the nav. Only the active section is shown at a time.
const SECTIONS = [
  { id: 'dashboard', label: 'Dashboard' },
  { id: 'providers', label: 'Providers' },
  { id: 'data', label: 'Data' },
  { id: 'query', label: 'Query' },
  { id: 'history', label: 'History' },
  { id: 'graph', label: 'Graph' },
  { id: 'config', label: 'Config' },
  { id: 'publish', label: 'Publish' },
]

export default function App() {
  const [activeSection, setActiveSection] = useState('dashboard')
  const [session, setSession] = useState(null)
  const [providers, setProviders] = useState(null)
  const [toast, setToast] = useState(null)
  const [modal, setModal] = useState(null)  // 'providers' | 'data' | 'query' | null
  const [navHistory, setNavHistory] = useState([])  // back stack of visited sections
  const [authNeeded, setAuthNeeded] = useState(false)  // admin token required

  const refreshSession = useCallback(async () => {
    try {
      const s = await api.session()
      setSession(s)
    } catch (e) {
      if (e.status === 401) { setAuthNeeded(true); return }
      setToast({ kind: 'err', msg: `Failed to load session: ${e.message}` })
    }
  }, [])

  const loadProviders = useCallback(async () => {
    try {
      const p = await api.providers()
      setProviders(p)
    } catch (e) {
      if (e.status === 401) { setAuthNeeded(true); return }
      setToast({ kind: 'err', msg: `Failed to load providers: ${e.message}` })
    }
  }, [])

  useEffect(() => {
    loadProviders()
    refreshSession()
  }, [loadProviders, refreshSession])

  // Switch to a section and scroll back to the top so the page starts clean.
  // Remember where we came from so the Back button can return there.
  const navigate = (id) => {
    if (id !== activeSection) setNavHistory((h) => [...h, activeSection])
    setActiveSection(id)
    window.scrollTo({ top: 0, behavior: 'auto' })
  }

  // Back button: pop the last visited section, or fall back to the dashboard.
  const canGoBack = navHistory.length > 0 || activeSection !== 'dashboard'
  const goBack = () => {
    if (navHistory.length > 0) {
      const prev = navHistory[navHistory.length - 1]
      setNavHistory((h) => h.slice(0, -1))
      setActiveSection(prev)
    } else {
      setActiveSection('dashboard')
    }
    window.scrollTo({ top: 0, behavior: 'auto' })
  }

  const notify = (kind, msg) => {
    setToast({ kind, msg })
    setTimeout(() => setToast(null), 4000)
  }

  const counts = {
    data: session?.n_sources || 0,
    history: session?.history_count || 0,
  }

  // Adaptive workflow: surface the single next action the user should take.
  const llm = session?.llm?.provider
  const emb = session?.embedder?.provider
  const providersReady = !!(session && llm && llm !== 'mock' && emb && emb !== 'hash')
  const dataReady = (session?.n_sources || 0) > 0
  const askedSomething = (session?.history_count || 0) > 0
  let nextStep = null
  if (session) {
    if (!providersReady) {
      nextStep = { go: 'providers', label: 'Connect a model',
        msg: <>Mock providers are active — <b>connect a real LLM &amp; embedder</b> to get grounded answers.</> }
    } else if (!dataReady) {
      nextStep = { go: 'data', label: 'Add data',
        msg: <>Your index is empty — <b>add documents, files or URLs</b> to query.</> }
    } else if (!askedSomething) {
      nextStep = { go: 'query', label: 'Ask a question',
        msg: <>Everything's ready — <b>ask your first question</b>.</> }
    }
  }
  // Show the slim banner everywhere except the dashboard (which has the
  // richer Getting-started stepper) and the page the action points to.
  const showBanner = nextStep && activeSection !== 'dashboard' && activeSection !== nextStep.go

  return (
    <div className="app">
      <header className="topnav">
        <div className="topnav-brand">
          <span
            className={`chev ${canGoBack ? '' : 'disabled'}`}
            onClick={canGoBack ? goBack : undefined}
            role="button"
            tabIndex={canGoBack ? 0 : -1}
            aria-label="Back"
            title={canGoBack ? 'Back' : ''}
            onKeyDown={(e) => {
              if (canGoBack && (e.key === 'Enter' || e.key === ' ')) { e.preventDefault(); goBack() }
            }}
          >
            ‹
          </span>
          <div
            className="brand-home"
            onClick={() => navigate('dashboard')}
            role="button"
            tabIndex={0}
            title="Go to dashboard"
            onKeyDown={(e) => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); navigate('dashboard') } }}
          >
            <div className="title">
              ragzero <span className="accent">console</span>
            </div>
            <div className="subtitle">
              session {session?.session_id?.slice(0, 8) || '——————'} ·{' '}
              {session?.n_sources || 0} sources · {session?.n_chunks ?? 0} chunks
            </div>
          </div>
        </div>
        <nav className="topnav-links">
          {SECTIONS.map((s) => (
            <span
              key={s.id}
              className={`topnav-link ${activeSection === s.id ? 'active' : ''}`}
              onClick={() => navigate(s.id)}
              role="button"
              tabIndex={0}
              aria-current={activeSection === s.id ? 'page' : undefined}
              onKeyDown={(e) => {
                if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); navigate(s.id) }
              }}
            >
              {s.label}
              {counts[s.id] > 0 && (
                <span className="topnav-badge">[{counts[s.id]}]</span>
              )}
            </span>
          ))}
        </nav>
      </header>

      <div className="statusbar">
        <div className="item">
          <span className="k">LLM</span>
          <span className="v accent">{session?.llm?.provider || '——'}</span>
        </div>
        <div className="item">
          <span className="k">EMBEDDER</span>
          <span className="v accent">{session?.embedder?.provider || '——'}</span>
        </div>
        <div className="item">
          <span className="k">STORE</span>
          <span className="v accent">{session?.vector_store?.provider || '——'}</span>
        </div>
        <div className="item">
          <span className="k">PIPELINE</span>
          <span className="v">{session?.pipeline_built ? 'BUILT' : 'PENDING'}</span>
        </div>
      </div>

      {showBanner && (
        <div className="flow-banner">
          <span className="fb-dot" />
          <span className="fb-msg">{nextStep.msg}</span>
          <span className="fb-cta">
            <button className="primary" onClick={() => setModal(nextStep.go)}>
              {nextStep.label} ›
            </button>
          </span>
        </div>
      )}

      <main className="main workspace-main">
        <Workspace
          active={activeSection}
          session={session}
          providers={providers}
          refreshSession={refreshSession}
          notify={notify}
          onNavigate={navigate}
          onQuickAction={setModal}
        />
      </main>

      <footer className="footer">
        © ragzero · production RAG with near-zero hallucination · 10-stage pipeline
      </footer>

      {modal && (
        <Modal
          icon={MODALS[modal].icon}
          title={MODALS[modal].title}
          subtitle={MODALS[modal].subtitle}
          onClose={() => setModal(null)}
        >
          {modal === 'providers' && (
            <ProvidersPage
              session={session}
              providers={providers}
              onChange={refreshSession}
              notify={notify}
            />
          )}
          {modal === 'data' && (
            <DataPage session={session} onChange={refreshSession} notify={notify} />
          )}
          {modal === 'query' && (
            <QueryPage session={session} onChange={refreshSession} notify={notify} />
          )}
        </Modal>
      )}

      {authNeeded && (
        <AdminGate onUnlock={() => {
          setAuthNeeded(false)
          loadProviders(); refreshSession()
        }} />
      )}

      {toast && <Toast {...toast} />}
    </div>
  )
}
