/* Workspace.

   Renders one section at a time — the section selected in the top nav.
   Each former route (Dashboard / Providers / Data / Query / History /
   Graph / Config) is shown on its own page; switching nav links swaps
   the whole panel rather than scrolling through a stack.

   Each section keeps the "3D" panel treatment from styles.css: layered
   box-shadows, a subtle yellow edge glow, and a hover lift. */
import Dashboard from './Dashboard'
import ProvidersPage from './ProvidersPage'
import DataPage from './DataPage'
import GraphPage from './GraphPage'
import QueryPage from './QueryPage'
import HistoryPage from './HistoryPage'
import ConfigPage from './ConfigPage'
import PublishPage from './PublishPage'


function Section({ id, subtitle, children }) {
  return (
    <section id={id} className="ws-section" data-section={id}>
      <div className="ws-section-tag">
        <span className="ws-section-id">▎{id.toUpperCase()}</span>
        {subtitle && <span className="ws-section-subtitle">{subtitle}</span>}
      </div>
      <div className="ws-section-body">
        {children}
      </div>
    </section>
  )
}


export default function Workspace({
  active, session, providers, refreshSession, notify, onNavigate, onQuickAction,
}) {
  return (
    <div className="workspace">
      {active === 'dashboard' && (
        <Section id="dashboard" subtitle="experiment · session overview">
          <Dashboard
            session={session}
            providers={providers}
            onNavigate={onNavigate}
            onQuickAction={onQuickAction}
            notify={notify}
          />
        </Section>
      )}

      {active === 'providers' && (
        <Section id="providers" subtitle="llm · embedder · vector store">
          <ProvidersPage
            session={session}
            providers={providers}
            onChange={refreshSession}
            notify={notify}
          />
        </Section>
      )}

      {active === 'data' && (
        <Section id="data" subtitle="ingest · chunk · embed">
          <DataPage session={session} onChange={refreshSession} notify={notify} />
        </Section>
      )}

      {active === 'query' && (
        <Section id="query" subtitle="retrieve · constrain · answer">
          <QueryPage session={session} onChange={refreshSession} notify={notify} />
        </Section>
      )}

      {active === 'history' && (
        <Section id="history" subtitle="recent answers · trace · citations">
          <HistoryPage notify={notify} />
        </Section>
      )}

      {active === 'graph' && (
        <Section id="graph" subtitle="entity extraction · graph rag">
          <GraphPage session={session} onChange={refreshSession} notify={notify} />
        </Section>
      )}

      {active === 'config' && (
        <Section id="config" subtitle="pipeline tuning · thresholds">
          <ConfigPage session={session} onChange={refreshSession} notify={notify} />
        </Section>
      )}

      {active === 'publish' && (
        <Section id="publish" subtitle="share a public ask-only assistant">
          <PublishPage session={session} notify={notify} />
        </Section>
      )}
    </div>
  )
}
