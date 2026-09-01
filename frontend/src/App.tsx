import { useState } from 'react';
import { CallCard } from './components/CallCard';
import { CallDetail } from './components/CallDetail';
import { useDashboard } from './useDashboard';
import type { Band } from './types';

const SOCKET_LABEL: Record<string, string> = {
  connecting: 'connecting…',
  open: 'live',
  closed: 'reconnecting…',
  error: 'connection error',
};

/**
 * The operations view: every call currently under analysis, ranked by risk.
 *
 * This is deliberately *not* the same screen as `/console`. The console answers "is this
 * one call real?" for an agent handling it. This answers "which of my 200 live calls
 * needs a human right now?" for a fraud desk or a telecom NOC — the persona the problem
 * statement calls the telecom operator / CISO.
 */
export default function App() {
  const { calls, alerts, health, socketState, stats, selected, selectCall, dismissAlerts } =
    useDashboard();
  const [bandFilter, setBandFilter] = useState<Band | 'ALL'>('ALL');
  const [hideClosed, setHideClosed] = useState(false);

  const visible = calls.filter((call) => {
    if (hideClosed && call.closed) return false;
    if (bandFilter !== 'ALL' && call.band !== bandFilter) return false;
    return true;
  });

  return (
    <div className="app">
      <header className="topbar">
        <div className="logo" aria-hidden="true">🛡️</div>
        <div>
          <h1>VoiceGuard Operations</h1>
          <p>Live voice-integrity monitoring · SIH26104</p>
        </div>

        <div className="topbar-right">
          <span className={`pill ${socketState === 'open' ? 'ok' : 'warn'}`}>
            <span className={`dot ${socketState === 'open' ? 'live' : ''}`} />
            {SOCKET_LABEL[socketState] ?? socketState}
          </span>
          {health && (
            <span className={`pill ${health.model_loaded ? 'ok' : 'warn'}`}
                  title={health.degraded.join(' · ') || 'All detectors nominal'}>
              {health.model_loaded ? 'trained model' : 'heuristic detector'}
            </span>
          )}
          <a className="pill link" href="/console">Analyse a recording ↗</a>
        </div>
      </header>

      {health && (
        <div className="privacy-strip">
          🔒 {health.retention.banner}
        </div>
      )}

      <section className="stat-row">
        <div className="stat"><span className="k">Live calls</span><span className="v">{stats.live}</span></div>
        <div className="stat danger"><span className="k">At risk</span><span className="v">{stats.atRisk}</span></div>
        <div className="stat"><span className="k">Alerts</span><span className="v">{stats.alerts}</span></div>
        <div className="stat"><span className="k">Mean latency</span><span className="v">{stats.meanLatency}<small>ms</small></span></div>
        <div className="stat">
          <span className="k">Capacity</span>
          <span className="v">
            {health ? `${health.sessions.active_sessions}/${health.sessions.capacity}` : '—'}
          </span>
        </div>
      </section>

      {alerts.length > 0 && (
        <section className="alert-bar">
          <div className="alert-bar-head">
            <strong>⚠️ {alerts.length} alert{alerts.length === 1 ? '' : 's'}</strong>
            <button type="button" onClick={dismissAlerts}>Clear</button>
          </div>
          <ul>
            {alerts.slice(0, 4).map((alert, index) => (
              <li key={`${alert.session_id}-${alert.created_at}-${index}`}>
                <span className={`band-chip band-${alert.band}`}>{alert.band}</span>
                <code>{alert.session_id}</code>
                <span className="alert-text">{alert.headline}</span>
              </li>
            ))}
          </ul>
        </section>
      )}

      <div className="controls">
        <div className="filters">
          {(['ALL', 'CRITICAL', 'HIGH', 'ELEVATED', 'LOW'] as const).map((band) => (
            <button
              key={band}
              type="button"
              className={bandFilter === band ? 'active' : ''}
              onClick={() => setBandFilter(band)}
            >
              {band}
            </button>
          ))}
        </div>
        <label className="toggle">
          <input type="checkbox" checked={hideClosed}
                 onChange={(e) => setHideClosed(e.target.checked)} />
          Hide finished calls
        </label>
      </div>

      <main className="layout">
        <div className="board">
          {visible.length === 0 ? (
            <div className="empty-board">
              <div className="empty-icon" aria-hidden="true">📡</div>
              <h2>No calls matching this filter</h2>
              <p>
                Start a stream from <a href="/console">the console</a> — or point any
                client at <code>WS /v1/stream</code> — and it will appear here within a
                second.
              </p>
            </div>
          ) : (
            <div className="grid">
              {visible.map((call) => (
                <CallCard
                  key={call.sessionId}
                  call={call}
                  isSelected={selected?.sessionId === call.sessionId}
                  onSelect={selectCall}
                />
              ))}
            </div>
          )}
        </div>

        <CallDetail call={selected} />
      </main>

      <footer className="foot">
        VoiceGuard · SIH26104 · <a href="/docs">API reference</a>
        {health && <> · v{health.version} · up {Math.round(health.uptime_seconds)}s</>}
      </footer>
    </div>
  );
}
