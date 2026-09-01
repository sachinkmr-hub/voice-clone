import type { Band, TrackedCall } from '../types';

const BAND_COLOR: Record<Band, string> = {
  LOW: '#059669', ELEVATED: '#d97706', HIGH: '#ea580c', CRITICAL: '#dc2626',
};

/** The drill-down for one call: score, evidence, and — importantly — which layers
 *  abstained and why. An operator who cannot tell "we checked and it was fine" from
 *  "we could not check" will eventually trust the wrong call. */
export function CallDetail({ call }: { call: TrackedCall | null }) {
  if (!call) {
    return (
      <aside className="detail empty-detail">
        <div className="empty-icon" aria-hidden="true">👈</div>
        <p>Select a call to see its evidence, layer breakdown and recommended action.</p>
      </aside>
    );
  }

  return (
    <aside className="detail">
      <header className="detail-head">
        <code>{call.sessionId}</code>
        <span className={`band-chip band-${call.band}`}>{call.band}</span>
      </header>

      <div className="detail-score" style={{ color: BAND_COLOR[call.band] }}>
        {Math.round(call.score)}<span className="unit">/100</span>
      </div>
      {call.headline && <p className="detail-headline">{call.headline}</p>}
      {call.action && (
        <div className={`detail-action band-${call.band}`}>{call.action}</div>
      )}

      <div className="detail-stats">
        <div><span className="k">Windows</span><span className="v">{call.windows}</span></div>
        <div><span className="k">Elapsed</span><span className="v">{call.elapsed.toFixed(1)}s</span></div>
        <div><span className="k">Latency</span><span className="v">{Math.round(call.latencyMs)}ms</span></div>
        <div><span className="k">Profile</span><span className="v">{call.profile}</span></div>
      </div>

      <section>
        <h3>Evidence</h3>
        {call.factors.length === 0 ? (
          <p className="muted small">No single factor is dominant.</p>
        ) : (
          call.factors.map((factor) => (
            <div className="factor" key={factor.code}>
              <div className="factor-head">
                <span>
                  <span className="layer-tag">{factor.layer}</span>
                  {factor.label}
                </span>
                <span className="factor-pct">{Math.round(factor.contribution * 100)}%</span>
              </div>
              {factor.detail && <div className="factor-detail">{factor.detail}</div>}
              <div className="factor-bar">
                <div className="factor-fill"
                     style={{ width: `${Math.min(100, factor.contribution * 100)}%` }} />
              </div>
            </div>
          ))
        )}
      </section>

      {call.layers.length > 0 && (
        <section>
          <h3>Layers</h3>
          {call.layers.map((layer) => (
            <div className="layer-row" key={layer.layer}>
              <div>
                <div className="layer-name">
                  {layer.label}
                  <span className={`status-tag status-${layer.status}`}>{layer.status}</span>
                </div>
                {layer.reason && <div className="layer-reason">{layer.reason}</div>}
              </div>
              <div className="layer-score">
                {layer.status === 'voted' ? `${Math.round(layer.score * 100)}/100` : '—'}
                <div className="layer-share">
                  {layer.status === 'voted'
                    ? `${Math.round(layer.weight_share * 100)}% of decision`
                    : 'abstained'}
                </div>
              </div>
            </div>
          ))}
        </section>
      )}

      {call.caveats.length > 0 && (
        <section className="caveats">
          <strong>Caveats</strong>
          <ul>{call.caveats.map((caveat) => <li key={caveat}>{caveat}</li>)}</ul>
        </section>
      )}
    </aside>
  );
}
