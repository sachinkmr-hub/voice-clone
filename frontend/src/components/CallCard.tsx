import type { Band, TrackedCall } from '../types';

const BAND_COLOR: Record<Band, string> = {
  LOW: '#059669',
  ELEVATED: '#d97706',
  HIGH: '#ea580c',
  CRITICAL: '#dc2626',
};

/** A compact sparkline of the call's score so far.
 *
 * Drawn as an inline SVG rather than pulling in a charting library: one path and three
 * threshold guides is the whole requirement, and a chart dependency would be larger than
 * the rest of this dashboard put together.
 */
export function Sparkline({ history, band }: { history: TrackedCall['history']; band: Band }) {
  if (history.length < 2) {
    return <div className="spark-empty">collecting…</div>;
  }
  const width = 260;
  const height = 46;
  const step = width / (history.length - 1);
  const y = (score: number) => height - (Math.max(0, Math.min(100, score)) / 100) * height;

  const line = history.map((point, i) => `${i * step},${y(point.score)}`).join(' ');
  const area = `0,${height} ${line} ${(history.length - 1) * step},${height}`;
  const id = `spark-${band}`;

  return (
    <svg className="spark" viewBox={`0 0 ${width} ${height}`} preserveAspectRatio="none"
         aria-hidden="true">
      <defs>
        <linearGradient id={id} x1="0" y1="0" x2="0" y2="1">
          <stop offset="0%" stopColor={BAND_COLOR[band]} stopOpacity="0.32" />
          <stop offset="100%" stopColor={BAND_COLOR[band]} stopOpacity="0" />
        </linearGradient>
      </defs>
      <line x1="0" y1={y(60)} x2={width} y2={y(60)} stroke="#ea580c"
            strokeWidth="1" strokeDasharray="3 4" opacity="0.3" />
      <polygon points={area} fill={`url(#${id})`} />
      <polyline points={line} fill="none" stroke={BAND_COLOR[band]} strokeWidth="2"
                strokeLinejoin="round" strokeLinecap="round" />
    </svg>
  );
}

interface Props {
  call: TrackedCall;
  onSelect: (id: string) => void;
  isSelected: boolean;
}

export function CallCard({ call, onSelect, isSelected }: Props) {
  const topFactor = call.factors[0];

  return (
    <button
      type="button"
      className={`call-card band-${call.band} ${call.closed ? 'closed' : ''} ${isSelected ? 'selected' : ''}`}
      onClick={() => onSelect(call.sessionId)}
      aria-label={`Call ${call.sessionId}, risk ${Math.round(call.score)} of 100, ${call.band}`}
    >
      <div className="call-head">
        <div>
          <code className="call-id">{call.sessionId}</code>
          <div className="call-meta">
            {call.profile} · {call.elapsed.toFixed(1)}s · {call.windows} windows
            {call.provisional && <span className="tag-provisional">settling</span>}
            {call.closed && <span className="tag-closed">{call.verdict ?? 'closed'}</span>}
          </div>
        </div>
        <div className="call-score" style={{ color: BAND_COLOR[call.band] }}>
          {Math.round(call.score)}
          <span className="call-score-suffix">/100</span>
        </div>
      </div>

      <Sparkline history={call.history} band={call.band} />

      <div className="call-foot">
        <span className={`band-chip band-${call.band}`}>{call.band}</span>
        {topFactor ? (
          <span className="call-factor" title={topFactor.detail}>{topFactor.label}</span>
        ) : (
          <span className="call-factor muted">no dominant factor</span>
        )}
        {call.alerted && <span className="alert-dot" title="Alert raised">●</span>}
      </div>
    </button>
  );
}
