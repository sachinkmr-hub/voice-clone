/** Wire types, mirroring `voiceguard/api/schemas.py`.
 *
 * Kept hand-written rather than generated from the OpenAPI document: the dashboard uses a
 * small, stable subset, and a generated client would drag in every admin schema for no
 * benefit. If the API changes shape, `npm run typecheck` fails here first.
 */

export type Band = 'LOW' | 'ELEVATED' | 'HIGH' | 'CRITICAL';

export interface Factor {
  code: string;
  label: string;
  contribution: number;
  value: number;
  detail: string;
  layer: string;
  direction: 'synthetic' | 'genuine';
}

export interface LayerSummary {
  layer: string;
  label: string;
  score: number;
  confidence: number;
  weight_share: number;
  status: 'voted' | 'abstained';
  reason: string;
  model_id: string;
}

export interface RiskEvent {
  type: 'risk';
  session_id: string;
  score: number;
  band: Band;
  action: string;
  headline: string;
  confidence: number;
  probability: number;
  provisional: boolean;
  speech_detected: boolean;
  window_index: number;
  elapsed_seconds: number;
  latency_ms: number;
  profile: string;
  threshold_shift: number;
  factors: Factor[];
  caveats: string[];
  layers?: LayerSummary[];
}

export interface AlertPayload {
  session_id: string;
  score: number;
  band: Band;
  action: string;
  headline: string;
  factors: Factor[];
  profile: string;
  created_at: number;
}

export interface SessionSummary {
  session_id: string;
  profile: string;
  language: string;
  identity: string | null;
  open: boolean;
  score: number;
  band: Band;
  created_at: number;
  last_activity: number;
  stats: {
    windows_analyzed?: number;
    audio_seconds?: number;
    mean_latency_ms?: number;
    peak_score?: number;
    alerts_raised?: number;
  };
  metadata?: Record<string, unknown>;
}

export interface CallReport {
  session_id: string;
  verdict: string;
  final_score: number;
  peak_score: number;
  band: Band;
  duration_seconds: number;
  stats: Record<string, number>;
  top_factors: Array<{ code: string; label: string; layer: string; count: number;
                       mean_contribution: number }>;
}

export interface Health {
  status: string;
  version: string;
  environment: string;
  model_loaded: boolean;
  degraded: string[];
  retention: { mode: string; banner: string; keeps_raw_audio: boolean };
  sessions: {
    active_sessions: number;
    closed_sessions: number;
    capacity: number;
    windows_analyzed: number;
    mean_latency_ms: number;
    alerting_sessions: number;
  };
  uptime_seconds: number;
}

/** Everything the dashboard tracks for one call, assembled from the event stream. */
export interface TrackedCall {
  sessionId: string;
  score: number;
  band: Band;
  headline: string;
  action: string;
  profile: string;
  factors: Factor[];
  layers: LayerSummary[];
  caveats: string[];
  history: Array<{ t: number; score: number; band: Band }>;
  latencyMs: number;
  elapsed: number;
  windows: number;
  provisional: boolean;
  alerted: boolean;
  closed: boolean;
  verdict?: string;
  lastUpdate: number;
}

export type SocketState = 'connecting' | 'open' | 'closed' | 'error';
