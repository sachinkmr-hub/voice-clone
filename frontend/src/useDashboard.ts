/** Dashboard state, driven by the `/v1/dashboard` WebSocket fan-out.
 *
 * The hook owns three things the components should not have to think about:
 *
 * 1. **Reconnection.** An ops screen left open overnight will lose its socket. Backoff is
 *    exponential and capped, and reconnecting replays the snapshot so nothing is missed.
 * 2. **Aggregation.** The server emits one event per analysis window per call. The hook
 *    folds those into a per-call record with a bounded score history.
 * 3. **Bounded memory.** A dashboard is a long-lived page: histories and the alert list
 *    are capped so a busy night cannot grow the tab until it is killed.
 */

import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import type {
  AlertPayload, Band, CallReport, Health, RiskEvent, SessionSummary, SocketState, TrackedCall,
} from './types';

const API_BASE = import.meta.env.VITE_API_BASE ?? '';
const MAX_HISTORY = 240;      // 2 minutes at 2 Hz
const MAX_ALERTS = 60;
const MAX_CLOSED = 40;
const RECONNECT_BASE_MS = 800;
const RECONNECT_MAX_MS = 15000;

function socketUrl(path: string): string {
  if (API_BASE) {
    return API_BASE.replace(/^http/, 'ws') + path;
  }
  const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
  return `${protocol}//${window.location.host}${path}`;
}

export async function apiGet<T>(path: string): Promise<T> {
  const response = await fetch(API_BASE + path);
  if (!response.ok) throw new Error(`${response.status} ${response.statusText}`);
  return response.json() as Promise<T>;
}

function emptyCall(sessionId: string): TrackedCall {
  return {
    sessionId, score: 0, band: 'LOW', headline: '', action: '', profile: 'default',
    factors: [], layers: [], caveats: [], history: [], latencyMs: 0, elapsed: 0,
    windows: 0, provisional: true, alerted: false, closed: false, lastUpdate: Date.now(),
  };
}

export function useDashboard() {
  const [calls, setCalls] = useState<Record<string, TrackedCall>>({});
  const [alerts, setAlerts] = useState<AlertPayload[]>([]);
  const [closed, setClosed] = useState<CallReport[]>([]);
  const [health, setHealth] = useState<Health | null>(null);
  const [socketState, setSocketState] = useState<SocketState>('connecting');
  const [selected, setSelected] = useState<string | null>(null);

  const socketRef = useRef<WebSocket | null>(null);
  const attemptRef = useRef(0);
  const timerRef = useRef<number | null>(null);
  const closedRef = useRef(false);

  const applyRisk = useCallback((event: RiskEvent) => {
    setCalls((current) => {
      const existing = current[event.session_id] ?? emptyCall(event.session_id);
      const history = [
        ...existing.history,
        { t: event.elapsed_seconds, score: event.score, band: event.band },
      ].slice(-MAX_HISTORY);

      return {
        ...current,
        [event.session_id]: {
          ...existing,
          score: event.score,
          band: event.band,
          headline: event.headline,
          action: event.action,
          profile: event.profile,
          factors: event.factors ?? [],
          layers: event.layers ?? existing.layers,
          caveats: event.caveats ?? [],
          history,
          latencyMs: event.latency_ms,
          elapsed: event.elapsed_seconds,
          windows: existing.windows + 1,
          provisional: event.provisional,
          lastUpdate: Date.now(),
        },
      };
    });
  }, []);

  const connect = useCallback(() => {
    if (closedRef.current) return;
    setSocketState('connecting');

    let socket: WebSocket;
    try {
      socket = new WebSocket(socketUrl('/v1/dashboard'));
    } catch {
      setSocketState('error');
      return;
    }
    socketRef.current = socket;

    socket.onopen = () => {
      attemptRef.current = 0;
      setSocketState('open');
    };

    socket.onmessage = (raw) => {
      let message: Record<string, unknown>;
      try {
        message = JSON.parse(raw.data as string);
      } catch {
        return;
      }

      switch (message.type) {
        case 'snapshot': {
          const sessions = (message.sessions ?? []) as SessionSummary[];
          setCalls((current) => {
            const next = { ...current };
            for (const session of sessions) {
              if (!session.open) continue;
              const existing = next[session.session_id] ?? emptyCall(session.session_id);
              next[session.session_id] = {
                ...existing,
                score: session.score,
                band: session.band,
                profile: session.profile,
                windows: session.stats?.windows_analyzed ?? existing.windows,
              };
            }
            return next;
          });
          setAlerts(((message.alerts ?? []) as Array<{ alert: AlertPayload }>)
            .map((row) => row.alert).filter(Boolean).slice(0, MAX_ALERTS));
          setHealth(message.health as Health);
          break;
        }
        case 'risk':
          applyRisk(message as unknown as RiskEvent);
          break;
        case 'alert': {
          const alert = (message.alert ?? message) as AlertPayload;
          setAlerts((current) => [alert, ...current].slice(0, MAX_ALERTS));
          setCalls((current) => {
            const existing = current[alert.session_id];
            return existing
              ? { ...current, [alert.session_id]: { ...existing, alerted: true } }
              : current;
          });
          break;
        }
        case 'session_started': {
          const session = message.session as SessionSummary;
          setCalls((current) => ({
            ...current,
            [session.session_id]: {
              ...emptyCall(session.session_id),
              profile: session.profile,
            },
          }));
          break;
        }
        case 'session_complete': {
          const report = message.session as CallReport;
          setClosed((current) => [report, ...current].slice(0, MAX_CLOSED));
          setCalls((current) => {
            const existing = current[report.session_id];
            if (!existing) return current;
            return {
              ...current,
              [report.session_id]: {
                ...existing, closed: true, verdict: report.verdict,
                score: report.final_score ?? existing.score,
              },
            };
          });
          break;
        }
        default:
          break;
      }
    };

    socket.onerror = () => setSocketState('error');

    socket.onclose = () => {
      setSocketState('closed');
      if (closedRef.current) return;
      // Exponential backoff, capped. An ops screen is left open for days; hammering a
      // restarting backend with reconnects helps nobody.
      const delay = Math.min(RECONNECT_MAX_MS, RECONNECT_BASE_MS * 2 ** attemptRef.current);
      attemptRef.current += 1;
      timerRef.current = window.setTimeout(connect, delay);
    };
  }, [applyRisk]);

  useEffect(() => {
    closedRef.current = false;
    connect();
    return () => {
      closedRef.current = true;
      if (timerRef.current) window.clearTimeout(timerRef.current);
      socketRef.current?.close();
    };
  }, [connect]);

  // Health is polled as well as pushed: the snapshot only arrives on (re)connect, and an
  // operator needs to see a degraded model without waiting for a socket bounce.
  useEffect(() => {
    let cancelled = false;
    const poll = async () => {
      try {
        const data = await apiGet<Health>('/v1/health');
        if (!cancelled) setHealth((current) => ({ ...(current ?? data), ...data }));
      } catch {
        /* the socket state already shows connectivity */
      }
    };
    poll();
    const handle = window.setInterval(poll, 10000);
    return () => { cancelled = true; window.clearInterval(handle); };
  }, []);

  // Retire calls that have gone quiet, so the board reflects what is actually live.
  useEffect(() => {
    const handle = window.setInterval(() => {
      const cutoff = Date.now() - 120000;
      setCalls((current) => {
        const next: Record<string, TrackedCall> = {};
        for (const [id, call] of Object.entries(current)) {
          if (!call.closed || call.lastUpdate > cutoff) next[id] = call;
        }
        return next;
      });
    }, 20000);
    return () => window.clearInterval(handle);
  }, []);

  const list = useMemo(() => {
    const rank: Record<Band, number> = { CRITICAL: 0, HIGH: 1, ELEVATED: 2, LOW: 3 };
    return Object.values(calls).sort((a, b) => {
      if (a.closed !== b.closed) return a.closed ? 1 : -1;
      if (rank[a.band] !== rank[b.band]) return rank[a.band] - rank[b.band];
      return b.score - a.score;
    });
  }, [calls]);

  const stats = useMemo(() => {
    const live = list.filter((c) => !c.closed);
    const latencies = live.map((c) => c.latencyMs).filter((v) => v > 0);
    return {
      live: live.length,
      atRisk: live.filter((c) => c.band === 'HIGH' || c.band === 'CRITICAL').length,
      alerts: alerts.length,
      meanLatency: latencies.length
        ? Math.round(latencies.reduce((a, b) => a + b, 0) / latencies.length)
        : 0,
    };
  }, [list, alerts]);

  return {
    calls: list, alerts, closed, health, socketState, stats,
    selected: selected ? calls[selected] ?? null : null,
    selectCall: setSelected,
    dismissAlerts: () => setAlerts([]),
  };
}
