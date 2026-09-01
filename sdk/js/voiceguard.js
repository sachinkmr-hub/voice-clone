/**
 * VoiceGuard JavaScript SDK.
 *
 * Zero dependencies. Runs in a browser (fetch + WebSocket are built in) and in Node 18+
 * (same globals). Ships as an ES module with a UMD-ish global fallback so a contact-centre
 * page can drop it in with a plain <script> tag — which is how most of them will integrate.
 *
 *   import { VoiceGuardClient } from './voiceguard.js';
 *
 *   const client = new VoiceGuardClient({ baseUrl: 'http://localhost:8000' });
 *   const result = await client.analyzeFile(fileInput.files[0], { profile: 'wire_transfer' });
 *   if (result.isHighRisk) showWarning(result.headline, result.action);
 */

const HIGH_RISK_BANDS = ['HIGH', 'CRITICAL'];
const INCONCLUSIVE_VERDICTS = ['insufficient_audio', 'inconclusive'];

export class VoiceGuardError extends Error {
  constructor(message, status, payload) {
    super(message);
    this.name = 'VoiceGuardError';
    this.status = status;
    this.payload = payload || {};
  }
}

/** One verdict, with the convenience accessors a call site actually wants. */
export class RiskResult {
  constructor(payload) {
    Object.assign(this, {
      sessionId: payload.session_id,
      score: payload.score ?? 0,
      band: payload.band ?? 'LOW',
      verdict: payload.verdict ?? '',
      action: payload.action ?? '',
      headline: payload.headline ?? '',
      peakScore: payload.peak_score ?? payload.score ?? 0,
      durationSeconds: payload.duration_seconds ?? 0,
      windowsAnalyzed: payload.windows_analyzed ?? 0,
      meanLatencyMs: payload.mean_latency_ms ?? payload.latency_ms ?? 0,
      factors: payload.factors ?? [],
      caveats: payload.caveats ?? [],
      layers: payload.layers ?? [],
      timeline: payload.timeline ?? [],
      raw: payload,
    });
  }

  get isHighRisk() { return HIGH_RISK_BANDS.includes(this.band); }
  get isSynthetic() { return ['likely_synthetic', 'suspicious'].includes(this.verdict); }

  /** True when there was not enough speech to decide.
   *  Check this before treating a low score as a pass: "we could not tell" and
   *  "it was genuine" are different answers and only one is safe to act on. */
  get inconclusive() { return INCONCLUSIVE_VERDICTS.includes(this.verdict); }

  get topFactor() { return this.factors[0] ?? null; }

  explain() {
    const lines = [this.headline || `Risk ${Math.round(this.score)}/100 (${this.band})`];
    for (const factor of this.factors.slice(0, 4)) {
      lines.push(`  - ${factor.label}: ${factor.detail}`);
    }
    for (const caveat of this.caveats) lines.push(`  ! ${caveat}`);
    if (this.action) lines.push(`  -> ${this.action}`);
    return lines.join('\n');
  }
}

export class VoiceGuardClient {
  constructor({ baseUrl = 'http://localhost:8000', apiKey = null, consent = null,
                timeoutMs = 30000 } = {}) {
    this.baseUrl = baseUrl.replace(/\/$/, '');
    this.apiKey = apiKey;
    this.consent = consent;
    this.timeoutMs = timeoutMs;
  }

  get headers() {
    const headers = {};
    if (this.apiKey) headers['X-API-Key'] = this.apiKey;
    if (this.consent) headers['X-Consent'] = this.consent;
    return headers;
  }

  async #request(method, path, { json, body, headers } = {}) {
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), this.timeoutMs);
    let response;
    try {
      response = await fetch(this.baseUrl + path, {
        method,
        signal: controller.signal,
        headers: { ...this.headers, ...(json ? { 'Content-Type': 'application/json' } : {}), ...headers },
        body: json ? JSON.stringify(json) : body,
      });
    } catch (error) {
      throw new VoiceGuardError(`could not reach ${this.baseUrl}${path}: ${error.message}`);
    } finally {
      clearTimeout(timer);
    }

    if (!response.ok) {
      let payload = {};
      let detail = response.statusText;
      try { payload = await response.json(); detail = payload.detail ?? detail; } catch { /* text body */ }
      throw new VoiceGuardError(`${response.status}: ${detail}`, response.status, payload);
    }
    return response.status === 204 ? null : response.json();
  }

  /* ------------------------------------------------------------------ analysis */

  /** Analyse a File/Blob — what a browser upload widget has. */
  async analyzeFile(file, options = {}) {
    const form = new FormData();
    form.append('file', file, file.name ?? 'call.wav');
    form.append('language', options.language ?? 'auto');
    form.append('profile', options.profile ?? 'default');
    if (options.transactionAmount) form.append('transaction_amount', String(options.transactionAmount));
    if (options.claimedRole) form.append('claimed_role', options.claimedRole);
    if (options.transcript) form.append('transcript', options.transcript);
    if (options.identity) form.append('identity', options.identity);
    if (options.knownContact !== undefined) form.append('known_contact', String(options.knownContact));
    if (options.callerIdVerified !== undefined) form.append('caller_id_verified', String(options.callerIdVerified));
    if (options.verbose) form.append('verbose', 'true');

    return new RiskResult(await this.#request('POST', '/v1/analyze/file', { body: form }));
  }

  /** Analyse raw bytes (ArrayBuffer / Uint8Array). */
  async analyzeBytes(buffer, options = {}) {
    const bytes = buffer instanceof Uint8Array ? buffer : new Uint8Array(buffer);
    // Chunked, because String.fromCharCode(...hugeArray) overflows the call stack.
    let binary = '';
    for (let i = 0; i < bytes.length; i += 8192) {
      binary += String.fromCharCode.apply(null, bytes.subarray(i, i + 8192));
    }
    return new RiskResult(await this.#request('POST', '/v1/analyze', {
      json: {
        audio_base64: btoa(binary),
        encoding: options.encoding ?? 'auto',
        sample_rate: options.sampleRate,
        language: options.language ?? 'auto',
        profile: options.profile ?? 'default',
        identity: options.identity,
        call_context: options.callContext,
        verbose: Boolean(options.verbose),
      },
    }));
  }

  /* ------------------------------------------------------------------ sessions */

  async createSession(options = {}) {
    const payload = await this.#request('POST', '/v1/sessions', {
      json: {
        profile: options.profile ?? 'default',
        language: options.language ?? 'auto',
        identity: options.identity,
        call_context: options.callContext,
        metadata: options.metadata ?? {},
      },
    });
    return payload.session_id;
  }

  getReport(sessionId, includeTrail = true) {
    return this.#request('GET', `/v1/sessions/${encodeURIComponent(sessionId)}/report?include_trail=${includeTrail}`);
  }

  closeSession(sessionId) {
    return this.#request('POST', `/v1/sessions/${encodeURIComponent(sessionId)}/close`);
  }

  /** Right to erasure — removes every stored row for this call. */
  deleteSession(sessionId) {
    return this.#request('DELETE', `/v1/sessions/${encodeURIComponent(sessionId)}`);
  }

  listSessions({ includeClosed = false, limit = 50 } = {}) {
    return this.#request('GET', `/v1/sessions?include_closed=${includeClosed}&limit=${limit}`);
  }

  /* --------------------------------------------------------------- integration */

  async requestApproval(sessionId, amount, options = {}) {
    const payload = await this.#request('POST', '/v1/integrations/bank/approval', {
      json: {
        session_id: sessionId, amount,
        currency: options.currency ?? 'INR',
        beneficiary: options.beneficiary ?? '',
        reference: options.reference ?? '',
        initiated_by: options.initiatedBy ?? '',
        profile: options.profile ?? 'wire_transfer',
      },
    });
    return {
      ...payload,
      allowed: payload.decision === 'allow',
      blocked: payload.decision === 'block',
      needsStepUp: payload.decision === 'step_up',
    };
  }

  health() { return this.#request('GET', '/v1/health'); }
  profiles() { return this.#request('GET', '/v1/admin/profiles'); }

  /* ----------------------------------------------------------------- streaming */

  /** Open a live streaming session. See {@link VoiceGuardStream}. */
  stream(options = {}) {
    return new VoiceGuardStream(this, options);
  }
}

/**
 * Live call streaming.
 *
 *   const stream = client.stream({ profile: 'wire_transfer' });
 *   stream.onRisk = (risk) => gauge.update(risk);
 *   await stream.start();
 *   stream.send(pcm16Chunk);
 *   const report = await stream.stop();
 */
export class VoiceGuardStream {
  constructor(client, { profile = 'default', language = 'auto', identity = null,
                        callContext = null, encoding = 'pcm16', sampleRate = 16000 } = {}) {
    this.client = client;
    this.options = { profile, language, identity, callContext, encoding, sampleRate };
    this.sessionId = null;
    this.socket = null;
    this.onRisk = null;
    this.onAlert = null;
    this.onError = null;
    this._finalResolve = null;
  }

  #url() {
    const base = this.client.baseUrl.replace(/^http/, 'ws');
    const query = this.client.apiKey ? `?api_key=${encodeURIComponent(this.client.apiKey)}` : '';
    return `${base}/v1/stream${query}`;
  }

  start() {
    return new Promise((resolve, reject) => {
      this.socket = new WebSocket(this.#url());
      this.socket.binaryType = 'arraybuffer';

      this.socket.onopen = () => {
        const start = {
          action: 'start', profile: this.options.profile, language: this.options.language,
          encoding: this.options.encoding, sample_rate: this.options.sampleRate,
        };
        if (this.options.identity) start.identity = this.options.identity;
        if (this.options.callContext) start.call_context = this.options.callContext;
        this.socket.send(JSON.stringify(start));
      };

      this.socket.onmessage = (event) => {
        const message = JSON.parse(event.data);
        switch (message.type) {
          case 'started':
            this.sessionId = message.session_id;
            resolve(message);
            break;
          case 'risk':
            this.onRisk?.(new RiskResult(message));
            break;
          case 'alert':
            this.onAlert?.(message);
            break;
          case 'final':
            this._finalResolve?.(message.report);
            break;
          case 'error':
            this.onError?.(new VoiceGuardError(message.error));
            break;
          default:
            break;
        }
      };

      this.socket.onerror = () => {
        const error = new VoiceGuardError('websocket error');
        this.onError?.(error);
        reject(error);
      };
    });
  }

  /** Push audio. `chunk` is an ArrayBuffer / TypedArray of PCM16 at the declared rate. */
  send(chunk) {
    if (!this.socket || this.socket.readyState !== WebSocket.OPEN) return false;
    this.socket.send(chunk);
    return true;
  }

  /** Stop the call and resolve with its report. */
  stop() {
    if (!this.socket) return Promise.resolve(null);
    return new Promise((resolve) => {
      this._finalResolve = resolve;
      try {
        this.socket.send(JSON.stringify({ action: 'stop' }));
      } catch {
        resolve(null);
      }
      // Never hang a caller's await on a socket that died mid-close.
      setTimeout(() => resolve(null), 6000);
    }).finally(() => {
      try { this.socket?.close(); } catch { /* already closed */ }
      this.socket = null;
    });
  }
}

/** Downsample Float32 audio to 16 kHz PCM16, the format the API expects.
 *  Averages over each source span rather than decimating: naive decimation folds aliases
 *  straight into the band the detector reads. */
export function toPcm16(float32, inputRate = 48000, targetRate = 16000) {
  let samples = float32;
  if (inputRate !== targetRate) {
    const ratio = inputRate / targetRate;
    const out = new Float32Array(Math.floor(float32.length / ratio));
    for (let i = 0; i < out.length; i++) {
      const start = Math.floor(i * ratio);
      const end = Math.min(float32.length, Math.floor((i + 1) * ratio));
      let sum = 0;
      for (let j = start; j < end; j++) sum += float32[j];
      out[i] = end > start ? sum / (end - start) : 0;
    }
    samples = out;
  }
  const buffer = new ArrayBuffer(samples.length * 2);
  const view = new DataView(buffer);
  for (let i = 0; i < samples.length; i++) {
    const s = Math.max(-1, Math.min(1, samples[i]));
    view.setInt16(i * 2, s < 0 ? s * 0x8000 : s * 0x7fff, true);
  }
  return buffer;
}

// Plain <script> fallback for pages that cannot use modules.
if (typeof window !== 'undefined') {
  window.VoiceGuard = { VoiceGuardClient, VoiceGuardStream, RiskResult, VoiceGuardError, toPcm16 };
}
