# VoiceGuard — Architecture

## 0. One-paragraph summary

Audio arrives as a stream of small chunks (WebSocket, REST batch, or file). A per-call
`CallSession` accumulates chunks into overlapping analysis windows, runs four independent
detection layers over each window, fuses their outputs into a single calibrated
probability, smooths that probability across the call, and publishes a risk event
containing the score, the ranked evidence, and the recommended action. Alerts fire when
the score crosses a configurable, use-case-specific threshold.

## 1. Layered detection

```
                       ┌──────────────────────── window (1.0 s, hop 0.5 s) ───────────────────────┐
                       │                                                                          │
  ┌────────────────────▼────────────────────┐   ┌──────────────────────────────────────────────┐
  │ L1 — Acoustic / spectral                │   │ L2 — Prosodic / behavioural                  │
  │  • mel + MFCC + Δ statistics            │   │  • F0 contour (autocorrelation pitch tracker)│
  │  • spectral centroid/rolloff/flatness   │   │  • jitter, shimmer, HNR                      │
  │  • high-band energy cliff (vocoder LPF) │   │  • pause count/length distribution           │
  │  • group-delay & phase-coherence stats  │   │  • speaking-rate and energy-envelope rhythm  │
  │  • modulation-spectrum depth            │   │  • micro-variation entropy                   │
  │  • silence noise-floor "digital zero"   │   │                                              │
  └────────────────────┬────────────────────┘   └───────────────────┬──────────────────────────┘
                       │                                            │
  ┌────────────────────▼────────────────────┐   ┌───────────────────▼──────────────────────────┐
  │ L3 — Cross-session speaker consistency  │   │ L4 — Call context                            │
  │  • lightweight speaker embedding        │   │  • origin / number reputation                │
  │  • cosine distance vs enrolled voice    │   │  • claimed identity vs enrolled identity     │
  │  • within-call embedding drift          │   │  • transaction amount & urgency language     │
  └────────────────────┬────────────────────┘   └───────────────────┬──────────────────────────┘
                       └──────────────► FUSION (calibrated) ◄───────┘
                                             │
                              risk 0-100 + factors + action
```

Each layer emits `LayerResult{score∈[0,1], confidence∈[0,1], factors[]}`. A layer that
cannot run (no enrolment for L3, no metadata for L4) reports `confidence=0` and is
excluded from the fusion weight normalisation rather than defaulting to "safe".

## 2. Streaming model

| Parameter | Default | Why |
|---|---|---|
| Sample rate | 16 kHz mono | telephony-compatible; all features defined at this rate |
| Window | 1.0 s | shortest window with a stable F0 contour |
| Hop | 0.5 s | 2 score updates/second → well inside the 5 s NFR |
| Warm-up | 3 windows | avoids a jumpy score in the first second |
| Smoothing | EWMA, α = 0.35, plus a max-of-recent guard | responsive but not twitchy |

Latency budget measured end-to-end (`scripts/bench_latency.py`): feature extraction is the
dominant term at ~15–35 ms per window on one CPU core, so the wall-clock latency is
essentially the hop size (0.5 s) plus transport.

## 3. Fusion and calibration

Raw layer scores are combined as a confidence-weighted logit average:

```
z = Σ wᵢ·cᵢ·logit(sᵢ) / Σ wᵢ·cᵢ            p = σ(a·z + b)
```

`a, b` are Platt-scaling parameters fitted on a held-out split so that "80" really means
"≈80 % of calls scoring here were synthetic in evaluation". Weights `wᵢ` are configurable
per deployment profile (a bank cares more about L3 because it has enrolment data; a
consumer app has only L1/L2).

## 4. Risk engine

The calibrated probability is mapped to a band using per-use-case thresholds:

| Band | Default range | Recommended action |
|---|---|---|
| `LOW` | 0–34 | proceed |
| `ELEVATED` | 35–59 | verify identity with a knowledge question |
| `HIGH` | 60–79 | call back on the registered number before acting |
| `CRITICAL` | 80–100 | block the transaction, escalate to supervisor |

Context multipliers (transaction amount, off-hours, unknown origin, urgency keywords)
adjust the band boundary, never the underlying probability — so the model output stays
auditable and the policy stays separable from the science.

## 5. Services and data flow

```
 client ──WS /v1/stream──► FastAPI ──► SessionManager ──► AnalysisPipeline ──► RiskEngine
                                │                                                │
                                ├──► AlertEngine ──► channels (ws/webhook/email/sms)
                                └──► AuditStore (SQLite; feature-only by default)

 client ──POST /v1/analyze──► same pipeline, single-shot
 dashboard ──WS /v1/dashboard──► fan-out of all live sessions
 core-banking ──POST /v1/integrations/bank/approval──► allow | step-up | block
```

The inference path is stateless apart from `SessionManager`, which holds per-call state in
memory keyed by `session_id`. Horizontal scaling therefore only needs sticky routing on
`session_id` (or an external store — the repository interface is already abstracted).

## 6. Privacy architecture

* **Default:** audio buffers live only inside the window ring-buffer and are overwritten;
  nothing touches disk. The audit record stores the feature vector, the score, the factors,
  and a SHA-256 of the audio chunk (for later dispute resolution) — not the audio.
* **Opt-in retention:** `RETENTION_MODE=raw_audio` with a TTL in seconds, enforced by a
  sweeper task.
* **Edge mode:** the entire `voiceguard.features` + `voiceguard.models` path is pure
  NumPy/SciPy and runs on-device; only the resulting score need leave the handset.

## 7. Failure and degradation matrix

| Missing | Behaviour |
|---|---|
| trained model artifact | heuristic detector, `model_id=heuristic-v1`, confidence ×0.8 |
| torch / transformers | classical features only, logged once at startup |
| speaker enrolment | L3 disabled (`confidence=0`), fusion renormalises |
| call metadata | L4 disabled, no context multiplier applied |
| silent / no-speech window | window skipped, score held, `speech_detected=false` |
