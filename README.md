# VoiceGuard — Real-Time AI Voice-Clone & Impersonation Detection

**SIH 2026 · PS SIH26104 · Blockchain & Cybersecurity · AICTE**

VoiceGuard is an end-to-end, privacy-preserving framework that listens to a live call,
analyses the speech **while the conversation is still in progress**, and emits a
continuously-updating **impersonation risk score (0–100)** with human-readable reasons —
then blocks the transaction before the money leaves.

```
  caller audio ──► chunker ──► ┌──────────────────────────────┐
   (upload /       (1.0 s win,  │  L1  spectral / vocoder      │
    mic / SIP)      0.5 s hop)  │  L2  prosody & micro-variation│──► fusion ──► risk 0-100
                                │  L3  cross-session speaker   │      + calibration   │
                                │  L4  call context            │                      │
                                └──────────────────────────────┘                      ▼
                                                          allow · step-up · BLOCK the transfer
```

| Measured | Result |
|---|---|
| End-to-end latency | **0.62 s** (NFR: < 5 s) |
| Same speaker — real voice vs a clone of it | **15/100 LOW** vs **79/100 CRITICAL** |
| Compute cost per call | **≈ ₹0.05**, CPU only |
| Tests | **156 passing** |

---

## 1. Quick start (60 seconds, no GPU, no dataset, no npm)

```bash
git clone https://github.com/sachinkmr-hub/voice-clone && cd voice-clone
python3 -m venv .venv && source .venv/bin/activate
pip install -r backend/requirements.txt
make run
```

Open **<http://127.0.0.1:8000/console>** and click *"Cloned CFO demanding an urgent wire"*.
No files needed — the server generates labelled demo audio on demand.

```bash
make demo-data     # build a bona-fide vs synthetic corpus (~20 s)
make train         # train + evaluate; prints Accuracy / AUC / EER
make test          # 156 tests, ~30 s
docker compose -f deploy/docker-compose.yml up --build   # whole stack
```

Also served by the running app: **`/pitch`** (the judging deck) · **`/docs`** (OpenAPI).

## 2. Three ways in, one engine

A post-hoc "upload a file" analyser cannot satisfy this problem statement — it asks for
detection *during live calls*, *before sensitive actions are taken*. So all three paths run
through the **same** chunker, windows and model:

| Path | Endpoint | Who it is for |
|---|---|---|
| **Upload a recording** | `POST /v1/analyze/file` | The front door. Zero permissions, works offline. |
| **Live call** | `WS /v1/stream` | Browser mic, WebRTC bridge, SIP tap. New score every 0.5 s. |
| **Approval gate** | `POST /v1/integrations/bank/approval` | Core banking, before releasing funds. |

## 3. What is in this repository

| Path | What it is |
|---|---|
| `backend/voiceguard/audio` | Decoding, resampling, energy+spectral VAD, streaming chunker |
| `backend/voiceguard/features` | 123 features: STFT/MFCC, spectral statistics, phase & modulation, F0/jitter/shimmer, vocoder-artifact probes, speaker embedding |
| `backend/voiceguard/models` | Four detection layers, rule calibration, registry with graceful degradation |
| `backend/voiceguard/scoring` | Fusion, calibration, risk policy, explanations |
| `backend/voiceguard/alerts` | Threshold engine + channels (WebSocket, webhook, email, SMS) |
| `backend/voiceguard/privacy` | Retention enforcement, anonymisation, transcript redaction |
| `backend/voiceguard/api` | FastAPI: REST + WebSocket + admin + mock bank integration |
| `backend/voiceguard/api/static` | **Zero-build web console** and the pitch deck — no npm required |
| `frontend/` | React + Vite operations dashboard (the fleet view) |
| `ml/` | Corpus builder, training, evaluation, leakage detection |
| `sdk/` | Python and JavaScript client SDKs |
| `deploy/` | Dockerfiles, Compose, nginx |
| `scripts/` | Latency benchmark, demo call driver, on-device inference |

## 4. Documentation

| Document | What it answers |
|---|---|
| [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) | How the four layers, fusion and streaming actually work |
| [`docs/MODEL_CARD.md`](docs/MODEL_CARD.md) | **Read this before quoting any number.** What the model was trained on, why the results are too good, seven known failure modes |
| [`docs/PRD_TRACEABILITY.md`](docs/PRD_TRACEABILITY.md) | Every FR/NFR mapped to the code that satisfies it |
| [`docs/FEASIBILITY.md`](docs/FEASIBILITY.md) | SMART breakdown, measured metrics, timeline, risk register |
| [`docs/MARKET.md`](docs/MARKET.md) | Does this exist already? What is missing, and where we sit |
| [`docs/SUSTAINABILITY.md`](docs/SUSTAINABILITY.md) | Unit economics and carbon, derived from one measured number |
| [`docs/IP.md`](docs/IP.md) | What is genuinely ours, what is prior art, what we gave away |
| [`docs/PRIVACY.md`](docs/PRIVACY.md) | Retention modes, anonymisation, the edge path |
| [`docs/DEMO_SCRIPT.md`](docs/DEMO_SCRIPT.md) | The six-minute demo, beat by beat |

## 5. Design principles

1. **Near-real-time by construction.** Every layer is streaming-friendly. No layer needs
   the call to end.
2. **Never a naked number.** Every score ships with ranked evidence in plain language, the
   counter-evidence, and its caveats.
3. **Abstention beats guessing.** A layer that cannot run reports `confidence = 0` and is
   removed from *both sides* of the fusion — it never votes "genuine" by default. The
   interface shows "abstained", which is a different answer from "it was fine".
4. **Policy is separable from the model.** Call context moves alerting *thresholds*, never
   the probability, so the same audio always scores the same and the audit trail holds up.
5. **Degrade, never crash.** Missing model, corrupt artifact, no torch — each downgrades one
   layer, reports why through `/v1/health`, and keeps serving.
6. **Privacy first.** Features-only by default; the TTL sweeper hard-deletes; the on-device
   path is a runnable script, not a claim.
7. **Honest about limits.** The model card says the shipped numbers are too good to be real,
   and CI fails the build on corpus leakage.

## 6. Results

Bootstrap corpus, **speaker-disjoint** splits, three vocoder families:

| Evaluation | Accuracy | AUC | EER |
|---|---|---|---|
| Classifier, window level | 0.974 | 0.998 | 2.65 % |
| Classifier, utterance level | 0.983 | 0.999 | 1.67 % |
| Full four-layer stack | 1.000 | 1.000 | 0.00 % |
| False positives on genuine audio | — | — | **1.7 %** @ 0.50 |

Unseen-vocoder evaluation (each family held out entirely, then tested only on it): 0.00 %
EER for all three, detection 95–100 %.

Latency on one CPU core: **p95 124 ms** per window against a 500 ms budget — 4× headroom.

> ⚠️ **These numbers are on simulated audio and are too good to be real.** They demonstrate
> the pipeline, not accuracy against ElevenLabs or RVC. `ml/train.py` reads ASVspoof and
> In-the-Wild layouts unchanged. See [`docs/MODEL_CARD.md`](docs/MODEL_CARD.md) §2.

## 7. Where each judging criterion is answered

| Criterion | Where |
|---|---|
| Relevance to the problem statement | [`docs/PRD_TRACEABILITY.md`](docs/PRD_TRACEABILITY.md) — every FR/NFR → code |
| Does it exist in market/industry? | [`docs/MARKET.md`](docs/MARKET.md) §1–2 — yes, partly; here is the gap |
| Feasibility (SMART) | [`docs/FEASIBILITY.md`](docs/FEASIBILITY.md) — each letter, with measured numbers |
| Timeline | [`docs/FEASIBILITY.md`](docs/FEASIBILITY.md) §5 |
| Usability | `/console` (3 clicks to a verdict) and the React dashboard |
| Scalability | [`docs/SUSTAINABILITY.md`](docs/SUSTAINABILITY.md) §1.1, `deploy/docker-compose.yml` |
| Economic sustainability | [`docs/SUSTAINABILITY.md`](docs/SUSTAINABILITY.md) §1 — ₹0.05/call, working shown |
| Environmental sustainability | [`docs/SUSTAINABILITY.md`](docs/SUSTAINABILITY.md) §2 — 0.05 g CO₂e/call |
| Intellectual property | [`docs/IP.md`](docs/IP.md) — one defensible method claim, prior art stated first |
| Presentation | `/pitch` — 10 slides, served by the system it describes |

## 8. Licence

MIT — see [`LICENSE`](LICENSE). Chosen deliberately: a security control that banks and
government departments can *audit* is more trustworthy than one they cannot.
