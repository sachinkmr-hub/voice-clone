# VoiceGuard — Real-Time AI Voice-Clone & Impersonation Detection

**SIH 2026 · PS SIH26104 · Blockchain & Cybersecurity · AICTE**

VoiceGuard is an end-to-end, privacy-preserving framework that listens to a live call,
analyses the speech **while the conversation is still in progress**, and emits a
continuously-updating **impersonation risk score (0–100)** with human-readable reasons —
so an agent, employee, or family member can act *before* money moves.

```
  caller audio ──► chunker ──► ┌──────────────────────────────┐
   (WebRTC/SIP/    (1.0 s win,  │  L1  spectral / vocoder      │
    file/mic)       0.5 s hop)  │  L2  prosody & micro-variation│──► fusion ──► risk 0-100
                                │  L3  cross-session speaker   │      + calibration   │
                                │  L4  call-context signals    │                      │
                                └──────────────────────────────┘                      │
                                                                                      ▼
                                            WebSocket ──► dashboard ──► alert / block / step-up auth
```

---

## 1. Why this exists

A 3-second sample of a CFO's voice from a webinar is enough for modern neural TTS to
produce a convincing clone. Caller ID, "does it sound like him?", and call-back policies
fail under time pressure. VoiceGuard adds a **machine second opinion** to every call.

## 2. What is in this repository

| Path | What it is |
|---|---|
| `backend/voiceguard/audio` | Decoding, resampling, energy+spectral VAD, streaming chunker |
| `backend/voiceguard/features` | STFT/MFCC/mel, spectral statistics, phase & modulation features, F0/jitter/shimmer prosody, vocoder-artifact probes, speaker embedding |
| `backend/voiceguard/models` | Acoustic detector, prosodic detector, speaker-consistency detector, model registry with graceful fallbacks |
| `backend/voiceguard/scoring` | Score fusion, calibration, risk engine with context enrichment, explainability |
| `backend/voiceguard/alerts` | Threshold engine + pluggable channels (WebSocket, webhook, email, SMS) |
| `backend/voiceguard/privacy` | Retention policy, feature-only logging, anonymisation |
| `backend/voiceguard/api` | FastAPI app: REST + WebSocket streaming + admin + mock bank integration |
| `ml/` | Corpus builder, training, evaluation (Accuracy / AUC / EER / DET) |
| `frontend/` | React + Vite real-time dashboard (live gauge, timeline, alerts, admin) |
| `backend/voiceguard/api/static` | Zero-build HTML console — full demo with **no npm required** |
| `sdk/` | Python and JavaScript client SDKs |
| `deploy/` | Dockerfiles + docker-compose for one-command bring-up |
| `docs/` | Architecture, API reference, privacy note, model card, demo script |

## 3. Quick start (60 seconds, no Node required)

```bash
git clone https://github.com/sachinkmr-hub/voice-clone && cd voice-clone
python3 -m venv .venv && source .venv/bin/activate
pip install -r backend/requirements.txt
make demo-data          # builds a small bona-fide vs synthetic corpus
make train              # fits the bootstrap detector, prints Accuracy/AUC/EER
make run                # http://127.0.0.1:8000/console
```

Open <http://127.0.0.1:8000/console>, press **Start streaming**, and feed it either the
microphone or one of the generated `data/corpus/*.wav` files. The gauge updates every
~0.5 s with the current risk, the contributing factors, and the recommended action.

With Docker:

```bash
docker compose -f deploy/docker-compose.yml up --build
```

## 4. Design principles

1. **Near-real-time by construction.** Every layer is streaming-friendly: features are
   computed on 1 s windows with 0.5 s hop, and the risk score is an exponentially-weighted
   fusion over the call so far. No layer needs the call to end.
2. **Never a naked number.** Every score ships with ranked contributing factors
   (`"high-frequency energy cliff at 7.8 kHz"`, `"pitch micro-variation 4.1σ below human range"`)
   and a recommended action. Judges, auditors and end-users all need the *why*.
3. **Degrade, never crash.** Torch missing? Trained model missing? The registry falls back
   to the calibrated DSP heuristic detector and marks its own confidence lower. The demo
   always runs.
4. **Privacy first.** Raw audio is never written to disk unless retention is explicitly
   enabled; the default audit trail stores feature vectors and hashes only.
5. **Honest about limits.** See `docs/MODEL_CARD.md` — we state exactly what the shipped
   model was trained on and where it will fail.

## 5. Status of each PRD requirement

See `docs/PRD_TRACEABILITY.md` for the full FR/NFR → code mapping.

## 6. Licence

MIT — see `LICENSE`.
