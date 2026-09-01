# PRD → Implementation traceability (SIH26104)

Every requirement from the problem statement, mapped to the code that satisfies it.
`✔` = implemented and tested, `◐` = implemented, demo-scope, `—` = deliberately out of scope.

## Functional requirements

| ID | Requirement | Status | Where |
|---|---|---|---|
| FR-1 | Ingest live/streamed audio via a defined API | ✔ | `api/routes_stream.py` (WebSocket `/v1/stream`), `api/routes_analyze.py` (REST batch + file upload), `audio/stream.py` (chunker) |
| FR-2 | Extract acoustic/spectral features, detect synthesis artifacts | ✔ | `features/spectral.py`, `features/artifacts.py`, `models/acoustic.py` |
| FR-3 | Analyse prosodic features (pitch, rhythm, pauses) | ✔ | `features/prosody.py`, `audio/vad.py`, `models/prosodic.py` |
| FR-4 | Continuously updating impersonation risk score 0–100 % | ✔ | `scoring/fusion.py`, `scoring/risk.py`, `pipeline/session.py` |
| FR-5 | Configurable threshold-based alerting | ✔ | `alerts/engine.py`, `config.py::RiskProfile`, `api/routes_admin.py` |
| FR-6 | Real-time UI dashboard | ✔ | `frontend/` (React) and `api/static/console.html` (zero-build) |
| FR-7 | Per-call risk report with confidence trail | ✔ | `storage/repository.py`, `GET /v1/sessions/{id}/report` |
| FR-8 | REST API for third-party integration | ✔ | `api/routes_analyze.py`, `api/routes_sessions.py`, `api/routes_integrations.py`, `sdk/` |
| FR-9 | 2–3 Indian languages/accents in the demo | ◐ | `features/*` are language-agnostic by design; `ml/datasets/build.py` synthesises Hindi/English/Tamil-profiled prosody bands; `config.py::LANGUAGE_PROFILES` holds per-language prosody priors. Real multilingual accuracy needs real corpora — see `docs/MODEL_CARD.md`. |
| FR-10 | Admin panel to configure thresholds per use case | ✔ | `api/routes_admin.py`, dashboard "Policy" tab |
| FR-11 | Anonymised / feature-only audio logging | ✔ | `privacy/anonymize.py`, `privacy/retention.py`, default `RETENTION_MODE=features_only` |

## Non-functional requirements

| Category | Target | Status | Where / evidence |
|---|---|---|---|
| Latency | risk update in 2–5 s | ✔ | 0.5 s hop; `scripts/bench_latency.py` reports per-window cost |
| Privacy | no raw audio persisted by default | ✔ | `privacy/retention.py`, audit rows store features + SHA-256 only |
| Scalability | stateless, containerised | ✔ | `deploy/`; only `SessionManager` is stateful, behind a repository interface |
| Accuracy | report EER | ✔ | `ml/evaluate.py` emits Accuracy / AUC / EER / DET points to `ml/artifacts/metrics.json` |
| Security | TLS transport, authenticated API | ✔ | bearer/API-key auth in `api/deps.py`; TLS terminated at the reverse proxy (`deploy/`) |
| Explainability | top contributing factors | ✔ | `scoring/explain.py`; every score carries ranked `factors[]` |
| Localization | multilingual/accented Indian speech | ◐ | language-agnostic features + per-language prosody priors (`config.py`) |

## Architecture sections of the PS

| PS §  | Item | Where |
|---|---|---|
| 6.1 | Multi-layer voice authenticity analysis | `features/`, `models/` — L1 spectral, L2 prosody, L3 cross-session |
| 6.2 | Real-time risk scoring engine | `scoring/risk.py` with context enrichment from call metadata |
| 6.3 | Alerting & user-interaction layer | `alerts/` (channels: websocket, webhook, email, sms) + dashboards |
| 6.4 | Privacy & compliance module | `privacy/` + edge-inference path (pure NumPy feature/model stack) |
| 6.5 | Platform & integration APIs | `api/`, `sdk/python`, `sdk/js`, mock core-banking flow |

## Out of scope (stated in the PRD)

| Item | Note |
|---|---|
| Carrier-grade telecom integration | simulated with a WebRTC/file call bridge |
| Video deepfake detection | audio only |
| Bank-wide voice-biometric enrolment | small demo enrolment store (`storage/repository.py`) |
