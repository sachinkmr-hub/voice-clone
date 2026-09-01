# Feasibility (SMART) — VoiceGuard, SIH26104

Every number in this document is either measured by a command in this repository or
derived from one by arithmetic that is shown. Where a figure is an assumption, it is
labelled **[assumption]** and the working is given so a judge can substitute their own.

---

## 1. Specific — what exactly is being built

Not "an AI system for voice security". Precisely this:

> A service that consumes a live call's audio in 1-second windows, emits a calibrated
> impersonation risk score (0–100) with ranked evidence every 0.5 seconds, and exposes a
> decision endpoint that a core-banking system calls before releasing funds.

The scope boundary is equally specific — these are **out**, and stated as out:

| Out of scope | Why |
|---|---|
| Video deepfake detection | Audio-only problem statement |
| Replay-attack detection (a recording of a real person) | Different problem; layers 1–2 cannot see it, and we say so in the model card |
| Speaker identification against an open population | We do closed-set enrolment comparison only |
| Adversarially-robust detection | Untested; listed as a known weakness |
| Carrier-grade SS7/IMS integration | Simulated with a WebSocket/SIP-shaped bridge |

## 2. Measurable — the numbers, and how to reproduce them

| Metric | Result | Reproduce with |
|---|---|---|
| Per-window latency, p95 | **124 ms** | `python scripts/bench_latency.py` |
| End-to-end user-visible latency | **0.62 s** (hop 0.5 s + p95) | same |
| NFR target | < 5 s | — |
| Concurrent calls per CPU core | **~4** | same (500 ms budget ÷ 124 ms p95) |
| Detection, utterance level (bootstrap corpus) | acc 0.983 · AUC 0.999 · **EER 1.67 %** | `make train` |
| Full four-layer stack, utterance level | acc 1.000 · **EER 0.00 %** | `python -m ml.evaluate --full-stack` |
| False-positive rate on genuine audio | **1.7 %** at threshold 0.50 | `make train` |
| Unseen-vocoder EER (each family held out) | 0.00 % / 0.00 % / 0.00 % | `make train` |
| Edge inference speed | **27 % of real time**, no GPU | `python scripts/edge_infer.py --demo cloned` |
| Edge uplink payload | **252 bytes**, no audio, no features | same |
| Test suite | **156 tests**, ~30 s | `make test` |
| Dashboard bundle | 50 kB gzipped | `npm run build` |

**The accuracy numbers are on simulated audio and are too good to be real.** That is
stated in `docs/MODEL_CARD.md` §2 rather than buried, along with what we did to keep them
honest and why an EER of 0 % on a real corpus would be a red flag. Published state of the
art on ASVspoof 2021 DF is 2–7 % EER; treat anything we report below that as unproven
until reproduced on real data.

## 3. Attainable — what is already done

This is not a plan. The following is running and committed:

- ✅ 123-feature DSP stack (spectral, prosodic, artifact, embedding) — pure NumPy/SciPy
- ✅ Four detection layers with an abstention contract, fusion, calibration, explanations
- ✅ FastAPI service: REST + WebSocket streaming + admin + audit + privacy enforcement
- ✅ Two front ends: zero-build console (upload + live mic + approval) and a React ops dashboard
- ✅ Training/evaluation pipeline with speaker-disjoint splits and automated leakage detection
- ✅ Python and JavaScript SDKs
- ✅ Docker Compose stack, CI on three Python versions
- ✅ 156 tests, all passing

Nothing in the demo is a mock: the "bank blocks the transfer" step calls the real endpoint,
which reads the real session, which was scored by the real model.

## 4. Realistic — the honest constraints

**What we know is weak.**

| Constraint | Reality | Mitigation |
|---|---|---|
| Shipped weights trained on simulated audio | The largest limitation of this submission | `ml/train.py` reads ASVspoof and In-the-Wild layouts unchanged; retraining is one command |
| No real Indian-accent spoof corpus | We could not source one in the time available | Features are language-agnostic; per-language prosody priors are in `config.py`; honest ◐ in the traceability matrix |
| Speaker embedding is classical, not ECAPA | Compressed distances, weaker than SOTA | Used only for closed-set enrolment and drift; WavLM backend switches on if torch is installed |
| Adversarial audio untested | Known to defeat this model class | Listed as an unmitigated weakness in the model card |
| No fairness audit on real data across age/gender/accent | Not possible without real data | Required before any real deployment; its absence is stated, not discovered later |

**What makes it realistic anyway.** The architecture assumes it will be wrong sometimes:
layers abstain instead of guessing, context moves thresholds rather than scores, the bank
gate refuses to clear on thin evidence, and every verdict carries its caveats. A detector
that is 85 % accurate and says so is deployable. One that is 99 % accurate on paper and
silent about its failure modes is not.

## 5. Timeline

**Already delivered (this repository).** Detection engine, service, both front ends, ML
pipeline, SDKs, deployment, CI, docs.

| Phase | Window | Deliverable | Done when |
|---|---|---|---|
| **Now → submission (20 Sep 2026)** | ~3 weeks | Retrain on ASVspoof 2019 LA + In-the-Wild; publish real EER next to the simulated one | `ml/artifacts/metrics.json` regenerated from a real corpus |
| | | Record 40–60 Indian-accent samples across 4 languages; clone them with two public TTS tools; measure | A second metrics file on real Indian audio |
| | | Rehearse the 6-minute demo (`docs/DEMO_SCRIPT.md`) | Timed under 6:00 with no live-failure path |
| **Post-SIH, 0–3 months** | | Pilot with one bank contact centre, shadow mode (score logged, never blocking) | 10k calls scored, FPR measured on real traffic |
| | | Swap classical embedding for ECAPA/WavLM; fairness audit across accent, age, gender | Audit published with per-group FPR |
| **3–9 months** | | Redis-backed session store (removes sticky routing); SIP/WebRTC bridge for a real PBX | Horizontal scale demonstrated at 500 concurrent calls |
| | | Adversarial robustness evaluation and hardening | Attack success rate published |
| **9–18 months** | | Telecom-side pilot; on-device SDK for Android; DPDP compliance review | Production deployment with a named customer |

The near-term items are sized to what two to four students can do alongside coursework.
The pilot items require a partner organisation, and we say so rather than pretending the
timeline is under our control.

## 6. Risk register

| Risk | Likelihood | Impact | Mitigation in the repo |
|---|---|---|---|
| Model does not transfer to real TTS | **High** | Severe | Multi-vocoder training, unseen-family evaluation, physics-grounded heuristic fallback that needs no training at all |
| Corpus artifacts inflate accuracy | High (it already happened once) | Severe | `detect_leakage` runs on every training run and fails CI; the DC-offset bug it caught is documented in the model card |
| False positives erode trust | Medium | High | One-sided prosodic rules, per-use-case thresholds, spectrum-with-action instead of a binary block, FPR reported at four thresholds |
| Latency fails under load | Low | Medium | 4× headroom measured; stateless service; capacity ceiling enforced with a 503 rather than silent degradation |
| Privacy objection blocks a pilot | Medium | High | Features-only default, TTL sweeper that hard-deletes, edge path demonstrated, right-to-erasure endpoint |
| A judge asks "which vocoder does this fail on?" | Certain | — | The per-family table is printed by `make train` and reproduced in the model card |
