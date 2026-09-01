# Market and competitive position

Judges ask, correctly: *does this already exist?* The honest answer is **yes, partly** —
and the interesting question is what is missing from what exists. This document says what
is out there, what it does not do, and where VoiceGuard sits.

> A note on numbers: this section contains **no fabricated market statistics**. Where a
> figure would normally go, we describe the mechanism instead. We do not have licensed
> analyst data, and inventing a "₹X crore TAM growing at Y % CAGR" would be worse than
> omitting it — a judge who knows the sector will spot a made-up number immediately, and a
> judge who does not is being misled.

---

## 1. Does the solution exist in the market?

**Audio deepfake detection exists as a category.** It is served by:

| Category | Examples of the type | What they do |
|---|---|---|
| Voice-biometric / anti-spoofing vendors | Established voice-biometrics companies serving bank contact centres | Speaker verification with a spoof-detection add-on, usually enrolment-first |
| Deepfake-detection startups | A growing set of AI-audio-forensics companies | Upload a file, get a "real/fake" verdict; some offer streaming |
| Cloud platform APIs | Major cloud speech services | Speech-to-text and speaker ID; spoof detection is largely absent or nascent |
| Academic / open source | ASVspoof challenge systems, AASIST, RawNet2, In-the-Wild benchmarks | State-of-the-art detection research, released as models and papers, not products |
| Telecom-side | STIR/SHAKEN, TRAI DLT, caller-ID reputation | Verify the *number*, not the *voice* |

So the technology is not novel in the abstract. **What is thin is the specific combination
this problem statement asks for.**

## 2. What is actually missing

| Gap | Why it matters for SIH26104 |
|---|---|
| **Post-hoc, not in-call** | Most accessible tools are upload-a-file forensics. The PS explicitly requires detection *"during live calls"* and *"before sensitive actions are taken"*. A verdict that arrives after the transfer is a forensics report, not a control. |
| **A score, not a decision** | Almost everything stops at "87 % fake". Nothing in the accessible tier closes the loop into "therefore this transaction does not complete". |
| **Opaque** | A bank agent cannot tell a customer "the model said so". Regulated environments need contributing factors, and automated-decision provisions (DPDP Act 2023 §, GDPR Art. 22-style) increasingly require them. |
| **Silence treated as innocence** | Ensemble detectors routinely let an unavailable check vote "genuine". That is the failure mode that makes them quietly useless in production, and it is invisible from the outside. |
| **Enrolment-first** | Voice-biometric vendors need an enrolled voiceprint. That works for a bank's own customers and fails completely for the CFO-impersonation and family-vishing cases, where the victim has no enrolment for the caller. |
| **Indian-language coverage** | Detection research is overwhelmingly English/Mandarin. Indian-language and code-mixed speech is under-represented in every public benchmark. |
| **Cost and deployability** | GPU-served transformer detectors are hard to justify per call for a mid-sized NBFC, and impossible to run on a handset. |

## 3. Where VoiceGuard sits

Not "better AI". A different set of engineering commitments:

| Commitment | Concretely |
|---|---|
| **In-call by construction** | 1 s windows, 0.5 s hop, score every half second. Measured 0.62 s end to end. Not a batch tool with streaming bolted on. |
| **Closes the loop** | `POST /v1/integrations/bank/approval` returns allow / step_up / block with its reason chain. The demo ends in a blocked transfer, not a number. |
| **Works with no enrolment** | Layers 1, 2 and 4 need nothing but the audio and the call metadata. Layer 3 strengthens the verdict when enrolment exists and *abstains* — visibly — when it does not. |
| **Abstention is a first-class state** | A layer that cannot run reports `confidence = 0` and is excluded from both numerator and denominator of the fusion. The UI shows "abstained", never a silent "genuine". |
| **Explainable to a non-expert** | Every score ships with ranked factors in plain language ("non-speech segments have no room tone"), the counter-evidence, and the caveats. |
| **Policy separated from model** | Context shifts alerting *thresholds*, never the probability, so the same audio always scores the same and the audit trail is defensible to a regulator. |
| **CPU-only, edge-capable** | ~5 paise per call; runs on a handset; `scripts/edge_infer.py` proves it rather than asserting it. |
| **Honest about limits** | A model card that says the shipped numbers are too good to be real, and an automated leakage detector that fails CI. This is a differentiator in a field where overclaiming is the norm. |

## 4. Who buys it, and why they cannot just build it

| Buyer | Why not build in-house |
|---|---|
| Bank / NBFC contact centre | The hard parts are calibration, false-positive control, and the audit trail — not the FFT. A bank has no DSP team and no spoof corpus. |
| Enterprise (finance/HR desks) | Needs an integration, not a research project. The SDKs are the product surface. |
| Telecom operator | Wants a per-subscriber network feature; the CPU-only footprint is what makes that arithmetic work at subscriber scale. |
| Government / law enforcement | Needs on-prem, no egress, and explainability for evidentiary use. The edge path and the factor trail are the requirements. |
| Consumer app | Needs it free and on-device. Only a CPU-only model can be that. |

## 5. Adoption path, in order of least resistance

1. **Shadow mode.** Score every call, block nothing, log everything. Costs the bank nothing
   and measures the real false-positive rate on their traffic — the number that actually
   decides whether this gets deployed.
2. **Advisory.** Show the agent the score and the reasons. Still no blocking.
3. **Step-up.** Above a threshold, require a call-back or a second factor. This is where
   fraud losses start falling.
4. **Blocking.** Only after the FPR is known and the thresholds are tuned per use case —
   which is exactly what the runtime-editable risk profiles exist for.

Skipping to step 4 is how this class of product gets switched off in its first month.

## 6. What would make us wrong

Stated plainly, because a judge will think of it anyway:

* If a major cloud provider ships real-time spoof detection as a checkbox in their speech
  API, the "does it exist" answer changes for the contact-centre segment overnight. The
  defensible ground would then be the on-prem, edge and explainability requirements.
* If synthesis quality improves to the point where the physical artifacts we measure are
  genuinely gone, layers 1–2 degrade and the product's centre of gravity has to move to
  layer 3 (enrolment) and layer 4 (context). The architecture already separates them, which
  is the point of the layering.
* Our accuracy claims are unproven on real data. If retraining on ASVspoof and In-the-Wild
  yields an EER far worse than published baselines, the honest response is to say so and
  fix the model — not to keep quoting the simulated numbers.
