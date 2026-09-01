# Intellectual property

Judging asks about "existence of intellectual property". This document separates what is
genuinely ours and potentially protectable, what is prior art we build on, and what we
have deliberately given away — because a claim of IP that dissolves under a five-minute
prior-art search is worse than no claim at all.

---

## 1. Honest baseline: what is *not* novel

Stated first, so that everything after it is credible.

| Element | Status |
|---|---|
| MFCC, mel filterbank, STFT, LPC | Textbook DSP, decades old |
| Autocorrelation pitch tracking, jitter/shimmer | Standard voice-quality measurement (Praat has done this for 30 years) |
| Detecting vocoder band-limitation and phase artifacts | Well-established in the ASVspoof literature |
| Gradient boosting on acoustic features | Standard machine learning |
| Platt scaling | Platt, 1999 |
| Confidence-weighted score fusion | Standard in biometrics |

**We invented none of these and do not claim to.** Anyone asserting a patent on "using
MFCCs to detect deepfakes" would be laughed out of an examination.

## 2. What we believe is genuinely ours

Four things, in descending order of how defensible we think they are.

### 2.1 Abstention-gated multi-layer fusion for voice authenticity — *strongest claim*

**The idea.** Each detection layer emits a *confidence in its own applicability*, and a
layer with zero confidence is removed from **both the numerator and the denominator** of
the fusion, rather than contributing a neutral vote. The system's overall confidence then
reflects how much of the available evidence could actually be evaluated, and the interface
surfaces "abstained" as a distinct state from "genuine".

**Why it is not obvious.** The standard ensemble treats a missing sub-model as 0.5, which
mathematically drags every verdict toward the prior. In voice authenticity this is
specifically dangerous: the layers most likely to be unavailable (enrolment comparison,
prosody on a short utterance) are the ones whose absence most often coincides with an
attack. The conventional design therefore fails *silently and in the attacker's favour*.

**Where it lives.** `voiceguard/models/base.py::LayerResult.unavailable`,
`voiceguard/scoring/fusion.py::ScoreFusion.fuse`.

**Protectability.** Plausibly a method claim: *"a method for authenticating a live voice
stream in which each of a plurality of analysis layers emits an applicability confidence,
layers below a threshold are excluded from the weighted fusion denominator, and the
excluded layers are reported to the operator as unevaluated."* Would need a real
prior-art search before filing.

### 2.2 Threshold-shifting context enrichment (score/policy separation)

**The idea.** Call context (transaction value, hour, caller-ID attestation, prior fraud
reports) adjusts the **alerting thresholds**, never the model's probability, with the total
shift capped.

**Why it matters.** If context moved the score, the displayed number would mean different
things on different calls and the audit trail would be indefensible — you could not explain
to a regulator why identical audio scored 45 on one call and 62 on another. Separating them
keeps the model output reproducible and auditable while still letting the institution be
more careful about a ₹42 lakh transfer than a balance enquiry. Most commercial risk engines
blend context into a single opaque score.

**Where it lives.** `voiceguard/scoring/risk.py::RiskEngine.context_threshold_shift`.

**Protectability.** Weaker as a patent (it is arguably a design decision rather than a
technical effect), but strong as a **regulatory and procurement differentiator**, which may
be worth more in this market than a patent would be.

### 2.3 Bona-fide-only rule anchor calibration

**The idea.** The interpretable evidence rules have their thresholds fitted from quantiles
of the **genuine** distribution only — never from the spoof side — and rules that fail to
separate are pruned automatically.

**Why it is not obvious.** Fitting on both classes produces anchors that encode "what this
vocoder did", which is exactly what fails on the next TTS release. Fitting on the genuine
side encodes "what real human speech looks like", which is a far more stable target. It is
a small change with a direct generalisation argument behind it.

**Where it lives.** `voiceguard/models/calibration.py::fit_rule_anchors`.

### 2.4 Corpus-leakage detection as a build gate

**The idea.** Before training, every feature is scored for single-feature AUC; anything
above 0.90 alone is flagged as suspected corpus leakage and **fails CI**.

**Why it matters.** It caught a real bug in this project: a DC-offset difference between
classes produced 100 % accuracy and 0 % EER from a corpus artifact. Most published
anti-spoofing work has no such gate, and the field has a documented history of results that
did not reproduce.

**Where it lives.** `ml/metrics.py::detect_leakage`, wired into
`.github/workflows/ci.yml`.

**Protectability.** Not patentable, and we would not try. It is a **methodological
contribution** — the kind of thing that belongs in a paper or a blog post, and it is a
credibility asset in a field where overclaiming is the norm.

## 3. Copyright and trade secret

* **Copyright** subsists automatically in ~9,000 lines of original source across the DSP
  stack, service, front ends, ML pipeline and SDKs. Authored by the team; no code was
  copied from another project.
* **Trade secret** is not applicable in any meaningful way — the repository is open, which
  is a deliberate choice (see §5).
* **Trademark**: "VoiceGuard" is a descriptive, commonly-used compound. If this were
  commercialised, the name would need a clearance search and would very likely have to
  change. We flag that now rather than discovering it later.

## 4. Defensive publication

Everything in §2 is published in this public repository with commit timestamps. That
establishes prior art against a third party patenting the same ideas, which — given that
the team's realistic near-term goal is adoption rather than licensing — is the more useful
outcome. If a patent is later pursued, the publication date starts the clock in
jurisdictions with a grace period and forecloses it in those without; that trade-off is
made knowingly.

## 5. Licensing

**MIT** (see `LICENSE`), chosen deliberately:

* A security control that banks, telecoms and government departments can **audit** is more
  trustworthy than one they cannot. For fraud detection specifically, "trust us" is a
  weak position.
* Adoption is the goal. A restrictive licence on a student project mostly prevents anyone
  from using it.
* The genuinely hard-to-copy assets are not the source: they are the trained weights on a
  real corpus, the calibration against a specific institution's traffic, the measured
  false-positive rate, and the operational relationship. None of those are in the licence.

Third-party dependencies (NumPy, SciPy, scikit-learn, FastAPI, React) are all
BSD/MIT/Apache-2.0 — no copyleft obligations, so an enterprise can deploy this without a
legal review of the dependency tree.

## 6. If we pursued protection

In priority order, and only with proper counsel:

1. Prior-art search on §2.1 (abstention-gated fusion) — the only claim we would seriously
   consider filing.
2. Provisional application covering §2.1 and §2.2 together as a system claim.
3. Trademark clearance and, in all likelihood, a rename.
4. Publish §2.3 and §2.4 as a short methods paper — more valuable as citable evidence of
   rigour than as property.

We are not going to overstate this: a hackathon submission with a working system and one
defensible method claim is a normal, healthy IP position. Claiming a portfolio would not be.
