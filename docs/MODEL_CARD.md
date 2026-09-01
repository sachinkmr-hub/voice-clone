# Model card — VoiceGuard acoustic detector

Read this before quoting any number from this repository.

| | |
|---|---|
| **Model** | `acoustic-gradient_boosting-v1` (scikit-learn GradientBoostingClassifier, 250 trees, depth 3) |
| **Input** | 123 hand-designed features over a 1.0 s window of 16 kHz mono speech |
| **Output** | Calibrated P(synthetic) ∈ [0, 1], fused with three other layers into a 0–100 risk score |
| **Shipped weights trained on** | The **simulated** bootstrap corpus in `data/corpus` |
| **Intended use** | A second opinion during live calls, alongside human judgement and existing controls |
| **Not for** | Automated, unreviewed decisions about a person; evidence in a legal proceeding; voice biometric identification |

---

## 1. The most important caveat

**The weights that ship in this repository were trained on synthetic audio produced by
this repository.** `voiceguard/simulation/voice.py` generates both classes: bona fide
speech from a source-filter vocal-tract model, and "cloned" speech by pushing that same
audio through a real mel-spectrogram bottleneck and one of three vocoders (Griffin-Lim,
a phase-preserving neural-vocoder proxy, and a hybrid).

That design gives the artifacts a genuine physical basis — mel inversion really is
rank-deficient, Griffin-Lim really does invent phase — which is why the model transfers to
*some* degree. But it is not a substitute for real data, and the numbers below describe
**the pipeline working**, not the product's accuracy against ElevenLabs, RVC, XTTS,
Tortoise, or any commercial cloning service. We have not tested against those.

To get numbers that mean something:

```bash
# ASVspoof 2019 LA (protocol layout is auto-detected)
python -m ml.train --corpus /path/to/ASVspoof2019_LA --layout asvspoof
# In-the-Wild, or your own bonafide/ + spoof/ folders
python -m ml.train --corpus /path/to/release_in_the_wild --layout folders
```

## 2. Measured results (bootstrap corpus)

120 utterances, 12 speakers, 3 vocoder families, ~55 % passed through a simulated
telephony channel. **Speaker-disjoint** train / calibration / test split.

| Evaluation | Accuracy | AUC | EER |
|---|---|---|---|
| Classifier, window level | 0.974 | 0.998 | 2.65 % |
| Classifier, utterance level | 0.983 | 0.999 | 1.67 % |
| **Full four-layer stack**, utterance level | 1.000 | 1.000 | 0.00 % |

Per synthesis family, at threshold 0.50:

| Family | n | Detected |
|---|---|---|
| griffin_lim | 20 | 1.000 |
| hybrid | 20 | 1.000 |
| neural | 20 | 0.950 |
| bona fide (false-positive rate) | 60 | **0.017** |

Unseen-vocoder evaluation — retrain with one family removed entirely, test only on it:

| Held-out family | EER | Detection @ 0.50 |
|---|---|---|
| griffin_lim | 0.00 % | 1.000 |
| hybrid | 0.00 % | 1.000 |
| neural | 0.00 % | 0.950 |

Latency, full stack, one CPU core: **133 ms mean / 165 ms p95 per 1 s window**, against a
500 ms hop budget and the 5 s NFR.

**These numbers are too good, and we are saying so.** An EER of 0 % on a real corpus would
be a red flag; here it reflects a task that is easier than reality because both classes
come from one generator with a bounded set of differences. Expect materially worse
performance on real data. Published state of the art on ASVspoof 2021 DF sits in the
2–7 % EER range; anything we report below that on real data should be disbelieved until
independently reproduced.

## 3. What we did to keep the numbers honest

These are the guards, not aspirations — each is code that runs on every training run.

**Speaker-disjoint splits** (`ml/features.py::split_by_speaker`). No speaker appears in
both train and test. A per-file split would let the model memorise voices.

**The same speaker appears in both classes.** Every synthetic speaker is used for both
bona fide and cloned utterances, so the classes cannot be separated on timbre.

**Channel effects applied to both classes** (`ml/datasets/build.py`). If only spoofed
audio were band-limited, the model would learn "narrow bandwidth = spoof" and be useless
on a PSTN call where *everyone* is band-limited.

**A shared capture chain on both classes.** DC blocker plus converter dither. This one
came from a real bug: an earlier corpus gave bona fide utterances a DC offset of ~4.4e-3
(the glottal pulse train is all-positive) against ~4e-5 for spoofed ones (the STFT
round-trip removes DC). The classifier reached 100 % accuracy reading that artifact, with
`quantization_levels` at 0.55 feature importance. Adding the DC blocker dropped it to
0.038 and moved the top features to `silence_floor_db` and `phase_advance_entropy`.

**An automated leakage detector** (`ml/metrics.py::detect_leakage`) runs before training
and prints loudly if any single feature reaches AUC ≥ 0.90 alone. It exists because of the
bug above: a perfect score from a corpus artifact is worse than a mediocre score from real
signal, because it is confidently wrong and invisible in the headline number.

**Unseen-vocoder evaluation by default.** Reported next to the headline, because a
detector that scores 0.99 on known vocoders and 0.60 on a new one will fail on the next
TTS release.

**Rule anchors fitted on the bona fide side only** (`voiceguard/models/calibration.py`),
so they encode "what real speech looks like" rather than "what this vocoder did".

## 4. Known failure modes

| Failure | Why | Mitigation in the product |
|---|---|---|
| **A high-quality commercial clone over a good line** | Our hardest simulated case is still easier than a well-tuned XTTS/ElevenLabs output. The band-limit and digital-silence tells largely vanish. | Prosody and speaker layers still contribute; the score degrades gracefully rather than snapping to "genuine". Enrolment (layer 3) is the strongest remaining signal. |
| **Heavy codec / poor line on a genuine caller** | Aggressive compression mimics several vocoder artifacts. | Channel augmentation in training; the risk engine presents a spectrum with recommended actions, never a hard binary block on the acoustic score alone. |
| **Very short utterances (< 3 s)** | Prosody needs voiced runs; layer 2 abstains below 25 voiced cycles. | The score is marked `provisional`, the bank integration refuses to clear on thin evidence, and the caveat is shown to the user. |
| **Non-Indian accents, children, elderly, atypical voices** | Language priors cover six Indian languages; the corpus has no children or pathological voices. | Prosodic rules fire only when perturbation is *below* the human range, never above, so a hoarse or distressed caller is not penalised. Still, expect a higher false-positive rate outside the covered population. |
| **Adversarial audio** | We have not tested against anti-detection adversarial perturbation, which is known to defeat classifiers of this family. | Not mitigated. Treat this as an open weakness. |
| **Replay of a genuine recording** | A recording of a real person is real audio; nothing in layers 1–2 will flag it. | Out of scope for this PS. Layer 3 within-call drift and layer 4 context may help; replay detection is a separate problem. |
| **Speaker embedding quality** | The shipped embedding is classical (MFCC + envelope + pitch statistics), not an x-vector/ECAPA network. Its cosine distances are compressed and it is not competitive for open-set speaker ID. | Used only for closed-set enrolment comparison and within-call drift. Install `torch` + `transformers` to switch to the WavLM backend. |

## 5. Fairness and harm considerations

A false positive here is not a neutral event: it tells someone that a person they are
talking to — possibly a family member in genuine distress — may be a machine.

* Prosodic rules are **one-sided by design**. They fire when micro-variation is below the
  human range, never above. Excess jitter and shimmer indicate illness, stress, age, or a
  bad line — exactly the population most likely to be making a genuine urgent call.
* The system reports a **spectrum with a recommended action**, not a binary verdict, and
  the recommended action at every band is a *verification step*, never "hang up".
* **Layers abstain rather than guessing.** No enrolment means the identity was not
  checked, and the UI says so — it does not silently report "genuine".
* We have **not** audited performance across gender, age or regional accent on real data.
  That audit is required before any deployment affecting real customers, and its absence
  is a limitation of this submission, not an oversight to be discovered later.

## 6. Reproducing everything above

```bash
make demo-data           # build the corpus (~20 s)
make train               # train + evaluate, writes ml/artifacts/metrics.json
make evaluate            # standalone evaluation, writes ml/artifacts/evaluation.json
python -m ml.evaluate --corpus data/corpus --full-stack   # includes the live pipeline
```

Every number in section 2 comes from `ml/artifacts/metrics.json` and
`ml/artifacts/evaluation.json`, both regenerated by those commands.
