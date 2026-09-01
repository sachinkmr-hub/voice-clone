"""Layer 2 — prosodic / behavioural detection.

The physical claim behind every rule here: a human vocal apparatus is a noisy biological
oscillator driven by breathing and real-time planning, and *cannot* produce the smooth
trajectories a neural acoustic model emits by construction. Reference ranges for healthy
conversational speech: jitter 0.3–1.5 %, shimmer 3–8 %, pause distribution heavy-tailed.

Note the deliberate asymmetry: rules fire when micro-variation is **below** the human
range, never when it is above. Excess perturbation means a bad line, a cold, or emotion —
none of which is evidence of cloning, and treating it as such is how a detector starts
flagging distressed callers, which is precisely the population most likely to be
*genuinely* calling about a fraud.
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence

from voiceguard.config import Layer
from voiceguard.models.base import Detector, EvidenceRule, LayerResult, RuleScorer

PROSODIC_RULES: Sequence[EvidenceRule] = (
    EvidenceRule(
        feature="f0_micro_var",
        label="Suppressed pitch micro-variation",
        low=0.020, high=0.004, weight=1.5,
        requires=("cycle_count",),
        detail_template="pitch residual variation {value:.4f} (human speech: 0.02–0.06)",
    ),
    EvidenceRule(
        feature="jitter_ppq5",
        label="Below-human cycle jitter",
        low=0.004, high=0.0004, weight=1.3,
        requires=("cycle_count",),
        detail_template="5-point period perturbation {value:.5f} (human: 0.003–0.015)",
    ),
    EvidenceRule(
        feature="shimmer_apq3",
        label="Below-human amplitude shimmer",
        low=0.035, high=0.006, weight=1.2,
        requires=("cycle_count",),
        detail_template="3-point amplitude perturbation {value:.4f} (human: 0.03–0.08)",
    ),
    EvidenceRule(
        feature="f0_contour_entropy",
        label="Mechanical pitch contour",
        low=0.72, high=0.38, weight=0.9,
        requires=("voiced_ratio",),
        detail_template="pitch contour entropy {value:.3f}",
    ),
    EvidenceRule(
        feature="pause_std",
        label="Uniform pause lengths",
        low=0.16, high=0.02, weight=1.0,
        requires=("pause_count",),
        detail_template="pause-duration spread {value:.3f} s (human pauses vary widely)",
    ),
    EvidenceRule(
        feature="silence_floor_std_db",
        label="Static background between words",
        low=3.5, high=0.4, weight=1.1,
        detail_template="background level varies by only {value:.2f} dB",
    ),
    EvidenceRule(
        feature="energy_range_db",
        label="Compressed loudness dynamics",
        low=22.0, high=7.0, weight=0.8,
        requires=("speech_ratio",),
        detail_template="speech level spans only {value:.1f} dB",
    ),
    EvidenceRule(
        feature="speech_run_std",
        label="Metronomic phrase lengths",
        low=0.35, high=0.05, weight=0.7,
        requires=("speech_run_mean",),
        detail_template="phrase-length spread {value:.3f} s",
    ),
    EvidenceRule(
        feature="env_lag1_autocorr",
        label="Over-smooth loudness envelope",
        low=0.55, high=0.92, weight=0.8,
        detail_template="envelope autocorrelation {value:.3f}",
    ),
    EvidenceRule(
        feature="period_cv",
        label="Near-perfect pitch periodicity",
        low=0.030, high=0.004, weight=1.0,
        requires=("cycle_count",),
        detail_template="period coefficient of variation {value:.4f}",
    ),
)


class ProsodicDetector(Detector):
    """Rule-based prosody scorer (optionally backed by a trained model)."""

    layer = Layer.PROSODIC.value
    model_id = "prosodic-heuristic-v1"

    #: A window with almost no voiced speech cannot support a prosody judgement.
    MIN_CYCLES = 25.0
    MIN_SPEECH_RATIO = 0.25

    def __init__(self, rules: Sequence[EvidenceRule] = PROSODIC_RULES,
                 model=None, feature_names: Optional[Sequence[str]] = None) -> None:
        self.scorer = RuleScorer(rules, bias=-1.2, gain=3.0)
        self.model = model
        self.feature_names = list(feature_names or [])
        if model is not None:
            self.model_id = "prosodic-model-v1"

    def analyze(self, features: Dict[str, float],
                context: Optional[dict] = None) -> LayerResult:
        if not features:
            return LayerResult.unavailable(self.layer, "no features extracted")

        cycles = float(features.get("cycle_count", 0.0))
        speech_ratio = float(features.get("speech_ratio", 0.0))
        if cycles < self.MIN_CYCLES or speech_ratio < self.MIN_SPEECH_RATIO:
            return LayerResult.unavailable(
                self.layer,
                f"insufficient voiced speech (cycles={cycles:.0f}, speech={speech_ratio:.2f})",
            )

        score, confidence, factors = self.scorer.score(features, layer=self.layer)
        if confidence <= 0:
            return LayerResult.unavailable(self.layer, "no prosodic rule could be evaluated")

        # More voiced material means a more trustworthy prosody measurement.
        evidence_scale = min(1.0, cycles / 120.0)
        confidence = confidence * (0.55 + 0.45 * evidence_scale)

        return LayerResult(
            layer=self.layer,
            score=score,
            confidence=float(confidence),
            factors=factors,
            model_id=self.model_id,
            features_used=len(factors),
        )
