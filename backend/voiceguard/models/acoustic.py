"""Layer 1 — acoustic / spectral synthesis detection.

Two implementations behind one interface:

:class:`AcousticHeuristicDetector`
    Rules derived from documented, physically-grounded properties of vocoded speech
    (mel-inversion band limitation, over-smoothed spectral trajectories, generated
    silence). It needs no training data, which is what makes the demo run on a clean
    checkout, and it is what the registry falls back to if a model artifact is missing.

:class:`AcousticModelDetector`
    A scikit-learn classifier over the full feature vector, fitted by ``ml/train.py``.
    It carries its own feature-name schema so that adding a feature later cannot silently
    shift the columns underneath a previously-trained model.

The trained model, when present, is the primary signal; the heuristic is kept alive in
parallel and blended in at low weight so that a model which meets an out-of-distribution
vocoder still has a physics-based opinion behind it.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np

from voiceguard.config import Layer
from voiceguard.models.base import Detector, EvidenceRule, LayerResult, RuleScorer, logit, sigmoid

#: Anchors are population values measured on bona fide speech; ``high`` is where the
#: property becomes hard to explain without a vocoder in the chain.
ACOUSTIC_RULES: Sequence[EvidenceRule] = (
    EvidenceRule(
        feature="hf_cliff_depth_db",
        label="High-frequency energy cliff",
        low=25.0, high=80.0, weight=1.6,
        detail_template="spectrum collapses by {value:.0f} dB above the cut-off",
    ),
    EvidenceRule(
        feature="hf_energy_ratio",
        label="Missing high-band energy",
        low=0.045, high=0.004, weight=1.1,
        detail_template="only {value:.3f} of speech-band energy survives above 6 kHz",
    ),
    EvidenceRule(
        feature="spec_rolloff95_mean",
        label="Narrow effective bandwidth",
        low=4200.0, high=2200.0, weight=0.9,
        detail_template="95 % of energy below {value:.0f} Hz",
    ),
    EvidenceRule(
        feature="mel_frame_corr",
        label="Unnatural frame-to-frame spectral change",
        low=0.92, high=0.74, weight=1.2,
        detail_template="inter-frame spectral correlation {value:.3f}",
    ),
    EvidenceRule(
        feature="env_second_diff",
        label="Envelope discontinuity",
        low=0.20, high=0.55, weight=0.9,
        detail_template="envelope second difference {value:.3f}",
    ),
    EvidenceRule(
        feature="digital_silence_score",
        label="Digitally clean silence",
        low=0.05, high=0.45, weight=1.5,
        detail_template="non-speech segments have no room tone (score {value:.2f})",
    ),
    EvidenceRule(
        feature="silence_exact_zero_ratio",
        label="Exact-zero samples",
        low=0.02, high=0.25, weight=1.0,
        detail_template="{value:.1%} of samples are numerically zero",
    ),
    EvidenceRule(
        feature="comb_strength",
        label="Periodic spectral ripple",
        low=0.35, high=0.75, weight=0.7,
        detail_template="regularly-spaced spectral ripple, autocorrelation {value:.2f}",
    ),
    EvidenceRule(
        feature="phase_coherence",
        label="Irregular phase structure",
        low=0.40, high=0.18, weight=0.8,
        detail_template="inter-frame phase coherence {value:.3f}",
    ),
    EvidenceRule(
        feature="residual_kurtosis",
        label="Non-impulsive excitation",
        low=6.0, high=2.8, weight=0.7,
        requires=("residual_peakiness",),
        detail_template="LPC residual kurtosis {value:.2f} (natural glottal pulses are spikier)",
    ),
    EvidenceRule(
        feature="spec_flatness_mean",
        label="Over-flat spectrum",
        low=0.02, high=0.14, weight=0.5,
        detail_template="mean spectral flatness {value:.3f}",
    ),
    EvidenceRule(
        feature="mod_syllabic_ratio",
        label="Weak syllabic modulation",
        low=0.55, high=0.18, weight=0.6,
        detail_template="only {value:.2f} of envelope energy in the 2–16 Hz syllabic band",
    ),
)


class AcousticHeuristicDetector(Detector):
    """Physics-grounded fallback detector for layer 1."""

    layer = Layer.ACOUSTIC.value
    model_id = "acoustic-heuristic-v1"

    def __init__(self, rules: Sequence[EvidenceRule] = ACOUSTIC_RULES) -> None:
        self.scorer = RuleScorer(rules, bias=-1.05, gain=3.4)

    def analyze(self, features: Dict[str, float],
                context: Optional[dict] = None) -> LayerResult:
        if not features:
            return LayerResult.unavailable(self.layer, "no features extracted")

        score, confidence, factors = self.scorer.score(features, layer=self.layer)
        if confidence <= 0:
            return LayerResult.unavailable(self.layer, "no acoustic rule could be evaluated")

        return LayerResult(
            layer=self.layer,
            score=score,
            # The heuristic is a prior, not a measurement: cap its confidence so a trained
            # model always outranks it when one is loaded.
            confidence=confidence * 0.8,
            factors=factors,
            model_id=self.model_id,
            features_used=len(factors),
        )


class AcousticModelDetector(Detector):
    """Trained classifier over the full feature vector."""

    layer = Layer.ACOUSTIC.value

    def __init__(
        self,
        model,
        feature_names: Sequence[str],
        *,
        model_id: str = "acoustic-model",
        scaler=None,
        heuristic: Optional[AcousticHeuristicDetector] = None,
        heuristic_blend: float = 0.25,
        importances: Optional[Dict[str, float]] = None,
    ) -> None:
        self.model = model
        self.feature_names = list(feature_names)
        self.model_id = model_id
        self.scaler = scaler
        self.heuristic = heuristic or AcousticHeuristicDetector()
        self.heuristic_blend = float(np.clip(heuristic_blend, 0.0, 1.0))
        self.importances = importances or {}

    # ------------------------------------------------------------------ inference
    def _vector(self, features: Dict[str, float]) -> np.ndarray:
        return np.array(
            [float(features.get(name, 0.0)) for name in self.feature_names],
            dtype=np.float64,
        ).reshape(1, -1)

    def _predict(self, vector: np.ndarray) -> float:
        if self.scaler is not None:
            vector = self.scaler.transform(vector)
        if hasattr(self.model, "predict_proba"):
            return float(self.model.predict_proba(vector)[0][-1])
        if hasattr(self.model, "decision_function"):
            return sigmoid(float(self.model.decision_function(vector)[0]))
        return float(self.model.predict(vector)[0])

    def analyze(self, features: Dict[str, float],
                context: Optional[dict] = None) -> LayerResult:
        if not features:
            return LayerResult.unavailable(self.layer, "no features extracted")

        try:
            probability = self._predict(self._vector(features))
        except Exception as exc:  # a broken artifact must not take the call down
            fallback = self.heuristic.analyze(features, context)
            fallback.note = f"model inference failed ({exc}); using heuristic"
            return fallback

        # Blend in the physics prior so an out-of-distribution vocoder still moves us.
        heuristic_result = self.heuristic.analyze(features, context)
        if heuristic_result.confidence > 0 and self.heuristic_blend > 0:
            blended = sigmoid(
                (1.0 - self.heuristic_blend) * logit(probability)
                + self.heuristic_blend * logit(heuristic_result.score)
            )
        else:
            blended = probability

        factors = self._explain(features, heuristic_result)
        return LayerResult(
            layer=self.layer,
            score=float(blended),
            confidence=0.95,
            factors=factors,
            model_id=self.model_id,
            features_used=len(self.feature_names),
            note="" if heuristic_result.confidence > 0 else "heuristic prior unavailable",
        )

    # --------------------------------------------------------------- explanation
    def _explain(self, features: Dict[str, float],
                 heuristic_result: LayerResult) -> List:
        """Explain the model's decision using the interpretable rules it agrees with.

        A gradient-boosted tree's own importances are global, not per-call, so we report
        the *rule-based* evidence for this window — which is what an analyst can act on —
        and order it by the model's global importance where we have it.
        """
        factors = list(heuristic_result.factors)
        if self.importances:
            for factor in factors:
                weight = self.importances.get(factor.code)
                if weight:
                    factor.contribution *= 1.0 + float(weight)
        total = sum(f.contribution for f in factors) or 1.0
        for factor in factors:
            factor.contribution = float(factor.contribution / total)
            factor.layer = self.layer
        return sorted(factors, key=lambda f: -f.contribution)

    def describe(self) -> dict:
        return {
            "layer": self.layer,
            "model_id": self.model_id,
            "type": type(self).__name__,
            "n_features": len(self.feature_names),
            "heuristic_blend": self.heuristic_blend,
        }
