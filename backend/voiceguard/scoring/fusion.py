"""Score fusion and calibration.

Layers are combined as a **confidence-weighted average in logit space**:

.. math::

    z = \\frac{\\sum_i w_i c_i \\,\\mathrm{logit}(s_i)}{\\sum_i w_i c_i}
    \\qquad
    p = \\sigma(a z + b)

Three properties matter and are all deliberate:

1. **Logit space, not probability space.** Averaging probabilities lets a confident
   "0.02 genuine" cancel a confident "0.98 synthetic" into a useless 0.5. In logit space
   the evidence adds the way log-odds should.
2. **Confidence gates participation.** A layer with ``confidence == 0`` (no enrolment, no
   voiced speech, no metadata) is dropped from *both* the numerator and the denominator.
   It does not get to vote "genuine" by default — which is the single most common way an
   ensemble spoof detector silently stops working in the field.
3. **Calibration is separate and fitted.** ``(a, b)`` come from Platt scaling on a held-out
   split (``ml/train.py``), so a displayed "80" means "in evaluation, roughly 80 % of
   calls scoring here were synthetic" rather than "the model felt strongly".
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

from voiceguard.config import DEFAULT_CALIBRATION, DEFAULT_FUSION_WEIGHTS
from voiceguard.models.base import Factor, LayerResult, logit, merge_factors, sigmoid

EPS = 1e-9


@dataclass
class FusionResult:
    """Outcome of fusing one window's layer results."""

    probability: float
    raw_probability: float
    confidence: float
    layers: List[LayerResult] = field(default_factory=list)
    contributions: Dict[str, float] = field(default_factory=dict)
    factors: List[Factor] = field(default_factory=list)
    participating: List[str] = field(default_factory=list)
    excluded: Dict[str, str] = field(default_factory=dict)

    @property
    def score_100(self) -> float:
        return float(round(self.probability * 100.0, 1))

    def as_dict(self) -> dict:
        return {
            "probability": round(self.probability, 4),
            "raw_probability": round(self.raw_probability, 4),
            "score": self.score_100,
            "confidence": round(self.confidence, 4),
            "participating_layers": list(self.participating),
            "excluded_layers": dict(self.excluded),
            "contributions": {k: round(v, 4) for k, v in self.contributions.items()},
            "factors": [f.as_dict() for f in self.factors],
            "layers": [layer.as_dict() for layer in self.layers],
        }


class ScoreFusion:
    """Confidence-weighted logit fusion with Platt calibration."""

    def __init__(
        self,
        weights: Optional[Dict[str, float]] = None,
        calibration: Sequence[float] = DEFAULT_CALIBRATION,
    ) -> None:
        self.weights = dict(DEFAULT_FUSION_WEIGHTS)
        if weights:
            self.weights.update({k: float(v) for k, v in weights.items()})
        self.calibration = (float(calibration[0]), float(calibration[1]))

    # ------------------------------------------------------------------- fusing
    def fuse(self, results: Sequence[LayerResult]) -> FusionResult:
        participating: List[str] = []
        excluded: Dict[str, str] = {}
        numerator = 0.0
        denominator = 0.0
        contributions: Dict[str, float] = {}

        for result in results:
            weight = self.weights.get(result.layer, 0.0)
            effective = weight * max(0.0, float(result.confidence))
            if effective <= EPS:
                excluded[result.layer] = result.note or "no confidence"
                continue
            participating.append(result.layer)
            numerator += effective * logit(result.score)
            denominator += effective
            contributions[result.layer] = effective

        if denominator <= EPS:
            # Nothing could be evaluated. Return the prior, with zero confidence, and let
            # the caller decide what to display — never a confident "genuine".
            return FusionResult(
                probability=0.5, raw_probability=0.5, confidence=0.0,
                layers=list(results), excluded=excluded,
            )

        z = numerator / denominator
        raw = sigmoid(z)
        a, b = self.calibration
        probability = sigmoid(a * z + b)

        # Normalise contributions to shares of the decision.
        contributions = {k: v / denominator for k, v in contributions.items()}

        # Overall confidence: how much of the *available* weight actually voted.
        total_weight = sum(self.weights.get(r.layer, 0.0) for r in results) or 1.0
        coverage = denominator / total_weight
        # Agreement: layers pulling in opposite directions should lower confidence.
        votes = [r.score for r in results if r.layer in participating]
        spread = (max(votes) - min(votes)) if len(votes) > 1 else 0.0
        confidence = float(max(0.05, min(1.0, coverage * (1.0 - 0.35 * spread))))

        return FusionResult(
            probability=float(probability),
            raw_probability=float(raw),
            confidence=confidence,
            layers=list(results),
            contributions=contributions,
            factors=merge_factors(results, limit=6, weights=contributions),
            participating=participating,
            excluded=excluded,
        )

    # -------------------------------------------------------------- maintenance
    def update_weights(self, weights: Dict[str, float]) -> None:
        for key, value in weights.items():
            if key in self.weights:
                self.weights[key] = float(value)

    def as_dict(self) -> dict:
        return {"weights": dict(self.weights), "calibration": list(self.calibration)}
