"""Detector interfaces and the evidence-rule machinery shared by the heuristic layers.

Design notes
------------
Every layer returns the same shape — :class:`LayerResult` — carrying a probability, a
*confidence in that probability*, and the ranked :class:`Factor` list that produced it.
Confidence is what lets a layer say "I have nothing useful here" (no enrolment, no speech,
no metadata) instead of silently voting "genuine", which is the failure mode that makes
ensemble spoof detectors quietly useless in production.

The heuristic detectors are built from :class:`EvidenceRule` objects rather than an opaque
scoring function so that the same object produces both the number and the explanation —
they cannot drift apart.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence  # noqa: F401

EPS = 1e-9


def sigmoid(z: float) -> float:
    if z >= 0:
        return 1.0 / (1.0 + math.exp(-min(z, 60.0)))
    ez = math.exp(max(z, -60.0))
    return ez / (1.0 + ez)


def logit(p: float, clip: float = 1e-4) -> float:
    p = min(max(float(p), clip), 1.0 - clip)
    return math.log(p / (1.0 - p))


def ramp(value: float, low: float, high: float) -> float:
    """Piecewise-linear 0→1 ramp. Handles ``low > high`` as a descending ramp."""
    if not math.isfinite(value):
        return 0.0
    if low == high:
        return 1.0 if value >= high else 0.0
    if low < high:
        return float(min(1.0, max(0.0, (value - low) / (high - low))))
    return float(min(1.0, max(0.0, (low - value) / (low - high))))


# --------------------------------------------------------------------------------------
# Explanation primitives
# --------------------------------------------------------------------------------------

@dataclass
class Factor:
    """One piece of ranked, human-readable evidence."""

    code: str
    label: str
    contribution: float           #: 0..1 — how strongly this pushed toward "synthetic"
    value: float = 0.0
    detail: str = ""
    layer: str = ""
    direction: str = "synthetic"  #: "synthetic" or "genuine"

    def as_dict(self) -> dict:
        return {
            "code": self.code,
            "label": self.label,
            "contribution": round(float(self.contribution), 4),
            "value": round(float(self.value), 6),
            "detail": self.detail,
            "layer": self.layer,
            "direction": self.direction,
        }


@dataclass
class LayerResult:
    """The output contract every detection layer honours."""

    layer: str
    score: float = 0.5               #: P(synthetic) in [0, 1]
    confidence: float = 0.0          #: 0 means "exclude me from the fusion"
    factors: List[Factor] = field(default_factory=list)
    model_id: str = "none"
    features_used: int = 0
    note: str = ""

    @classmethod
    def unavailable(cls, layer: str, note: str) -> "LayerResult":
        """A layer that could not run. Confidence 0 keeps it out of the fusion entirely."""
        return cls(layer=layer, score=0.5, confidence=0.0, model_id="none", note=note)

    def top_factors(self, n: int = 3) -> List[Factor]:
        return sorted(self.factors, key=lambda f: -abs(f.contribution))[:n]

    def as_dict(self) -> dict:
        return {
            "layer": self.layer,
            "score": round(float(self.score), 4),
            "confidence": round(float(self.confidence), 4),
            "model_id": self.model_id,
            "features_used": self.features_used,
            "note": self.note,
            "factors": [f.as_dict() for f in self.top_factors(5)],
        }


# --------------------------------------------------------------------------------------
# Evidence rules
# --------------------------------------------------------------------------------------

@dataclass
class EvidenceRule:
    """A single interpretable rule over one feature.

    ``low``/``high`` anchor a ramp: ``low`` is where the feature stops looking human and
    ``high`` is where it is unambiguously synthetic. Set ``low > high`` for features where
    a *smaller* value is the suspicious one (e.g. pitch micro-variation).

    ``weight`` is this rule's share of the layer's heuristic score. ``requires`` names
    features that must be present and non-zero for the rule to fire at all — this is how
    we avoid, say, reading a jitter of exactly 0.0 as damning when the window simply had
    no voiced speech in it.
    """

    feature: str
    label: str
    low: float
    high: float
    weight: float = 1.0
    detail_template: str = "{label} = {value:.4g}"
    requires: Sequence[str] = ()
    minimum_evidence: float = 0.0

    def evaluate(self, features: Dict[str, float]) -> Optional[Factor]:
        if self.feature not in features:
            return None
        for required in self.requires:
            if abs(float(features.get(required, 0.0))) <= EPS:
                return None

        value = float(features[self.feature])
        strength = ramp(value, self.low, self.high)
        if strength < self.minimum_evidence:
            strength = 0.0

        return Factor(
            code=self.feature,
            label=self.label,
            contribution=strength,
            value=value,
            detail=self.detail_template.format(label=self.label, value=value),
        )


class RuleScorer:
    """Turns a set of :class:`EvidenceRule` into a calibrated score plus factors."""

    def __init__(self, rules: Sequence[EvidenceRule], *, bias: float = -1.15,
                 gain: float = 3.2) -> None:
        self.rules = list(rules)
        self.bias = bias    #: prior log-odds — most calls are genuine
        self.gain = gain    #: how sharply accumulated evidence moves the score

    def score(self, features: Dict[str, float], layer: str = "") -> tuple:
        """Return ``(probability, confidence, factors)``."""
        factors: List[Factor] = []
        total_weight = 0.0
        weighted = 0.0

        for rule in self.rules:
            factor = rule.evaluate(features)
            if factor is None:
                continue
            factor.layer = layer
            factor.contribution *= rule.weight
            factors.append(factor)
            total_weight += rule.weight
            weighted += factor.contribution

        if total_weight <= EPS:
            return 0.5, 0.0, []

        evidence = weighted / total_weight
        probability = sigmoid(self.bias + self.gain * (2.0 * evidence - 0.5))
        # Confidence grows with how many rules could actually be evaluated.
        coverage = total_weight / max(sum(r.weight for r in self.rules), EPS)
        confidence = float(min(1.0, 0.35 + 0.65 * coverage))

        # Normalise contributions back to a 0..1 share for display.
        for factor in factors:
            factor.contribution = float(factor.contribution / max(total_weight, EPS))
        return float(probability), confidence, factors


# --------------------------------------------------------------------------------------
# Detector interface
# --------------------------------------------------------------------------------------

class Detector:
    """Base class for every detection layer."""

    layer: str = "unknown"
    model_id: str = "none"

    def analyze(self, features: Dict[str, float], context: Optional[dict] = None) -> LayerResult:
        raise NotImplementedError

    def describe(self) -> dict:
        return {"layer": self.layer, "model_id": self.model_id,
                "type": type(self).__name__}


def merge_factors(
    results: Iterable[LayerResult],
    limit: int = 5,
    weights: Optional[Dict[str, float]] = None,
) -> List[Factor]:
    """Rank factors across layers, keeping the strongest per feature code.

    ``weights`` is each layer's share of the fused decision. Passing it matters: without
    it, a context signal like "caller ID not attested" — which is one twelfth of the
    decision by weight — can head the explanation of a call flagged on acoustic grounds.
    Users then read the tool as a caller-ID checker, which is not what it is, and stop
    trusting the voice verdict.
    """
    best: Dict[str, Factor] = {}
    for result in results:
        if result.confidence <= 0:
            continue
        share = float(weights.get(result.layer, 1.0)) if weights else 1.0
        for factor in result.factors:
            factor.layer = factor.layer or result.layer
            scaled = Factor(**{**factor.__dict__})
            scaled.contribution = factor.contribution * result.confidence * share
            current = best.get(factor.code)
            if current is None or scaled.contribution > current.contribution:
                best[factor.code] = scaled

    ranked = sorted(best.values(), key=lambda f: -f.contribution)[:limit]
    total = sum(f.contribution for f in ranked)
    if total > EPS:
        for factor in ranked:
            factor.contribution = float(factor.contribution / total)
    return ranked
