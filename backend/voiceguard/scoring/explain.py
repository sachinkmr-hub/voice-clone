"""Explainability.

The NFR is "risk score accompanied by top contributing factors". In practice the bar is
higher than that: a contact-centre agent has to be able to *say something true to the
customer on the line*, in one sentence, without knowing what a group delay is. So this
module produces three registers of the same explanation:

``headline``
    One sentence for the person on the call.
``factors``
    Ranked technical evidence with values, for the analyst and the audit trail.
``layer_summary``
    Which layers voted, which abstained and why — so nobody mistakes "we could not check"
    for "we checked and it was fine".

It also emits the *counter-evidence* (what pointed toward genuine), because a one-sided
explanation is how analysts learn to distrust a tool.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Sequence

from voiceguard.config import RECOMMENDED_ACTIONS, RiskBand
from voiceguard.models.base import Factor, LayerResult
from voiceguard.scoring.fusion import FusionResult

LAYER_LABELS = {
    "acoustic": "Acoustic / spectral analysis",
    "prosodic": "Prosody & micro-variation",
    "speaker": "Cross-session speaker check",
    "context": "Call context",
}

BAND_PHRASES = {
    RiskBand.LOW.value: "shows no significant sign of synthetic speech",
    RiskBand.ELEVATED.value: "shows some characteristics associated with synthetic speech",
    RiskBand.HIGH.value: "shows several strong indicators of synthetic or cloned speech",
    RiskBand.CRITICAL.value: "is very likely a synthetic or cloned voice",
}


@dataclass
class Explanation:
    """A layered, human-first account of one risk decision."""

    headline: str
    band: str
    action: str
    factors: List[Factor] = field(default_factory=list)
    counter_factors: List[Factor] = field(default_factory=list)
    layer_summary: List[dict] = field(default_factory=list)
    caveats: List[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "headline": self.headline,
            "band": self.band,
            "action": self.action,
            "factors": [f.as_dict() for f in self.factors],
            "counter_factors": [f.as_dict() for f in self.counter_factors],
            "layers": list(self.layer_summary),
            "caveats": list(self.caveats),
        }

    def as_text(self) -> str:
        lines = [self.headline, ""]
        if self.factors:
            lines.append("Why:")
            lines.extend(f"  • {f.label} — {f.detail}" for f in self.factors)
        if self.counter_factors:
            lines.append("Against:")
            lines.extend(f"  • {f.label} — {f.detail}" for f in self.counter_factors)
        if self.caveats:
            lines.append("Caveats:")
            lines.extend(f"  • {c}" for c in self.caveats)
        lines.append(f"Recommended action: {self.action}")
        return "\n".join(lines)


def _layer_summary(results: Sequence[LayerResult], fusion: FusionResult) -> List[dict]:
    summary = []
    for result in results:
        summary.append({
            "layer": result.layer,
            "label": LAYER_LABELS.get(result.layer, result.layer),
            "score": round(float(result.score), 4),
            "confidence": round(float(result.confidence), 4),
            "weight_share": round(float(fusion.contributions.get(result.layer, 0.0)), 4),
            "status": "voted" if result.layer in fusion.participating else "abstained",
            "reason": result.note,
            "model_id": result.model_id,
        })
    return summary


def _caveats(results: Sequence[LayerResult], fusion: FusionResult,
             speech_ratio: float, elapsed: float) -> List[str]:
    caveats: List[str] = []
    if elapsed < 3.0:
        caveats.append(
            f"Only {elapsed:.1f} s of audio analysed so far — the score is still settling."
        )
    if speech_ratio < 0.35:
        caveats.append(
            "Most of this audio is silence or background; the acoustic evidence is thin."
        )
    if "speaker" in fusion.excluded:
        caveats.append(
            "No enrolled voice sample for the claimed identity, so the identity itself "
            "could not be verified — only whether the audio looks synthetic."
        )
    if fusion.confidence < 0.5:
        caveats.append(
            "Detection layers disagree or few could run; treat this score as provisional."
        )
    for result in results:
        if result.model_id.endswith("heuristic-v1") and result.layer == "acoustic":
            caveats.append(
                "Running on the built-in heuristic detector (no trained model artifact "
                "loaded); accuracy is lower than the reported benchmark."
            )
            break
    return caveats


def build_explanation(
    fusion: FusionResult,
    score: float,
    band: str,
    *,
    speech_ratio: float = 1.0,
    elapsed_seconds: float = 0.0,
    action: str = "",
    max_factors: int = 4,
) -> Explanation:
    """Assemble the full explanation for one risk decision."""
    ranked = sorted(fusion.factors, key=lambda f: -f.contribution)
    factors = [f for f in ranked if f.contribution > 0.01][:max_factors]

    # Counter-evidence: layers that actively voted "genuine" despite the overall verdict.
    counter: List[Factor] = []
    for result in fusion.layers:
        if result.confidence > 0.2 and result.score < 0.35:
            counter.append(Factor(
                code=f"{result.layer}_genuine",
                label=f"{LAYER_LABELS.get(result.layer, result.layer)} looks genuine",
                contribution=float(result.confidence * (1.0 - result.score)),
                value=float(result.score),
                detail=f"this layer scored {result.score * 100:.0f}/100 for synthesis",
                layer=result.layer,
                direction="genuine",
            ))

    phrase = BAND_PHRASES.get(band, BAND_PHRASES[RiskBand.LOW.value])
    if factors:
        lead = factors[0].label.lower()
        headline = (
            f"Risk {score:.0f}/100 — this voice {phrase}; "
            f"the strongest single indicator is {lead}."
        )
    else:
        headline = f"Risk {score:.0f}/100 — this voice {phrase}."

    return Explanation(
        headline=headline,
        band=band,
        action=action or RECOMMENDED_ACTIONS.get(band, ""),
        factors=factors,
        counter_factors=sorted(counter, key=lambda f: -f.contribution)[:2],
        layer_summary=_layer_summary(fusion.layers, fusion),
        caveats=_caveats(fusion.layers, fusion, speech_ratio, elapsed_seconds),
    )


def factor_frequency(explanations: Sequence[Explanation]) -> Dict[str, int]:
    """How often each factor appeared — used for the per-call confidence trail (FR-7)."""
    counts: Dict[str, int] = {}
    for explanation in explanations:
        for factor in explanation.factors:
            counts[factor.code] = counts.get(factor.code, 0) + 1
    return dict(sorted(counts.items(), key=lambda kv: -kv[1]))
