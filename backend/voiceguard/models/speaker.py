"""Layer 3 — cross-session speaker consistency.

Two independent checks, both of which are *identity* checks rather than *synthesis*
checks, which makes them complementary to layers 1 and 2:

**Enrolment comparison.** If we hold genuine samples for the identity the caller claims,
compare the live embedding against that centroid. The distance is normalised by the
enrolment's own within-speaker spread, so a speaker whose enrolment is naturally variable
does not get penalised for it.

**Within-call drift.** Even with no enrolment at all, an impersonation attempt that
splices a real greeting onto synthesised instructions — or that swaps to a clone mid-call
after a human hand-off — shows a step change in embedding space that a single speaker
talking continuously does not produce.

A missing enrolment is reported as *unavailable*, never as *genuine*. That distinction is
the whole reason the fusion carries confidences.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np

from voiceguard.config import Layer
from voiceguard.features.embedding import (
    MIN_ENROLMENT_SPREAD,
    EnrolmentStore,
    cosine_distance,
)
from voiceguard.models.base import Detector, Factor, LayerResult, ramp, sigmoid


class SpeakerConsistencyDetector(Detector):
    """Compares the live voice against enrolment and against its own recent history."""

    layer = Layer.SPEAKER.value
    model_id = "speaker-consistency-v1"

    # These thresholds are calibrated to the *classical* embedding shipped in
    # features/embedding.py, whose cosine distances are compressed: same speaker,
    # different take lands near 0.00-0.02, different speaker near 0.03-0.30. A neural
    # (WavLM) embedding spreads much wider, so these are attributes rather than
    # constants and the registry overrides them when that backend is active.

    #: Distance at which a claimed identity starts looking wrong, and where it is wrong.
    #: Only used as the single-sample fallback — with three or more enrolment samples the
    #: spread-normalised z path below is strictly better, because absolute distance
    #: depends on how different the two speakers happen to be while z asks the right
    #: question: "is this further from the profile than this speaker's own takes are?"
    MATCH_DISTANCE = 0.010
    MISMATCH_DISTANCE = 0.100
    #: Same, expressed in within-speaker spreads (used when the enrolment has several samples).
    MATCH_Z = 1.5
    MISMATCH_Z = 6.0
    #: A drift larger than this between consecutive windows suggests a splice.
    DRIFT_LOW = 0.020
    DRIFT_HIGH = 0.120

    #: Threshold sets per embedding backend: (match, mismatch, drift_low, drift_high).
    BACKEND_THRESHOLDS = {
        "classical": (0.010, 0.100, 0.020, 0.120),
        "wavlm": (0.150, 0.550, 0.150, 0.500),
    }

    def __init__(self, enrolment: Optional[EnrolmentStore] = None,
                 history_size: int = 12) -> None:
        self.enrolment = enrolment or EnrolmentStore()
        self.history_size = history_size
        thresholds = self.BACKEND_THRESHOLDS.get(self.enrolment.embedder.backend)
        if thresholds:
            (self.MATCH_DISTANCE, self.MISMATCH_DISTANCE,
             self.DRIFT_LOW, self.DRIFT_HIGH) = thresholds

    # ------------------------------------------------------------------- analysis
    def analyze(self, features: Dict[str, float],
                context: Optional[dict] = None) -> LayerResult:
        """``context`` must carry ``embedding`` and may carry ``identity`` / ``history``."""
        context = context or {}
        embedding = context.get("embedding")
        if embedding is None or not np.any(np.asarray(embedding)):
            return LayerResult.unavailable(self.layer, "no speaker embedding for this window")

        embedding = np.asarray(embedding, dtype=np.float32)
        identity = context.get("identity")
        history: List[np.ndarray] = list(context.get("history") or [])

        factors: List[Factor] = []
        scores: List[tuple] = []   # (score, weight)

        enrolment_match = None
        if identity and self.enrolment.has(identity):
            enrolment_match = self.enrolment.compare(identity, embedding)

        if enrolment_match:
            distance = float(enrolment_match["distance"])
            samples = float(enrolment_match["samples"])
            # With several enrolment samples we can normalise by the speaker's own spread;
            # with one sample that estimate is meaningless, so use the absolute distance.
            if samples >= 3 and enrolment_match["enrolment_spread"] > MIN_ENROLMENT_SPREAD:
                strength = ramp(float(enrolment_match["z_distance"]), self.MATCH_Z, self.MISMATCH_Z)
                detail = (
                    f"voice sits {enrolment_match['z_distance']:.1f} within-speaker spreads "
                    f"from the enrolled profile for '{identity}'"
                )
            else:
                strength = ramp(distance, self.MATCH_DISTANCE, self.MISMATCH_DISTANCE)
                detail = (
                    f"embedding distance {distance:.3f} from the enrolled profile "
                    f"for '{identity}' ({int(samples)} sample(s))"
                )
            factors.append(Factor(
                code="speaker_enrolment_distance",
                label="Voice does not match the enrolled speaker",
                contribution=strength,
                value=distance,
                detail=detail,
                layer=self.layer,
            ))
            scores.append((strength, 1.0))

        drift = 0.0
        if history:
            recent = history[-self.history_size :]
            distances = [cosine_distance(embedding, past) for past in recent]
            drift = float(np.max(distances)) if distances else 0.0
            strength = ramp(drift, self.DRIFT_LOW, self.DRIFT_HIGH)
            if strength > 0:
                factors.append(Factor(
                    code="speaker_within_call_drift",
                    label="Voice changed during the call",
                    contribution=strength,
                    value=drift,
                    detail=f"embedding moved {drift:.3f} from earlier in this same call",
                    layer=self.layer,
                ))
            scores.append((strength, 0.6))

        if not scores:
            return LayerResult.unavailable(
                self.layer,
                "no enrolment for the claimed identity and no call history yet",
            )

        total_weight = sum(w for _, w in scores)
        evidence = sum(s * w for s, w in scores) / total_weight
        probability = sigmoid(-0.9 + 3.6 * (evidence - 0.15))

        # Confidence: an enrolment comparison is far stronger evidence than drift alone.
        confidence = 0.85 if enrolment_match else 0.45
        if enrolment_match and float(enrolment_match["samples"]) < 3:
            confidence *= 0.75
        if len(history) < 3 and not enrolment_match:
            confidence *= 0.6

        for factor in factors:
            factor.contribution = float(factor.contribution / max(total_weight, 1e-9))

        return LayerResult(
            layer=self.layer,
            score=float(probability),
            confidence=float(confidence),
            factors=factors,
            model_id=self.model_id,
            features_used=len(factors),
            note="enrolment comparison" if enrolment_match else "within-call drift only",
        )

    # ----------------------------------------------------------------- enrolment
    def enrol(self, identity: str, audio: np.ndarray, sample_rate: int) -> int:
        return self.enrolment.enrol(identity, audio, sample_rate)

    def describe(self) -> dict:
        return {
            "layer": self.layer,
            "model_id": self.model_id,
            "type": type(self).__name__,
            "enrolled_identities": self.enrolment.identities(),
            "backend": self.enrolment.embedder.backend,
        }
