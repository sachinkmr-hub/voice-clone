"""Tests for fusion, the risk engine, smoothing and explanations."""

import pytest

from voiceguard.config import DEFAULT_PROFILES, RiskBand
from voiceguard.models.base import Factor, LayerResult
from voiceguard.models.context import CallContext
from voiceguard.scoring.explain import build_explanation
from voiceguard.scoring.fusion import ScoreFusion
from voiceguard.scoring.risk import RiskEngine, ScoreSmoother


def layer(name, score, confidence, factors=None, note=""):
    return LayerResult(layer=name, score=score, confidence=confidence,
                       factors=factors or [], model_id=f"{name}-test", note=note)


# ------------------------------------------------------------------------------ fusion

def test_fusion_excludes_zero_confidence_layers_from_the_denominator():
    fusion = ScoreFusion()
    with_abstention = fusion.fuse([
        layer("acoustic", 0.9, 1.0),
        layer("speaker", 0.5, 0.0, note="no enrolment"),
    ])
    alone = fusion.fuse([layer("acoustic", 0.9, 1.0)])

    # An abstaining layer must not dilute the verdict toward 0.5.
    assert with_abstention.probability == pytest.approx(alone.probability, abs=1e-6)
    assert "speaker" in with_abstention.excluded
    assert with_abstention.participating == ["acoustic"]


def test_fusion_returns_the_prior_with_zero_confidence_when_nothing_ran():
    result = ScoreFusion().fuse([layer("acoustic", 0.9, 0.0), layer("prosodic", 0.9, 0.0)])
    assert result.probability == 0.5
    assert result.confidence == 0.0


def test_fusion_adds_evidence_in_logit_space():
    fusion = ScoreFusion()
    one = fusion.fuse([layer("acoustic", 0.8, 1.0)]).probability
    both = fusion.fuse([layer("acoustic", 0.8, 1.0), layer("prosodic", 0.8, 1.0)]).probability
    # Two layers agreeing should not average *down* below what one alone reported.
    assert both == pytest.approx(one, abs=0.02)


def test_fusion_confidence_drops_when_layers_disagree():
    fusion = ScoreFusion()
    agree = fusion.fuse([layer("acoustic", 0.85, 1.0), layer("prosodic", 0.85, 1.0)])
    disagree = fusion.fuse([layer("acoustic", 0.95, 1.0), layer("prosodic", 0.05, 1.0)])
    assert disagree.confidence < agree.confidence


def test_fusion_weights_are_configurable():
    fusion = ScoreFusion({"acoustic": 0.9, "context": 0.1})
    result = fusion.fuse([layer("acoustic", 0.9, 1.0), layer("context", 0.1, 1.0)])
    assert result.contributions["acoustic"] > result.contributions["context"]
    assert result.probability > 0.5           # the heavier layer wins


def test_calibration_shifts_the_reported_probability():
    raw = ScoreFusion(calibration=(1.0, 0.0)).fuse([layer("acoustic", 0.8, 1.0)])
    shifted = ScoreFusion(calibration=(1.0, -1.5)).fuse([layer("acoustic", 0.8, 1.0)])
    assert shifted.probability < raw.probability
    assert shifted.raw_probability == pytest.approx(raw.raw_probability)


# ------------------------------------------------------------------------- risk engine

def test_bands_follow_the_profile_thresholds():
    engine = RiskEngine(DEFAULT_PROFILES["default"])
    results = [layer("acoustic", 0.9, 1.0)]
    for score, expected in ((10, RiskBand.LOW), (40, RiskBand.ELEVATED),
                            (65, RiskBand.HIGH), (90, RiskBand.CRITICAL)):
        assessment = engine.assess(results, score_override=score)
        assert assessment.band == expected.value
        assert assessment.action


def test_wire_transfer_profile_is_more_sensitive_than_contact_center():
    results = [layer("acoustic", 0.9, 1.0)]
    engine = RiskEngine(profiles=dict(DEFAULT_PROFILES))
    wire = engine.assess(results, profile_name="wire_transfer", score_override=50)
    desk = engine.assess(results, profile_name="contact_center", score_override=50)
    assert wire.band == RiskBand.HIGH.value
    assert desk.band == RiskBand.ELEVATED.value


def test_context_moves_thresholds_not_the_probability():
    engine = RiskEngine()
    results = [layer("acoustic", 0.75, 1.0)]
    risky = CallContext(transaction_amount=5_000_000, known_contact=False,
                        caller_id_verified=False, local_hour=3, prior_fraud_reports=2)

    plain = engine.assess(results, score_override=50.0)
    shifted = engine.assess(results, call_context=risky, score_override=50.0)

    assert shifted.probability == pytest.approx(plain.probability)   # model untouched
    assert shifted.score == plain.score
    assert shifted.threshold_shift > 5.0
    assert shifted.band != plain.band                                # policy moved


def test_threshold_shift_is_capped():
    engine = RiskEngine()
    absurd = CallContext(transaction_amount=10_000_000_000, prior_fraud_reports=99,
                         known_contact=False, caller_id_verified=False, local_hour=3)
    assert engine.context_threshold_shift(absurd) <= RiskEngine.MAX_THRESHOLD_SHIFT


def test_known_good_context_raises_the_bar():
    engine = RiskEngine()
    trusted = CallContext(known_contact=True, caller_id_verified=True, local_hour=11)
    assert engine.context_threshold_shift(trusted) < 0


def test_effective_thresholds_reflect_the_shift():
    engine = RiskEngine()
    context = CallContext(transaction_amount=5_000_000, known_contact=False)
    thresholds = engine.effective_thresholds(context, "wire_transfer")
    assert thresholds["critical"] < DEFAULT_PROFILES["wire_transfer"].critical
    assert thresholds["elevated"] < thresholds["high"] < thresholds["critical"]


def test_assessment_serialises_both_ways():
    engine = RiskEngine()
    assessment = engine.assess([layer("acoustic", 0.9, 1.0)], score_override=88)
    brief, verbose = assessment.as_dict(), assessment.as_dict(verbose=True)
    assert brief["band"] == RiskBand.CRITICAL.value
    assert "layers" in brief and "fusion" not in brief
    assert "fusion" in verbose and "explanation" in verbose


# --------------------------------------------------------------------------- smoothing

def test_smoother_damps_a_single_spike():
    smoother = ScoreSmoother(alpha=0.35)
    for _ in range(5):
        smoother.update(10.0)
    spiked = smoother.update(95.0)
    assert 10.0 < spiked < 60.0        # reacts, but is not dragged all the way


def test_smoother_holds_sustained_evidence():
    """A burst of strong evidence must not be averaged away by later quiet windows."""
    smoother = ScoreSmoother(alpha=0.35, memory=8)
    for _ in range(5):
        smoother.update(85.0)
    after_quiet = smoother.update(20.0)
    assert after_quiet > 45.0
    assert smoother.peak() == 85.0


def test_smoother_converges_on_a_steady_signal():
    smoother = ScoreSmoother(alpha=0.5)
    for _ in range(30):
        value = smoother.update(70.0)
    assert value == pytest.approx(70.0, abs=1.0)


def test_smoother_reset():
    smoother = ScoreSmoother()
    smoother.update(50.0)
    smoother.reset()
    assert smoother.value is None and smoother.history == []


# ------------------------------------------------------------------------ explanations

def test_explanation_reports_abstaining_layers_as_abstained():
    fusion = ScoreFusion().fuse([
        layer("acoustic", 0.8, 1.0, [Factor("hf_cliff_depth_db", "Cliff", 0.6)]),
        layer("speaker", 0.5, 0.0, note="no enrolment for the claimed identity"),
    ])
    explanation = build_explanation(fusion, 80.0, RiskBand.CRITICAL.value)

    statuses = {row["layer"]: row["status"] for row in explanation.layer_summary}
    assert statuses["speaker"] == "abstained"
    assert statuses["acoustic"] == "voted"
    assert any("No enrolled voice sample" in c for c in explanation.caveats)


def test_explanation_warns_early_in_the_call():
    fusion = ScoreFusion().fuse([layer("acoustic", 0.8, 1.0)])
    explanation = build_explanation(fusion, 80.0, RiskBand.CRITICAL.value, elapsed_seconds=1.0)
    assert any("still settling" in c for c in explanation.caveats)


def test_explanation_warns_when_mostly_silence():
    fusion = ScoreFusion().fuse([layer("acoustic", 0.8, 1.0)])
    explanation = build_explanation(fusion, 80.0, RiskBand.HIGH.value,
                                    speech_ratio=0.1, elapsed_seconds=20.0)
    assert any("silence or background" in c for c in explanation.caveats)


def test_explanation_surfaces_counter_evidence():
    fusion = ScoreFusion().fuse([
        layer("acoustic", 0.9, 1.0, [Factor("hf_cliff_depth_db", "Cliff", 0.8)]),
        layer("prosodic", 0.15, 0.9),
    ])
    explanation = build_explanation(fusion, 70.0, RiskBand.HIGH.value)
    assert explanation.counter_factors
    assert explanation.counter_factors[0].direction == "genuine"


def test_explanation_text_is_human_readable():
    fusion = ScoreFusion().fuse([
        layer("acoustic", 0.9, 1.0,
              [Factor("digital_silence_score", "Digitally clean silence", 0.7,
                      detail="non-speech segments have no room tone")]),
    ])
    text = build_explanation(fusion, 88.0, RiskBand.CRITICAL.value,
                             action="Block the transfer.").as_text()
    assert "Risk 88/100" in text
    assert "Digitally clean silence" in text
    assert "Recommended action: Block the transfer." in text
