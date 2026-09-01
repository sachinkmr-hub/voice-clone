"""Tests for the detection layers, the registry and rule calibration."""

import numpy as np
import pytest

from voiceguard.features.embedding import EnrolmentStore, classical_embedding
from voiceguard.features.extractor import FeatureExtractor
from voiceguard.models.acoustic import ACOUSTIC_RULES, AcousticHeuristicDetector, AcousticModelDetector
from voiceguard.models.base import EvidenceRule, LayerResult, RuleScorer, merge_factors, ramp
from voiceguard.models.calibration import directional_auc, fit_rule_anchors
from voiceguard.models.context import CallContext, ContextDetector, urgency_hits
from voiceguard.models.prosodic import ProsodicDetector
from voiceguard.models.registry import ModelRegistry
from voiceguard.models.speaker import SpeakerConsistencyDetector
from voiceguard.simulation import synthesize_bonafide, synthesize_cloned

SR = 16000


@pytest.fixture(scope="module")
def extractor():
    return FeatureExtractor()


@pytest.fixture(scope="module")
def genuine_features(extractor):
    return extractor.extract(synthesize_bonafide(3.0, seed=31)[SR : 2 * SR]).all_features()


@pytest.fixture(scope="module")
def cloned_features(extractor):
    audio = synthesize_cloned(3.0, seed=31, method="neural")
    return extractor.extract(audio[SR : 2 * SR]).all_features()


# ------------------------------------------------------------------------------- base

def test_ramp_handles_both_directions():
    assert ramp(5.0, 0.0, 10.0) == pytest.approx(0.5)
    assert ramp(5.0, 10.0, 0.0) == pytest.approx(0.5)
    assert ramp(-1.0, 0.0, 10.0) == 0.0
    assert ramp(99.0, 0.0, 10.0) == 1.0
    assert ramp(float("nan"), 0.0, 1.0) == 0.0


def test_rule_requires_gate_blocks_meaningless_zeroes():
    rule = EvidenceRule("jitter", "Jitter", low=0.01, high=0.001, requires=("cycles",))
    assert rule.evaluate({"jitter": 0.0, "cycles": 0.0}) is None      # no voiced speech
    assert rule.evaluate({"jitter": 0.0, "cycles": 100.0}) is not None


def test_rule_scorer_with_no_applicable_rules_reports_zero_confidence():
    scorer = RuleScorer([EvidenceRule("absent", "Absent", 0.0, 1.0)])
    score, confidence, factors = scorer.score({"other": 1.0})
    assert score == 0.5 and confidence == 0.0 and factors == []


def test_layer_result_unavailable_is_not_a_genuine_vote():
    result = LayerResult.unavailable("speaker", "no enrolment")
    assert result.confidence == 0.0
    assert result.score == 0.5


def test_merge_factors_respects_layer_weights():
    from voiceguard.models.base import Factor

    acoustic = LayerResult("acoustic", 0.9, 1.0, [Factor("a", "Acoustic thing", 0.5)])
    context = LayerResult("context", 0.9, 1.0, [Factor("c", "Context thing", 0.9)])

    unweighted = merge_factors([acoustic, context])
    assert unweighted[0].code == "c"          # raw contribution wins

    weighted = merge_factors([acoustic, context], weights={"acoustic": 0.85, "context": 0.15})
    assert weighted[0].code == "a"            # the layer that drove the decision leads
    assert sum(f.contribution for f in weighted) == pytest.approx(1.0, abs=1e-6)


# --------------------------------------------------------------------------- acoustic

def test_acoustic_heuristic_scores_cloned_above_genuine(genuine_features, cloned_features):
    detector = AcousticHeuristicDetector()
    genuine = detector.analyze(genuine_features)
    cloned = detector.analyze(cloned_features)
    assert cloned.score > genuine.score
    assert genuine.confidence > 0 and cloned.confidence > 0
    assert cloned.factors


def test_acoustic_heuristic_handles_empty_features():
    result = AcousticHeuristicDetector().analyze({})
    assert result.confidence == 0.0


def test_acoustic_model_detector_falls_back_when_inference_raises(genuine_features):
    class Broken:
        def predict_proba(self, _):
            raise RuntimeError("bad artifact")

    detector = AcousticModelDetector(Broken(), list(genuine_features), model_id="broken")
    result = detector.analyze(genuine_features)
    assert "model inference failed" in result.note
    assert result.confidence > 0          # still produced a usable verdict


def test_acoustic_model_detector_blends_with_the_heuristic(genuine_features):
    class AlwaysSynthetic:
        def predict_proba(self, _):
            return np.array([[0.01, 0.99]])

    names = sorted(genuine_features)
    blended = AcousticModelDetector(AlwaysSynthetic(), names, heuristic_blend=0.5)
    pure = AcousticModelDetector(AlwaysSynthetic(), names, heuristic_blend=0.0)

    # The physics prior says "genuine", so blending must pull the score down from 0.99.
    assert blended.analyze(genuine_features).score < pure.analyze(genuine_features).score


def test_acoustic_model_uses_its_own_feature_order():
    class Echo:
        def predict_proba(self, vector):
            return np.array([[0.0, float(vector[0][0])]])

    detector = AcousticModelDetector(Echo(), ["target", "other"], heuristic_blend=0.0)
    assert detector.analyze({"target": 0.8, "other": 0.1}).score == pytest.approx(0.8)
    # A feature the model never saw must not shift the columns.
    assert detector.analyze({"aaa_new": 9.9, "target": 0.8}).score == pytest.approx(0.8)


# --------------------------------------------------------------------------- prosodic

def test_prosodic_abstains_without_voiced_speech():
    result = ProsodicDetector().analyze({"cycle_count": 2.0, "speech_ratio": 0.9})
    assert result.confidence == 0.0
    assert "insufficient voiced speech" in result.note


def test_prosodic_confidence_grows_with_voiced_material(genuine_features):
    detector = ProsodicDetector()
    thin = dict(genuine_features, cycle_count=30.0)
    rich = dict(genuine_features, cycle_count=400.0)
    assert detector.analyze(rich).confidence > detector.analyze(thin).confidence


def test_prosodic_flags_flat_micro_variation(genuine_features):
    detector = ProsodicDetector()
    flat = dict(genuine_features, f0_micro_var=0.0005, jitter_ppq5=0.00002,
                shimmer_apq3=0.001, period_cv=0.0005, pause_std=0.0,
                silence_floor_std_db=0.05, cycle_count=300.0, speech_ratio=0.8)
    lively = dict(genuine_features, f0_micro_var=0.05, jitter_ppq5=0.008,
                  shimmer_apq3=0.06, period_cv=0.05, pause_std=0.3,
                  silence_floor_std_db=5.0, cycle_count=300.0, speech_ratio=0.8)
    assert detector.analyze(flat).score > detector.analyze(lively).score


# ---------------------------------------------------------------------------- speaker

def test_speaker_abstains_without_enrolment_or_history():
    result = SpeakerConsistencyDetector().analyze({}, {"embedding": np.ones(64) / 8})
    assert result.confidence == 0.0


def test_speaker_abstains_without_an_embedding():
    assert SpeakerConsistencyDetector().analyze({}, {"embedding": np.zeros(64)}).confidence == 0.0


def test_speaker_flags_a_mismatched_enrolled_identity():
    from voiceguard.simulation.voice import SpeakerTimbre

    store = EnrolmentStore()
    boss = SpeakerTimbre.random(np.random.default_rng(1), "boss")
    impostor = SpeakerTimbre.random(np.random.default_rng(77), "impostor")
    for seed in range(4):
        store.enrol("boss", synthesize_bonafide(2.0, timbre=boss, seed=seed), SR)

    detector = SpeakerConsistencyDetector(store)
    same = detector.analyze({}, {
        "embedding": classical_embedding(synthesize_bonafide(2.0, timbre=boss, seed=50), SR),
        "identity": "boss"})
    other = detector.analyze({}, {
        "embedding": classical_embedding(synthesize_bonafide(2.0, timbre=impostor, seed=51), SR),
        "identity": "boss"})

    assert other.score > same.score
    assert same.confidence > 0 and other.confidence > 0


def test_speaker_detects_within_call_drift():
    from voiceguard.simulation.voice import SpeakerTimbre

    first = classical_embedding(
        synthesize_bonafide(2.0, timbre=SpeakerTimbre.random(np.random.default_rng(2)), seed=1), SR)
    later = classical_embedding(
        synthesize_bonafide(2.0, timbre=SpeakerTimbre.random(np.random.default_rng(300)), seed=2), SR)

    detector = SpeakerConsistencyDetector()
    stable = detector.analyze({}, {"embedding": first, "history": [first, first, first]})
    switched = detector.analyze({}, {"embedding": later, "history": [first, first, first]})
    assert switched.score > stable.score


# ---------------------------------------------------------------------------- context

def test_urgency_hits_finds_english_and_hindi_pressure_language():
    assert urgency_hits("Please transfer it urgently and don't tell anyone")
    assert urgency_hits("yeh turant karo, kisi ko mat batana")
    assert urgency_hits("Good morning, calling about last week's invoice") == []


def test_context_abstains_without_metadata():
    assert ContextDetector().analyze({}, {}).confidence == 0.0
    assert ContextDetector().analyze({}, None).confidence == 0.0


def test_context_scores_a_classic_vishing_setup_high():
    benign = CallContext(known_contact=True, caller_id_verified=True, origin="pstn",
                         local_hour=14, transaction_amount=5000)
    hostile = CallContext(known_contact=False, caller_id_verified=False, origin="international",
                          local_hour=2, transaction_amount=4_500_000, claimed_role="CFO",
                          prior_fraud_reports=2, first_contact=True,
                          transcript="urgent, share the OTP now, don't tell anyone")

    detector = ContextDetector()
    assert detector.analyze({}, {"call_context": hostile}).score > \
           detector.analyze({}, {"call_context": benign}).score


def test_context_accepts_a_plain_dict():
    result = ContextDetector().analyze({}, {"call_context": {"known_contact": False,
                                                             "transaction_amount": 900000}})
    assert result.confidence > 0


# ------------------------------------------------------------------------- calibration

def test_directional_auc_orients_itself():
    genuine = np.random.default_rng(0).normal(0.0, 1.0, 200)
    spoof = np.random.default_rng(1).normal(3.0, 1.0, 200)
    assert directional_auc(genuine, spoof, ascending=True) > 0.9
    assert directional_auc(genuine, spoof, ascending=False) < 0.1


def test_fit_rule_anchors_moves_anchors_off_the_genuine_distribution():
    rng = np.random.default_rng(5)
    genuine = [{"f0_micro_var": float(v)} for v in rng.normal(0.05, 0.01, 200)]
    spoof = [{"f0_micro_var": float(v)} for v in rng.normal(0.005, 0.002, 200)]
    rules = [EvidenceRule("f0_micro_var", "Micro-variation", low=0.020, high=0.004)]

    fitted, report = fit_rule_anchors(rules, genuine, spoof)
    assert len(fitted) == 1
    # Descending rule: `low` should now sit inside the genuine spread, not below it.
    assert fitted[0].low > 0.03
    assert fitted[0].high < fitted[0].low
    assert report["f0_micro_var"]["action"] == "refitted"


def test_fit_rule_anchors_prunes_useless_rules():
    rng = np.random.default_rng(6)
    rows = [{"noise": float(v)} for v in rng.normal(0.0, 1.0, 200)]
    other = [{"noise": float(v)} for v in rng.normal(0.0, 1.0, 200)]
    fitted, report = fit_rule_anchors(
        [EvidenceRule("noise", "Noise", low=0.0, high=1.0)], rows, other)
    assert fitted == []
    assert "pruned" in report["noise"]["action"]


def test_fit_rule_anchors_keeps_defaults_on_small_samples():
    rules = list(ACOUSTIC_RULES[:2])
    fitted, report = fit_rule_anchors(rules, [{"hf_cliff_depth_db": 1.0}] * 5)
    assert fitted[0].low == rules[0].low
    assert "too few samples" in report["hf_cliff_depth_db"]["action"]


# ----------------------------------------------------------------------------- registry

def test_registry_degrades_without_a_model_artifact(tmp_path):
    from voiceguard.config import Settings

    settings = Settings(model_dir=str(tmp_path), model_file="missing.joblib")
    registry = ModelRegistry(settings)

    assert registry.bundle is None
    assert registry.is_degraded
    assert any("no trained acoustic model" in note for note in registry.degraded)
    assert isinstance(registry.detector("acoustic"), AcousticHeuristicDetector)
    assert set(registry.detectors()) == {"acoustic", "prosodic", "speaker", "context"}


def test_registry_survives_a_corrupt_artifact(tmp_path):
    from voiceguard.config import Settings

    bad = tmp_path / "bad.joblib"
    bad.write_bytes(b"not a joblib file at all")
    registry = ModelRegistry(Settings(model_dir=str(tmp_path), model_file="bad.joblib"))

    assert registry.bundle is None
    assert any("could not load" in note for note in registry.degraded)
    assert registry.detector("acoustic") is not None      # system still comes up


def test_registry_describe_is_serialisable(tmp_path):
    from voiceguard.config import Settings
    import json

    registry = ModelRegistry(Settings(model_dir=str(tmp_path), model_file="none.joblib"))
    assert json.loads(json.dumps(registry.describe()))["model_loaded"] is False
