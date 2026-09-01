"""Tests for the metrics, splitting and training pipeline.

These guard the numbers we publish. The leakage detector and the speaker-disjoint split
in particular are the difference between a benchmark and a press release.
"""

import numpy as np
import pytest

from ml.metrics import (
    auc_score,
    detect_leakage,
    equal_error_rate,
    evaluate,
    per_method_breakdown,
    platt_scaling,
    roc_curve,
)


# ----------------------------------------------------------------------------- metrics

def test_auc_of_perfect_and_random_separation():
    y = np.array([0] * 50 + [1] * 50)
    perfect = np.array([0.1] * 50 + [0.9] * 50)
    inverted = np.array([0.9] * 50 + [0.1] * 50)
    assert auc_score(y, perfect) == pytest.approx(1.0)
    assert auc_score(y, inverted) == pytest.approx(0.0)
    assert auc_score(y, np.full(100, 0.5)) == pytest.approx(0.5)


def test_auc_handles_a_single_class():
    assert auc_score(np.zeros(10), np.random.rand(10)) == 0.5


def test_eer_of_a_perfect_detector_is_zero():
    y = np.array([0] * 40 + [1] * 40)
    scores = np.array([0.05] * 40 + [0.95] * 40)
    eer, threshold = equal_error_rate(y, scores)
    assert eer == pytest.approx(0.0, abs=1e-6)
    assert 0.05 < threshold <= 0.95


def test_eer_of_a_coin_flip_is_about_half():
    rng = np.random.default_rng(0)
    y = np.array([0] * 500 + [1] * 500)
    eer, _ = equal_error_rate(y, rng.random(1000))
    assert 0.4 < eer < 0.6


def test_eer_is_symmetric_around_the_crossing():
    """FRR and FAR must actually be equal at the reported operating point."""
    rng = np.random.default_rng(3)
    y = np.array([0] * 300 + [1] * 300)
    scores = np.concatenate([rng.normal(0.35, 0.15, 300), rng.normal(0.65, 0.15, 300)])
    scores = np.clip(scores, 0, 1)
    eer, threshold = equal_error_rate(y, scores)

    frr = float((scores[y == 0] >= threshold).mean())
    far = float((scores[y == 1] < threshold).mean())
    assert abs(frr - far) < 0.06
    assert eer == pytest.approx((frr + far) / 2, abs=0.06)


def test_roc_curve_is_monotonic_and_bounded():
    rng = np.random.default_rng(1)
    y = np.array([0] * 100 + [1] * 100)
    scores = np.concatenate([rng.normal(0.3, 0.2, 100), rng.normal(0.7, 0.2, 100)])
    fpr, tpr, _ = roc_curve(y, scores)
    assert (np.diff(fpr) >= -1e-9).all()
    assert (np.diff(tpr) >= -1e-9).all()
    assert fpr[0] == 0.0 and tpr[0] == 0.0
    assert fpr[-1] == pytest.approx(1.0) and tpr[-1] == pytest.approx(1.0)


def test_roc_handles_ties():
    y = np.array([0, 0, 1, 1])
    fpr, tpr, _ = roc_curve(y, np.array([0.5, 0.5, 0.5, 0.5]))
    assert np.isfinite(fpr).all() and np.isfinite(tpr).all()


def test_evaluate_reports_a_full_metric_set():
    y = np.array([0] * 30 + [1] * 30)
    scores = np.array([0.2] * 30 + [0.8] * 30)
    metrics = evaluate(y, scores)

    assert metrics.accuracy == 1.0
    assert metrics.confusion == {"tp": 30, "tn": 30, "fp": 0, "fn": 0}
    assert metrics.n_genuine == 30 and metrics.n_spoof == 30
    assert metrics.fpr_at_threshold["0.50"] == 0.0
    assert metrics.tpr_at_threshold["0.50"] == 1.0
    assert "eer_percent" in metrics.as_dict()
    assert metrics.summary_line()


def test_evaluate_threshold_changes_the_operating_point():
    y = np.array([0] * 20 + [1] * 20)
    scores = np.array([0.4] * 20 + [0.6] * 20)
    assert evaluate(y, scores, decision_threshold=0.5).accuracy == 1.0
    assert evaluate(y, scores, decision_threshold=0.7).recall == 0.0


def test_per_method_breakdown_separates_families():
    y = np.array([0, 0, 1, 1, 1, 1])
    scores = np.array([0.1, 0.2, 0.9, 0.9, 0.3, 0.2])
    methods = ["human", "human", "vocoder_a", "vocoder_a", "vocoder_b", "vocoder_b"]

    breakdown = per_method_breakdown(y, scores, methods)
    assert breakdown["bonafide"]["false_positive_rate"] == 0.0
    assert breakdown["vocoder_a"]["detection_rate"] == 1.0
    assert breakdown["vocoder_b"]["detection_rate"] == 0.0   # the number an average hides


def test_platt_scaling_improves_calibration():
    rng = np.random.default_rng(2)
    y = np.concatenate([np.zeros(300), np.ones(300)])
    # Systematically over-confident scores.
    raw = np.clip(np.concatenate([rng.normal(0.35, 0.1, 300),
                                  rng.normal(0.95, 0.03, 300)]), 1e-3, 1 - 1e-3)
    a, b = platt_scaling(raw, y)
    z = np.log(raw / (1 - raw))
    calibrated = 1.0 / (1.0 + np.exp(-(a * z + b)))

    def log_loss(p):
        p = np.clip(p, 1e-9, 1 - 1e-9)
        return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))

    assert log_loss(calibrated) <= log_loss(raw) + 1e-6


# ---------------------------------------------------------------------------- leakage

def test_leakage_detector_catches_a_giveaway_feature():
    rng = np.random.default_rng(4)
    y = np.array([0] * 100 + [1] * 100)
    honest = rng.normal(0, 1, 200)
    # The DC-offset bug, reproduced: one feature differs by orders of magnitude.
    leaking = np.concatenate([rng.normal(4.4e-3, 5e-4, 100), rng.normal(4e-5, 5e-6, 100)])
    X = np.column_stack([honest, leaking])

    flagged = detect_leakage(X, y, ["honest_feature", "dc_offset"])
    assert len(flagged) == 1
    assert flagged[0]["feature"] == "dc_offset"
    assert flagged[0]["auc"] > 0.95
    assert flagged[0]["direction"] == "higher_in_genuine"


def test_leakage_detector_is_quiet_on_a_clean_corpus():
    rng = np.random.default_rng(5)
    y = np.array([0] * 100 + [1] * 100)
    # Genuinely discriminative but not a giveaway: overlapping distributions.
    X = np.column_stack([
        np.concatenate([rng.normal(0.0, 1.0, 100), rng.normal(0.8, 1.0, 100)]),
        rng.normal(0, 1, 200),
    ])
    assert detect_leakage(X, y, ["useful", "noise"]) == []


def test_leakage_detector_ignores_constant_features():
    y = np.array([0] * 50 + [1] * 50)
    X = np.column_stack([np.ones(100), np.full(100, np.nan)])
    assert detect_leakage(X, y, ["constant", "broken"]) == []


# ---------------------------------------------------------------------------- splitting

def test_split_by_speaker_is_disjoint():
    from ml.features import FeatureMatrix, split_by_speaker

    speakers = [f"spk{i % 10:02d}" for i in range(200)]
    matrix = FeatureMatrix(
        X=np.random.rand(200, 4), y=np.random.randint(0, 2, 200),
        feature_names=["a", "b", "c", "d"], utterance_ids=list(range(200)),
        speakers=speakers, methods=["m"] * 200, languages=["auto"] * 200,
        paths=[f"p{i}" for i in range(200)],
    )
    train_mask, test_mask = split_by_speaker(matrix, 0.3, seed=1)

    train_speakers = {s for s, m in zip(speakers, train_mask) if m}
    test_speakers = {s for s, m in zip(speakers, test_mask) if m}
    assert train_speakers and test_speakers
    assert not (train_speakers & test_speakers)
    assert (train_mask | test_mask).all()          # every row is used exactly once
    assert not (train_mask & test_mask).any()


def test_split_holdout_method_isolates_one_family():
    from ml.features import FeatureMatrix, split_holdout_method

    methods = ["human"] * 40 + ["vocoder_a"] * 30 + ["vocoder_b"] * 30
    labels = np.array([0] * 40 + [1] * 60)
    matrix = FeatureMatrix(
        X=np.random.rand(100, 3), y=labels, feature_names=["a", "b", "c"],
        utterance_ids=list(range(100)), speakers=["s"] * 100, methods=methods,
        languages=["auto"] * 100, paths=[f"p{i}" for i in range(100)],
    )
    train_mask, test_mask = split_holdout_method(matrix, "vocoder_a")

    assert "vocoder_a" not in {m for m, keep in zip(methods, train_mask) if keep}
    test_methods = {m for m, keep in zip(methods, test_mask) if keep}
    assert test_methods == {"human", "vocoder_a"}     # unseen attack + genuine controls


def test_aggregate_to_utterances_uses_a_high_quantile():
    """A short suspicious burst inside a long call must survive aggregation."""
    from ml.features import FeatureMatrix, aggregate_to_utterances

    matrix = FeatureMatrix(
        X=np.zeros((8, 2)), y=np.array([1] * 8), feature_names=["a", "b"],
        utterance_ids=[0] * 8, speakers=["s"] * 8, methods=["m"] * 8,
        languages=["auto"] * 8, paths=["p"] * 8,
    )
    window_scores = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.9, 0.95, 0.9])
    scores, truth, _ = aggregate_to_utterances(matrix, window_scores)

    assert len(scores) == 1
    assert scores[0] > 0.5          # a mean would give 0.4 and miss it
    assert truth[0] == 1


# ---------------------------------------------------------------------------- training

def test_train_end_to_end_on_a_tiny_corpus(tmp_path):
    """Full loop: build a small corpus, train, verify the artifact is loadable."""
    from ml.datasets.build import build_corpus
    from ml.train import train
    from voiceguard.config import Settings
    from voiceguard.models.registry import ModelRegistry

    corpus = tmp_path / "corpus"
    manifest = build_corpus(str(corpus), per_class=8, speakers=4, seed=3,
                            duration_range=(2.0, 2.5))
    assert manifest["n_utterances"] == 16
    assert set(manifest["conditions"])                       # conditions recorded

    out = tmp_path / "artifacts"
    report = train(str(corpus), str(out), max_windows=3, seed=3, skip_unseen=True)

    assert (out / "bootstrap_model.joblib").exists()
    assert (out / "metrics.json").exists()
    assert report["split"]["strategy"].startswith("speaker-disjoint")
    assert not set(report["split"]["train_speakers"]) & set(report["split"]["test_speakers"])

    registry = ModelRegistry(Settings(model_dir=str(out),
                                      model_file="bootstrap_model.joblib"))
    assert registry.bundle is not None
    assert registry.bundle.feature_names
    assert len(registry.calibration) == 2
    assert registry.detector("acoustic").model_id.startswith("acoustic-")


def test_corpus_builder_gives_both_classes_the_same_conditions(tmp_path):
    """Condition must never correlate with the label, or it becomes the label."""
    from ml.datasets.build import build_corpus

    manifest = build_corpus(str(tmp_path / "c"), per_class=20, speakers=5, seed=9,
                            duration_range=(1.5, 2.0))
    genuine = [u for u in manifest["utterances"] if u["label"] == 0]
    spoof = [u for u in manifest["utterances"] if u["label"] == 1]

    genuine_telephony = sum(1 for u in genuine if u["condition"] == "telephony")
    spoof_telephony = sum(1 for u in spoof if u["condition"] == "telephony")
    # Same Bernoulli draw for both classes: allow sampling noise, reject a systematic split.
    assert abs(genuine_telephony - spoof_telephony) <= 8

    # And the same speakers must appear on both sides, so timbre cannot separate them.
    assert {u["speaker_id"] for u in genuine} == {u["speaker_id"] for u in spoof}


def test_loaders_detect_the_folder_layout(tmp_path):
    from ml.datasets.loaders import detect_layout, load_corpus
    from voiceguard.audio.io import write_wav
    from voiceguard.simulation import synthesize_bonafide, synthesize_cloned

    (tmp_path / "bonafide").mkdir()
    (tmp_path / "spoof").mkdir()
    write_wav(str(tmp_path / "bonafide" / "a.wav"), synthesize_bonafide(1.5, seed=1), 16000)
    write_wav(str(tmp_path / "spoof" / "b.wav"), synthesize_cloned(1.5, seed=1), 16000)

    assert detect_layout(str(tmp_path)) == "folders"
    items = load_corpus(str(tmp_path))
    assert {i.label for i in items} == {0, 1}


def test_loader_raises_on_a_missing_corpus():
    from ml.datasets.loaders import load_corpus

    with pytest.raises(FileNotFoundError):
        load_corpus("/nonexistent/corpus/path")
