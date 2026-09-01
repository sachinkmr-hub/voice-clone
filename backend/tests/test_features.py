"""Tests for the feature stack.

Beyond "does it run", these assert the *direction* of the signals we tell judges and
users about — e.g. that a band-limited vocoder output really does trip the cliff probe,
and that generated silence really does read as digital.
"""

import numpy as np
import pytest
from scipy.signal import butter, sosfilt

from voiceguard.features.artifacts import artifact_feature_dict, bandwidth_features, silence_features
from voiceguard.features.embedding import EnrolmentStore, classical_embedding, cosine_distance
from voiceguard.features.extractor import FeatureExtractor
from voiceguard.features.prosody import prosody_feature_dict, track_pitch
from voiceguard.features.spectral import (
    mel_filterbank,
    mel_spectrogram,
    mfcc,
    modulation_features,
    spectral_feature_dict,
    stft,
)
from voiceguard.simulation import CLONE_METHODS, synthesize_bonafide, synthesize_cloned

SR = 16000


@pytest.fixture(scope="module")
def bonafide():
    return synthesize_bonafide(3.0, seed=11, language="hi-IN")


@pytest.fixture(scope="module")
def cloned():
    return synthesize_cloned(3.0, seed=11, language="hi-IN", method="neural")


# ---------------------------------------------------------------------------- spectral

def test_stft_and_mel_shapes():
    audio = np.random.randn(SR).astype(np.float32) * 0.1
    spec = stft(audio)
    assert spec.shape[1] == 512 // 2 + 1
    assert mel_spectrogram(audio).shape[1] == 40
    assert mfcc(audio).shape[1] == 20


def test_mel_filterbank_is_normalised_and_positive():
    fb = mel_filterbank(SR, 512, 40)
    assert fb.shape == (40, 257)
    assert (fb >= 0).all()
    assert (fb.sum(axis=1) > 0).all()


def test_modulation_peak_tracks_the_real_envelope_rate():
    t = np.arange(2 * SR) / SR
    carrier = np.sin(2 * np.pi * 200 * t)
    for rate in (3.0, 6.0):
        signal = (0.4 * carrier * (0.5 + 0.5 * np.sin(2 * np.pi * rate * t))).astype(np.float32)
        assert modulation_features(signal, SR)["mod_peak_hz"] == pytest.approx(rate, abs=1.0)


def test_spectral_features_are_finite_for_every_input():
    for audio in (
        np.zeros(SR, dtype=np.float32),
        np.ones(SR, dtype=np.float32),
        (np.random.randn(SR) * 0.1).astype(np.float32),
        np.zeros(300, dtype=np.float32),
    ):
        features = spectral_feature_dict(audio, SR)
        assert features and all(np.isfinite(v) for v in features.values())


def test_lowpass_moves_the_spectral_centroid_down():
    noise = (np.random.randn(SR) * 0.2).astype(np.float32)
    sos = butter(8, 3000 / (SR / 2), btype="low", output="sos")
    filtered = sosfilt(sos, noise).astype(np.float32)
    assert (
        spectral_feature_dict(filtered, SR)["spec_centroid_mean"]
        < spectral_feature_dict(noise, SR)["spec_centroid_mean"]
    )


# ----------------------------------------------------------------------------- prosody

def test_pitch_tracker_recovers_a_known_f0():
    t = np.arange(SR) / SR
    for f0 in (110.0, 180.0, 250.0):
        signal = sum((1.0 / k) * np.sin(2 * np.pi * k * f0 * t) for k in range(1, 12))
        track = track_pitch((0.4 * signal).astype(np.float32), SR)
        voiced = track.voiced_f0()
        assert voiced.size > 10
        assert float(np.median(voiced)) == pytest.approx(f0, rel=0.05)


def test_pitch_tracker_marks_noise_as_unvoiced():
    track = track_pitch((np.random.randn(SR) * 0.1).astype(np.float32), SR)
    assert track.voiced_ratio < 0.5


def test_jitter_separates_perturbed_from_steady_periods():
    rng = np.random.default_rng(4)
    t = np.arange(2 * SR) / SR

    def make(jitter):
        f0 = 170.0 * (1.0 + jitter * rng.standard_normal(len(t)))
        phase = 2 * np.pi * np.cumsum(f0) / SR
        signal = sum((1.0 / k) * np.sin(k * phase) for k in range(1, 20))
        return (0.35 * signal).astype(np.float32)

    steady = prosody_feature_dict(make(0.0002), SR)
    rough = prosody_feature_dict(make(0.02), SR)
    assert rough["jitter_ppq5"] > steady["jitter_ppq5"] * 3
    assert rough["cycle_count"] > 100


def test_prosody_features_finite_on_silence():
    features = prosody_feature_dict(np.zeros(SR, dtype=np.float32), SR)
    assert all(np.isfinite(v) for v in features.values())
    assert features["voiced_ratio"] == 0.0


def test_language_profile_shifts_the_z_scores():
    audio = synthesize_bonafide(2.0, seed=7)
    hindi = prosody_feature_dict(audio, SR, language="hi-IN")
    tamil = prosody_feature_dict(audio, SR, language="ta-IN")
    assert hindi["f0_z_mean"] != tamil["f0_z_mean"]


# --------------------------------------------------------------------------- artifacts

def test_band_limitation_is_detected_as_a_cliff():
    noise = (np.random.randn(2 * SR) * 0.2).astype(np.float32)
    sos = butter(10, 5000 / (SR / 2), btype="low", output="sos")
    limited = sosfilt(sos, noise).astype(np.float32)

    wide = bandwidth_features(noise, SR)
    narrow = bandwidth_features(limited, SR)
    assert narrow["hf_cliff_hz"] > 0
    assert narrow["hf_cliff_depth_db"] > wide["hf_cliff_depth_db"]
    assert narrow["hf_energy_ratio"] < wide["hf_energy_ratio"]


def test_digital_silence_scores_higher_than_room_tone():
    speech = synthesize_bonafide(2.0, seed=3)
    digital = speech.copy()
    digital[int(0.5 * SR) : int(1.2 * SR)] = 0.0
    roomy = speech.copy()
    roomy[int(0.5 * SR) : int(1.2 * SR)] = (
        0.002 * np.random.randn(int(0.7 * SR))
    ).astype(np.float32)

    assert (
        silence_features(digital, SR)["digital_silence_score"]
        > silence_features(roomy, SR)["digital_silence_score"]
    )


def test_artifact_features_finite_for_degenerate_inputs():
    for audio in (np.zeros(SR, dtype=np.float32), np.ones(SR, dtype=np.float32),
                  np.zeros(200, dtype=np.float32)):
        features = artifact_feature_dict(audio, SR)
        assert features and all(np.isfinite(v) for v in features.values())


@pytest.mark.parametrize("method", CLONE_METHODS)
def test_every_vocoder_family_is_distinguishable_somewhere(method):
    """Each synthesis family must move at least one probe away from bona fide."""
    genuine = artifact_feature_dict(synthesize_bonafide(3.0, seed=21), SR)
    fake = artifact_feature_dict(synthesize_cloned(3.0, seed=21, method=method), SR)
    signals = [
        fake["digital_silence_score"] > genuine["digital_silence_score"],
        fake["mel_frame_corr"] < genuine["mel_frame_corr"],
        fake["env_second_diff"] > genuine["env_second_diff"],
        fake["hf_cliff_depth_db"] > genuine["hf_cliff_depth_db"],
    ]
    assert sum(signals) >= 2, f"{method} left too few tells: {signals}"


# --------------------------------------------------------------------------- embedding

def test_embedding_is_unit_norm_and_stable():
    audio = synthesize_bonafide(2.0, seed=5)
    vec = classical_embedding(audio, SR)
    assert vec.shape == (64,)
    assert float(np.linalg.norm(vec)) == pytest.approx(1.0, abs=1e-4)
    assert np.allclose(vec, classical_embedding(audio, SR))


def test_embedding_separates_speakers_more_than_it_separates_takes():
    from voiceguard.simulation.voice import SpeakerTimbre

    rng = np.random.default_rng(9)
    speaker_a = SpeakerTimbre.random(rng, "a")
    speaker_b = SpeakerTimbre.random(np.random.default_rng(99), "b")

    take1 = classical_embedding(synthesize_bonafide(2.5, timbre=speaker_a, seed=1), SR)
    take2 = classical_embedding(synthesize_bonafide(2.5, timbre=speaker_a, seed=2), SR)
    other = classical_embedding(synthesize_bonafide(2.5, timbre=speaker_b, seed=3), SR)

    assert cosine_distance(take1, take2) < cosine_distance(take1, other)


def test_embedding_of_silence_is_zero():
    assert not np.any(classical_embedding(np.zeros(SR, dtype=np.float32), SR))


def test_enrolment_store_compare_and_clear():
    store = EnrolmentStore()
    from voiceguard.simulation.voice import SpeakerTimbre

    timbre = SpeakerTimbre.random(np.random.default_rng(2), "cfo")
    for seed in range(3):
        store.enrol("cfo", synthesize_bonafide(2.0, timbre=timbre, seed=seed), SR)

    assert store.has("cfo") and not store.has("nobody")
    assert store.identities() == ["cfo"]

    match = store.compare("cfo", classical_embedding(
        synthesize_bonafide(2.0, timbre=timbre, seed=42), SR))
    assert match is not None and match["samples"] == 3.0
    assert 0.0 <= match["distance"] <= 2.0

    assert store.compare("cfo", np.zeros(64, dtype=np.float32)) is None
    store.clear("cfo")
    assert not store.has("cfo")


# --------------------------------------------------------------------------- extractor

def test_extractor_produces_a_stable_schema(bonafide):
    extractor = FeatureExtractor()
    schema = extractor.schema()
    bundle = extractor.extract(bonafide[:SR], language="hi-IN")

    assert len(schema) > 100
    assert schema == extractor.schema()          # cached, deterministic
    assert bundle.vector(schema).shape == (len(schema),)
    assert np.isfinite(bundle.vector(schema)).all()


def test_extractor_reports_speech_and_level(bonafide):
    bundle = FeatureExtractor().extract(bonafide[:SR], language="hi-IN")
    assert bundle.speech_detected
    assert bundle.speech_ratio > 0.3
    assert -60.0 < bundle.level_dbfs < 0.0
    assert bundle.extraction_ms > 0.0
    assert bundle.embedding.shape == (64,)


def test_extractor_short_window_is_safe():
    bundle = FeatureExtractor().extract(np.zeros(100, dtype=np.float32))
    assert not bundle.speech_detected
    assert bundle.all_features() == {}


def test_extractor_meets_the_latency_budget(bonafide):
    """One 1 s window must cost far less than the 500 ms hop it has to keep up with."""
    extractor = FeatureExtractor()
    extractor.extract(bonafide[:SR])  # warm caches
    timings = [extractor.extract(bonafide[:SR]).extraction_ms for _ in range(3)]
    assert float(np.median(timings)) < 300.0


def test_bonafide_and_cloned_differ_on_the_full_vector(bonafide, cloned):
    extractor = FeatureExtractor()
    schema = extractor.schema()
    a = extractor.extract(bonafide[:SR], language="hi-IN").vector(schema)
    b = extractor.extract(cloned[:SR], language="hi-IN").vector(schema)
    assert not np.allclose(a, b)
