"""Feature orchestration.

One call to :meth:`FeatureExtractor.extract` turns a window of audio into everything the
detection layers need, computing each shared intermediate (VAD pass, pitch track) exactly
once. The resulting :class:`FeatureBundle` carries both the named dictionary (for
explanations and audit) and a stable ordered vector (for the trained model).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from voiceguard.audio.io import rms
from voiceguard.audio.vad import SpeechSegments, detect_speech
from voiceguard.config import MIN_ANALYSIS_SAMPLES, SAMPLE_RATE
from voiceguard.features.artifacts import artifact_feature_dict
from voiceguard.features.embedding import SpeakerEmbedder
from voiceguard.features.prosody import (
    contour_features,
    jitter_shimmer,
    rhythm_features,
    track_pitch,
)
from voiceguard.features.spectral import spectral_feature_dict
from voiceguard.config import language_profile


@dataclass
class FeatureBundle:
    """All features for one analysis window."""

    spectral: Dict[str, float] = field(default_factory=dict)
    prosodic: Dict[str, float] = field(default_factory=dict)
    artifacts: Dict[str, float] = field(default_factory=dict)
    embedding: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float32))

    speech_detected: bool = False
    speech_ratio: float = 0.0
    level_dbfs: float = -120.0
    duration: float = 0.0
    language: str = "auto"
    extraction_ms: float = 0.0

    def all_features(self) -> Dict[str, float]:
        merged: Dict[str, float] = {}
        merged.update(self.spectral)
        merged.update(self.prosodic)
        merged.update(self.artifacts)
        return merged

    def names(self) -> List[str]:
        return sorted(self.all_features())

    def vector(self, names: Optional[List[str]] = None) -> np.ndarray:
        """Ordered feature vector. Pass ``names`` to match a trained model's schema."""
        features = self.all_features()
        keys = names if names is not None else sorted(features)
        return np.array([float(features.get(k, 0.0)) for k in keys], dtype=np.float32)

    def as_dict(self) -> dict:
        return {
            "speech_detected": self.speech_detected,
            "speech_ratio": round(self.speech_ratio, 4),
            "level_dbfs": round(self.level_dbfs, 2),
            "duration": round(self.duration, 3),
            "language": self.language,
            "extraction_ms": round(self.extraction_ms, 2),
            "features": {k: round(float(v), 6) for k, v in self.all_features().items()},
        }


class FeatureExtractor:
    """Stateless feature front end (safe to share across sessions and threads)."""

    def __init__(
        self,
        sample_rate: int = SAMPLE_RATE,
        *,
        embedder: Optional[SpeakerEmbedder] = None,
        compute_embedding: bool = True,
    ) -> None:
        self.sample_rate = sample_rate
        self.embedder = embedder or SpeakerEmbedder()
        self.compute_embedding = compute_embedding
        self._schema: Optional[List[str]] = None

    # ------------------------------------------------------------------ extraction
    def extract(
        self,
        x: np.ndarray,
        *,
        language: Optional[str] = None,
        segments: Optional[SpeechSegments] = None,
    ) -> FeatureBundle:
        started = time.perf_counter()
        x = np.asarray(x, dtype=np.float32).ravel()
        bundle = FeatureBundle(
            language=language or "auto",
            duration=len(x) / float(self.sample_rate),
        )

        if x.size < MIN_ANALYSIS_SAMPLES:
            bundle.extraction_ms = (time.perf_counter() - started) * 1000.0
            return bundle

        level = rms(x)
        bundle.level_dbfs = float(20.0 * np.log10(max(level, 1e-12)))

        segments = segments if segments is not None else detect_speech(x, self.sample_rate)
        bundle.speech_detected = segments.has_speech
        bundle.speech_ratio = segments.speech_ratio

        # Shared intermediates: the pitch track is the single most expensive thing here,
        # and three separate feature families need it.
        track = track_pitch(x, self.sample_rate)
        profile = language_profile(language)
        f0_hint = float(np.median(track.voiced_f0())) if track.voiced.any() else 0.0

        bundle.spectral = spectral_feature_dict(x, self.sample_rate)

        prosodic: Dict[str, float] = {}
        prosodic.update(contour_features(track, profile))
        prosodic.update(jitter_shimmer(track, x, self.sample_rate))
        prosodic.update(rhythm_features(x, segments, profile, self.sample_rate))
        prosodic["speech_ratio"] = float(segments.speech_ratio)
        bundle.prosodic = prosodic

        bundle.artifacts = artifact_feature_dict(
            x, self.sample_rate, segments=segments, f0_hint=f0_hint
        )

        if self.compute_embedding:
            bundle.embedding = self.embedder.embed(x, self.sample_rate)

        # Guarantee finiteness so no downstream model ever sees a NaN.
        for group in (bundle.spectral, bundle.prosodic, bundle.artifacts):
            for key, value in list(group.items()):
                if not np.isfinite(value):
                    group[key] = 0.0

        bundle.extraction_ms = (time.perf_counter() - started) * 1000.0
        return bundle

    # --------------------------------------------------------------------- schema
    def schema(self) -> List[str]:
        """The canonical ordered feature-name list, derived from a probe window.

        Models persist this list so that a feature added in a later version cannot
        silently shift the columns of an already-trained model.
        """
        if self._schema is None:
            probe = np.zeros(int(self.sample_rate * 1.0), dtype=np.float32)
            probe[::97] = 0.05  # a little structure so every branch produces its keys
            self._schema = self.extract(probe).names()
        return list(self._schema)


_DEFAULT_EXTRACTOR: Optional[FeatureExtractor] = None


def default_extractor() -> FeatureExtractor:
    global _DEFAULT_EXTRACTOR
    if _DEFAULT_EXTRACTOR is None:
        _DEFAULT_EXTRACTOR = FeatureExtractor()
    return _DEFAULT_EXTRACTOR


def extract_features(
    x: np.ndarray,
    sample_rate: int = SAMPLE_RATE,
    *,
    language: Optional[str] = None,
) -> FeatureBundle:
    """Convenience wrapper around the process-wide default extractor."""
    extractor = default_extractor() if sample_rate == SAMPLE_RATE else FeatureExtractor(sample_rate)
    return extractor.extract(x, language=language)
