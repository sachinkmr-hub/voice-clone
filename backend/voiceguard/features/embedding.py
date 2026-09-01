"""Speaker embedding for layer-3 cross-session consistency.

This is a deliberately classical, dependency-free embedding: cepstral statistics with
mean normalisation, a coarse long-term spectral envelope, and pitch statistics, all
L2-normalised. It is not competitive with an x-vector/ECAPA network for open-set speaker
recognition, and we do not claim it is — see ``docs/MODEL_CARD.md``. What it *is* good at
is the job layer 3 actually needs:

* comparing a live call against a small set of enrolled genuine samples for the same
  claimed identity (a closed-set, few-speaker decision), and
* measuring embedding drift *within* one call, which catches a splice between a real
  human hand-off and a synthesised segment.

If ``torch``/``transformers`` are installed, :func:`available_backends` reports the
neural path and :class:`SpeakerEmbedder` will prefer it.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np

from voiceguard.config import SAMPLE_RATE
from voiceguard.features.prosody import track_pitch
from voiceguard.features.spectral import magnitude_spectrogram, mel_filterbank, mfcc

EPS = 1e-12
EMBEDDING_DIM = 64


def available_backends() -> List[str]:
    """Which embedding backends this installation can use."""
    backends = ["classical"]
    try:  # pragma: no cover - optional dependency
        import torch  # noqa: F401
        import transformers  # noqa: F401

        backends.append("wavlm")
    except Exception:
        pass
    return backends


def classical_embedding(x: np.ndarray, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    """A 64-dimensional, L2-normalised speaker embedding."""
    x = np.asarray(x, dtype=np.float32)
    if x.size < int(0.2 * sample_rate):
        return np.zeros(EMBEDDING_DIM, dtype=np.float32)

    coeffs = mfcc(x, sample_rate)
    if coeffs.shape[0] < 2:
        return np.zeros(EMBEDDING_DIM, dtype=np.float32)

    # Cepstral mean normalisation removes the channel (handset / codec) so that the same
    # speaker over PSTN and over VoIP still lands close together. c0 is dropped entirely:
    # it is frame energy, i.e. gain, not identity.
    cmn = coeffs - coeffs.mean(axis=0, keepdims=True)
    mfcc_mean = coeffs[:, 1:21].mean(axis=0)
    mfcc_std = cmn[:, 1:21].std(axis=0)

    mag = magnitude_spectrogram(x)
    power = (mag.astype(np.float64) ** 2).mean(axis=0)
    fb = mel_filterbank(sample_rate, n_mels=16)
    envelope = np.log(fb @ power + 1e-10)
    envelope = envelope - envelope.mean()

    track = track_pitch(x, sample_rate)
    voiced = track.voiced_f0()
    if voiced.size:
        f0_stats = np.array([
            np.log(max(float(np.mean(voiced)), 1.0)),
            float(np.std(voiced) / max(np.mean(voiced), EPS)),
            float(np.percentile(voiced, 90) / max(np.mean(voiced), EPS)),
            track.voiced_ratio,
        ])
    else:
        f0_stats = np.zeros(4)

    # Each block is normalised on its own before concatenation, otherwise the block with
    # the largest raw units (the cepstral means) drowns out pitch and envelope entirely.
    blocks = [
        (mfcc_mean, 0.45),
        (mfcc_std, 0.25),
        (envelope, 0.18),
        (f0_stats, 0.12),
    ]
    parts = []
    for block, weight in blocks:
        block = np.nan_to_num(np.asarray(block, dtype=np.float32))
        norm = float(np.linalg.norm(block))
        parts.append((block / norm * weight) if norm > EPS else block * 0.0)

    raw = np.concatenate(parts).astype(np.float32)
    if raw.size < EMBEDDING_DIM:
        raw = np.concatenate([raw, np.zeros(EMBEDDING_DIM - raw.size, dtype=np.float32)])
    raw = np.nan_to_num(raw[:EMBEDDING_DIM])

    norm = float(np.linalg.norm(raw))
    return (raw / norm).astype(np.float32) if norm > EPS else raw.astype(np.float32)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    if a.size == 0 or b.size == 0 or a.size != b.size:
        return 0.0
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom < EPS:
        return 0.0
    return float(np.clip(np.dot(a, b) / denom, -1.0, 1.0))


def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    return 1.0 - cosine_similarity(a, b)


class SpeakerEmbedder:
    """Embedding front end with an optional neural backend."""

    def __init__(self, backend: str = "auto") -> None:
        backends = available_backends()
        if backend == "auto":
            backend = "wavlm" if "wavlm" in backends else "classical"
        if backend not in backends:
            backend = "classical"
        self.backend = backend
        self._model = None
        self._processor = None

    @property
    def dim(self) -> int:
        return EMBEDDING_DIM if self.backend == "classical" else 768

    def _ensure_neural(self) -> bool:  # pragma: no cover - optional dependency
        if self._model is not None:
            return True
        try:
            import torch
            from transformers import AutoFeatureExtractor, AutoModel

            name = "microsoft/wavlm-base-plus-sv"
            self._processor = AutoFeatureExtractor.from_pretrained(name)
            self._model = AutoModel.from_pretrained(name).eval()
            self._torch = torch
            return True
        except Exception:
            self.backend = "classical"
            return False

    def embed(self, x: np.ndarray, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
        if self.backend == "wavlm" and self._ensure_neural():  # pragma: no cover
            inputs = self._processor(x, sampling_rate=sample_rate, return_tensors="pt")
            with self._torch.no_grad():
                hidden = self._model(**inputs).last_hidden_state
            vec = hidden.mean(dim=1).squeeze().cpu().numpy().astype(np.float32)
            norm = float(np.linalg.norm(vec))
            return vec / norm if norm > EPS else vec
        return classical_embedding(x, sample_rate)


class EnrolmentStore:
    """In-memory enrolment of genuine speaker samples, keyed by identity."""

    def __init__(self, embedder: Optional[SpeakerEmbedder] = None) -> None:
        self.embedder = embedder or SpeakerEmbedder()
        self._profiles: Dict[str, List[np.ndarray]] = {}

    def enrol(self, identity: str, x: np.ndarray, sample_rate: int = SAMPLE_RATE) -> int:
        vec = self.embedder.embed(x, sample_rate)
        if not np.any(vec):
            return len(self._profiles.get(identity, []))
        self._profiles.setdefault(identity, []).append(vec)
        return len(self._profiles[identity])

    def enrol_vector(self, identity: str, vec: np.ndarray) -> int:
        vec = np.asarray(vec, dtype=np.float32).ravel()
        if not np.any(vec):
            return len(self._profiles.get(identity, []))
        self._profiles.setdefault(identity, []).append(vec)
        return len(self._profiles[identity])

    def has(self, identity: Optional[str]) -> bool:
        return bool(identity) and identity in self._profiles and bool(self._profiles[identity])

    def centroid(self, identity: str) -> Optional[np.ndarray]:
        vectors = self._profiles.get(identity)
        if not vectors:
            return None
        stacked = np.stack(vectors)
        mean = stacked.mean(axis=0)
        norm = float(np.linalg.norm(mean))
        return (mean / norm).astype(np.float32) if norm > EPS else mean.astype(np.float32)

    def spread(self, identity: str) -> float:
        """Mean within-speaker distance — the natural variability of this enrolment."""
        vectors = self._profiles.get(identity) or []
        if len(vectors) < 2:
            return 0.0
        centroid = self.centroid(identity)
        return float(np.mean([cosine_distance(v, centroid) for v in vectors]))

    def compare(self, identity: str, vec: np.ndarray) -> Optional[Dict[str, float]]:
        centroid = self.centroid(identity)
        if centroid is None or not np.any(vec):
            return None
        distance = cosine_distance(vec, centroid)
        spread = self.spread(identity)
        return {
            "distance": distance,
            "similarity": 1.0 - distance,
            "enrolment_spread": spread,
            "samples": float(len(self._profiles.get(identity, []))),
            # How many "within-speaker spreads" away this call sits from the centroid.
            "z_distance": float(distance / max(spread, 0.05)),
        }

    def identities(self) -> List[str]:
        return sorted(self._profiles)

    def clear(self, identity: Optional[str] = None) -> None:
        if identity is None:
            self._profiles.clear()
        else:
            self._profiles.pop(identity, None)
