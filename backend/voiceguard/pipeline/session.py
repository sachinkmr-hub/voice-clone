"""Per-call session state.

A :class:`CallSession` is the only stateful object in the inference path. It owns the
chunker, the running smoothed score, the embedding history used for within-call drift,
and the assessment trail that becomes the call report (FR-7).

Everything else — feature extraction, the detectors, the risk engine — is stateless and
shared, which is what keeps horizontal scaling to "route a session id consistently".
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from voiceguard.audio.io import sha256_hex
from voiceguard.audio.stream import StreamChunker, Window
from voiceguard.config import (
    HOP_SECONDS,
    SAMPLE_RATE,
    WARMUP_WINDOWS,
    WINDOW_SECONDS,
    RiskBand,
    Settings,
    get_settings,
)
from voiceguard.features.extractor import FeatureBundle, FeatureExtractor
from voiceguard.models.base import LayerResult
from voiceguard.models.context import CallContext
from voiceguard.models.registry import ModelRegistry, get_registry
from voiceguard.scoring.risk import RiskAssessment, RiskEngine, ScoreSmoother


@dataclass
class SessionStats:
    """Running counters, surfaced in the report and on the dashboard."""

    windows_analyzed: int = 0
    windows_skipped: int = 0
    bytes_ingested: int = 0
    audio_seconds: float = 0.0
    speech_seconds: float = 0.0
    total_latency_ms: float = 0.0
    peak_score: float = 0.0
    alerts_raised: int = 0

    @property
    def mean_latency_ms(self) -> float:
        return self.total_latency_ms / self.windows_analyzed if self.windows_analyzed else 0.0

    def as_dict(self) -> dict:
        return {
            "windows_analyzed": self.windows_analyzed,
            "windows_skipped": self.windows_skipped,
            "bytes_ingested": self.bytes_ingested,
            "audio_seconds": round(self.audio_seconds, 2),
            "speech_seconds": round(self.speech_seconds, 2),
            "mean_latency_ms": round(self.mean_latency_ms, 2),
            "peak_score": round(self.peak_score, 1),
            "alerts_raised": self.alerts_raised,
        }


class CallSession:
    """One live or replayed call."""

    def __init__(
        self,
        session_id: Optional[str] = None,
        *,
        profile: str = "default",
        language: str = "auto",
        identity: Optional[str] = None,
        call_context: Optional[CallContext] = None,
        registry: Optional[ModelRegistry] = None,
        extractor: Optional[FeatureExtractor] = None,
        engine: Optional[RiskEngine] = None,
        settings: Optional[Settings] = None,
        sample_rate: int = SAMPLE_RATE,
        metadata: Optional[dict] = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.id = session_id or f"call_{uuid.uuid4().hex[:12]}"
        self.profile = profile
        self.language = language
        self.identity = identity
        self.call_context = call_context
        self.metadata: Dict[str, object] = dict(metadata or {})
        self.sample_rate = sample_rate

        self.registry = registry or get_registry()
        self.extractor = extractor or FeatureExtractor(sample_rate)
        self.engine = engine or RiskEngine(fusion=None)
        if self.registry.fusion_weights:
            self.engine.fusion.update_weights(self.registry.fusion_weights)
        self.engine.fusion.calibration = self.registry.calibration

        self.chunker = StreamChunker(sample_rate, WINDOW_SECONDS, HOP_SECONDS)
        self.smoother = ScoreSmoother()

        self.created_at = time.time()
        self.last_activity = self.created_at
        self.closed_at: Optional[float] = None
        self.stats = SessionStats()

        self.assessments: List[RiskAssessment] = []
        self.embedding_history: List[np.ndarray] = []
        self.chunk_hashes: List[str] = []
        #: Populated only when retention mode is raw_audio.
        self.retained_audio: List[np.ndarray] = []

    # ------------------------------------------------------------------ properties
    @property
    def is_open(self) -> bool:
        return self.closed_at is None

    @property
    def current_score(self) -> float:
        return float(self.smoother.value) if self.smoother.value is not None else 0.0

    @property
    def latest(self) -> Optional[RiskAssessment]:
        return self.assessments[-1] if self.assessments else None

    @property
    def band(self) -> str:
        return self.latest.band if self.latest else RiskBand.LOW.value

    def idle_seconds(self) -> float:
        return time.time() - self.last_activity

    # -------------------------------------------------------------------- ingest
    def ingest(self, samples: np.ndarray, *, raw_bytes: Optional[bytes] = None
               ) -> List[RiskAssessment]:
        """Feed audio in; get back an assessment for every completed window."""
        self.last_activity = time.time()
        if raw_bytes:
            self.stats.bytes_ingested += len(raw_bytes)
            self.chunk_hashes.append(sha256_hex(raw_bytes))
        if self.settings.retention_mode == "raw_audio":
            self.retained_audio.append(np.asarray(samples, dtype=np.float32))

        windows = self.chunker.push(samples)
        return [a for a in (self._analyze(w) for w in windows) if a is not None]

    def flush(self) -> List[RiskAssessment]:
        """Analyse whatever is left at end-of-call."""
        return [a for a in (self._analyze(w) for w in self.chunker.flush()) if a is not None]

    # ------------------------------------------------------------------ analysis
    def _analyze(self, window: Window) -> Optional[RiskAssessment]:
        started = time.perf_counter()
        bundle = self.extractor.extract(window.samples, language=self.language)
        self.stats.audio_seconds = self.chunker.elapsed_seconds

        if not bundle.speech_detected:
            # Hold the score rather than decaying it: silence is not evidence of
            # innocence, and letting the gauge drift down during a pause would let an
            # attacker wait out an alert.
            self.stats.windows_skipped += 1
            return None

        self.stats.windows_analyzed += 1
        self.stats.speech_seconds += bundle.speech_ratio * window.duration

        layer_results = self._run_layers(bundle)
        provisional = self.stats.windows_analyzed <= WARMUP_WINDOWS

        # Score the window first, then smooth, then re-assess with the smoothed value so
        # that the number shown and the reasons under it describe the same thing.
        window_assessment = self.engine.assess(
            layer_results,
            call_context=self.call_context,
            profile_name=self.profile,
        )
        smoothed = self.smoother.update(window_assessment.score)

        latency_ms = (time.perf_counter() - started) * 1000.0
        self.stats.total_latency_ms += latency_ms
        self.stats.peak_score = max(self.stats.peak_score, smoothed)

        assessment = self.engine.assess(
            layer_results,
            call_context=self.call_context,
            profile_name=self.profile,
            window_index=window.index,
            elapsed_seconds=window.end_time,
            speech_ratio=bundle.speech_ratio,
            speech_detected=True,
            provisional=provisional,
            latency_ms=latency_ms,
            score_override=smoothed,
        )

        if bundle.embedding.size and np.any(bundle.embedding):
            self.embedding_history.append(bundle.embedding)
            if len(self.embedding_history) > 32:
                self.embedding_history = self.embedding_history[-32:]

        self.assessments.append(assessment)
        if len(self.assessments) > 600:            # ~5 minutes at 2 Hz
            self.assessments = self.assessments[-600:]
        return assessment

    def _run_layers(self, bundle: FeatureBundle) -> List[LayerResult]:
        features = bundle.all_features()
        context = {
            "embedding": bundle.embedding,
            "identity": self.identity,
            "history": self.embedding_history,
            "call_context": self.call_context,
        }
        results: List[LayerResult] = []
        for name, detector in self.registry.detectors().items():
            try:
                results.append(detector.analyze(features, context))
            except Exception as exc:  # one bad layer must not kill the call
                results.append(LayerResult.unavailable(name, f"layer error: {exc}"))
        return results

    # -------------------------------------------------------------------- report
    def close(self) -> None:
        if self.closed_at is None:
            self.closed_at = time.time()

    def report(self, *, include_trail: bool = True) -> dict:
        """The per-call risk report with its confidence trail (FR-7)."""
        scores = [a.score for a in self.assessments]
        bands: Dict[str, int] = {}
        for assessment in self.assessments:
            bands[assessment.band] = bands.get(assessment.band, 0) + 1

        factor_counts: Dict[str, dict] = {}
        for assessment in self.assessments:
            for factor in assessment.explanation.factors:
                entry = factor_counts.setdefault(
                    factor.code,
                    {"label": factor.label, "layer": factor.layer, "count": 0,
                     "mean_contribution": 0.0},
                )
                entry["count"] += 1
                entry["mean_contribution"] += factor.contribution
        for entry in factor_counts.values():
            entry["mean_contribution"] = round(
                entry["mean_contribution"] / max(entry["count"], 1), 4)

        verdict = self._verdict(scores)
        report = {
            "session_id": self.id,
            "profile": self.profile,
            "language": self.language,
            "identity": self.identity,
            "created_at": self.created_at,
            "closed_at": self.closed_at,
            "duration_seconds": round(self.stats.audio_seconds, 2),
            "verdict": verdict,
            "final_score": round(self.current_score, 1),
            "peak_score": round(self.stats.peak_score, 1),
            "mean_score": round(float(np.mean(scores)), 1) if scores else 0.0,
            "band": self.band,
            "band_histogram": bands,
            "stats": self.stats.as_dict(),
            "top_factors": sorted(
                ({"code": code, **data} for code, data in factor_counts.items()),
                key=lambda row: (-row["count"], -row["mean_contribution"]),
            )[:8],
            "call_context": self.call_context.as_dict() if self.call_context else None,
            "chunk_hashes": self.chunk_hashes[:64],
            "model": {
                "degraded": self.registry.degraded,
                "detectors": {k: v.model_id for k, v in self.registry.detectors().items()},
            },
        }
        if include_trail:
            report["trail"] = [a.as_dict() for a in self.assessments]
        return report

    def _verdict(self, scores: List[float]) -> str:
        if not scores:
            return "insufficient_audio"
        if self.stats.audio_seconds < 2.0:
            return "insufficient_audio"
        peak = self.stats.peak_score
        if peak >= 80:
            return "likely_synthetic"
        if peak >= 60:
            return "suspicious"
        if peak >= 35:
            return "inconclusive"
        return "likely_genuine"

    def as_dict(self) -> dict:
        return {
            "session_id": self.id,
            "profile": self.profile,
            "language": self.language,
            "identity": self.identity,
            "open": self.is_open,
            "created_at": self.created_at,
            "last_activity": self.last_activity,
            "score": round(self.current_score, 1),
            "band": self.band,
            "stats": self.stats.as_dict(),
            "metadata": self.metadata,
        }
