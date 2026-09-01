"""Model registry — loading, versioning and graceful degradation.

The registry is the single place that decides *which* detector implementation each layer
gets, and it is written so that the system always comes up. Missing artifact, corrupt
artifact, scikit-learn version mismatch, no torch — each of those downgrades one layer to
its heuristic implementation and records why, rather than failing the process. The reason
is surfaced through ``GET /v1/health`` so an operator can see they are running degraded.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from voiceguard.config import Layer, Settings, get_settings
from voiceguard.features.embedding import EnrolmentStore
from voiceguard.models.acoustic import AcousticHeuristicDetector, AcousticModelDetector
from voiceguard.models.base import Detector
from voiceguard.models.context import ContextDetector
from voiceguard.models.prosodic import ProsodicDetector
from voiceguard.models.speaker import SpeakerConsistencyDetector

logger = logging.getLogger("voiceguard.models")

#: Bumped whenever the persisted artifact layout changes.
ARTIFACT_FORMAT_VERSION = 1


@dataclass
class ModelBundle:
    """What ``ml/train.py`` persists and the registry consumes."""

    model: object
    feature_names: List[str]
    model_id: str = "acoustic-model"
    scaler: object = None
    calibration: tuple = (1.0, 0.0)
    fusion_weights: Dict[str, float] = field(default_factory=dict)
    importances: Dict[str, float] = field(default_factory=dict)
    metrics: Dict[str, float] = field(default_factory=dict)
    trained_on: str = ""
    format_version: int = ARTIFACT_FORMAT_VERSION

    def as_dict(self) -> dict:
        return {
            "model_id": self.model_id,
            "n_features": len(self.feature_names),
            "calibration": list(self.calibration),
            "fusion_weights": dict(self.fusion_weights),
            "metrics": dict(self.metrics),
            "trained_on": self.trained_on,
            "format_version": self.format_version,
        }


class ModelRegistry:
    """Owns one detector per layer plus the shared enrolment store."""

    def __init__(self, settings: Optional[Settings] = None,
                 enrolment: Optional[EnrolmentStore] = None) -> None:
        self.settings = settings or get_settings()
        self.enrolment = enrolment or EnrolmentStore()
        self.bundle: Optional[ModelBundle] = None
        self.degraded: List[str] = []
        self._detectors: Dict[str, Detector] = {}
        self.reload()

    # ------------------------------------------------------------------- loading
    def reload(self) -> None:
        """(Re)build every detector. Safe to call at runtime after training."""
        self.degraded = []
        self.bundle = self._load_bundle()

        if self.bundle is not None:
            acoustic: Detector = AcousticModelDetector(
                model=self.bundle.model,
                feature_names=self.bundle.feature_names,
                model_id=self.bundle.model_id,
                scaler=self.bundle.scaler,
                heuristic=AcousticHeuristicDetector(),
                importances=self.bundle.importances,
            )
        else:
            acoustic = AcousticHeuristicDetector()

        self._detectors = {
            Layer.ACOUSTIC.value: acoustic,
            Layer.PROSODIC.value: ProsodicDetector(),
            Layer.SPEAKER.value: SpeakerConsistencyDetector(self.enrolment),
            Layer.CONTEXT.value: ContextDetector(),
        }

        backends = self.enrolment.embedder.backend
        if backends == "classical":
            self.degraded.append(
                "speaker embedding: classical backend (install torch + transformers for WavLM)"
            )

    def _load_bundle(self) -> Optional[ModelBundle]:
        path = self.settings.model_path()
        if not os.path.exists(path):
            self.degraded.append(
                f"no trained acoustic model at {path}; using the heuristic detector "
                f"(run `make train`)"
            )
            return None
        try:
            import joblib

            raw = joblib.load(path)
        except Exception as exc:
            self.degraded.append(f"could not load {path}: {exc}; using the heuristic detector")
            return None

        try:
            if isinstance(raw, ModelBundle):
                bundle = raw
            elif isinstance(raw, dict):
                bundle = ModelBundle(
                    model=raw["model"],
                    feature_names=list(raw["feature_names"]),
                    model_id=raw.get("model_id", "acoustic-model"),
                    scaler=raw.get("scaler"),
                    calibration=tuple(raw.get("calibration", (1.0, 0.0))),
                    fusion_weights=dict(raw.get("fusion_weights", {})),
                    importances=dict(raw.get("importances", {})),
                    metrics=dict(raw.get("metrics", {})),
                    trained_on=raw.get("trained_on", ""),
                    format_version=int(raw.get("format_version", ARTIFACT_FORMAT_VERSION)),
                )
            else:
                raise TypeError(f"unexpected artifact type {type(raw)!r}")
        except Exception as exc:
            self.degraded.append(f"artifact at {path} is malformed: {exc}")
            return None

        if bundle.format_version != ARTIFACT_FORMAT_VERSION:
            self.degraded.append(
                f"artifact format v{bundle.format_version} != expected "
                f"v{ARTIFACT_FORMAT_VERSION}; retrain to remove this warning"
            )
        logger.info("loaded acoustic model %s (%d features)",
                    bundle.model_id, len(bundle.feature_names))
        return bundle

    # ------------------------------------------------------------------- access
    def detector(self, layer: str) -> Optional[Detector]:
        return self._detectors.get(layer)

    def detectors(self) -> Dict[str, Detector]:
        return dict(self._detectors)

    @property
    def calibration(self) -> tuple:
        return self.bundle.calibration if self.bundle else (1.0, 0.0)

    @property
    def fusion_weights(self) -> Dict[str, float]:
        return dict(self.bundle.fusion_weights) if self.bundle else {}

    @property
    def is_degraded(self) -> bool:
        return bool(self.degraded)

    def feature_names(self) -> List[str]:
        return list(self.bundle.feature_names) if self.bundle else []

    def describe(self) -> dict:
        return {
            "model_loaded": self.bundle is not None,
            "bundle": self.bundle.as_dict() if self.bundle else None,
            "detectors": {name: d.describe() for name, d in self._detectors.items()},
            "degraded": list(self.degraded),
        }


_REGISTRY: Optional[ModelRegistry] = None


def get_registry() -> ModelRegistry:
    """Process-wide registry (detectors are stateless, so sharing them is safe)."""
    global _REGISTRY
    if _REGISTRY is None:
        _REGISTRY = ModelRegistry()
    return _REGISTRY


def reset_registry() -> None:
    global _REGISTRY
    _REGISTRY = None
