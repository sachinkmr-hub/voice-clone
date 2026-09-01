"""Detection layers and the model registry."""

from voiceguard.models.acoustic import AcousticHeuristicDetector, AcousticModelDetector
from voiceguard.models.base import Detector, EvidenceRule, Factor, LayerResult, RuleScorer
from voiceguard.models.calibration import fit_rule_anchors, summarise_report
from voiceguard.models.context import CallContext, ContextDetector
from voiceguard.models.prosodic import ProsodicDetector
from voiceguard.models.registry import ModelBundle, ModelRegistry, get_registry, reset_registry
from voiceguard.models.speaker import SpeakerConsistencyDetector

__all__ = [
    "AcousticHeuristicDetector",
    "AcousticModelDetector",
    "CallContext",
    "ContextDetector",
    "Detector",
    "EvidenceRule",
    "Factor",
    "LayerResult",
    "ModelBundle",
    "ModelRegistry",
    "ProsodicDetector",
    "RuleScorer",
    "SpeakerConsistencyDetector",
    "fit_rule_anchors",
    "summarise_report",
    "get_registry",
    "reset_registry",
]
