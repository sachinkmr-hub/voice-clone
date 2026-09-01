"""Feature extraction: spectral (L1), prosodic (L2), synthesis artifacts, embeddings (L3)."""

from voiceguard.features.artifacts import artifact_feature_dict
from voiceguard.features.embedding import (
    EnrolmentStore,
    SpeakerEmbedder,
    classical_embedding,
    cosine_distance,
    cosine_similarity,
)
from voiceguard.features.extractor import FeatureBundle, FeatureExtractor, extract_features
from voiceguard.features.prosody import prosody_feature_dict, track_pitch
from voiceguard.features.spectral import spectral_feature_dict

__all__ = [
    "artifact_feature_dict",
    "prosody_feature_dict",
    "spectral_feature_dict",
    "track_pitch",
    "classical_embedding",
    "cosine_distance",
    "cosine_similarity",
    "EnrolmentStore",
    "SpeakerEmbedder",
    "FeatureBundle",
    "FeatureExtractor",
    "extract_features",
]
