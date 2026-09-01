"""Speech simulation for demos, tests and bootstrap training data."""

from voiceguard.simulation.voice import (
    CLONE_METHODS,
    SpeakerTimbre,
    griffin_lim,
    mel_vocoder_roundtrip,
    synthesize_bonafide,
    synthesize_cloned,
)

__all__ = [
    "CLONE_METHODS",
    "SpeakerTimbre",
    "griffin_lim",
    "mel_vocoder_roundtrip",
    "synthesize_bonafide",
    "synthesize_cloned",
]
