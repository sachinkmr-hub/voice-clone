"""VoiceGuard Python SDK — client for the real-time voice-clone detection API."""

from voiceguard_sdk.client import (
    ApprovalResult,
    Factor,
    RiskResult,
    StreamSession,
    VoiceGuardClient,
    VoiceGuardError,
    chunk_pcm16,
)

__version__ = "1.0.0"
__all__ = [
    "VoiceGuardClient", "VoiceGuardError", "RiskResult", "ApprovalResult",
    "StreamSession", "Factor", "chunk_pcm16", "__version__",
]
