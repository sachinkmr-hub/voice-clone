"""Audio ingestion: decoding, resampling, voice activity detection, stream chunking."""

from voiceguard.audio.io import (
    decode_audio_bytes,
    decode_pcm16,
    load_audio,
    resample,
    to_float_mono,
    write_wav,
)
from voiceguard.audio.stream import StreamChunker, Window
from voiceguard.audio.vad import SpeechSegments, detect_speech

__all__ = [
    "decode_audio_bytes",
    "decode_pcm16",
    "load_audio",
    "resample",
    "to_float_mono",
    "write_wav",
    "StreamChunker",
    "Window",
    "SpeechSegments",
    "detect_speech",
]
