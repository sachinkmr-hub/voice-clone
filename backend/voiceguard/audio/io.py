"""Audio decoding, resampling and file helpers.

The whole pipeline works on ``float32`` mono at :data:`voiceguard.config.SAMPLE_RATE`.
Everything entering the system is funnelled through :func:`decode_audio_bytes` or
:func:`load_audio` so that no downstream module has to think about formats.
"""

from __future__ import annotations

import hashlib
import io
import math
import struct
import wave
from typing import Optional, Tuple

import numpy as np
from scipy import signal

from voiceguard.config import SAMPLE_RATE

try:  # soundfile handles wav/flac/ogg; it is a hard requirement but we degrade anyway
    import soundfile as sf
except Exception:  # pragma: no cover - exercised only on stripped installs
    sf = None


EPS = 1e-12


# --------------------------------------------------------------------------------------
# Conversion primitives
# --------------------------------------------------------------------------------------

def to_float_mono(data: np.ndarray) -> np.ndarray:
    """Coerce any numeric array into 1-D ``float32`` in roughly [-1, 1]."""
    arr = np.asarray(data)
    if arr.ndim > 1:
        arr = arr.mean(axis=-1) if arr.shape[-1] <= 8 else arr.mean(axis=0)
    arr = arr.astype(np.float32, copy=False)
    if arr.size == 0:
        return arr
    if np.issubdtype(np.asarray(data).dtype, np.integer):
        max_val = float(np.iinfo(np.asarray(data).dtype).max)
        arr = arr / max_val
    peak = float(np.max(np.abs(arr))) if arr.size else 0.0
    if peak > 1.0:  # e.g. int32 samples handed to us as float
        arr = arr / peak
    return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)


def decode_pcm16(raw: bytes) -> np.ndarray:
    """Decode headerless little-endian signed 16-bit PCM."""
    usable = len(raw) - (len(raw) % 2)
    if usable <= 0:
        return np.zeros(0, dtype=np.float32)
    ints = np.frombuffer(raw[:usable], dtype="<i2")
    return (ints.astype(np.float32) / 32768.0).copy()


def _looks_like_container(raw: bytes) -> bool:
    head = raw[:12]
    return (
        head[:4] in (b"RIFF", b"fLaC", b"OggS", b".snd")
        or head[4:8] == b"ftyp"
        or head[:3] == b"ID3"
    )


def decode_audio_bytes(
    raw: bytes,
    *,
    encoding: str = "auto",
    sample_rate: Optional[int] = None,
    target_rate: int = SAMPLE_RATE,
) -> Tuple[np.ndarray, int]:
    """Decode an audio payload into ``(float32 mono, target_rate)``.

    ``encoding`` may be ``auto`` (sniff a container header, else assume PCM16),
    ``pcm16``, ``float32`` or ``container``.
    """
    if not raw:
        return np.zeros(0, dtype=np.float32), target_rate

    if encoding == "float32":
        usable = len(raw) - (len(raw) % 4)
        samples = np.frombuffer(raw[:usable], dtype="<f4").astype(np.float32).copy()
        src_rate = sample_rate or target_rate
    elif encoding == "pcm16":
        samples = decode_pcm16(raw)
        src_rate = sample_rate or target_rate
    elif encoding == "container" or (encoding == "auto" and _looks_like_container(raw)):
        samples, src_rate = _decode_container(raw)
    else:
        samples = decode_pcm16(raw)
        src_rate = sample_rate or target_rate

    samples = to_float_mono(samples)
    if src_rate != target_rate:
        samples = resample(samples, src_rate, target_rate)
    return samples, target_rate


def _decode_container(raw: bytes) -> Tuple[np.ndarray, int]:
    if sf is not None:
        try:
            data, src_rate = sf.read(io.BytesIO(raw), dtype="float32", always_2d=False)
            return to_float_mono(data), int(src_rate)
        except Exception:
            pass
    # Last-resort: stdlib wave for plain PCM RIFF files.
    try:
        with wave.open(io.BytesIO(raw), "rb") as handle:
            frames = handle.readframes(handle.getnframes())
            width = handle.getsampwidth()
            channels = handle.getnchannels()
            src_rate = handle.getframerate()
        if width == 2:
            data = decode_pcm16(frames)
        elif width == 1:
            data = (np.frombuffer(frames, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
        else:
            count = len(frames) // width
            data = np.array(
                [
                    int.from_bytes(frames[i * width : (i + 1) * width], "little", signed=True)
                    for i in range(count)
                ],
                dtype=np.float32,
            ) / float(2 ** (8 * width - 1))
        if channels > 1:
            data = data[: (len(data) // channels) * channels].reshape(-1, channels).mean(axis=1)
        return to_float_mono(data), int(src_rate)
    except Exception:
        return np.zeros(0, dtype=np.float32), SAMPLE_RATE


def resample(x: np.ndarray, src_rate: int, dst_rate: int) -> np.ndarray:
    """Band-limited polyphase resampling."""
    if src_rate == dst_rate or x.size == 0:
        return np.asarray(x, dtype=np.float32)
    gcd = math.gcd(int(src_rate), int(dst_rate))
    up, down = int(dst_rate // gcd), int(src_rate // gcd)
    return signal.resample_poly(x, up, down).astype(np.float32)


# --------------------------------------------------------------------------------------
# Files
# --------------------------------------------------------------------------------------

def load_audio(path: str, target_rate: int = SAMPLE_RATE) -> Tuple[np.ndarray, int]:
    """Load any file soundfile understands (falls back to stdlib ``wave``)."""
    with open(path, "rb") as handle:
        raw = handle.read()
    return decode_audio_bytes(raw, encoding="container", target_rate=target_rate)


def write_wav(path: str, x: np.ndarray, sample_rate: int = SAMPLE_RATE) -> None:
    """Write ``float32`` mono to a 16-bit PCM WAV without needing soundfile."""
    data = np.clip(to_float_mono(x), -1.0, 1.0)
    ints = (data * 32767.0).astype("<i2")
    with wave.open(path, "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(int(sample_rate))
        handle.writeframes(ints.tobytes())


def float_to_pcm16_bytes(x: np.ndarray) -> bytes:
    data = np.clip(to_float_mono(x), -1.0, 1.0)
    return (data * 32767.0).astype("<i2").tobytes()


# --------------------------------------------------------------------------------------
# Misc helpers used across the pipeline
# --------------------------------------------------------------------------------------

def sha256_hex(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def preemphasis(x: np.ndarray, coeff: float = 0.97) -> np.ndarray:
    if x.size < 2:
        return np.asarray(x, dtype=np.float32)
    out = np.empty_like(x, dtype=np.float32)
    out[0] = x[0]
    out[1:] = x[1:] - coeff * x[:-1]
    return out


def rms(x: np.ndarray) -> float:
    if x.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(x, dtype=np.float64)) + EPS))


def db(value: float, floor_db: float = -120.0) -> float:
    return float(max(floor_db, 20.0 * math.log10(max(value, 1e-12))))


def peak_normalize(x: np.ndarray, target: float = 0.95) -> np.ndarray:
    peak = float(np.max(np.abs(x))) if x.size else 0.0
    if peak < 1e-9:
        return np.asarray(x, dtype=np.float32)
    return (x * (target / peak)).astype(np.float32)


def duration_seconds(x: np.ndarray, sample_rate: int = SAMPLE_RATE) -> float:
    return float(len(x)) / float(sample_rate) if sample_rate else 0.0


def pack_float32(x: np.ndarray) -> bytes:
    return np.asarray(x, dtype="<f4").tobytes()


def unpack_float32(raw: bytes) -> np.ndarray:
    usable = len(raw) - (len(raw) % 4)
    return np.frombuffer(raw[:usable], dtype="<f4").astype(np.float32).copy()


def struct_version() -> str:  # tiny helper so the module has a stable import-time symbol
    return struct.pack("<I", 1).hex()
