"""Voice activity detection.

A lightweight, dependency-free VAD good enough to (a) skip silent windows so the risk
score is not driven by room tone, and (b) produce the pause statistics that layer L2
(prosody) needs. It combines three cheap cues per 20 ms frame:

* short-time energy relative to an adaptive noise floor,
* spectral flatness (broadband noise sits near 0.56, voiced speech below 0.05),
* zero-crossing rate (rejects DC/rumble, keeps fricatives).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple

import numpy as np

from voiceguard.config import SAMPLE_RATE

EPS = 1e-12
FRAME_MS = 20.0
#: Below this, a 16-bit frame carries no recoverable signal. Deliberately far below a
#: "normal" speaking level so that quiet callers on low-gain trunks are not discarded.
ABSOLUTE_SILENCE_DB = -78.0


@dataclass
class SpeechSegments:
    """Result of a VAD pass over one buffer."""

    mask: np.ndarray                     #: bool, one entry per frame
    frame_length: int
    hop_length: int
    sample_rate: int
    speech_ratio: float = 0.0
    pause_durations: List[float] = field(default_factory=list)
    speech_durations: List[float] = field(default_factory=list)
    noise_floor_db: float = -90.0
    frame_energy_db: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float32))

    @property
    def has_speech(self) -> bool:
        return bool(self.mask.any())

    def speech_samples(self, x: np.ndarray) -> np.ndarray:
        """Concatenate only the speech frames of ``x``."""
        if not self.mask.any():
            return np.zeros(0, dtype=np.float32)
        pieces = []
        for idx in np.flatnonzero(self.mask):
            start = idx * self.hop_length
            pieces.append(x[start : start + self.frame_length])
        return np.concatenate(pieces).astype(np.float32) if pieces else np.zeros(0, np.float32)

    def segments(self) -> List[Tuple[float, float]]:
        """Speech spans as ``(start_seconds, end_seconds)``."""
        spans: List[Tuple[float, float]] = []
        start_idx = None
        for i, flag in enumerate(self.mask):
            if flag and start_idx is None:
                start_idx = i
            elif not flag and start_idx is not None:
                spans.append(self._span(start_idx, i))
                start_idx = None
        if start_idx is not None:
            spans.append(self._span(start_idx, len(self.mask)))
        return spans

    def _span(self, i0: int, i1: int) -> Tuple[float, float]:
        sec = self.hop_length / float(self.sample_rate)
        return (i0 * sec, i1 * sec + self.frame_length / float(self.sample_rate))


def _frame(x: np.ndarray, frame_length: int, hop_length: int) -> np.ndarray:
    if x.size < frame_length:
        pad = np.zeros(frame_length - x.size, dtype=np.float32)
        x = np.concatenate([x.astype(np.float32), pad])
    n_frames = 1 + (len(x) - frame_length) // hop_length
    idx = np.arange(frame_length)[None, :] + hop_length * np.arange(n_frames)[:, None]
    return x[idx]


def _smooth_mask(mask: np.ndarray, min_speech: int, min_pause: int) -> np.ndarray:
    """Remove speech blips shorter than ``min_speech`` and bridge pauses shorter than ``min_pause``."""
    out = mask.copy()
    # bridge short gaps
    i = 0
    while i < len(out):
        if not out[i]:
            j = i
            while j < len(out) and not out[j]:
                j += 1
            if 0 < i and j < len(out) and (j - i) < min_pause:
                out[i:j] = True
            i = j
        else:
            i += 1
    # drop short blips
    i = 0
    while i < len(out):
        if out[i]:
            j = i
            while j < len(out) and out[j]:
                j += 1
            if (j - i) < min_speech:
                out[i:j] = False
            i = j
        else:
            i += 1
    return out


def detect_speech(
    x: np.ndarray,
    sample_rate: int = SAMPLE_RATE,
    *,
    energy_margin_db: float = 9.0,
    flatness_threshold: float = 0.50,
    min_speech_ms: float = 60.0,
    min_pause_ms: float = 120.0,
) -> SpeechSegments:
    """Run the VAD over ``x`` and return frame-level speech decisions."""
    x = np.asarray(x, dtype=np.float32)
    frame_length = max(64, int(sample_rate * FRAME_MS / 1000.0))
    hop_length = frame_length // 2

    if x.size < frame_length:
        return SpeechSegments(
            mask=np.zeros(0, dtype=bool),
            frame_length=frame_length,
            hop_length=hop_length,
            sample_rate=sample_rate,
        )

    frames = _frame(x, frame_length, hop_length)
    window = np.hanning(frame_length).astype(np.float32)
    windowed = frames * window

    energy = np.sqrt(np.mean(np.square(windowed.astype(np.float64)), axis=1) + EPS)
    energy_db = 20.0 * np.log10(np.maximum(energy, 1e-12))

    spec = np.abs(np.fft.rfft(windowed, axis=1)) + EPS
    power = spec**2
    geo = np.exp(np.mean(np.log(power), axis=1))
    arith = np.mean(power, axis=1)
    flatness = geo / np.maximum(arith, EPS)

    zcr = np.mean(np.abs(np.diff(np.sign(frames), axis=1)) > 0, axis=1)

    # Adaptive noise floor: the 15th percentile of frame energy is a robust estimate of
    # background level even when speech occupies most of the buffer.
    noise_floor = float(np.percentile(energy_db, 15))
    threshold = noise_floor + energy_margin_db
    dynamic = float(np.percentile(energy_db, 95)) - noise_floor
    if dynamic < 6.0:
        # Essentially constant level. This is *not* the same as "silent": a window filled
        # edge to edge with continuous speech also has no internal contrast, and the
        # relative threshold above would then classify all of it as background. With no
        # contrast to exploit, decide on absolute level alone (the 16-bit noise floor,
        # not a "normal" speaking level — a quiet caller on a low-gain trunk is still a
        # real call) and let spectral flatness reject steady room tone and hum.
        mask = (energy_db > ABSOLUTE_SILENCE_DB) & (flatness < flatness_threshold)
    else:
        mask = (energy_db > threshold) & (flatness < flatness_threshold) & (zcr < 0.55)

    min_speech = max(1, int(min_speech_ms / (hop_length * 1000.0 / sample_rate)))
    min_pause = max(1, int(min_pause_ms / (hop_length * 1000.0 / sample_rate)))
    mask = _smooth_mask(mask, min_speech, min_pause)

    result = SpeechSegments(
        mask=mask,
        frame_length=frame_length,
        hop_length=hop_length,
        sample_rate=sample_rate,
        speech_ratio=float(np.mean(mask)) if mask.size else 0.0,
        noise_floor_db=noise_floor,
        frame_energy_db=energy_db.astype(np.float32),
    )

    sec_per_frame = hop_length / float(sample_rate)
    run_value, run_len = None, 0
    for flag in mask:
        if flag == run_value:
            run_len += 1
            continue
        if run_value is not None:
            (result.speech_durations if run_value else result.pause_durations).append(
                run_len * sec_per_frame
            )
        run_value, run_len = bool(flag), 1
    if run_value is not None:
        (result.speech_durations if run_value else result.pause_durations).append(
            run_len * sec_per_frame
        )
    return result
