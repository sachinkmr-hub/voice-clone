"""Streaming chunker.

Callers push arbitrarily-sized audio blobs; the chunker emits fixed-size overlapping
analysis :class:`Window` objects. This is what turns "a socket dribbling 20 ms Opus
frames" into "1.0 s windows every 0.5 s" without any layer above needing a buffer.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Iterator, List

import numpy as np

from voiceguard.config import HOP_SECONDS, SAMPLE_RATE, WINDOW_SECONDS


@dataclass
class Window:
    """One analysis window."""

    index: int
    samples: np.ndarray
    sample_rate: int
    start_time: float          #: seconds from the start of the call
    received_at: float = field(default_factory=time.time)

    @property
    def duration(self) -> float:
        return len(self.samples) / float(self.sample_rate)

    @property
    def end_time(self) -> float:
        return self.start_time + self.duration


class StreamChunker:
    """Fixed-size overlapping window producer with a bounded internal buffer."""

    def __init__(
        self,
        sample_rate: int = SAMPLE_RATE,
        window_seconds: float = WINDOW_SECONDS,
        hop_seconds: float = HOP_SECONDS,
        *,
        max_buffer_seconds: float = 30.0,
    ) -> None:
        self.sample_rate = int(sample_rate)
        self.window_size = max(1, int(round(window_seconds * sample_rate)))
        self.hop_size = max(1, int(round(hop_seconds * sample_rate)))
        self.max_buffer = int(max_buffer_seconds * sample_rate)
        self._buffer = np.zeros(0, dtype=np.float32)
        self._consumed = 0      #: samples already emitted as window starts
        self._index = 0
        self.total_samples = 0

    # ---------------------------------------------------------------- properties
    @property
    def buffered_seconds(self) -> float:
        return len(self._buffer) / float(self.sample_rate)

    @property
    def elapsed_seconds(self) -> float:
        return self.total_samples / float(self.sample_rate)

    @property
    def windows_emitted(self) -> int:
        return self._index

    # ------------------------------------------------------------------- pushing
    def push(self, samples: np.ndarray) -> List[Window]:
        """Append audio and return every complete window it unlocked."""
        chunk = np.asarray(samples, dtype=np.float32).ravel()
        if chunk.size:
            self._buffer = np.concatenate([self._buffer, chunk])
            self.total_samples += chunk.size
        if len(self._buffer) > self.max_buffer:
            drop = len(self._buffer) - self.max_buffer
            self._buffer = self._buffer[drop:]
            self._consumed += drop
        return list(self._drain())

    def _drain(self) -> Iterator[Window]:
        while len(self._buffer) >= self.window_size:
            samples = self._buffer[: self.window_size].copy()
            window = Window(
                index=self._index,
                samples=samples,
                sample_rate=self.sample_rate,
                start_time=self._consumed / float(self.sample_rate),
            )
            self._index += 1
            self._buffer = self._buffer[self.hop_size :]
            self._consumed += self.hop_size
            yield window

    def flush(self, *, min_seconds: float = 0.4) -> List[Window]:
        """Emit a final short window at end-of-call if enough audio remains."""
        if len(self._buffer) < int(min_seconds * self.sample_rate):
            self._buffer = np.zeros(0, dtype=np.float32)
            return []
        window = Window(
            index=self._index,
            samples=self._buffer.copy(),
            sample_rate=self.sample_rate,
            start_time=self._consumed / float(self.sample_rate),
        )
        self._index += 1
        self._consumed += len(self._buffer)
        self._buffer = np.zeros(0, dtype=np.float32)
        return [window]

    def reset(self) -> None:
        self._buffer = np.zeros(0, dtype=np.float32)
        self._consumed = 0
        self._index = 0
        self.total_samples = 0


def iter_windows(
    x: np.ndarray,
    sample_rate: int = SAMPLE_RATE,
    window_seconds: float = WINDOW_SECONDS,
    hop_seconds: float = HOP_SECONDS,
) -> List[Window]:
    """Convenience: split an offline buffer exactly the way the live path would."""
    chunker = StreamChunker(sample_rate, window_seconds, hop_seconds,
                            max_buffer_seconds=max(30.0, len(x) / sample_rate + 1))
    windows = chunker.push(x)
    windows.extend(chunker.flush())
    return windows
