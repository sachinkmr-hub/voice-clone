"""Tests for decoding, VAD and the streaming chunker."""

import numpy as np
import pytest

from voiceguard.audio.io import (
    decode_audio_bytes,
    decode_pcm16,
    float_to_pcm16_bytes,
    load_audio,
    peak_normalize,
    resample,
    rms,
    to_float_mono,
    write_wav,
)
from voiceguard.audio.stream import StreamChunker, iter_windows
from voiceguard.audio.vad import detect_speech

SR = 16000


@pytest.fixture()
def tone():
    t = np.arange(SR) / SR
    return (0.4 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)


def test_pcm16_roundtrip(tone):
    decoded = decode_pcm16(float_to_pcm16_bytes(tone))
    assert decoded.shape == tone.shape
    assert np.max(np.abs(decoded - tone)) < 1e-3


def test_decode_auto_detects_wav_container(tmp_path, tone):
    path = tmp_path / "a.wav"
    write_wav(str(path), tone, SR)
    samples, rate = decode_audio_bytes(path.read_bytes(), encoding="auto")
    assert rate == SR
    assert abs(len(samples) - len(tone)) <= 1


def test_decode_handles_odd_length_and_empty():
    assert decode_pcm16(b"\x01").size == 0
    samples, rate = decode_audio_bytes(b"", encoding="pcm16")
    assert samples.size == 0 and rate == SR


def test_load_audio_resamples(tmp_path, tone):
    path = tmp_path / "b.wav"
    write_wav(str(path), tone, 8000)
    samples, rate = load_audio(str(path), target_rate=SR)
    assert rate == SR
    assert abs(len(samples) - 2 * len(tone)) < 100  # 8 kHz file, 16 kHz target


def test_to_float_mono_handles_int_and_stereo():
    stereo = np.stack([np.ones(100), -np.ones(100)], axis=-1)
    assert np.allclose(to_float_mono(stereo), 0.0)
    ints = (np.ones(50) * 16384).astype(np.int16)
    assert abs(float(to_float_mono(ints)[0]) - 0.5) < 1e-3


def test_resample_changes_length(tone):
    assert len(resample(tone, SR, 8000)) == pytest.approx(len(tone) // 2, abs=2)
    assert len(resample(tone, SR, SR)) == len(tone)


def test_peak_normalize_and_rms(tone):
    normalized = peak_normalize(tone, 0.5)
    assert float(np.max(np.abs(normalized))) == pytest.approx(0.5, abs=1e-3)
    assert rms(np.zeros(10)) < 1e-5


# ------------------------------------------------------------------------------- VAD

def test_vad_finds_speech_and_pause():
    t = np.arange(3 * SR) / SR
    signal = (0.3 * np.sin(2 * np.pi * 180 * t) * (1 + 0.4 * np.sin(2 * np.pi * 4 * t))).astype(
        np.float32
    )
    signal[SR : int(1.6 * SR)] = (1e-4 * np.random.randn(int(0.6 * SR))).astype(np.float32)

    segments = detect_speech(signal, SR)
    assert segments.has_speech
    assert 0.5 < segments.speech_ratio < 0.95
    assert any(0.4 < p < 0.8 for p in segments.pause_durations)
    assert len(segments.segments()) == 2


def test_vad_rejects_pure_silence():
    segments = detect_speech(np.zeros(SR, dtype=np.float32), SR)
    assert not segments.has_speech
    assert segments.speech_ratio == 0.0


def test_vad_handles_buffer_shorter_than_one_frame():
    segments = detect_speech(np.zeros(10, dtype=np.float32), SR)
    assert segments.mask.size == 0
    assert not segments.has_speech


# --------------------------------------------------------------------------- chunker

def test_chunker_emits_overlapping_windows():
    audio = np.random.randn(3 * SR).astype(np.float32) * 0.1
    chunker = StreamChunker(SR, 1.0, 0.5)
    windows = chunker.push(audio)

    assert len(windows) == 5  # starts at 0.0, 0.5, 1.0, 1.5, 2.0
    assert all(w.samples.shape[0] == SR for w in windows)
    assert [round(w.start_time, 2) for w in windows] == [0.0, 0.5, 1.0, 1.5, 2.0]
    assert [w.index for w in windows] == [0, 1, 2, 3, 4]


def test_chunker_is_incremental():
    audio = np.random.randn(3 * SR).astype(np.float32) * 0.1
    chunker = StreamChunker(SR, 1.0, 0.5)
    collected = []
    for start in range(0, len(audio), 1600):  # 100 ms pushes
        collected.extend(chunker.push(audio[start : start + 1600]))
    assert len(collected) == 5
    assert chunker.elapsed_seconds == pytest.approx(3.0, abs=0.01)


def test_chunker_flush_emits_tail_once():
    chunker = StreamChunker(SR, 1.0, 0.5)
    chunker.push(np.random.randn(int(1.8 * SR)).astype(np.float32) * 0.1)
    tail = chunker.flush()
    assert len(tail) == 1
    assert chunker.flush() == []


def test_chunker_drops_short_tail():
    chunker = StreamChunker(SR, 1.0, 0.5)
    chunker.push(np.random.randn(int(0.2 * SR)).astype(np.float32))
    assert chunker.flush() == []


def test_chunker_bounds_its_buffer():
    chunker = StreamChunker(SR, 1.0, 0.5, max_buffer_seconds=2.0)
    chunker.push(np.random.randn(10 * SR).astype(np.float32) * 0.1)
    assert chunker.buffered_seconds <= 2.0


def test_iter_windows_matches_streaming_path():
    audio = np.random.randn(2 * SR).astype(np.float32) * 0.1
    offline = iter_windows(audio, SR, 1.0, 0.5)
    assert len(offline) >= 3
    assert offline[0].samples.shape[0] == SR
