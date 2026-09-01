"""Layer-1 acoustic / spectral analysis.

Pure NumPy/SciPy so the whole feature stack can run on-device (see docs/PRIVACY.md).
Everything here operates on one analysis window of float32 mono audio.

Feature families
----------------
``stft`` / ``mel`` / ``mfcc``
    Standard front end. MFCC means and variances plus their deltas capture the coarse
    timbre; neural vocoders reproduce these well, so they are context rather than
    evidence — but their *variance* over time is informative (TTS is smoother).

spectral shape statistics
    Centroid, spread, rolloff, flatness, flux, slope, entropy and sub-band energy ratios.

phase statistics
    Group-delay deviation and inter-frame phase-advance consistency. Most vocoders
    reconstruct magnitude faithfully and phase approximately, which shows up as an
    unnaturally *regular* phase structure.

modulation spectrum
    Energy of the temporal envelope in the 2–16 Hz syllabic band. Human speech peaks
    near 4 Hz with a broad spread; synthetic speech is typically flatter and shallower.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
from scipy.fftpack import dct

from voiceguard.config import FFT_SIZE, HOP_LENGTH, N_MELS, N_MFCC, SAMPLE_RATE

EPS = 1e-12


# --------------------------------------------------------------------------------------
# Front end
# --------------------------------------------------------------------------------------

def frame_signal(x: np.ndarray, frame_length: int, hop_length: int) -> np.ndarray:
    """Frame a signal into ``(n_frames, frame_length)`` without copying more than needed."""
    x = np.asarray(x, dtype=np.float32)
    if x.size < frame_length:
        x = np.concatenate([x, np.zeros(frame_length - x.size, dtype=np.float32)])
    n_frames = 1 + (len(x) - frame_length) // hop_length
    idx = np.arange(frame_length)[None, :] + hop_length * np.arange(n_frames)[:, None]
    return x[idx]


def stft(
    x: np.ndarray,
    n_fft: int = FFT_SIZE,
    hop_length: int = HOP_LENGTH,
) -> np.ndarray:
    """Complex STFT with a Hann window, shape ``(n_frames, n_fft // 2 + 1)``."""
    frames = frame_signal(x, n_fft, hop_length)
    window = np.hanning(n_fft).astype(np.float32)
    return np.fft.rfft(frames * window, axis=1)


def magnitude_spectrogram(x: np.ndarray, n_fft: int = FFT_SIZE,
                          hop_length: int = HOP_LENGTH) -> np.ndarray:
    return np.abs(stft(x, n_fft, hop_length)).astype(np.float32)


def hz_to_mel(hz: np.ndarray | float) -> np.ndarray | float:
    return 2595.0 * np.log10(1.0 + np.asarray(hz, dtype=np.float64) / 700.0)


def mel_to_hz(mel: np.ndarray | float) -> np.ndarray | float:
    return 700.0 * (10.0 ** (np.asarray(mel, dtype=np.float64) / 2595.0) - 1.0)


_MEL_CACHE: Dict[Tuple[int, int, int, float, float], np.ndarray] = {}


def mel_filterbank(
    sample_rate: int = SAMPLE_RATE,
    n_fft: int = FFT_SIZE,
    n_mels: int = N_MELS,
    fmin: float = 20.0,
    fmax: float | None = None,
) -> np.ndarray:
    """Slaney-style triangular mel filterbank, shape ``(n_mels, n_fft // 2 + 1)``."""
    fmax = fmax or sample_rate / 2.0
    key = (sample_rate, n_fft, n_mels, fmin, fmax)
    cached = _MEL_CACHE.get(key)
    if cached is not None:
        return cached

    n_bins = n_fft // 2 + 1
    fft_freqs = np.linspace(0.0, sample_rate / 2.0, n_bins)
    mel_points = np.linspace(hz_to_mel(fmin), hz_to_mel(fmax), n_mels + 2)
    hz_points = mel_to_hz(mel_points)

    fb = np.zeros((n_mels, n_bins), dtype=np.float32)
    for m in range(n_mels):
        left, centre, right = hz_points[m], hz_points[m + 1], hz_points[m + 2]
        left_slope = (fft_freqs - left) / max(centre - left, EPS)
        right_slope = (right - fft_freqs) / max(right - centre, EPS)
        fb[m] = np.maximum(0.0, np.minimum(left_slope, right_slope))
        norm = right - left
        if norm > 0:
            fb[m] *= 2.0 / norm
    _MEL_CACHE[key] = fb
    return fb


def mel_spectrogram(
    x: np.ndarray,
    sample_rate: int = SAMPLE_RATE,
    n_fft: int = FFT_SIZE,
    hop_length: int = HOP_LENGTH,
    n_mels: int = N_MELS,
) -> np.ndarray:
    power = magnitude_spectrogram(x, n_fft, hop_length) ** 2
    fb = mel_filterbank(sample_rate, n_fft, n_mels)
    return (power @ fb.T).astype(np.float32)


def mfcc(
    x: np.ndarray,
    sample_rate: int = SAMPLE_RATE,
    n_mfcc: int = N_MFCC,
    n_fft: int = FFT_SIZE,
    hop_length: int = HOP_LENGTH,
    n_mels: int = N_MELS,
) -> np.ndarray:
    mel = mel_spectrogram(x, sample_rate, n_fft, hop_length, n_mels)
    log_mel = np.log(mel + 1e-8)
    coeffs = dct(log_mel, type=2, axis=1, norm="ortho")[:, :n_mfcc]
    return coeffs.astype(np.float32)


def delta(matrix: np.ndarray, width: int = 2) -> np.ndarray:
    """Regression-style first difference along the time axis."""
    if matrix.shape[0] < 2:
        return np.zeros_like(matrix)
    padded = np.pad(matrix, ((width, width), (0, 0)), mode="edge")
    denom = 2.0 * sum(i**2 for i in range(1, width + 1))
    out = np.zeros_like(matrix, dtype=np.float32)
    for i in range(1, width + 1):
        out += i * (padded[width + i : width + i + matrix.shape[0]]
                    - padded[width - i : width - i + matrix.shape[0]])
    return (out / denom).astype(np.float32)


# --------------------------------------------------------------------------------------
# Spectral shape descriptors
# --------------------------------------------------------------------------------------

@dataclass
class SpectralShape:
    centroid: np.ndarray
    spread: np.ndarray
    rolloff85: np.ndarray
    rolloff95: np.ndarray
    flatness: np.ndarray
    entropy: np.ndarray
    slope: np.ndarray
    flux: np.ndarray
    crest: np.ndarray


def spectral_shape(mag: np.ndarray, sample_rate: int = SAMPLE_RATE) -> SpectralShape:
    """Per-frame spectral shape descriptors from a magnitude spectrogram."""
    mag = np.maximum(mag.astype(np.float64), EPS)
    n_bins = mag.shape[1]
    freqs = np.linspace(0.0, sample_rate / 2.0, n_bins)
    power = mag**2
    total = power.sum(axis=1) + EPS
    prob = power / total[:, None]

    centroid = prob @ freqs
    spread = np.sqrt(np.maximum(prob @ (freqs**2) - centroid**2, 0.0))

    cumulative = np.cumsum(power, axis=1) / total[:, None]
    rolloff85 = freqs[np.argmax(cumulative >= 0.85, axis=1)]
    rolloff95 = freqs[np.argmax(cumulative >= 0.95, axis=1)]

    geo = np.exp(np.mean(np.log(power), axis=1))
    flatness = geo / (power.mean(axis=1) + EPS)
    entropy = -np.sum(prob * np.log(prob + EPS), axis=1) / np.log(n_bins)

    log_f = np.log(freqs + 1.0)
    log_p = np.log(power + EPS)
    fmean = log_f.mean()
    slope = ((log_p - log_p.mean(axis=1, keepdims=True)) @ (log_f - fmean)) / (
        np.sum((log_f - fmean) ** 2) + EPS
    )

    if mag.shape[0] > 1:
        norm = mag / (np.linalg.norm(mag, axis=1, keepdims=True) + EPS)
        flux = np.concatenate([[0.0], np.sqrt(np.sum(np.diff(norm, axis=0) ** 2, axis=1))])
    else:
        flux = np.zeros(mag.shape[0])

    crest = mag.max(axis=1) / (mag.mean(axis=1) + EPS)

    return SpectralShape(
        centroid=centroid, spread=spread, rolloff85=rolloff85, rolloff95=rolloff95,
        flatness=flatness, entropy=entropy, slope=slope, flux=flux, crest=crest,
    )


def band_energy_ratios(
    mag: np.ndarray,
    sample_rate: int = SAMPLE_RATE,
    bands: Tuple[Tuple[float, float], ...] = (
        (0.0, 300.0), (300.0, 1000.0), (1000.0, 2000.0),
        (2000.0, 4000.0), (4000.0, 6000.0), (6000.0, 8000.0),
    ),
) -> Dict[str, float]:
    """Fraction of total energy in each band, averaged over frames."""
    power = mag.astype(np.float64) ** 2
    freqs = np.linspace(0.0, sample_rate / 2.0, power.shape[1])
    total = power.sum() + EPS
    out: Dict[str, float] = {}
    for low, high in bands:
        sel = (freqs >= low) & (freqs < high)
        label = f"band_{int(low)}_{int(high)}"
        out[label] = float(power[:, sel].sum() / total) if sel.any() else 0.0
    return out


# --------------------------------------------------------------------------------------
# Phase structure
# --------------------------------------------------------------------------------------

def phase_features(
    x: np.ndarray,
    n_fft: int = FFT_SIZE,
    hop_length: int = HOP_LENGTH,
) -> Dict[str, float]:
    """Group-delay and phase-advance statistics.

    Vocoders (Griffin-Lim, HiFi-GAN, WaveNet-style) reconstruct phase either
    algorithmically or from a learned prior. Both give phase structure that is *more
    regular* across frequency and time than a real microphone recording of a real room.
    """
    spec = stft(x, n_fft, hop_length)
    if spec.shape[0] < 3:
        return {
            "phase_group_delay_std": 0.0,
            "phase_advance_dev": 0.0,
            "phase_advance_entropy": 0.0,
            "phase_coherence": 0.0,
        }

    phase = np.angle(spec)
    mag = np.abs(spec)
    # Weight by magnitude: phase in near-empty bins is meaningless noise.
    weight = mag / (mag.sum(axis=1, keepdims=True) + EPS)

    unwrapped = np.unwrap(phase, axis=1)
    group_delay = -np.diff(unwrapped, axis=1)
    gd_weight = weight[:, 1:] / (weight[:, 1:].sum(axis=1, keepdims=True) + EPS)
    gd_mean = np.sum(gd_weight * group_delay, axis=1, keepdims=True)
    gd_std = np.sqrt(np.sum(gd_weight * (group_delay - gd_mean) ** 2, axis=1))

    # Expected phase advance between consecutive frames for bin k is 2π·k·hop/n_fft.
    bins = np.arange(spec.shape[1])
    expected = 2.0 * np.pi * bins * hop_length / n_fft
    advance = np.diff(unwrapped, axis=0)
    deviation = np.angle(np.exp(1j * (advance - expected[None, :])))
    dev_weight = weight[1:] / (weight[1:].sum() + EPS)
    advance_dev = float(np.sum(dev_weight * np.abs(deviation)))

    hist, _ = np.histogram(deviation.ravel(), bins=24, range=(-np.pi, np.pi), density=True)
    hist = hist / (hist.sum() + EPS)
    advance_entropy = float(-np.sum(hist * np.log(hist + EPS)) / np.log(len(hist)))

    coherence = float(np.abs(np.mean(np.exp(1j * deviation))))

    return {
        "phase_group_delay_std": float(np.mean(gd_std)),
        "phase_advance_dev": advance_dev,
        "phase_advance_entropy": advance_entropy,
        "phase_coherence": coherence,
    }


# --------------------------------------------------------------------------------------
# Modulation spectrum
# --------------------------------------------------------------------------------------

def modulation_features(
    x: np.ndarray,
    sample_rate: int = SAMPLE_RATE,
    hop_length: int = HOP_LENGTH,
) -> Dict[str, float]:
    """Temporal-envelope modulation statistics (syllabic rhythm energy)."""
    mel = mel_spectrogram(x, sample_rate, hop_length=hop_length)
    if mel.shape[0] < 8:
        return {
            "mod_peak_hz": 0.0,
            "mod_depth": 0.0,
            "mod_syllabic_ratio": 0.0,
            "mod_high_ratio": 0.0,
            "mod_flatness": 0.0,
        }

    envelope = np.log(mel.sum(axis=1) + 1e-8)
    envelope = envelope - envelope.mean()
    frame_rate = sample_rate / float(hop_length)

    n = int(2 ** np.ceil(np.log2(len(envelope))))
    spectrum = np.abs(np.fft.rfft(envelope * np.hanning(len(envelope)), n=n))
    freqs = np.fft.rfftfreq(n, d=1.0 / frame_rate)
    power = spectrum**2
    total = power.sum() + EPS

    syllabic = (freqs >= 2.0) & (freqs <= 16.0)
    high = freqs > 16.0
    peak_idx = int(np.argmax(power[1:]) + 1) if len(power) > 1 else 0

    valid = power[1:] + EPS
    geo = np.exp(np.mean(np.log(valid)))
    flatness = float(geo / (valid.mean() + EPS))

    return {
        "mod_peak_hz": float(freqs[peak_idx]),
        "mod_depth": float(np.std(envelope)),
        "mod_syllabic_ratio": float(power[syllabic].sum() / total),
        "mod_high_ratio": float(power[high].sum() / total),
        "mod_flatness": flatness,
    }


# --------------------------------------------------------------------------------------
# Aggregation
# --------------------------------------------------------------------------------------

def _stats(name: str, values: np.ndarray) -> Dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return {f"{name}_mean": 0.0, f"{name}_std": 0.0}
    return {
        f"{name}_mean": float(np.nan_to_num(values.mean())),
        f"{name}_std": float(np.nan_to_num(values.std())),
    }


def spectral_feature_dict(x: np.ndarray, sample_rate: int = SAMPLE_RATE) -> Dict[str, float]:
    """The complete layer-1 feature dictionary for one window."""
    mag = magnitude_spectrogram(x)
    shape = spectral_shape(mag, sample_rate)

    features: Dict[str, float] = {}
    features.update(_stats("spec_centroid", shape.centroid))
    features.update(_stats("spec_spread", shape.spread))
    features.update(_stats("spec_rolloff85", shape.rolloff85))
    features.update(_stats("spec_rolloff95", shape.rolloff95))
    features.update(_stats("spec_flatness", shape.flatness))
    features.update(_stats("spec_entropy", shape.entropy))
    features.update(_stats("spec_slope", shape.slope))
    features.update(_stats("spec_flux", shape.flux))
    features.update(_stats("spec_crest", shape.crest))
    features.update(band_energy_ratios(mag, sample_rate))

    coeffs = mfcc(x, sample_rate)
    d_coeffs = delta(coeffs)
    # Keep the first 13 MFCC statistics: enough timbre detail without exploding the vector.
    for i in range(min(13, coeffs.shape[1])):
        features[f"mfcc{i}_mean"] = float(np.nan_to_num(coeffs[:, i].mean()))
        features[f"mfcc{i}_std"] = float(np.nan_to_num(coeffs[:, i].std()))
    features["mfcc_delta_energy"] = float(np.nan_to_num(np.mean(np.abs(d_coeffs))))
    features["mfcc_delta_std"] = float(np.nan_to_num(np.std(d_coeffs)))

    features.update(phase_features(x))
    features.update(modulation_features(x, sample_rate))
    return features
