"""Synthesis-artifact probes.

These are the features that carry most of the discriminative weight, because they target
things a vocoder *has* to get wrong rather than things it merely tends to get wrong.

+------------------------------+------------------------------------------------------+
| Probe                        | Why a synthesiser trips it                           |
+==============================+======================================================+
| high-frequency energy cliff  | Mel-inversion and most neural vocoders are trained   |
|                              | at a fixed bandwidth; above it the spectrum falls off|
|                              | far more steeply than any microphone or codec does.  |
+------------------------------+------------------------------------------------------+
| digital-silence floor        | Real rooms have tone. Generated silence is either    |
|                              | numerically zero or an unnaturally steady low shelf. |
+------------------------------+------------------------------------------------------+
| spectral comb / periodicity  | Transposed-convolution upsampling in GAN vocoders    |
|                              | leaves regularly-spaced spectral ripple.             |
+------------------------------+------------------------------------------------------+
| LPC-residual kurtosis        | Natural glottal excitation is impulsive (high        |
|                              | kurtosis); learned excitation is closer to Gaussian. |
+------------------------------+------------------------------------------------------+
| envelope over-smoothness     | Acoustic models predict smooth trajectories; the     |
|                              | second derivative of a real envelope is much rougher.|
+------------------------------+------------------------------------------------------+
| harmonic over-regularity     | Real harmonics wander in frequency and amplitude.    |
+------------------------------+------------------------------------------------------+
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
from scipy.linalg import solve_toeplitz
from scipy.signal import lfilter

from voiceguard.audio.vad import SpeechSegments, detect_speech
from voiceguard.config import FFT_SIZE, HOP_LENGTH, SAMPLE_RATE
from voiceguard.features.spectral import (
    magnitude_spectrogram,
    mel_filterbank,
    mel_spectrogram,
)

EPS = 1e-12


# --------------------------------------------------------------------------------------
# Long-term average spectrum probes
# --------------------------------------------------------------------------------------

def _ltas_db(x: np.ndarray, sample_rate: int) -> tuple:
    """Long-term average spectrum in dB plus its frequency axis."""
    mag = magnitude_spectrogram(x)
    power = (mag.astype(np.float64) ** 2).mean(axis=0)
    freqs = np.linspace(0.0, sample_rate / 2.0, power.size)
    ltas = 10.0 * np.log10(power + 1e-16)
    return ltas, freqs


def _smooth(values: np.ndarray, width: int) -> np.ndarray:
    if values.size < width or width < 2:
        return values
    kernel = np.ones(width) / float(width)
    return np.convolve(values, kernel, mode="same")


def bandwidth_features(
    x: np.ndarray,
    sample_rate: int = SAMPLE_RATE,
    ltas_cache: Optional[tuple] = None,
) -> Dict[str, float]:
    """Detect a band-limitation cliff and quantify high-band content."""
    ltas, freqs = ltas_cache if ltas_cache is not None else _ltas_db(x, sample_rate)
    if ltas.size < 16:
        return {
            "hf_cliff_hz": 0.0, "hf_cliff_depth_db": 0.0, "hf_energy_ratio": 0.0,
            "hf_shelf_flatness": 0.0, "ltas_tilt_db_per_khz": 0.0,
        }

    smooth = _smooth(ltas, 5)
    peak_db = float(np.max(smooth))
    # Search only above 2.5 kHz: below that, a "cliff" is just the natural spectral tilt.
    search = freqs >= 2500.0
    cliff_hz, cliff_depth = 0.0, 0.0
    if search.sum() > 6:
        band = smooth[search]
        band_freqs = freqs[search]
        # Steepest fall over a ~500 Hz span, measured in dB per kHz.
        span = max(2, int(500.0 / max(band_freqs[1] - band_freqs[0], EPS)))
        if band.size > span + 1:
            drops = band[:-span] - band[span:]
            idx = int(np.argmax(drops))
            slope_db_per_khz = float(drops[idx] / (span * (band_freqs[1] - band_freqs[0]) / 1000.0))
            # Only call it a cliff if the level *stays* down afterwards.
            after = band[idx + span :]
            sustained = float(peak_db - np.mean(after)) if after.size else 0.0
            if slope_db_per_khz > 12.0 and sustained > 18.0:
                cliff_hz = float(band_freqs[idx + span // 2])
                cliff_depth = sustained

    power = 10.0 ** (ltas / 10.0)
    speech_band = (freqs >= 300.0) & (freqs <= 3400.0)
    high_band = freqs >= 6000.0
    hf_ratio = float(power[high_band].sum() / (power[speech_band].sum() + EPS)) if high_band.any() else 0.0

    if high_band.sum() > 3:
        shelf = power[high_band] + EPS
        geo = np.exp(np.mean(np.log(shelf)))
        shelf_flatness = float(geo / (shelf.mean() + EPS))
    else:
        shelf_flatness = 0.0

    valid = freqs > 100.0
    tilt = float(
        np.polyfit(freqs[valid] / 1000.0, smooth[valid], 1)[0]
    ) if valid.sum() > 4 else 0.0

    return {
        "hf_cliff_hz": cliff_hz,
        "hf_cliff_depth_db": cliff_depth,
        "hf_energy_ratio": hf_ratio,
        "hf_shelf_flatness": shelf_flatness,
        "ltas_tilt_db_per_khz": tilt,
    }


def comb_features(
    x: np.ndarray,
    sample_rate: int = SAMPLE_RATE,
    ltas_cache: Optional[tuple] = None,
) -> Dict[str, float]:
    """Detect regularly-spaced spectral ripple left by upsampling layers."""
    ltas, freqs = ltas_cache if ltas_cache is not None else _ltas_db(x, sample_rate)
    if ltas.size < 32:
        return {"comb_strength": 0.0, "comb_spacing_hz": 0.0, "ltas_ripple_db": 0.0}

    detrended = ltas - _smooth(ltas, 15)
    detrended = detrended - detrended.mean()
    ripple = float(np.std(detrended))

    # Autocorrelate along the *frequency* axis: a comb shows a strong non-zero-lag peak.
    norm = float(np.dot(detrended, detrended)) + EPS
    acf = np.correlate(detrended, detrended, mode="full")[detrended.size - 1 :] / norm
    lo = 3
    hi = min(acf.size, detrended.size // 3)
    if hi <= lo:
        return {"comb_strength": 0.0, "comb_spacing_hz": 0.0, "ltas_ripple_db": ripple}
    peak_lag = int(np.argmax(acf[lo:hi]) + lo)
    bin_hz = float(freqs[1] - freqs[0]) if freqs.size > 1 else 0.0

    return {
        "comb_strength": float(max(0.0, acf[peak_lag])),
        "comb_spacing_hz": float(peak_lag * bin_hz),
        "ltas_ripple_db": ripple,
    }


# --------------------------------------------------------------------------------------
# Silence / noise-floor probes
# --------------------------------------------------------------------------------------

def silence_features(
    x: np.ndarray,
    sample_rate: int = SAMPLE_RATE,
    segments: Optional[SpeechSegments] = None,
) -> Dict[str, float]:
    """Characterise the non-speech portions: real rooms are never this clean."""
    x = np.asarray(x, dtype=np.float64)
    segments = segments if segments is not None else detect_speech(x, sample_rate)

    exact_zero_ratio = float(np.mean(x == 0.0)) if x.size else 0.0
    near_zero_ratio = float(np.mean(np.abs(x) < 1e-5)) if x.size else 0.0

    energy_db = segments.frame_energy_db
    if segments.mask.size == energy_db.size and (~segments.mask).any():
        silence_db = energy_db[~segments.mask]
    else:
        silence_db = np.zeros(0, dtype=np.float32)

    if silence_db.size >= 2:
        floor = float(np.median(silence_db))
        floor_std = float(np.std(silence_db))
        floor_range = float(np.ptp(silence_db))
    else:
        floor, floor_std, floor_range = float(segments.noise_floor_db), 0.0, 0.0

    # A room-tone floor sits around -70..-45 dBFS with several dB of wobble. Anything
    # below -85 dB with near-zero variance is a generated silence.
    digital_silence = float(
        np.clip((-70.0 - floor) / 25.0, 0.0, 1.0) * np.clip(1.0 - floor_std / 4.0, 0.0, 1.0)
    )

    return {
        "silence_exact_zero_ratio": exact_zero_ratio,
        "silence_near_zero_ratio": near_zero_ratio,
        "silence_floor_db_abs": floor,
        "silence_floor_std_db": floor_std,
        "silence_floor_range_db": floor_range,
        "digital_silence_score": digital_silence,
    }


# --------------------------------------------------------------------------------------
# Excitation probes
# --------------------------------------------------------------------------------------

def _lpc(frame: np.ndarray, order: int) -> np.ndarray:
    """Levinson-Durbin LPC coefficients via a Toeplitz solve."""
    autocorr = np.correlate(frame, frame, mode="full")[frame.size - 1 :]
    autocorr = autocorr[: order + 1]
    if autocorr.size < order + 1 or autocorr[0] <= EPS:
        return np.zeros(order)
    autocorr = autocorr + np.concatenate([[1e-6 * autocorr[0]], np.zeros(order)])
    try:
        coeffs = solve_toeplitz((autocorr[:order], autocorr[:order]), autocorr[1 : order + 1])
    except Exception:  # pragma: no cover - singular frames
        return np.zeros(order)
    return np.nan_to_num(coeffs)


def residual_features(x: np.ndarray, sample_rate: int = SAMPLE_RATE,
                      order: int = 16) -> Dict[str, float]:
    """Statistics of the LPC residual — a proxy for the excitation signal."""
    x = np.asarray(x, dtype=np.float64)
    frame_len = int(0.032 * sample_rate)
    hop = frame_len          # no overlap: LPC statistics are stable without it
    max_frames = 24          # caps cost on long buffers; plenty for a stable median
    if x.size < frame_len * 2:
        return {"residual_kurtosis": 0.0, "residual_skew_abs": 0.0,
                "residual_peakiness": 0.0, "residual_autocorr": 0.0}

    kurts, skews, peaks, acs = [], [], [], []
    n_frames = 1 + (x.size - frame_len) // hop
    stride = max(1, n_frames // max_frames)
    window = np.hanning(frame_len)
    for i in range(0, n_frames, stride):
        frame = x[i * hop : i * hop + frame_len] * window
        if float(np.sqrt(np.mean(frame**2))) < 1e-5:
            continue
        coeffs = _lpc(frame, order)
        if not np.any(coeffs):
            continue
        residual = lfilter(np.concatenate([[1.0], -coeffs]), [1.0], frame)
        std = float(np.std(residual))
        if std < EPS:
            continue
        z = residual / std
        kurts.append(float(np.mean(z**4)))
        skews.append(abs(float(np.mean(z**3))))
        peaks.append(float(np.max(np.abs(z))))
        denom = float(np.dot(residual, residual)) + EPS
        acs.append(float(np.dot(residual[:-1], residual[1:]) / denom))

    if not kurts:
        return {"residual_kurtosis": 0.0, "residual_skew_abs": 0.0,
                "residual_peakiness": 0.0, "residual_autocorr": 0.0}
    return {
        "residual_kurtosis": float(np.median(kurts)),
        "residual_skew_abs": float(np.median(skews)),
        "residual_peakiness": float(np.median(peaks)),
        "residual_autocorr": float(np.median(acs)),
    }


def envelope_smoothness(
    x: np.ndarray,
    sample_rate: int = SAMPLE_RATE,
    mel_cache: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """How rough is the amplitude envelope? Generated speech is over-smooth."""
    mel = mel_cache if mel_cache is not None else mel_spectrogram(x, sample_rate, hop_length=HOP_LENGTH)
    if mel.shape[0] < 6:
        return {"env_roughness": 0.0, "env_second_diff": 0.0, "env_lag1_autocorr": 0.0,
                "mel_frame_corr": 0.0}

    env = np.log(mel.sum(axis=1) + 1e-8)
    env = env - env.mean()
    scale = float(np.std(env)) + EPS

    first = np.diff(env)
    second = np.diff(env, n=2)
    denom = float(np.dot(env, env)) + EPS

    # Frame-to-frame correlation of the mel spectra themselves: TTS frames are more
    # self-similar than natural frames at the same speaking rate.
    normed = mel / (np.linalg.norm(mel, axis=1, keepdims=True) + EPS)
    frame_corr = float(np.mean(np.sum(normed[:-1] * normed[1:], axis=1)))

    return {
        "env_roughness": float(np.mean(np.abs(first)) / scale),
        "env_second_diff": float(np.mean(np.abs(second)) / scale),
        "env_lag1_autocorr": float(np.dot(env[:-1], env[1:]) / denom),
        "mel_frame_corr": frame_corr,
    }


def harmonic_regularity(x: np.ndarray, sample_rate: int = SAMPLE_RATE,
                        f0_hint: float = 0.0,
                        mag_cache: Optional[np.ndarray] = None) -> Dict[str, float]:
    """How precisely do harmonics sit on integer multiples of F0, and how steady are they?"""
    mag = mag_cache if mag_cache is not None else magnitude_spectrogram(x)
    if mag.shape[0] < 4 or f0_hint <= 0:
        return {"harmonic_dev_hz": 0.0, "harmonic_amp_cv": 0.0, "harmonic_count": 0.0}

    freqs = np.linspace(0.0, sample_rate / 2.0, mag.shape[1])
    bin_hz = float(freqs[1] - freqs[0])
    avg = mag.mean(axis=0)

    deviations, amplitudes = [], []
    for k in range(1, 13):
        target = k * f0_hint
        if target >= sample_rate / 2.0 - 2 * bin_hz:
            break
        lo = int(max(0, (target - 0.35 * f0_hint) / bin_hz))
        hi = int(min(avg.size - 1, (target + 0.35 * f0_hint) / bin_hz))
        if hi <= lo + 1:
            continue
        local = avg[lo : hi + 1]
        peak = int(np.argmax(local)) + lo
        deviations.append(abs(freqs[peak] - target))
        amplitudes.append(float(avg[peak]))

    if len(amplitudes) < 3:
        return {"harmonic_dev_hz": 0.0, "harmonic_amp_cv": 0.0, "harmonic_count": 0.0}

    amps = np.array(amplitudes)
    return {
        "harmonic_dev_hz": float(np.mean(deviations)),
        "harmonic_amp_cv": float(np.std(amps) / (np.mean(amps) + EPS)),
        "harmonic_count": float(len(amplitudes)),
    }


def signal_hygiene(x: np.ndarray) -> Dict[str, float]:
    """Clipping, DC offset and effective quantisation — cheap sanity probes."""
    x = np.asarray(x, dtype=np.float64)
    if x.size == 0:
        return {"clipping_ratio": 0.0, "dc_offset": 0.0, "quantization_levels": 0.0,
                "crest_factor": 0.0}
    peak = float(np.max(np.abs(x)))
    clipping = float(np.mean(np.abs(x) > 0.985)) if peak > 0.985 else 0.0
    rms = float(np.sqrt(np.mean(x**2) + EPS))

    # Effective quantisation: how many distinct 16-bit codes does this window use,
    # normalised by how many it could use given its own amplitude range?
    codes = np.unique(np.round(x * 32768.0).astype(np.int32))
    span = max(1.0, float(np.ptp(np.round(x * 32768.0))) if x.size > 1 else 1.0)
    levels = float(codes.size / span)

    return {
        "clipping_ratio": clipping,
        "dc_offset": float(abs(np.mean(x))),
        "quantization_levels": levels,
        "crest_factor": float(peak / (rms + EPS)),
    }


# --------------------------------------------------------------------------------------
# Aggregation
# --------------------------------------------------------------------------------------

def artifact_feature_dict(
    x: np.ndarray,
    sample_rate: int = SAMPLE_RATE,
    *,
    segments: Optional[SpeechSegments] = None,
    f0_hint: float = 0.0,
) -> Dict[str, float]:
    """The complete synthesis-artifact feature dictionary for one window."""
    # The magnitude spectrogram, its long-term average and the mel spectrogram are each
    # needed by several probes; compute them once per window.
    mag = magnitude_spectrogram(x)
    power = (mag.astype(np.float64) ** 2).mean(axis=0)
    ltas_cache = (10.0 * np.log10(power + 1e-16),
                  np.linspace(0.0, sample_rate / 2.0, power.size))
    mel_cache = (mag.astype(np.float64) ** 2) @ mel_filterbank(sample_rate).T

    features: Dict[str, float] = {}
    features.update(bandwidth_features(x, sample_rate, ltas_cache))
    features.update(comb_features(x, sample_rate, ltas_cache))
    features.update(silence_features(x, sample_rate, segments))
    features.update(residual_features(x, sample_rate))
    features.update(envelope_smoothness(x, sample_rate, mel_cache))
    features.update(harmonic_regularity(x, sample_rate, f0_hint, mag))
    features.update(signal_hygiene(x))
    return {k: float(np.nan_to_num(v)) for k, v in features.items()}
