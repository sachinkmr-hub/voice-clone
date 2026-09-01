"""Layer-2 prosodic / behavioural analysis.

Neural TTS has become extremely good at *timbre* and merely good at *timing*. The
residual gap shows up in the small stuff:

* **Micro-variation.** A human larynx never repeats a pitch period exactly; cycle-to-cycle
  jitter of 0.3–1.5 % and shimmer of 3–8 % are normal. Vocoder output is usually far
  smoother, because the acoustic model predicts a smooth F0 contour and the vocoder
  renders it faithfully.
* **Pause structure.** Human pauses follow a heavy-tailed distribution driven by
  breathing and planning. TTS pauses are punctuation-driven and cluster tightly.
* **Energy dynamics.** Real rooms and real mouths move; synthetic level contours are flat.

All measurements are z-scored against a per-language population prior
(:data:`voiceguard.config.LANGUAGE_PROFILES`) so that, e.g., the wider conversational
pitch range of Hindi is not read as an anomaly.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
from scipy.signal import butter, filtfilt, find_peaks

from voiceguard.audio.vad import SpeechSegments, detect_speech
from voiceguard.config import SAMPLE_RATE, LanguageProfile, language_profile

EPS = 1e-12
F0_MIN_HZ = 60.0
F0_MAX_HZ = 420.0
PITCH_FRAME_MS = 40.0
PITCH_HOP_MS = 10.0
VOICING_THRESHOLD = 0.36


@dataclass
class PitchTrack:
    """Frame-synchronous F0 estimates."""

    f0: np.ndarray                       #: Hz, 0.0 where unvoiced
    voiced: np.ndarray                   #: bool
    confidence: np.ndarray               #: normalised autocorrelation peak
    amplitude: np.ndarray                #: per-frame RMS
    hop_seconds: float
    periods: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float64))

    @property
    def voiced_ratio(self) -> float:
        return float(np.mean(self.voiced)) if self.voiced.size else 0.0

    def voiced_f0(self) -> np.ndarray:
        return self.f0[self.voiced] if self.voiced.any() else np.zeros(0)


def _parabolic_peak(values: np.ndarray, index: int) -> float:
    """Sub-sample peak location by parabolic interpolation — needed for usable jitter."""
    if index <= 0 or index >= len(values) - 1:
        return float(index)
    a, b, c = values[index - 1], values[index], values[index + 1]
    denom = a - 2.0 * b + c
    if abs(denom) < EPS:
        return float(index)
    return float(index + 0.5 * (a - c) / denom)


def track_pitch(
    x: np.ndarray,
    sample_rate: int = SAMPLE_RATE,
    *,
    fmin: float = F0_MIN_HZ,
    fmax: float = F0_MAX_HZ,
) -> PitchTrack:
    """Autocorrelation pitch tracker with sub-sample period refinement.

    Chosen over a neural tracker deliberately: it is ~2 ms per window, has no model
    dependency (so it runs in the on-device path), and its period estimates are precise
    enough for jitter because of the parabolic refinement step.
    """
    x = np.asarray(x, dtype=np.float64)
    frame_len = int(sample_rate * PITCH_FRAME_MS / 1000.0)
    hop_len = int(sample_rate * PITCH_HOP_MS / 1000.0)
    hop_seconds = hop_len / float(sample_rate)

    if x.size < frame_len:
        empty = np.zeros(0)
        return PitchTrack(empty, empty.astype(bool), empty, empty, hop_seconds)

    min_lag = max(2, int(sample_rate / fmax))
    max_lag = min(frame_len - 2, int(sample_rate / fmin))
    n_frames = 1 + (len(x) - frame_len) // hop_len

    f0 = np.zeros(n_frames)
    conf = np.zeros(n_frames)
    amp = np.zeros(n_frames)
    periods = np.zeros(n_frames)
    window = np.hanning(frame_len)

    for i in range(n_frames):
        frame = x[i * hop_len : i * hop_len + frame_len]
        amp[i] = float(np.sqrt(np.mean(frame**2) + EPS))
        frame = (frame - frame.mean()) * window
        energy = float(np.dot(frame, frame))
        if energy < 1e-9:
            continue

        # Autocorrelation via FFT, normalised by the sliding energy so long lags are
        # not penalised (this is the "normalised cross-correlation" formulation).
        n_fft = int(2 ** np.ceil(np.log2(2 * frame_len)))
        spec = np.fft.rfft(frame, n=n_fft)
        acf = np.fft.irfft(spec * np.conj(spec), n=n_fft)[: max_lag + 1]
        cumsq = np.concatenate([[0.0], np.cumsum(frame**2)])
        norms = np.array(
            [np.sqrt(max(energy * (cumsq[frame_len] - cumsq[lag]), EPS))
             for lag in range(max_lag + 1)]
        )
        nacf = acf / norms
        nacf[:min_lag] = 0.0

        peak = int(np.argmax(nacf[min_lag : max_lag + 1]) + min_lag)
        score = float(nacf[peak])
        if score < VOICING_THRESHOLD:
            continue

        refined = _parabolic_peak(nacf, peak)
        if refined <= 0:
            continue
        period = refined / float(sample_rate)
        freq = 1.0 / period
        if fmin <= freq <= fmax:
            f0[i] = freq
            periods[i] = period
            conf[i] = score

    voiced = f0 > 0
    return PitchTrack(f0=f0, voiced=voiced, confidence=conf, amplitude=amp,
                      hop_seconds=hop_seconds, periods=periods)


# --------------------------------------------------------------------------------------
# Perturbation measures
# --------------------------------------------------------------------------------------

def _voiced_runs(voiced: np.ndarray, min_len: int = 4) -> List[np.ndarray]:
    runs, start = [], None
    for i, flag in enumerate(voiced):
        if flag and start is None:
            start = i
        elif not flag and start is not None:
            if i - start >= min_len:
                runs.append(np.arange(start, i))
            start = None
    if start is not None and len(voiced) - start >= min_len:
        runs.append(np.arange(start, len(voiced)))
    return runs


def extract_cycles(
    x: np.ndarray,
    track: PitchTrack,
    sample_rate: int = SAMPLE_RATE,
) -> List[tuple]:
    """Locate individual glottal cycles inside each voiced run.

    Frame-level F0 (40 ms analysis frames) averages over ~7 pitch periods and therefore
    *destroys* exactly the cycle-to-cycle perturbation we want to measure. So for
    jitter/shimmer we go back to the waveform: low-pass just above F0 to isolate the
    fundamental, pick its peaks with a minimum-distance constraint derived from the
    tracked F0, and refine each peak parabolically. Returns
    ``[(periods_seconds, cycle_amplitudes), ...]`` — one tuple per voiced run.
    """
    runs = _voiced_runs(track.voiced, min_len=5)
    if not runs or x.size == 0:
        return []

    hop = max(1, int(round(track.hop_seconds * sample_rate)))
    frame_len = int(sample_rate * PITCH_FRAME_MS / 1000.0)
    nyquist = 0.5 * sample_rate
    out: List[tuple] = []

    for run in runs:
        start = int(run[0] * hop)
        end = min(len(x), int(run[-1] * hop) + frame_len)
        seg = np.asarray(x[start:end], dtype=np.float64)
        run_f0 = track.f0[run]
        run_f0 = run_f0[run_f0 > 0]
        if run_f0.size == 0:
            continue
        f0_mean = float(np.mean(run_f0))
        if f0_mean <= 0 or seg.size < 4 * sample_rate / f0_mean:
            continue

        cutoff = min(0.9 * nyquist, max(2.2 * f0_mean, 220.0))
        try:
            b, a = butter(4, cutoff / nyquist, btype="low")
            filtered = filtfilt(b, a, seg)
        except Exception:  # pragma: no cover - degenerate segment
            filtered = seg

        min_distance = max(2, int(0.72 * sample_rate / f0_mean))
        peaks, _ = find_peaks(filtered, distance=min_distance)
        if peaks.size < 4:
            continue

        refined = np.array([_parabolic_peak(filtered, int(p)) for p in peaks])
        periods = np.diff(refined) / float(sample_rate)
        valid = (periods > 1.0 / F0_MAX_HZ) & (periods < 1.0 / F0_MIN_HZ)
        if valid.sum() < 3:
            continue

        half = max(1, int(0.5 * sample_rate / f0_mean))
        amplitudes = np.array([
            float(np.max(np.abs(seg[max(0, int(p) - half) : min(seg.size, int(p) + half + 1)])))
            for p in peaks
        ])
        out.append((periods[valid], amplitudes[: periods.size][valid]))
    return out


def _perturbation(values: np.ndarray) -> tuple:
    """(local perturbation, 3-point smoothed perturbation) as a fraction of the mean."""
    values = np.asarray(values, dtype=np.float64)
    values = values[values > 0]
    if values.size < 3:
        return 0.0, 0.0
    mean = float(np.mean(values))
    if mean <= EPS:
        return 0.0, 0.0
    local = float(np.mean(np.abs(np.diff(values))) / mean)
    smoothed = np.convolve(values, np.ones(3) / 3.0, mode="valid")
    centre = values[1 : 1 + smoothed.size]
    smooth_pert = float(np.mean(np.abs(centre - smoothed)) / mean)
    return local, smooth_pert


def jitter_shimmer(
    track: PitchTrack,
    x: Optional[np.ndarray] = None,
    sample_rate: int = SAMPLE_RATE,
) -> Dict[str, float]:
    """Cycle-level jitter (period perturbation) and shimmer (amplitude perturbation).

    Typical healthy human conversational speech: jitter 0.3–1.5 %, shimmer 3–8 %.
    Vocoder output usually lands well below both, which is one of the strongest and most
    explainable signals we have.
    """
    empty = {
        "jitter_local": 0.0, "jitter_ppq5": 0.0, "shimmer_local": 0.0,
        "shimmer_apq3": 0.0, "hnr_db": 0.0, "cycle_count": 0.0,
        "period_cv": 0.0, "amplitude_cv": 0.0,
    }
    if x is None:
        return empty

    cycles = extract_cycles(np.asarray(x, dtype=np.float64), track, sample_rate)
    if not cycles:
        return empty

    jitters, ppq5s, shimmers, apq3s = [], [], [], []
    period_cvs, amp_cvs = [], []
    total_cycles = 0

    for periods, amplitudes in cycles:
        total_cycles += int(periods.size)
        j_local, _ = _perturbation(periods)
        if j_local:
            jitters.append(j_local)
        if periods.size >= 5:
            mean_p = float(np.mean(periods))
            smoothed = np.convolve(periods, np.ones(5) / 5.0, mode="valid")
            centre = periods[2 : 2 + smoothed.size]
            ppq5s.append(float(np.mean(np.abs(centre - smoothed)) / max(mean_p, EPS)))
            period_cvs.append(float(np.std(periods) / max(mean_p, EPS)))

        s_local, s_apq3 = _perturbation(amplitudes)
        if s_local:
            shimmers.append(s_local)
            apq3s.append(s_apq3)
            amp_cvs.append(float(np.std(amplitudes) / max(np.mean(amplitudes), EPS)))

    conf = track.confidence[track.voiced]
    conf = np.clip(conf[conf > 0], 1e-4, 0.999)
    hnr = float(np.mean(10.0 * np.log10(conf / (1.0 - conf)))) if conf.size else 0.0

    def _avg(values: List[float]) -> float:
        return float(np.mean(values)) if values else 0.0

    return {
        "jitter_local": _avg(jitters),
        "jitter_ppq5": _avg(ppq5s),
        "shimmer_local": _avg(shimmers),
        "shimmer_apq3": _avg(apq3s),
        "hnr_db": hnr,
        "cycle_count": float(total_cycles),
        "period_cv": _avg(period_cvs),
        "amplitude_cv": _avg(amp_cvs),
    }


def contour_features(track: PitchTrack, profile: LanguageProfile) -> Dict[str, float]:
    """Shape and roughness of the F0 contour, z-scored against the language prior."""
    voiced_f0 = track.voiced_f0()
    if voiced_f0.size < 4:
        return {
            "f0_mean": 0.0, "f0_std": 0.0, "f0_range": 0.0, "f0_cv": 0.0,
            "f0_delta_std": 0.0, "f0_micro_var": 0.0, "f0_contour_entropy": 0.0,
            "f0_z_mean": 0.0, "f0_z_std": 0.0, "voiced_ratio": track.voiced_ratio,
        }

    mean = float(np.mean(voiced_f0))
    std = float(np.std(voiced_f0))
    semitones = 12.0 * np.log2(np.maximum(voiced_f0, EPS) / max(mean, EPS))

    deltas = np.diff(voiced_f0)
    kernel = np.ones(5) / 5.0
    if voiced_f0.size >= 5:
        smoothed = np.convolve(voiced_f0, kernel, mode="valid")
        residual = voiced_f0[2 : 2 + len(smoothed)] - smoothed
        micro = float(np.std(residual) / max(mean, EPS))
    else:
        micro = 0.0

    hist, _ = np.histogram(semitones, bins=16, range=(-12.0, 12.0), density=True)
    hist = hist / (hist.sum() + EPS)
    entropy = float(-np.sum(hist * np.log(hist + EPS)) / np.log(len(hist)))

    return {
        "f0_mean": mean,
        "f0_std": std,
        "f0_range": float(np.percentile(voiced_f0, 95) - np.percentile(voiced_f0, 5)),
        "f0_cv": float(std / max(mean, EPS)),
        "f0_delta_std": float(np.std(deltas)),
        "f0_micro_var": micro,
        "f0_contour_entropy": entropy,
        "f0_z_mean": float((mean - profile.f0_mean_hz) / max(profile.f0_std_hz, EPS)),
        "f0_z_std": float((std - profile.f0_std_hz) / max(profile.f0_std_hz, EPS)),
        "voiced_ratio": track.voiced_ratio,
    }


def rhythm_features(
    x: np.ndarray,
    segments: SpeechSegments,
    profile: LanguageProfile,
    sample_rate: int = SAMPLE_RATE,
) -> Dict[str, float]:
    """Pause structure, speaking rate proxy and energy dynamics."""
    pauses = np.array([p for p in segments.pause_durations if p > 0.05])
    speeches = np.array(segments.speech_durations or [0.0])

    energy_db = segments.frame_energy_db
    if energy_db.size:
        energy_db = energy_db[np.isfinite(energy_db)]
    speech_db = energy_db[segments.mask] if segments.mask.size == energy_db.size and segments.mask.any() else energy_db
    silence_db = (
        energy_db[~segments.mask]
        if segments.mask.size == energy_db.size and (~segments.mask).any()
        else np.zeros(0)
    )

    # Syllable-rate proxy: peaks of the smoothed energy envelope.
    frame_rate = sample_rate / float(max(segments.hop_length, 1))
    rate = 0.0
    if speech_db.size > 4:
        env = speech_db - speech_db.mean()
        if env.size >= 5:
            env = np.convolve(env, np.ones(3) / 3.0, mode="same")
        peaks = np.sum((env[1:-1] > env[:-2]) & (env[1:-1] > env[2:]) & (env[1:-1] > 0))
        duration = env.size / max(frame_rate, EPS)
        rate = float(peaks / max(duration, EPS))

    return {
        "pause_count": float(len(pauses)),
        "pause_mean": float(np.mean(pauses)) if pauses.size else 0.0,
        "pause_std": float(np.std(pauses)) if pauses.size else 0.0,
        "pause_ratio": float(1.0 - segments.speech_ratio),
        "pause_ratio_z": float(
            ((1.0 - segments.speech_ratio) - profile.pause_ratio) / max(profile.pause_ratio, EPS)
        ),
        "speech_run_mean": float(np.mean(speeches)),
        "speech_run_std": float(np.std(speeches)),
        "syllable_rate": rate,
        "syllable_rate_z": float((rate - profile.syllable_rate_hz) / max(profile.syllable_rate_hz, EPS)),
        "energy_std_db": float(np.std(speech_db)) if speech_db.size else 0.0,
        "energy_range_db": float(np.ptp(speech_db)) if speech_db.size > 1 else 0.0,
        "silence_floor_db": float(np.median(silence_db)) if silence_db.size else float(segments.noise_floor_db),
        "silence_floor_std": float(np.std(silence_db)) if silence_db.size > 1 else 0.0,
    }


def prosody_feature_dict(
    x: np.ndarray,
    sample_rate: int = SAMPLE_RATE,
    *,
    language: Optional[str] = None,
    segments: Optional[SpeechSegments] = None,
) -> Dict[str, float]:
    """The complete layer-2 feature dictionary for one window."""
    profile = language_profile(language)
    segments = segments if segments is not None else detect_speech(x, sample_rate)
    track = track_pitch(x, sample_rate)

    features: Dict[str, float] = {}
    features.update(contour_features(track, profile))
    features.update(jitter_shimmer(track, x, sample_rate))
    features.update(rhythm_features(x, segments, profile, sample_rate))
    features["speech_ratio"] = float(segments.speech_ratio)
    return features
