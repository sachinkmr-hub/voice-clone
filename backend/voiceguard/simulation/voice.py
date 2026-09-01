"""A source-filter speech simulator, plus a real vocoder round-trip.

Why this module exists
----------------------
Judging a spoof detector on audio it has never heard is the only honest test, and the
right corpora for that are ASVspoof 2019/2021 and In-the-Wild (``ml/train.py`` reads both).
But a hackathon repo also has to *run* on a laptop with no dataset download, so this module
provides a self-contained bootstrap:

``synthesize_bonafide``
    A source-filter vocal tract: a jittered, shimmered glottal pulse train driving
    time-varying formant resonators, with fricative bursts, breath noise, a real room-tone
    floor and human pause structure. It is not intelligible speech, but every acoustic
    property our detector measures is in the human range.

``synthesize_cloned``
    The same speaker content, then pushed through :func:`mel_vocoder_roundtrip` — an
    actual mel-spectrogram inversion with Griffin-Lim phase reconstruction, which is the
    same lossy path a real neural TTS vocoder takes. The artifacts it leaves (phase
    incoherence, mel-band-limited high end, over-smoothed envelope) are therefore *real
    vocoder artifacts*, not hand-drawn ones — which is what makes the bootstrap model
    transfer at all.

The generated audio is labelled honestly everywhere it is used, and
``docs/MODEL_CARD.md`` states plainly that numbers measured on it are a smoke test, not a
claim about ElevenLabs or RVC.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from scipy.signal import butter, lfilter, sosfilt

from voiceguard.config import SAMPLE_RATE
from voiceguard.features.spectral import mel_filterbank

EPS = 1e-12


# --------------------------------------------------------------------------------------
# Speaker definition
# --------------------------------------------------------------------------------------

@dataclass
class SpeakerTimbre:
    """A synthetic speaker: pitch range, formant targets and voice quality."""

    name: str = "speaker"
    f0_base: float = 165.0
    f0_range_semitones: float = 4.5
    formants: Tuple[float, ...] = (700.0, 1220.0, 2600.0, 3400.0)
    bandwidths: Tuple[float, ...] = (90.0, 110.0, 170.0, 250.0)
    spectral_tilt: float = 1.15
    breathiness: float = 0.02

    @staticmethod
    def random(rng: np.random.Generator, name: str = "speaker") -> "SpeakerTimbre":
        female = rng.random() < 0.5
        f0 = float(rng.uniform(180.0, 240.0) if female else rng.uniform(95.0, 155.0))
        scale = 1.14 if female else 1.0
        return SpeakerTimbre(
            name=name,
            f0_base=f0,
            f0_range_semitones=float(rng.uniform(3.0, 7.0)),
            formants=(
                float(rng.uniform(560.0, 820.0) * scale),
                float(rng.uniform(1050.0, 1450.0) * scale),
                float(rng.uniform(2350.0, 2900.0) * scale),
                float(rng.uniform(3200.0, 3800.0) * scale),
            ),
            bandwidths=(
                float(rng.uniform(70.0, 120.0)),
                float(rng.uniform(90.0, 150.0)),
                float(rng.uniform(140.0, 220.0)),
                float(rng.uniform(200.0, 300.0)),
            ),
            spectral_tilt=float(rng.uniform(0.95, 1.4)),
            breathiness=float(rng.uniform(0.01, 0.045)),
        )


# --------------------------------------------------------------------------------------
# Building blocks
# --------------------------------------------------------------------------------------

def _resonator(x: np.ndarray, freq: float, bandwidth: float, sample_rate: int) -> np.ndarray:
    """Single two-pole formant resonator (Klatt-style)."""
    freq = float(np.clip(freq, 60.0, sample_rate / 2.0 - 120.0))
    r = float(np.exp(-np.pi * bandwidth / sample_rate))
    theta = 2.0 * np.pi * freq / sample_rate
    a = [1.0, -2.0 * r * np.cos(theta), r * r]
    gain = (1.0 - 2.0 * r * np.cos(theta) + r * r)
    return lfilter([gain], a, x)


def _syllable_envelope(
    n: int,
    sample_rate: int,
    rng: np.random.Generator,
    rate_hz: float,
    regular: bool,
) -> np.ndarray:
    """Amplitude envelope built from overlapping syllable-shaped bumps."""
    duration = n / float(sample_rate)
    n_syll = max(1, int(duration * rate_hz))
    env = np.full(n, 0.05, dtype=np.float64)
    t = np.arange(n) / float(sample_rate)
    position = 0.0
    for _ in range(n_syll * 2):
        if position >= duration:
            break
        length = 1.0 / rate_hz * (1.0 if regular else float(rng.uniform(0.6, 1.6)))
        peak = position + length * 0.45
        width = length * (0.30 if regular else float(rng.uniform(0.22, 0.42)))
        height = 1.0 if regular else float(rng.uniform(0.55, 1.0))
        env += height * np.exp(-0.5 * ((t - peak) / max(width, 1e-3)) ** 2)
        position += length * (1.0 if regular else float(rng.uniform(0.85, 1.25)))
    return env / max(float(env.max()), EPS)


def _formant_trajectory(
    n: int,
    sample_rate: int,
    rng: np.random.Generator,
    target: float,
    smooth: bool,
) -> np.ndarray:
    """Slowly-varying formant centre frequency (vowel transitions)."""
    duration = max(n / float(sample_rate), 1e-3)
    n_points = max(2, int(duration * (3.0 if smooth else 5.0)))
    spread = 0.06 if smooth else 0.16
    points = target * (1.0 + rng.uniform(-spread, spread, n_points))
    return np.interp(np.linspace(0, n_points - 1, n), np.arange(n_points), points)


def _glottal_source(
    n: int,
    sample_rate: int,
    rng: np.random.Generator,
    timbre: SpeakerTimbre,
    *,
    jitter: float,
    shimmer: float,
    contour_smooth: bool,
) -> Tuple[np.ndarray, np.ndarray]:
    """Jittered/shimmered glottal pulse train. Returns ``(source, f0_track)``."""
    t = np.arange(n) / float(sample_rate)
    span = timbre.f0_base * (2.0 ** (timbre.f0_range_semitones / 12.0) - 1.0)

    # Intonation: a slow declination plus a couple of phrase accents.
    contour = -0.25 * span * (t / max(t[-1], EPS))
    for _ in range(2 if contour_smooth else 4):
        centre = float(rng.uniform(0.0, max(t[-1], EPS)))
        width = float(rng.uniform(0.25, 0.7))
        contour += float(rng.uniform(-span, span)) * np.exp(
            -0.5 * ((t - centre) / width) ** 2
        )
    f0 = timbre.f0_base + contour

    if jitter > 0:
        # Random-walk jitter, band-limited: real cycle perturbation is correlated.
        walk = np.cumsum(rng.standard_normal(n)) / np.sqrt(max(n, 1))
        walk = walk - walk.mean()
        f0 = f0 * (1.0 + jitter * (rng.standard_normal(n) * 0.7 + walk * 0.3))
    f0 = np.clip(f0, 60.0, 400.0)

    # Pulse train by phase accumulation: a pulse each time the phase wraps.
    phase = np.cumsum(f0) / float(sample_rate)
    source = np.zeros(n, dtype=np.float64)
    pulse_idx = np.flatnonzero(np.diff(np.floor(phase)) > 0) + 1
    if pulse_idx.size:
        amps = np.ones(pulse_idx.size)
        if shimmer > 0:
            amps = amps * (1.0 + shimmer * rng.standard_normal(pulse_idx.size))
        source[pulse_idx] = np.clip(amps, 0.05, 3.0)

    # Rosenberg-like pulse shaping: a one-pole lowpass gives the -12 dB/oct glottal tilt.
    sos = butter(2, min(0.45, 900.0 / (0.5 * sample_rate)), btype="low", output="sos")
    source = sosfilt(sos, source)
    source += timbre.breathiness * rng.standard_normal(n)
    return source, f0


def _apply_pauses(
    x: np.ndarray,
    sample_rate: int,
    rng: np.random.Generator,
    *,
    digital: bool,
    pause_ratio: float,
) -> np.ndarray:
    """Insert pauses. ``digital=True`` writes true zeros (the TTS giveaway)."""
    out = x.copy()
    duration = len(x) / float(sample_rate)
    budget = duration * pause_ratio
    placed = 0.0
    guard = 0
    while placed < budget and guard < 40:
        guard += 1
        length = float(rng.uniform(0.12, 0.30)) if digital else float(rng.uniform(0.10, 0.55))
        start = float(rng.uniform(0.1, max(0.2, duration - length - 0.1)))
        i0, i1 = int(start * sample_rate), int((start + length) * sample_rate)
        if i1 >= len(out):
            continue
        if digital:
            out[i0:i1] = 0.0
        else:
            # A human pause is not silence: it is room tone plus breath.
            room = 0.0016 * rng.standard_normal(i1 - i0)
            if rng.random() < 0.4:
                breath_len = min(i1 - i0, int(0.18 * sample_rate))
                breath = 0.006 * rng.standard_normal(breath_len)
                sos = butter(2, [900 / (0.5 * sample_rate), 3200 / (0.5 * sample_rate)],
                             btype="band", output="sos")
                room[:breath_len] += sosfilt(sos, breath)
            out[i0:i1] = room
        placed += length
    return out


# --------------------------------------------------------------------------------------
# Vocoder round-trip (the honest part of the synthetic path)
# --------------------------------------------------------------------------------------

def _stft_complex(x: np.ndarray, n_fft: int, hop: int) -> np.ndarray:
    window = np.hanning(n_fft)
    n_frames = 1 + max(0, (len(x) - n_fft) // hop)
    frames = np.stack([x[i * hop : i * hop + n_fft] * window for i in range(n_frames)])
    return np.fft.rfft(frames, axis=1)


def _istft(spec: np.ndarray, n_fft: int, hop: int, length: int) -> np.ndarray:
    window = np.hanning(n_fft)
    frames = np.fft.irfft(spec, n=n_fft, axis=1) * window
    out = np.zeros(length + n_fft, dtype=np.float64)
    norm = np.zeros(length + n_fft, dtype=np.float64)
    for i, frame in enumerate(frames):
        start = i * hop
        out[start : start + n_fft] += frame
        norm[start : start + n_fft] += window**2

    # At the very first and last samples only one window tail overlaps, so the norm tends
    # to zero and dividing by it manufactures an enormous edge spike — which then eats the
    # whole dynamic range when the result is peak-normalised. Suppress those samples
    # instead of amplifying them.
    floor = 1e-3 * float(norm.max()) if norm.size and norm.max() > 0 else 1.0
    result = np.zeros_like(out)
    usable = norm > floor
    result[usable] = out[usable] / norm[usable]
    return result[:length]


def griffin_lim(
    magnitude: np.ndarray,
    n_fft: int,
    hop: int,
    length: int,
    n_iter: int = 32,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """Reconstruct a waveform from a magnitude spectrogram by iterative phase estimation.

    This is the classic vocoder failure mode: magnitude is preserved, phase is invented.
    Even at 32 iterations the phase never becomes as coherent as a real recording's, which
    is precisely what ``features.spectral.phase_features`` measures.
    """
    rng = rng or np.random.default_rng(0)
    angles = np.exp(2j * np.pi * rng.random(magnitude.shape))
    signal = _istft(magnitude * angles, n_fft, hop, length)
    for _ in range(n_iter):
        spec = _stft_complex(signal, n_fft, hop)
        frames = min(spec.shape[0], magnitude.shape[0])
        angles = np.exp(1j * np.angle(spec[:frames]))
        signal = _istft(magnitude[:frames] * angles, n_fft, hop, length)
    return signal


def _mel_bottleneck(
    x: np.ndarray,
    sample_rate: int,
    n_fft: int,
    hop: int,
    n_mels: int,
    smoothing: float,
) -> tuple:
    """Analyse → mel → invert. Returns ``(approx_magnitude, original_complex_spec)``.

    This is the lossy step every mel-based TTS shares, regardless of vocoder: the mel
    projection is rank-deficient (513 bins → 80 mels), so fine spectral detail and the
    top of the band cannot be recovered.
    """
    spec = _stft_complex(x, n_fft, hop)
    mag = np.abs(spec)
    fb = mel_filterbank(sample_rate, n_fft=n_fft, n_mels=n_mels, fmin=40.0,
                        fmax=sample_rate / 2.0)
    mel = mag @ fb.T

    if smoothing > 0 and mel.shape[0] > 4:
        # The acoustic model predicts a smooth mel trajectory; this stands in for that.
        kernel = np.array([smoothing / 2, 1.0 - smoothing, smoothing / 2])
        mel = np.apply_along_axis(lambda m: np.convolve(m, kernel, mode="same"), 0, mel)

    inverse = np.linalg.pinv(fb.T)
    return np.maximum(mel @ inverse, 0.0), spec


def mel_vocoder_roundtrip(
    x: np.ndarray,
    sample_rate: int = SAMPLE_RATE,
    *,
    n_fft: int = 1024,
    hop: int = 256,
    n_mels: int = 80,
    n_iter: int = 32,
    smoothing: float = 0.35,
    method: str = "griffin_lim",
    phase_jitter: float = 0.0,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """Push audio through a mel bottleneck and resynthesise it, like a TTS vocoder.

    ``method`` selects the vocoder family, because they leave *different* fingerprints and
    a detector trained on only one of them learns that one vocoder rather than "cloning":

    ``griffin_lim``
        Classic / low-end TTS. Phase is discarded and re-estimated iteratively, leaving
        incoherent phase and audibly rough cycle structure.
    ``neural``
        Proxy for a modern GAN vocoder (HiFi-GAN, BigVGAN). Phase is reconstructed well,
        so the tell-tales are the *smoothness* ones — over-smoothed mel trajectories,
        band limitation, and suppressed micro-variation — not phase.
    ``hybrid``
        Partial phase corruption, for the diffusion/flow vocoders that sit between.
    """
    rng = rng or np.random.default_rng(0)
    x = np.asarray(x, dtype=np.float64)
    if x.size < n_fft * 2:
        return x.astype(np.float32)

    approx, spec = _mel_bottleneck(x, sample_rate, n_fft, hop, n_mels, smoothing)

    if method == "griffin_lim":
        out = griffin_lim(approx, n_fft, hop, len(x), n_iter=n_iter, rng=rng)
    else:
        # A good neural vocoder reproduces phase nearly correctly; the residual error is
        # small and structured rather than random.
        phase = np.angle(spec)
        if method == "hybrid":
            phase_jitter = phase_jitter or 0.55
        if phase_jitter > 0:
            drift = np.cumsum(rng.standard_normal(phase.shape) * 0.08, axis=0)
            phase = phase + phase_jitter * drift
        frames = min(approx.shape[0], phase.shape[0])
        out = _istft(approx[:frames] * np.exp(1j * phase[:frames]), n_fft, hop, len(x))

    peak = float(np.max(np.abs(out))) if out.size else 0.0
    if peak > EPS:
        out = out * (float(np.max(np.abs(x))) / peak)
    return out.astype(np.float32)


# --------------------------------------------------------------------------------------
# Public generators
# --------------------------------------------------------------------------------------

def _render(
    timbre: SpeakerTimbre,
    duration: float,
    sample_rate: int,
    rng: np.random.Generator,
    *,
    jitter: float,
    shimmer: float,
    syllable_rate: float,
    regular_rhythm: bool,
    smooth_formants: bool,
) -> np.ndarray:
    n = int(duration * sample_rate)
    source, _ = _glottal_source(
        n, sample_rate, rng, timbre,
        jitter=jitter, shimmer=shimmer, contour_smooth=smooth_formants,
    )

    # Vocal tract: sum of formant resonators with slowly-moving centre frequencies.
    # Piecewise-constant per 25 ms block keeps this cheap while still moving.
    block = max(1, int(0.025 * sample_rate))
    voiced = np.zeros(n, dtype=np.float64)
    trajectories = [
        _formant_trajectory(n, sample_rate, rng, f, smooth_formants)
        for f in timbre.formants
    ]
    for start in range(0, n, block):
        end = min(n, start + block)
        segment = source[start:end]
        acc = np.zeros(end - start, dtype=np.float64)
        for idx, (bw, traj) in enumerate(zip(timbre.bandwidths, trajectories)):
            freq = float(traj[start])
            acc += _resonator(segment, freq, bw, sample_rate) / (idx + 1) ** timbre.spectral_tilt
        voiced[start:end] = acc

    # Fricatives: shaped noise bursts, which is where the real high-band energy lives.
    fricative = np.zeros(n, dtype=np.float64)
    n_bursts = max(1, int(duration * 1.6))
    sos = butter(4, [3000 / (0.5 * sample_rate), min(0.98, 7600 / (0.5 * sample_rate))],
                 btype="band", output="sos")
    for _ in range(n_bursts):
        length = int(float(rng.uniform(0.04, 0.11)) * sample_rate)
        start = int(rng.uniform(0, max(1, n - length)))
        burst = sosfilt(sos, rng.standard_normal(length))
        fricative[start : start + length] += burst * float(rng.uniform(0.05, 0.16))

    envelope = _syllable_envelope(n, sample_rate, rng, syllable_rate, regular_rhythm)
    signal = voiced * envelope + fricative * (0.35 + 0.65 * envelope)

    peak = float(np.max(np.abs(signal))) if signal.size else 0.0
    if peak > EPS:
        signal = signal / peak * 0.72
    return signal


def synthesize_bonafide(
    duration: float = 4.0,
    sample_rate: int = SAMPLE_RATE,
    *,
    timbre: Optional[SpeakerTimbre] = None,
    seed: Optional[int] = None,
    language: str = "auto",
    snr_db: float = 34.0,
) -> np.ndarray:
    """Human-profile speech: real micro-variation, room tone, irregular rhythm."""
    rng = np.random.default_rng(seed)
    timbre = timbre or SpeakerTimbre.random(rng)
    rate = {"hi-IN": 5.0, "ta-IN": 5.2, "en-IN": 4.4}.get(language, 4.6)

    signal = _render(
        timbre, duration, sample_rate, rng,
        jitter=float(rng.uniform(0.008, 0.020)),
        shimmer=float(rng.uniform(0.06, 0.13)),
        syllable_rate=rate * float(rng.uniform(0.85, 1.15)),
        regular_rhythm=False,
        smooth_formants=False,
    )
    signal = _apply_pauses(signal, sample_rate, rng, digital=False,
                           pause_ratio=float(rng.uniform(0.14, 0.28)))

    # Room tone + a little pink-ish noise: the noise floor a real handset always has.
    noise = rng.standard_normal(len(signal))
    sos = butter(1, 0.35, btype="low", output="sos")
    room = sosfilt(sos, noise) * 0.6 + noise * 0.4
    amplitude = float(np.sqrt(np.mean(signal**2) + EPS)) * (10 ** (-snr_db / 20.0))
    signal = signal + room / (np.std(room) + EPS) * amplitude
    return np.clip(signal, -1.0, 1.0).astype(np.float32)


#: The synthesis families the bootstrap corpus mixes over. Keeping several is the whole
#: point: a detector trained on one vocoder learns that vocoder, not synthetic speech.
CLONE_METHODS: Tuple[str, ...] = ("griffin_lim", "neural", "hybrid")


def synthesize_cloned(
    duration: float = 4.0,
    sample_rate: int = SAMPLE_RATE,
    *,
    timbre: Optional[SpeakerTimbre] = None,
    seed: Optional[int] = None,
    language: str = "auto",
    method: str = "auto",
    vocoder_iterations: int = 24,
    smoothing: Optional[float] = None,
    band_limit_hz: Optional[float] = None,
) -> np.ndarray:
    """Cloned-profile speech: flat micro-variation, then a real vocoder round-trip.

    ``method`` is one of :data:`CLONE_METHODS`, or ``"auto"`` to pick one at random —
    which is what the corpus builder uses so the training set spans vocoder families.
    """
    rng = np.random.default_rng(seed)
    timbre = timbre or SpeakerTimbre.random(rng)
    rate = {"hi-IN": 5.0, "ta-IN": 5.2, "en-IN": 4.4}.get(language, 4.6)
    if method == "auto":
        method = str(rng.choice(CLONE_METHODS))

    # Per-family defaults: a good neural vocoder is *smoother* and wider-band than
    # Griffin-Lim, so a single setting would misrepresent both.
    if smoothing is None:
        smoothing = {"griffin_lim": 0.40, "neural": 0.55, "hybrid": 0.45}.get(method, 0.45)
    if band_limit_hz is None:
        band_limit_hz = float(rng.choice([6800.0, 7200.0, 7600.0, 0.0]))

    signal = _render(
        timbre, duration, sample_rate, rng,
        jitter=float(rng.uniform(0.0002, 0.0018)),   # far below the human 0.3–1.5 %
        shimmer=float(rng.uniform(0.004, 0.020)),
        syllable_rate=rate,
        regular_rhythm=True,
        smooth_formants=True,
    )

    signal = mel_vocoder_roundtrip(
        signal, sample_rate, n_iter=vocoder_iterations, smoothing=smoothing,
        method=method, rng=rng,
    ).astype(np.float64)

    if band_limit_hz and band_limit_hz > 0:
        sos = butter(8, min(0.99, band_limit_hz / (0.5 * sample_rate)),
                     btype="low", output="sos")
        signal = sosfilt(sos, signal)

    signal = _apply_pauses(signal, sample_rate, rng, digital=True,
                           pause_ratio=float(rng.uniform(0.10, 0.20)))
    peak = float(np.max(np.abs(signal))) if signal.size else 0.0
    if peak > EPS:
        signal = signal / peak * 0.72
    return np.clip(signal, -1.0, 1.0).astype(np.float32)
