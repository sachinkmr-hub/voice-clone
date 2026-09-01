"""Build the bootstrap corpus.

Writes a bona fide / synthetic corpus plus a JSON manifest, so that a clean checkout can
train and evaluate without downloading anything. Two properties of the manifest matter far
more than the audio itself:

``speaker_id``
    Every utterance is tagged with the synthetic speaker that produced it, and the
    train/test split is made **speaker-disjoint**. Splitting by file instead would let the
    model memorise speakers and report an accuracy that evaporates on the first real call.

``method``
    Each spoofed utterance records which vocoder family produced it, so evaluation can
    hold one family out entirely and measure performance on an *unseen attack* — the only
    number that says anything about ElevenLabs, RVC, or whatever ships next month.

Run::

    python -m ml.datasets.build --out data/corpus --per-class 120
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from dataclasses import asdict, dataclass
from typing import Dict, List

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "backend"))

import numpy as np

from voiceguard.audio.io import write_wav
from voiceguard.config import SAMPLE_RATE
from voiceguard.simulation.voice import (
    CLONE_METHODS,
    SpeakerTimbre,
    synthesize_bonafide,
    synthesize_cloned,
)

LANGUAGES = ("hi-IN", "en-IN", "ta-IN", "bn-IN", "mr-IN", "te-IN")


@dataclass
class Utterance:
    """One manifest row."""

    path: str
    label: int            #: 0 = bona fide, 1 = synthetic
    speaker_id: str
    language: str
    method: str           #: "human" for bona fide, else the vocoder family
    condition: str        #: "raw" (file straight from source) or "telephony"
    duration: float
    seed: int


def _capture_chain(audio: np.ndarray, rng: np.random.Generator,
                   *, dither: bool = True) -> np.ndarray:
    """The numerical tail every real recording has, applied identically to both classes.

    Without this, the corpus leaks. Measured on an earlier build: bona fide utterances
    carried a DC offset of ~4.4e-3 (the glottal pulse train is all-positive) while spoofed
    ones sat at ~4e-5 (the STFT round-trip removes DC), giving a single feature an AUC of
    0.81 and letting the classifier hit a meaningless 100 %. Every real capture chain —
    every ADC, every codec — has a DC blocker, so a corpus without one is not merely
    unrealistic, it is actively misleading.

    The same argument applies to dither: a real converter has a noise floor, so both
    classes get one.
    """
    from scipy.signal import lfilter

    out = np.asarray(audio, dtype=np.float64)
    out = out - out.mean()
    if out.size > 1:
        # One-pole DC blocker, y[n] = x[n] - x[n-1] + a*y[n-1], corner near 13 Hz.
        out = lfilter([1.0, -1.0], [1.0, -0.9975], out)

    if dither:
        # Converter noise floor. Real 16-bit capture never produces mathematically clean
        # samples, and giving only one class a clean floor is the quantisation leak.
        # Applied in the telephony condition only: a WAV exported straight out of a TTS
        # tool genuinely *is* numerically clean, and that is a real, usable signal for
        # the file-upload path — just not one that survives a phone network.
        amplitude = 10.0 ** (float(rng.uniform(-82.0, -66.0)) / 20.0)
        out = out + rng.standard_normal(out.size) * amplitude
    return out


def _channel_effects(audio: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Apply a random telephony channel to *both* classes.

    This is the single most important line of defence against a fake result. If only the
    synthetic side went through a band-limiting codec, the classifier would learn "narrow
    bandwidth = spoof" and score ~100 % here while being worthless on a real PSTN call,
    where *every* caller is band-limited. Applying the same channel distribution to both
    classes forces the model onto properties of the voice rather than of the line.
    """
    from scipy.signal import butter, sosfilt

    out = audio.astype(np.float64)
    nyquist = 0.5 * SAMPLE_RATE

    channel = rng.choice(["wideband", "narrowband", "mobile", "voip"])
    if channel == "narrowband":       # classic PSTN 300-3400 Hz
        sos = butter(6, [300 / nyquist, 3400 / nyquist], btype="band", output="sos")
        out = sosfilt(sos, out)
    elif channel == "mobile":         # AMR-ish band plus mild companding
        sos = butter(6, [200 / nyquist, 3800 / nyquist], btype="band", output="sos")
        out = sosfilt(sos, out)
        out = np.sign(out) * np.abs(out) ** float(rng.uniform(0.85, 1.0))
    elif channel == "voip":           # wider, but with packet-loss style dropouts
        sos = butter(4, min(0.99, 7000 / nyquist), btype="low", output="sos")
        out = sosfilt(sos, out)
        for _ in range(int(rng.integers(0, 3))):
            start = int(rng.uniform(0, max(1, len(out) - 400)))
            out[start : start + int(rng.uniform(80, 400))] *= float(rng.uniform(0.0, 0.3))

    if rng.random() < 0.5:            # additive line noise, both classes
        snr_db = float(rng.uniform(18.0, 40.0))
        noise = rng.standard_normal(len(out))
        amplitude = float(np.sqrt(np.mean(out**2) + 1e-12)) * (10 ** (-snr_db / 20.0))
        out = out + noise / (np.std(noise) + 1e-12) * amplitude

    return out


def build_corpus(
    out_dir: str,
    per_class: int = 120,
    *,
    speakers: int = 24,
    duration_range: tuple = (3.0, 7.0),
    seed: int = 1234,
    apply_channel: bool = True,
) -> Dict:
    """Generate the corpus and write ``manifest.json``."""
    rng = np.random.default_rng(seed)
    random.seed(seed)
    os.makedirs(out_dir, exist_ok=True)

    # A fixed speaker pool shared by both classes: the same voice appears as both bona
    # fide and cloned, which is exactly the attack (someone's real voice, then their
    # clone) and stops the model from separating classes on timbre alone.
    pool = [SpeakerTimbre.random(np.random.default_rng(seed + i), f"spk{i:03d}")
            for i in range(speakers)]

    utterances: List[Utterance] = []
    started = time.time()

    def emit(index: int, label: int, method: str) -> Utterance:
        """Generate one utterance. Both classes take the identical post-processing path."""
        speaker_idx = index % speakers
        timbre = pool[speaker_idx]
        language = str(rng.choice(LANGUAGES))
        duration = float(rng.uniform(*duration_range))
        utterance_seed = int(rng.integers(0, 2**31 - 1))

        if label == 0:
            audio = synthesize_bonafide(duration, timbre=timbre, seed=utterance_seed,
                                        language=language)
        else:
            audio = synthesize_cloned(duration, timbre=timbre, seed=utterance_seed,
                                      language=language, method=method)

        # Condition is drawn from the same distribution for both classes, so it can never
        # act as a class label. Roughly half the corpus is "as captured on a phone line",
        # the rest is "the file as exported", because the product has to work on both.
        condition = "telephony" if (apply_channel and rng.random() < 0.55) else "raw"
        audio = _capture_chain(audio, rng, dither=(condition == "telephony"))
        if condition == "telephony":
            audio = _channel_effects(audio, rng)

        peak = float(np.max(np.abs(audio))) if audio.size else 0.0
        if peak > 1e-9:
            audio = audio / peak * float(rng.uniform(0.3, 0.95))
        audio = np.clip(audio, -1.0, 1.0).astype(np.float32)

        prefix = "bonafide" if label == 0 else f"synthetic_{method}"
        name = f"{prefix}_{index:04d}_spk{speaker_idx:03d}_{condition}.wav"
        write_wav(os.path.join(out_dir, name), audio, SAMPLE_RATE)
        return Utterance(name, label, timbre.name, language, method, condition,
                         round(len(audio) / SAMPLE_RATE, 2), utterance_seed)

    for index in range(per_class):
        utterances.append(emit(index, 0, "human"))
    for index in range(per_class):
        utterances.append(emit(index, 1, CLONE_METHODS[index % len(CLONE_METHODS)]))

    manifest = {
        "created_at": time.time(),
        "sample_rate": SAMPLE_RATE,
        "n_utterances": len(utterances),
        "n_speakers": speakers,
        "per_class": per_class,
        "methods": list(CLONE_METHODS),
        "languages": list(LANGUAGES),
        "channel_effects": apply_channel,
        "conditions": {
            c: sum(1 for u in utterances if u.condition == c)
            for c in sorted({u.condition for u in utterances})
        },
        "seed": seed,
        "build_seconds": round(time.time() - started, 1),
        "source": "voiceguard.simulation (bootstrap corpus)",
        "warning": (
            "Simulated audio. Numbers measured on this corpus are a smoke test of the "
            "pipeline, NOT a claim about real TTS systems. See docs/MODEL_CARD.md and "
            "train on ASVspoof / In-the-Wild before quoting accuracy anywhere."
        ),
        "utterances": [asdict(u) for u in utterances],
    }
    with open(os.path.join(out_dir, "manifest.json"), "w") as handle:
        json.dump(manifest, handle, indent=2)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description="Build the VoiceGuard bootstrap corpus")
    parser.add_argument("--out", default="data/corpus")
    parser.add_argument("--per-class", type=int, default=120)
    parser.add_argument("--speakers", type=int, default=24)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--no-channel", action="store_true",
                        help="skip channel simulation (not recommended — see build.py)")
    args = parser.parse_args()

    manifest = build_corpus(args.out, args.per_class, speakers=args.speakers,
                            seed=args.seed, apply_channel=not args.no_channel)
    print(f"Wrote {manifest['n_utterances']} utterances to {args.out} "
          f"in {manifest['build_seconds']}s")
    print(f"  {manifest['per_class']} bona fide + {manifest['per_class']} synthetic "
          f"across {manifest['n_speakers']} speakers and "
          f"{len(manifest['methods'])} vocoder families")
    print(f"  channel effects applied to BOTH classes: {manifest['channel_effects']}")
    print(f"\n  {manifest['warning']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
