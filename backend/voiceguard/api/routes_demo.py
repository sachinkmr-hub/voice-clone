"""Demo endpoints.

A live demo fails when the presenter has to find a file. These endpoints generate
labelled sample audio on the fly so the console always has something to analyse, and
expose the scenario definitions the front end uses for its one-click walkthrough.

Everything here is clearly marked as simulated. It is a demonstration aid, never a
benchmark: see ``docs/MODEL_CARD.md``.
"""

from __future__ import annotations

import io
import wave
from typing import Optional

import numpy as np
from fastapi import APIRouter, HTTPException, Query, status
from fastapi.responses import Response

from voiceguard.config import SAMPLE_RATE
from voiceguard.simulation.voice import (
    CLONE_METHODS,
    SpeakerTimbre,
    synthesize_bonafide,
    synthesize_cloned,
)

router = APIRouter(prefix="/v1/demo", tags=["demo"])

MAX_SECONDS = 20.0


def _wav_bytes(audio: np.ndarray, sample_rate: int = SAMPLE_RATE) -> bytes:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes((np.clip(audio, -1.0, 1.0) * 32767.0).astype("<i2").tobytes())
    return buffer.getvalue()


@router.get("/sample", summary="Generate a labelled demo recording")
def sample(
    kind: str = Query("bonafide", pattern="^(bonafide|cloned)$"),
    method: str = Query("auto", description="Vocoder family for kind=cloned"),
    seconds: float = Query(6.0, ge=1.0, le=MAX_SECONDS),
    seed: Optional[int] = Query(None),
    language: str = Query("hi-IN"),
    speaker: int = Query(1, ge=0, le=99, description="Same value = same synthetic voice"),
):
    """Return a WAV the console can analyse.

    ``speaker`` is the point of the endpoint: request a bona fide sample and a cloned
    sample with the *same* speaker value and you get the actual attack — one person's real
    voice and a clone of it — which is far more convincing than two unrelated clips.
    """
    if kind == "cloned" and method != "auto" and method not in CLONE_METHODS:
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST,
            f"Unknown method {method!r}. Choose from: {', '.join(CLONE_METHODS)}, or 'auto'.",
        )

    timbre = SpeakerTimbre.random(np.random.default_rng(1000 + speaker), f"demo{speaker}")
    resolved_seed = seed if seed is not None else int(np.random.default_rng().integers(0, 10**6))

    if kind == "bonafide":
        audio = synthesize_bonafide(seconds, timbre=timbre, seed=resolved_seed,
                                    language=language)
        label = "bonafide"
    else:
        audio = synthesize_cloned(seconds, timbre=timbre, seed=resolved_seed,
                                  language=language, method=method)
        label = f"cloned-{method}"

    return Response(
        content=_wav_bytes(audio),
        media_type="audio/wav",
        headers={
            "Content-Disposition": f'inline; filename="demo_{label}_spk{speaker}.wav"',
            "X-VoiceGuard-Label": label,
            "X-VoiceGuard-Simulated": "true",
            "Cache-Control": "no-store",
        },
    )


@router.get("/scenarios", summary="Demo scenarios for the console walkthrough")
def scenarios():
    """The scripted narrative the console offers as one-click buttons."""
    return {
        "note": (
            "All demo audio is simulated by voiceguard.simulation and is labelled as such. "
            "It demonstrates the pipeline; it is not evidence of accuracy against real "
            "cloning tools. See docs/MODEL_CARD.md."
        ),
        "methods": list(CLONE_METHODS),
        "scenarios": [
            {
                "id": "genuine_cfo",
                "title": "Genuine call from the CFO",
                "kind": "bonafide",
                "speaker": 7,
                "language": "hi-IN",
                "expect": "LOW",
                "story": "The real CFO calls the payments desk about a routine transfer.",
                "call_context": {
                    "known_contact": True, "caller_id_verified": True, "origin": "pstn",
                    "claimed_role": "CFO", "transaction_amount": 250000, "local_hour": 14,
                },
            },
            {
                "id": "cloned_cfo_wire",
                "title": "Cloned CFO demanding an urgent wire",
                "kind": "cloned",
                "method": "neural",
                "speaker": 7,
                "language": "hi-IN",
                "expect": "HIGH or CRITICAL",
                "story": (
                    "The same voice — but synthesised — calls after hours from an "
                    "unverified number demanding a ₹42 lakh transfer before close."
                ),
                "call_context": {
                    "known_contact": False, "caller_id_verified": False, "origin": "voip",
                    "claimed_role": "CFO", "transaction_amount": 4200000, "local_hour": 22,
                    "transcript": "This is urgent, do not tell anyone, release it now.",
                },
            },
            {
                "id": "cloned_family",
                "title": "Vishing call impersonating a family member",
                "kind": "cloned",
                "method": "griffin_lim",
                "speaker": 22,
                "language": "ta-IN",
                "expect": "HIGH",
                "story": "A senior citizen receives a distress call in a familiar voice.",
                "call_context": {
                    "known_contact": False, "caller_id_verified": False,
                    "origin": "international", "local_hour": 23, "prior_fraud_reports": 1,
                    "transcript": "Please send money immediately, don't tell anyone.",
                },
            },
            {
                "id": "hybrid_agent",
                "title": "Hybrid vocoder, contact-centre profile",
                "kind": "cloned",
                "method": "hybrid",
                "speaker": 3,
                "language": "en-IN",
                "expect": "ELEVATED or HIGH",
                "story": "A less obvious synthesis method against the low-false-positive profile.",
                "call_context": {"known_contact": True, "origin": "voip", "local_hour": 11},
            },
        ],
    }
