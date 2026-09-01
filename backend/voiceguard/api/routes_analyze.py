"""Single-shot analysis: upload a recording, get a verdict.

This is the front door — the endpoint behind the website's "drop a call recording here"
box. It runs the *same* streaming pipeline as the live path (same chunker, same windows,
same smoothing), just fed from a file, so an uploaded recording and a live call produce
comparable numbers. That equivalence is deliberate: a demo that scored differently from
production would teach everyone the wrong thing about the product.
"""

from __future__ import annotations

import base64
import binascii
import logging
from typing import List, Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status

from voiceguard.api.deps import AppState, check_consent, get_state, require_api_key
from voiceguard.api.schemas import AnalyzeRequest, AnalyzeResponse
from voiceguard.audio.io import decode_audio_bytes, duration_seconds, sha256_hex
from voiceguard.config import SAMPLE_RATE
from voiceguard.models.context import CallContext
from voiceguard.pipeline.manager import SessionLimitExceeded
from voiceguard.privacy.anonymize import redact_transcript

logger = logging.getLogger("voiceguard.api.analyze")
router = APIRouter(prefix="/v1", tags=["analyze"])

#: Upper bound on a single upload. 25 MB of 16 kHz mono is ~13 minutes, which is longer
#: than any call that needs a *live* verdict; longer recordings should be chunked by the
#: caller so they get progressive results rather than one long wait.
MAX_UPLOAD_BYTES = 25 * 1024 * 1024
MIN_ANALYZABLE_SECONDS = 0.5


def _decode(raw: bytes, encoding: str, sample_rate: Optional[int]):
    if not raw:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "Empty audio payload.")
    if len(raw) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            f"Audio exceeds the {MAX_UPLOAD_BYTES // (1024 * 1024)} MB limit. "
            f"Split it, or use the streaming endpoint for long calls.",
        )
    samples, rate = decode_audio_bytes(raw, encoding=encoding, sample_rate=sample_rate,
                                       target_rate=SAMPLE_RATE)
    if samples.size == 0:
        raise HTTPException(
            status.HTTP_422_UNPROCESSABLE_ENTITY,
            "Could not decode the audio. Supported: WAV/FLAC/OGG containers, or raw "
            "pcm16/float32 with an explicit sample_rate.",
        )
    if duration_seconds(samples, rate) < MIN_ANALYZABLE_SECONDS:
        raise HTTPException(
            status.HTTP_422_UNPROCESSABLE_ENTITY,
            f"Need at least {MIN_ANALYZABLE_SECONDS} s of audio to analyse.",
        )
    return samples, rate


async def _run(
    state: AppState,
    samples,
    *,
    raw: bytes,
    language: str,
    profile: str,
    identity: Optional[str],
    call_context: Optional[CallContext],
    session_id: Optional[str],
    verbose: bool,
) -> AnalyzeResponse:
    """Replay a buffer through the streaming pipeline and summarise the result."""
    try:
        session = state.sessions.create(
            session_id=session_id, profile=profile, language=language,
            identity=identity, call_context=call_context,
            metadata={"mode": "upload", "bytes": len(raw)},
        )
    except SessionLimitExceeded as exc:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, str(exc)) from exc

    state.repository.open_session(session)
    audio_hash = sha256_hex(raw)
    timeline: List[dict] = []

    # Feed in 0.5 s pushes so the window/hop behaviour is byte-for-byte what the live
    # WebSocket path does.
    step = int(0.5 * SAMPLE_RATE)
    assessments = []
    for start in range(0, len(samples), step):
        assessments.extend(session.ingest(samples[start : start + step]))
    assessments.extend(session.flush())

    for assessment in assessments:
        state.repository.record_assessment(session.id, assessment, audio_sha256=audio_hash)
        timeline.append({
            "t": round(assessment.elapsed_seconds, 2),
            "score": round(assessment.score, 1),
            "band": assessment.band,
            "confidence": round(assessment.confidence, 3),
            "provisional": assessment.provisional,
        })
        record = await state.alerts.maybe_alert(
            session.id, assessment, profile=state.profile(profile),
            metadata={"mode": "upload"})
        if record:
            state.repository.record_alert(session.id, record)
            session.stats.alerts_raised += 1

    report = session.report(include_trail=verbose)
    state.sessions.close(session.id)
    state.repository.close_session(session, report)
    await state.publish({"type": "session_complete", "session": report})

    latest = session.latest
    if latest is None:
        return AnalyzeResponse(
            session_id=session.id, verdict="insufficient_audio", score=0.0, peak_score=0.0,
            band="LOW", action="Not enough speech to analyse.",
            headline="No speech detected in this recording.",
            duration_seconds=report["duration_seconds"], windows_analyzed=0,
            mean_latency_ms=0.0,
            caveats=["The recording contained no detectable speech."],
            timeline=timeline, report=report if verbose else None,
        )

    return AnalyzeResponse(
        session_id=session.id,
        verdict=report["verdict"],
        score=report["final_score"],
        peak_score=report["peak_score"],
        band=latest.band,
        action=latest.action,
        headline=latest.explanation.headline,
        duration_seconds=report["duration_seconds"],
        windows_analyzed=report["stats"]["windows_analyzed"],
        mean_latency_ms=report["stats"]["mean_latency_ms"],
        factors=[f.as_dict() for f in latest.explanation.factors],
        caveats=latest.explanation.caveats,
        layers=latest.explanation.layer_summary,
        timeline=timeline,
        report=report if verbose else None,
    )


@router.post(
    "/analyze",
    response_model=AnalyzeResponse,
    dependencies=[Depends(require_api_key), Depends(check_consent)],
    summary="Analyse a base64 audio payload",
)
async def analyze(request: AnalyzeRequest, state: AppState = Depends(get_state)):
    try:
        raw = base64.b64decode(request.audio_base64, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise HTTPException(status.HTTP_400_BAD_REQUEST,
                            f"audio_base64 is not valid base64: {exc}") from exc

    samples, _ = _decode(raw, request.encoding, request.sample_rate)
    context = None
    if request.call_context:
        payload = request.call_context.model_dump()
        payload["transcript"] = redact_transcript(payload.get("transcript", ""))
        context = CallContext.from_dict(payload)

    return await _run(
        state, samples, raw=raw, language=request.language, profile=request.profile,
        identity=request.identity, call_context=context, session_id=request.session_id,
        verbose=request.verbose,
    )


@router.post(
    "/analyze/file",
    response_model=AnalyzeResponse,
    dependencies=[Depends(require_api_key), Depends(check_consent)],
    summary="Analyse an uploaded recording (multipart)",
)
async def analyze_file(
    file: UploadFile = File(..., description="WAV/FLAC/OGG call recording"),
    language: str = Form("auto"),
    profile: str = Form("default"),
    identity: Optional[str] = Form(None),
    transaction_amount: float = Form(0.0),
    known_contact: Optional[bool] = Form(None),
    caller_id_verified: Optional[bool] = Form(None),
    claimed_role: str = Form(""),
    transcript: str = Form(""),
    verbose: bool = Form(False),
    state: AppState = Depends(get_state),
):
    """The endpoint the website's upload box posts to."""
    raw = await file.read()
    samples, _ = _decode(raw, "auto", None)

    context = None
    if any([transaction_amount, known_contact is not None, caller_id_verified is not None,
            claimed_role, transcript]):
        context = CallContext(
            transaction_amount=transaction_amount,
            known_contact=known_contact,
            caller_id_verified=caller_id_verified,
            claimed_role=claimed_role,
            transcript=redact_transcript(transcript),
        )

    return await _run(
        state, samples, raw=raw, language=language, profile=profile, identity=identity,
        call_context=context, session_id=None, verbose=verbose,
    )
