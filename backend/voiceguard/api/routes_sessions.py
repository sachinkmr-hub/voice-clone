"""Session inspection, reporting, enrolment and erasure."""

from __future__ import annotations

import base64
import binascii
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, status

from voiceguard.api.deps import AppState, get_state, require_api_key
from voiceguard.api.schemas import (
    EnrolRequest,
    EnrolResponse,
    SessionCreateRequest,
    SessionResponse,
)
from voiceguard.audio.io import decode_audio_bytes, duration_seconds
from voiceguard.config import SAMPLE_RATE
from voiceguard.models.context import CallContext
from voiceguard.pipeline.manager import SessionLimitExceeded

router = APIRouter(prefix="/v1", tags=["sessions"], dependencies=[Depends(require_api_key)])

#: Enrolment needs enough speech to characterise a voice; below this the centroid is
#: dominated by whatever phoneme happened to be spoken.
MIN_ENROLMENT_SECONDS = 1.5


@router.post("/sessions", response_model=SessionResponse, status_code=status.HTTP_201_CREATED,
             summary="Open a session for streaming")
def create_session(request: SessionCreateRequest, state: AppState = Depends(get_state)):
    context = CallContext.from_dict(request.call_context.model_dump()) if request.call_context else None
    try:
        session = state.sessions.create(
            profile=request.profile, language=request.language, identity=request.identity,
            call_context=context, metadata=request.metadata,
        )
    except SessionLimitExceeded as exc:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, str(exc)) from exc
    state.repository.open_session(session)
    return SessionResponse(**session.as_dict())


@router.get("/sessions", response_model=List[SessionResponse], summary="List sessions")
def list_sessions(
    include_closed: bool = Query(False),
    limit: int = Query(50, ge=1, le=500),
    state: AppState = Depends(get_state),
):
    rows = state.sessions.list_sessions(include_closed=include_closed, limit=limit)
    out: List[SessionResponse] = []
    for row in rows:
        out.append(SessionResponse(
            session_id=row["session_id"],
            profile=row.get("profile", "default"),
            language=row.get("language", "auto"),
            identity=row.get("identity"),
            open=row.get("open", False),
            score=row.get("score", row.get("final_score", 0.0)) or 0.0,
            band=row.get("band", "LOW"),
            created_at=row.get("created_at", 0.0),
            last_activity=row.get("last_activity", row.get("closed_at") or 0.0) or 0.0,
            stats=row.get("stats", {}),
            metadata=row.get("metadata", {}),
        ))
    return out


@router.get("/sessions/{session_id}", summary="Live state of one session")
def get_session(session_id: str, state: AppState = Depends(get_state)):
    session = state.sessions.get(session_id)
    if session is not None:
        return session.as_dict()
    stored = state.repository.get_session(session_id)
    if stored is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"No session {session_id!r}.")
    return stored


@router.get("/sessions/{session_id}/report", summary="Full risk report with confidence trail")
def get_report(
    session_id: str,
    include_trail: bool = Query(True),
    state: AppState = Depends(get_state),
) -> Dict[str, Any]:
    """FR-7: the auditable record of how this call was scored."""
    session = state.sessions.get(session_id)
    if session is not None:
        return session.report(include_trail=include_trail)

    stored = state.repository.get_session(session_id)
    if stored is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"No session {session_id!r}.")

    report = stored.get("report") or {
        "session_id": session_id,
        "verdict": stored.get("verdict"),
        "final_score": stored.get("final_score"),
        "peak_score": stored.get("peak_score"),
        "duration_seconds": stored.get("duration_seconds"),
    }
    if include_trail:
        report["trail"] = state.repository.assessments_for(session_id)
    return report


@router.post("/sessions/{session_id}/close", summary="Close a session and return its report")
def close_session(session_id: str, state: AppState = Depends(get_state)):
    session = state.sessions.get(session_id)
    if session is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"No open session {session_id!r}.")
    report = state.sessions.close(session_id)
    state.repository.close_session(session, report)
    state.alerts.reset_session(session_id)
    return report


@router.delete("/sessions/{session_id}", summary="Erase everything held about a session")
def delete_session(session_id: str, state: AppState = Depends(get_state)):
    """Right to erasure. Hard deletes; there is no soft-delete flag by design."""
    live = state.sessions.delete(session_id)
    purged = state.repository.purge_session(session_id)
    if not live and not any(purged.values()):
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"No session {session_id!r}.")
    state.alerts.reset_session(session_id)
    return {"session_id": session_id, "deleted": True, "removed_rows": purged,
            "was_live": live}


# --------------------------------------------------------------------------------------
# Enrolment (layer 3)
# --------------------------------------------------------------------------------------

@router.post("/enrol", response_model=EnrolResponse, summary="Enrol a genuine voice sample")
def enrol(request: EnrolRequest, state: AppState = Depends(get_state)):
    """Register known-genuine audio for an identity, enabling the cross-session check.

    Three or more samples are strongly preferred: with fewer, the within-speaker spread
    cannot be estimated and the comparison falls back to a raw distance threshold, which
    is markedly weaker.
    """
    try:
        raw = base64.b64decode(request.audio_base64, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, f"invalid base64: {exc}") from exc

    samples, rate = decode_audio_bytes(raw, encoding=request.encoding,
                                       sample_rate=request.sample_rate,
                                       target_rate=SAMPLE_RATE)
    if duration_seconds(samples, rate) < MIN_ENROLMENT_SECONDS:
        raise HTTPException(
            status.HTTP_422_UNPROCESSABLE_ENTITY,
            f"Enrolment needs at least {MIN_ENROLMENT_SECONDS} s of clear speech.",
        )

    store = state.registry.enrolment
    count = store.enrol(request.identity, samples, rate)
    if count == 0:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY,
                            "Could not extract a voice profile from this audio.")

    centroid = store.centroid(request.identity)
    if centroid is not None:
        state.repository.save_enrolment(request.identity, store.embedder.embed(samples, rate))

    note = ("Enrolment active." if count >= 3 else
            f"{3 - count} more sample(s) recommended: with fewer than three, the "
            f"within-speaker spread cannot be estimated and the check is weaker.")
    return EnrolResponse(
        identity=request.identity,
        samples=count,
        embedding_dim=int(centroid.size) if centroid is not None else 0,
        within_speaker_spread=round(store.spread(request.identity), 5),
        note=note,
    )


@router.get("/enrol", summary="List enrolled identities")
def list_enrolments(state: AppState = Depends(get_state)):
    store = state.registry.enrolment
    return {
        "backend": store.embedder.backend,
        "identities": [
            {"identity": identity,
             "samples": len(store._profiles.get(identity, [])),
             "spread": round(store.spread(identity), 5)}
            for identity in store.identities()
        ],
    }


@router.delete("/enrol/{identity}", summary="Delete an enrolled identity")
def delete_enrolment(identity: str, state: AppState = Depends(get_state)):
    store = state.registry.enrolment
    if not store.has(identity):
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"No enrolment for {identity!r}.")
    store.clear(identity)
    removed = state.repository.delete_enrolment(identity)
    return {"identity": identity, "deleted": True, "rows_removed": removed}
