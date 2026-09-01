"""Live streaming endpoints.

``/v1/stream``
    The call-side socket. A client opens it, sends a JSON ``start`` frame, then pushes
    binary audio chunks; every completed analysis window comes back as a ``risk`` frame.
    This is the path that satisfies "detect during the call" — the whole point of the PS.

``/v1/dashboard``
    The watcher-side socket. Read-only fan-out of every session's events plus alerts, for
    the ops dashboard and the CISO view.

Protocol notes
--------------
Binary frames are audio; text frames are control JSON. That split means a browser can
send ``MediaRecorder`` output or raw PCM without any framing header, and it keeps the
control channel readable in a network trace during a demo.
"""

from __future__ import annotations

import json
import logging
import time
from typing import Optional

from fastapi import APIRouter, Depends, WebSocket, WebSocketDisconnect

from voiceguard.api.deps import AppState, authorize_websocket, get_state_ws
from voiceguard.audio.io import decode_audio_bytes
from voiceguard.config import SAMPLE_RATE
from voiceguard.models.context import CallContext
from voiceguard.pipeline.manager import SessionLimitExceeded
from voiceguard.privacy.anonymize import redact_transcript

logger = logging.getLogger("voiceguard.api.stream")
router = APIRouter(tags=["stream"])

#: Guard against a client that opens a socket and pushes unbounded audio.
MAX_STREAM_SECONDS = 3600.0


async def _send(websocket: WebSocket, payload: dict) -> None:
    await websocket.send_text(json.dumps(payload, default=str))


@router.websocket("/v1/stream")
async def stream(websocket: WebSocket, state: AppState = Depends(get_state_ws)):
    """Bidirectional live-call analysis."""
    if not await authorize_websocket(websocket):
        return
    await websocket.accept()

    session = None
    encoding = "pcm16"
    input_rate = SAMPLE_RATE

    try:
        while True:
            message = await websocket.receive()

            if message.get("type") == "websocket.disconnect":
                break

            # ---------------------------------------------------------- control
            if message.get("text") is not None:
                try:
                    frame = json.loads(message["text"])
                except json.JSONDecodeError:
                    await _send(websocket, {"type": "error", "error": "invalid JSON control frame"})
                    continue

                action = frame.get("action", "")

                if action == "start":
                    if session is not None:
                        state.sessions.close(session.id)
                    context = None
                    if frame.get("call_context"):
                        payload = dict(frame["call_context"])
                        payload["transcript"] = redact_transcript(payload.get("transcript", ""))
                        context = CallContext.from_dict(payload)
                    try:
                        session = state.sessions.create(
                            profile=frame.get("profile", "default"),
                            language=frame.get("language", "auto"),
                            identity=frame.get("identity"),
                            call_context=context,
                            metadata={"mode": "stream"},
                        )
                    except SessionLimitExceeded as exc:
                        await _send(websocket, {"type": "error", "error": str(exc)})
                        await websocket.close(code=1013, reason="server at capacity")
                        return

                    encoding = frame.get("encoding", "pcm16")
                    input_rate = int(frame.get("sample_rate", SAMPLE_RATE))
                    state.repository.open_session(session)
                    await _send(websocket, {
                        "type": "started",
                        "session_id": session.id,
                        "profile": session.profile,
                        "language": session.language,
                        "window_seconds": 1.0,
                        "hop_seconds": 0.5,
                        "retention": state.retention_policy().as_dict(),
                    })
                    await state.publish({"type": "session_started",
                                         "session": session.as_dict()})

                elif action == "context" and session is not None:
                    payload = dict(frame.get("call_context") or {})
                    payload["transcript"] = redact_transcript(payload.get("transcript", ""))
                    session.call_context = CallContext.from_dict(payload)
                    await _send(websocket, {"type": "context_updated"})

                elif action == "stop":
                    break

                elif action == "ping":
                    await _send(websocket, {"type": "pong", "t": time.time()})

                else:
                    await _send(websocket, {
                        "type": "error",
                        "error": f"unknown action {action!r}; send start | context | stop | ping",
                    })
                continue

            # ------------------------------------------------------------ audio
            chunk = message.get("bytes")
            if chunk is None:
                continue
            if session is None:
                await _send(websocket, {
                    "type": "error",
                    "error": "send a {\"action\":\"start\"} frame before streaming audio",
                })
                continue
            if session.stats.audio_seconds > MAX_STREAM_SECONDS:
                await _send(websocket, {"type": "error", "error": "stream length limit reached"})
                break

            samples, _ = decode_audio_bytes(
                chunk, encoding=encoding, sample_rate=input_rate, target_rate=SAMPLE_RATE)
            if samples.size == 0:
                continue

            for assessment in session.ingest(samples, raw_bytes=chunk):
                payload = assessment.as_dict()
                payload["session_id"] = session.id
                await _send(websocket, {"type": "risk", **payload})

                state.repository.record_assessment(session.id, assessment)
                await state.publish({"type": "risk", "session_id": session.id, **payload})

                record = await state.alerts.maybe_alert(
                    session.id, assessment, profile=state.profile(session.profile))
                if record:
                    state.repository.record_alert(session.id, record)
                    session.stats.alerts_raised += 1
                    await _send(websocket, {"type": "alert", **record["alert"]})

    except WebSocketDisconnect:
        logger.info("stream client disconnected")
    except Exception as exc:  # never leak a stack trace to the socket
        logger.exception("stream error")
        try:
            await _send(websocket, {"type": "error", "error": str(exc)})
        except Exception:
            pass
    finally:
        if session is not None:
            for assessment in session.flush():
                state.repository.record_assessment(session.id, assessment)
            report = state.sessions.close(session.id)
            if report:
                state.repository.close_session(session, report)
                try:
                    await _send(websocket, {"type": "final", "report": report})
                except Exception:
                    pass
                await state.publish({"type": "session_complete", "session": report})
        try:
            await websocket.close()
        except Exception:
            pass


@router.websocket("/v1/dashboard")
async def dashboard(websocket: WebSocket, state: AppState = Depends(get_state_ws)):
    """Read-only fan-out of every live session's risk events and alerts."""
    if not await authorize_websocket(websocket):
        return
    await state.hub.connect(websocket)
    try:
        await websocket.send_text(json.dumps({
            "type": "snapshot",
            "sessions": state.sessions.list_sessions(include_closed=True, limit=25),
            "alerts": state.alerts.recent(20),
            "health": state.health(),
        }, default=str))

        while True:
            # The dashboard is read-only; we only read to notice a disconnect and to
            # answer keepalive pings.
            message = await websocket.receive()
            if message.get("type") == "websocket.disconnect":
                break
            if message.get("text"):
                try:
                    if json.loads(message["text"]).get("action") == "ping":
                        await websocket.send_text(json.dumps({"type": "pong", "t": time.time()}))
                except Exception:
                    pass
    except WebSocketDisconnect:
        pass
    except Exception:
        logger.exception("dashboard socket error")
    finally:
        await state.hub.disconnect(websocket)
