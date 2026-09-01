"""FastAPI application factory.

Run with::

    uvicorn voiceguard.api.app:app --app-dir backend --host 0.0.0.0 --port 8000

Then open http://127.0.0.1:8000/console for the zero-build web console, or
http://127.0.0.1:8000/docs for the interactive API reference.
"""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles

from voiceguard.api.deps import AppState
from voiceguard.api import (
    routes_admin,
    routes_analyze,
    routes_demo,
    routes_integrations,
    routes_sessions,
    routes_stream,
)
from voiceguard.config import Settings, get_settings

logging.basicConfig(
    level=os.getenv("VG_LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
)
logger = logging.getLogger("voiceguard")

STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")

DESCRIPTION = """
Real-time detection of AI-generated and cloned voices during live calls.

**Three ways in, one engine behind them:**

* `POST /v1/analyze/file` — upload a call recording (the website's front door)
* `WS /v1/stream` — live streaming from a browser mic, WebRTC bridge or SIP tap
* `POST /v1/integrations/bank/approval` — gate a transaction on the live voice risk

Every verdict carries a 0–100 risk score, the ranked evidence behind it, and a
recommended action. Layers that cannot run *abstain* rather than voting "genuine" —
check the `layers[].status` field.
"""

TAGS = [
    {"name": "analyze", "description": "Single-shot analysis of a recording."},
    {"name": "stream", "description": "Live WebSocket analysis and the dashboard feed."},
    {"name": "sessions", "description": "Session state, reports, enrolment and erasure."},
    {"name": "admin", "description": "Health, risk profiles, fusion weights, audit."},
    {"name": "integrations", "description": "Reference core-banking approval gate."},
    {"name": "demo", "description": "Generated demo audio and scenarios (clearly labelled as simulated)."},
]


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()

    @asynccontextmanager
    async def lifespan(application: FastAPI):
        state = AppState(settings)
        application.state.vg = state
        state.sweeper.start()

        logger.info("VoiceGuard %s starting (%s)", settings.version, settings.environment)
        logger.info("%s", state.retention_policy().banner())
        if state.registry.is_degraded:
            for note in state.registry.degraded:
                logger.warning("degraded: %s", note)
        if not settings.auth_required:
            logger.warning(
                "API authentication is DISABLED (VG_AUTH_REQUIRED=false) — fine for a "
                "demo, never for a deployment handling real calls."
            )
        try:
            yield
        finally:
            await state.sweeper.stop()
            state.shutdown()
            logger.info("VoiceGuard stopped")

    app = FastAPI(
        title="VoiceGuard — AI Voice-Clone Detection",
        description=DESCRIPTION,
        version=settings.version,
        openapi_tags=TAGS,
        lifespan=lifespan,
        contact={"name": "SIH26104 — VoiceGuard"},
        license_info={"name": "MIT"},
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(routes_analyze.router)
    app.include_router(routes_stream.router)
    app.include_router(routes_sessions.router)
    app.include_router(routes_admin.router)
    app.include_router(routes_integrations.router)
    app.include_router(routes_demo.router)

    if os.path.isdir(STATIC_DIR):
        app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

    @app.get("/", include_in_schema=False)
    def index():
        return RedirectResponse("/console")

    @app.get("/console", include_in_schema=False)
    def console():
        path = os.path.join(STATIC_DIR, "console.html")
        if not os.path.exists(path):
            return JSONResponse(
                {"error": "console not built", "hint": "see /docs for the API"},
                status_code=404,
            )
        return FileResponse(path)

    @app.exception_handler(Exception)
    async def unhandled(request: Request, exc: Exception):
        # Never leak a stack trace to a caller; the log has the detail.
        logger.exception("unhandled error on %s %s", request.method, request.url.path)
        return JSONResponse(
            status_code=500,
            content={"error": "internal_error",
                     "detail": "The request could not be completed. See server logs."},
        )

    return app


app = create_app()
