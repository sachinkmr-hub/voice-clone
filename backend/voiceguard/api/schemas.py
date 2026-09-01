"""Request/response models for the public API.

Kept in one place so the SDKs, the OpenAPI document and the dashboard all agree on the
wire format. Field names match the JSON exactly — no aliasing — because integrators read
the OpenAPI page far more often than they read this file.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


# --------------------------------------------------------------------------------------
# Shared
# --------------------------------------------------------------------------------------

class CallContextModel(BaseModel):
    """Call metadata used by layer 4 and by the threshold policy."""

    caller_number: Optional[str] = Field(None, description="Caller's number, hashed before storage")
    known_contact: Optional[bool] = Field(None, description="Number is a registered contact")
    caller_id_verified: Optional[bool] = Field(
        None, description="Network attestation for the caller ID (STIR/SHAKEN style)")
    origin: str = Field("", description="pstn | mobile | voip | webrtc | international")
    claimed_identity: str = ""
    claimed_role: str = Field("", description="e.g. 'CFO' — impersonation of authority")
    transaction_amount: float = Field(0.0, ge=0)
    currency: str = "INR"
    local_hour: Optional[int] = Field(None, ge=0, le=23)
    first_contact: Optional[bool] = None
    prior_fraud_reports: int = Field(0, ge=0)
    transcript: str = Field("", description="Optional live transcript; digits are redacted before storage")
    account_age_days: Optional[int] = Field(None, ge=0)


class FactorModel(BaseModel):
    code: str
    label: str
    contribution: float
    value: float = 0.0
    detail: str = ""
    layer: str = ""
    direction: str = "synthetic"


class LayerSummaryModel(BaseModel):
    layer: str
    label: str
    score: float
    confidence: float
    weight_share: float
    status: str
    reason: str = ""
    model_id: str = ""


class RiskResponse(BaseModel):
    """One risk verdict."""

    session_id: str
    score: float = Field(..., ge=0, le=100, description="Impersonation risk, 0–100")
    band: str = Field(..., description="LOW | ELEVATED | HIGH | CRITICAL")
    action: str
    probability: float
    confidence: float
    headline: str
    provisional: bool = False
    speech_detected: bool = True
    window_index: int = 0
    elapsed_seconds: float = 0.0
    latency_ms: float = 0.0
    profile: str = "default"
    threshold_shift: float = 0.0
    factors: List[FactorModel] = []
    caveats: List[str] = []
    layers: List[LayerSummaryModel] = []


# --------------------------------------------------------------------------------------
# Analyse
# --------------------------------------------------------------------------------------

class AnalyzeRequest(BaseModel):
    """Single-shot analysis of a base64 audio payload."""

    audio_base64: str = Field(..., description="Audio bytes, base64-encoded")
    encoding: str = Field("auto", description="auto | container | pcm16 | float32")
    sample_rate: Optional[int] = Field(None, description="Required for raw pcm16/float32")
    language: str = Field("auto", description="auto | hi-IN | en-IN | ta-IN | bn-IN | mr-IN | te-IN")
    profile: str = "default"
    identity: Optional[str] = Field(None, description="Claimed identity, for the enrolment check")
    call_context: Optional[CallContextModel] = None
    session_id: Optional[str] = None
    verbose: bool = False


class AnalyzeResponse(BaseModel):
    session_id: str
    verdict: str
    score: float
    peak_score: float
    band: str
    action: str
    headline: str
    duration_seconds: float
    windows_analyzed: int
    mean_latency_ms: float
    factors: List[FactorModel] = []
    caveats: List[str] = []
    layers: List[LayerSummaryModel] = []
    timeline: List[Dict[str, Any]] = []
    report: Optional[Dict[str, Any]] = None


# --------------------------------------------------------------------------------------
# Sessions
# --------------------------------------------------------------------------------------

class SessionCreateRequest(BaseModel):
    profile: str = "default"
    language: str = "auto"
    identity: Optional[str] = None
    call_context: Optional[CallContextModel] = None
    metadata: Dict[str, Any] = {}


class SessionResponse(BaseModel):
    session_id: str
    profile: str
    language: str
    identity: Optional[str] = None
    open: bool = True
    score: float = 0.0
    band: str = "LOW"
    created_at: float
    last_activity: float
    stats: Dict[str, Any] = {}
    metadata: Dict[str, Any] = {}


class EnrolRequest(BaseModel):
    identity: str = Field(..., min_length=1)
    audio_base64: str
    encoding: str = "auto"
    sample_rate: Optional[int] = None


class EnrolResponse(BaseModel):
    identity: str
    samples: int
    embedding_dim: int
    within_speaker_spread: float
    note: str = ""


# --------------------------------------------------------------------------------------
# Admin
# --------------------------------------------------------------------------------------

class ProfileModel(BaseModel):
    name: str
    elevated: float = Field(..., ge=0, le=100)
    high: float = Field(..., ge=0, le=100)
    critical: float = Field(..., ge=0, le=100)
    description: str = ""
    alert_channels: List[str] = ["websocket"]


class FusionWeightsModel(BaseModel):
    acoustic: Optional[float] = Field(None, ge=0, le=1)
    prosodic: Optional[float] = Field(None, ge=0, le=1)
    speaker: Optional[float] = Field(None, ge=0, le=1)
    context: Optional[float] = Field(None, ge=0, le=1)


class HealthResponse(BaseModel):
    status: str
    version: str
    environment: str
    model_loaded: bool
    degraded: List[str] = []
    detectors: Dict[str, Any] = {}
    retention: Dict[str, Any] = {}
    sessions: Dict[str, Any] = {}
    uptime_seconds: float = 0.0


# --------------------------------------------------------------------------------------
# Integrations
# --------------------------------------------------------------------------------------

class ApprovalRequest(BaseModel):
    """A mock core-banking approval, gated on the live voice risk."""

    session_id: str
    amount: float = Field(..., ge=0)
    currency: str = "INR"
    beneficiary: str = ""
    initiated_by: str = ""
    reference: str = ""
    profile: str = "wire_transfer"


class ApprovalResponse(BaseModel):
    decision: str = Field(..., description="allow | step_up | block")
    reference: str
    session_id: str
    risk_score: float
    band: str
    reasons: List[str] = []
    required_verification: List[str] = []
    message: str
    evaluated_at: float
