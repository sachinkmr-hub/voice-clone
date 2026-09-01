"""VoiceGuard Python SDK.

A thin, dependency-light client. The only hard dependency is ``httpx``; ``websockets`` is
needed for the streaming client and imported lazily so that a caller who only ever uploads
recordings does not have to install it.

    from voiceguard_sdk import VoiceGuardClient

    client = VoiceGuardClient("http://localhost:8000", api_key="demo-key-sih26104")
    result = client.analyze_file("call.wav", profile="wire_transfer")
    if result.blocked:
        escalate(result.reasons)

Design choices worth knowing about:

* **Verdicts are objects, not dicts.** ``result.is_high_risk`` reads better at a call site
  than ``result["band"] in ("HIGH", "CRITICAL")`` and does not break when a band is added.
* **Nothing raises on "risky".** A high score is a normal, expected outcome, not an error.
  Only transport and validation problems raise.
* **The streaming client is a context manager** that guarantees the session is closed, so
  a crashed integration does not leak sessions on the server.
"""

from __future__ import annotations

import base64
import json
import os
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional

__all__ = [
    "VoiceGuardClient",
    "VoiceGuardError",
    "RiskResult",
    "ApprovalResult",
    "StreamSession",
]

DEFAULT_TIMEOUT = 30.0
HIGH_RISK_BANDS = ("HIGH", "CRITICAL")


class VoiceGuardError(RuntimeError):
    """Transport, authentication or validation failure. Never raised for a high score."""

    def __init__(self, message: str, status_code: Optional[int] = None,
                 payload: Optional[dict] = None) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.payload = payload or {}


@dataclass
class Factor:
    code: str
    label: str
    contribution: float
    detail: str = ""
    layer: str = ""
    value: float = 0.0

    def __str__(self) -> str:
        return f"{self.label} ({self.contribution:.0%})"


@dataclass
class RiskResult:
    """The verdict for one recording or one call."""

    session_id: str
    score: float
    band: str
    verdict: str = ""
    action: str = ""
    headline: str = ""
    peak_score: float = 0.0
    duration_seconds: float = 0.0
    windows_analyzed: int = 0
    mean_latency_ms: float = 0.0
    factors: List[Factor] = field(default_factory=list)
    caveats: List[str] = field(default_factory=list)
    layers: List[dict] = field(default_factory=list)
    timeline: List[dict] = field(default_factory=list)
    raw: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_high_risk(self) -> bool:
        return self.band in HIGH_RISK_BANDS

    @property
    def is_synthetic(self) -> bool:
        """Whether the *call-level* verdict says synthetic. Prefer this over a raw score."""
        return self.verdict in ("likely_synthetic", "suspicious")

    @property
    def inconclusive(self) -> bool:
        """True when there was not enough speech to decide.

        Check this before treating a low score as a pass — "we could not tell" and
        "it was genuine" are different answers, and only one of them is safe to act on.
        """
        return self.verdict in ("insufficient_audio", "inconclusive")

    @property
    def top_factor(self) -> Optional[Factor]:
        return self.factors[0] if self.factors else None

    def explain(self) -> str:
        lines = [self.headline or f"Risk {self.score:.0f}/100 ({self.band})"]
        lines += [f"  - {factor.label}: {factor.detail}" for factor in self.factors[:4]]
        if self.caveats:
            lines += [f"  ! {caveat}" for caveat in self.caveats]
        if self.action:
            lines.append(f"  -> {self.action}")
        return "\n".join(lines)

    @classmethod
    def from_payload(cls, payload: dict) -> "RiskResult":
        return cls(
            session_id=payload.get("session_id", ""),
            score=float(payload.get("score", 0.0)),
            band=payload.get("band", "LOW"),
            verdict=payload.get("verdict", ""),
            action=payload.get("action", ""),
            headline=payload.get("headline", ""),
            peak_score=float(payload.get("peak_score", payload.get("score", 0.0))),
            duration_seconds=float(payload.get("duration_seconds", 0.0)),
            windows_analyzed=int(payload.get("windows_analyzed", 0)),
            mean_latency_ms=float(payload.get("mean_latency_ms", payload.get("latency_ms", 0.0))),
            factors=[Factor(**{k: f.get(k) for k in
                              ("code", "label", "contribution", "detail", "layer", "value")
                              if k in f})
                     for f in payload.get("factors", [])],
            caveats=list(payload.get("caveats", [])),
            layers=list(payload.get("layers", [])),
            timeline=list(payload.get("timeline", [])),
            raw=payload,
        )


@dataclass
class ApprovalResult:
    """The core-banking gate's decision."""

    decision: str
    reference: str
    session_id: str
    risk_score: float
    band: str
    message: str = ""
    reasons: List[str] = field(default_factory=list)
    required_verification: List[str] = field(default_factory=list)
    raw: Dict[str, Any] = field(default_factory=dict)

    @property
    def allowed(self) -> bool:
        return self.decision == "allow"

    @property
    def blocked(self) -> bool:
        return self.decision == "block"

    @property
    def needs_step_up(self) -> bool:
        return self.decision == "step_up"


class VoiceGuardClient:
    """Synchronous client for the VoiceGuard API."""

    def __init__(
        self,
        base_url: str = "http://localhost:8000",
        api_key: Optional[str] = None,
        *,
        timeout: float = DEFAULT_TIMEOUT,
        consent: Optional[str] = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key or os.getenv("VOICEGUARD_API_KEY")
        self.timeout = timeout
        self.consent = consent
        self._client = None

    # ------------------------------------------------------------------ plumbing
    @property
    def headers(self) -> Dict[str, str]:
        headers: Dict[str, str] = {}
        if self.api_key:
            headers["X-API-Key"] = self.api_key
        if self.consent:
            headers["X-Consent"] = self.consent
        return headers

    def _http(self):
        if self._client is None:
            try:
                import httpx
            except ImportError as exc:  # pragma: no cover
                raise VoiceGuardError(
                    "The VoiceGuard SDK needs httpx: pip install httpx") from exc
            self._client = httpx.Client(base_url=self.base_url, timeout=self.timeout,
                                        headers=self.headers)
        return self._client

    def _request(self, method: str, path: str, **kwargs) -> Any:
        try:
            response = self._http().request(method, path, **kwargs)
        except Exception as exc:
            raise VoiceGuardError(f"could not reach {self.base_url}{path}: {exc}") from exc

        if response.status_code >= 400:
            try:
                payload = response.json()
                detail = payload.get("detail", response.text)
            except Exception:
                payload, detail = {}, response.text
            raise VoiceGuardError(f"{response.status_code}: {detail}",
                                  response.status_code, payload)
        return None if response.status_code == 204 else response.json()

    def close(self) -> None:
        if self._client is not None:
            self._client.close()
            self._client = None

    def __enter__(self) -> "VoiceGuardClient":
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()

    # ------------------------------------------------------------------ analysis
    def analyze_file(
        self,
        path: str,
        *,
        language: str = "auto",
        profile: str = "default",
        identity: Optional[str] = None,
        transaction_amount: float = 0.0,
        claimed_role: str = "",
        transcript: str = "",
        known_contact: Optional[bool] = None,
        caller_id_verified: Optional[bool] = None,
        verbose: bool = False,
    ) -> RiskResult:
        """Analyse a recording on disk."""
        data: Dict[str, Any] = {"language": language, "profile": profile,
                                "verbose": str(verbose).lower()}
        if identity:
            data["identity"] = identity
        if transaction_amount:
            data["transaction_amount"] = str(transaction_amount)
        if claimed_role:
            data["claimed_role"] = claimed_role
        if transcript:
            data["transcript"] = transcript
        if known_contact is not None:
            data["known_contact"] = str(known_contact).lower()
        if caller_id_verified is not None:
            data["caller_id_verified"] = str(caller_id_verified).lower()

        with open(path, "rb") as handle:
            files = {"file": (os.path.basename(path), handle.read(), "audio/wav")}
        return RiskResult.from_payload(
            self._request("POST", "/v1/analyze/file", data=data, files=files))

    def analyze_bytes(
        self,
        audio: bytes,
        *,
        encoding: str = "auto",
        sample_rate: Optional[int] = None,
        language: str = "auto",
        profile: str = "default",
        identity: Optional[str] = None,
        call_context: Optional[dict] = None,
        verbose: bool = False,
    ) -> RiskResult:
        """Analyse an in-memory buffer (container bytes, or raw pcm16/float32)."""
        payload = {
            "audio_base64": base64.b64encode(audio).decode(),
            "encoding": encoding,
            "language": language,
            "profile": profile,
            "verbose": verbose,
        }
        if sample_rate:
            payload["sample_rate"] = sample_rate
        if identity:
            payload["identity"] = identity
        if call_context:
            payload["call_context"] = call_context
        return RiskResult.from_payload(self._request("POST", "/v1/analyze", json=payload))

    # ----------------------------------------------------------------- sessions
    def create_session(self, *, profile: str = "default", language: str = "auto",
                       identity: Optional[str] = None,
                       call_context: Optional[dict] = None) -> str:
        payload = {"profile": profile, "language": language}
        if identity:
            payload["identity"] = identity
        if call_context:
            payload["call_context"] = call_context
        return self._request("POST", "/v1/sessions", json=payload)["session_id"]

    def get_report(self, session_id: str, *, include_trail: bool = True) -> dict:
        return self._request("GET", f"/v1/sessions/{session_id}/report",
                             params={"include_trail": include_trail})

    def close_session(self, session_id: str) -> dict:
        return self._request("POST", f"/v1/sessions/{session_id}/close")

    def delete_session(self, session_id: str) -> dict:
        """Right to erasure — removes every stored row for this call."""
        return self._request("DELETE", f"/v1/sessions/{session_id}")

    def list_sessions(self, *, include_closed: bool = False, limit: int = 50) -> List[dict]:
        return self._request("GET", "/v1/sessions",
                             params={"include_closed": include_closed, "limit": limit})

    # ---------------------------------------------------------------- enrolment
    def enrol(self, identity: str, audio: bytes, *, encoding: str = "auto",
              sample_rate: Optional[int] = None) -> dict:
        """Register a genuine sample, enabling the cross-session check for this identity."""
        payload = {"identity": identity,
                   "audio_base64": base64.b64encode(audio).decode(),
                   "encoding": encoding}
        if sample_rate:
            payload["sample_rate"] = sample_rate
        return self._request("POST", "/v1/enrol", json=payload)

    def enrol_file(self, identity: str, path: str) -> dict:
        with open(path, "rb") as handle:
            return self.enrol(identity, handle.read())

    def list_enrolments(self) -> dict:
        return self._request("GET", "/v1/enrol")

    # -------------------------------------------------------------- integration
    def request_approval(
        self,
        session_id: str,
        amount: float,
        *,
        profile: str = "wire_transfer",
        currency: str = "INR",
        beneficiary: str = "",
        reference: str = "",
        initiated_by: str = "",
    ) -> ApprovalResult:
        """Gate a transaction on the live voice risk."""
        payload = self._request("POST", "/v1/integrations/bank/approval", json={
            "session_id": session_id, "amount": amount, "currency": currency,
            "beneficiary": beneficiary, "reference": reference,
            "initiated_by": initiated_by, "profile": profile,
        })
        return ApprovalResult(
            decision=payload["decision"], reference=payload["reference"],
            session_id=payload["session_id"], risk_score=payload["risk_score"],
            band=payload["band"], message=payload.get("message", ""),
            reasons=list(payload.get("reasons", [])),
            required_verification=list(payload.get("required_verification", [])),
            raw=payload,
        )

    # ---------------------------------------------------------------- operations
    def health(self) -> dict:
        return self._request("GET", "/v1/health")

    def profiles(self) -> List[dict]:
        return self._request("GET", "/v1/admin/profiles")

    def set_profile(self, name: str, *, elevated: float, high: float,
                    critical: float, description: str = "",
                    alert_channels: Optional[List[str]] = None) -> dict:
        return self._request("PUT", f"/v1/admin/profiles/{name}", json={
            "name": name, "elevated": elevated, "high": high, "critical": critical,
            "description": description, "alert_channels": alert_channels or ["websocket"],
        })

    # ----------------------------------------------------------------- streaming
    def stream(self, **kwargs) -> "StreamSession":
        """Open a live streaming session. Use as a context manager."""
        return StreamSession(self, **kwargs)


class StreamSession:
    """Live streaming client.

    ::

        with client.stream(profile="wire_transfer") as stream:
            for chunk in microphone_chunks():
                for risk in stream.send(chunk):
                    if risk.is_high_risk:
                        warn_agent(risk)
    """

    def __init__(
        self,
        client: VoiceGuardClient,
        *,
        profile: str = "default",
        language: str = "auto",
        identity: Optional[str] = None,
        call_context: Optional[dict] = None,
        encoding: str = "pcm16",
        sample_rate: int = 16000,
        on_alert: Optional[Callable[[dict], None]] = None,
    ) -> None:
        self.client = client
        self.profile = profile
        self.language = language
        self.identity = identity
        self.call_context = call_context
        self.encoding = encoding
        self.sample_rate = sample_rate
        self.on_alert = on_alert
        self.session_id: Optional[str] = None
        self.final_report: Optional[dict] = None
        self._socket = None

    def _url(self) -> str:
        base = self.client.base_url.replace("https://", "wss://").replace("http://", "ws://")
        url = f"{base}/v1/stream"
        if self.client.api_key:
            url += f"?api_key={self.client.api_key}"
        return url

    def __enter__(self) -> "StreamSession":
        try:
            from websockets.sync.client import connect
        except ImportError as exc:  # pragma: no cover
            raise VoiceGuardError(
                "Streaming needs the websockets package: pip install websockets") from exc

        self._socket = connect(self._url(), open_timeout=self.client.timeout)
        start = {"action": "start", "profile": self.profile, "language": self.language,
                 "encoding": self.encoding, "sample_rate": self.sample_rate}
        if self.identity:
            start["identity"] = self.identity
        if self.call_context:
            start["call_context"] = self.call_context
        self._socket.send(json.dumps(start))

        message = json.loads(self._socket.recv())
        if message.get("type") != "started":
            raise VoiceGuardError(f"stream did not start: {message}")
        self.session_id = message["session_id"]
        return self

    def send(self, chunk: bytes, *, drain: bool = True) -> List[RiskResult]:
        """Push audio; return any risk verdicts that became available."""
        if self._socket is None:
            raise VoiceGuardError("stream is not open — use it as a context manager")
        self._socket.send(chunk)
        return self._drain() if drain else []

    def _drain(self, timeout: float = 0.01) -> List[RiskResult]:
        """Read whatever is already queued without blocking on the next window."""
        results: List[RiskResult] = []
        while True:
            try:
                raw = self._socket.recv(timeout=timeout)
            except TimeoutError:
                break
            except Exception:
                break
            message = json.loads(raw)
            kind = message.get("type")
            if kind == "risk":
                results.append(RiskResult.from_payload(message))
            elif kind == "alert" and self.on_alert:
                self.on_alert(message)
            elif kind == "final":
                self.final_report = message.get("report")
        return results

    def finish(self) -> Optional[dict]:
        """Stop the call and return its report."""
        if self._socket is None:
            return self.final_report
        try:
            self._socket.send(json.dumps({"action": "stop"}))
            for _ in range(200):
                try:
                    message = json.loads(self._socket.recv(timeout=5.0))
                except Exception:
                    break
                if message.get("type") == "final":
                    self.final_report = message.get("report")
                    break
        except Exception:
            pass
        return self.final_report

    def __exit__(self, *exc_info) -> None:
        try:
            self.finish()
        finally:
            if self._socket is not None:
                try:
                    self._socket.close()
                except Exception:
                    pass
                self._socket = None


def chunk_pcm16(samples: Iterable[float], chunk_ms: int = 500,
                sample_rate: int = 16000) -> Iterable[bytes]:
    """Helper: turn a float iterable into PCM16 chunks of the right size."""
    import array
    import math

    per_chunk = int(sample_rate * chunk_ms / 1000)
    buffer = array.array("h")
    for value in samples:
        clamped = max(-1.0, min(1.0, float(value)))
        buffer.append(int(clamped * 32767) if not math.isnan(clamped) else 0)
        if len(buffer) >= per_chunk:
            yield buffer.tobytes()
            buffer = array.array("h")
    if buffer:
        yield buffer.tobytes()
