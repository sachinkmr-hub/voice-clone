"""Central configuration.

Everything tunable lives here so that a deployment can be re-profiled without touching
detection code. Values may be overridden by environment variables prefixed ``VG_``
(e.g. ``VG_SAMPLE_RATE=8000``) or by an ``.env`` file.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from enum import Enum
from functools import lru_cache
from typing import Dict, List

# --------------------------------------------------------------------------------------
# Audio / streaming constants
# --------------------------------------------------------------------------------------

SAMPLE_RATE = int(os.getenv("VG_SAMPLE_RATE", "16000"))
WINDOW_SECONDS = float(os.getenv("VG_WINDOW_SECONDS", "1.0"))
HOP_SECONDS = float(os.getenv("VG_HOP_SECONDS", "0.5"))
FFT_SIZE = int(os.getenv("VG_FFT_SIZE", "512"))
HOP_LENGTH = int(os.getenv("VG_HOP_LENGTH", "160"))  # 10 ms at 16 kHz
N_MELS = int(os.getenv("VG_N_MELS", "40"))
N_MFCC = int(os.getenv("VG_N_MFCC", "20"))
MIN_ANALYSIS_SAMPLES = int(0.25 * SAMPLE_RATE)

# Number of windows to observe before publishing a non-provisional score.
WARMUP_WINDOWS = int(os.getenv("VG_WARMUP_WINDOWS", "3"))
# Exponentially-weighted moving average factor for score smoothing.
SCORE_EWMA_ALPHA = float(os.getenv("VG_SCORE_EWMA_ALPHA", "0.35"))
# Number of recent windows kept for the "sustained evidence" guard.
RECENT_WINDOW_MEMORY = int(os.getenv("VG_RECENT_WINDOW_MEMORY", "8"))


# --------------------------------------------------------------------------------------
# Detection layers
# --------------------------------------------------------------------------------------

class Layer(str, Enum):
    ACOUSTIC = "acoustic"
    PROSODIC = "prosodic"
    SPEAKER = "speaker"
    CONTEXT = "context"


#: Fusion weights per layer. Renormalised over the layers that actually reported.
DEFAULT_FUSION_WEIGHTS: Dict[str, float] = {
    Layer.ACOUSTIC.value: float(os.getenv("VG_W_ACOUSTIC", "0.45")),
    Layer.PROSODIC.value: float(os.getenv("VG_W_PROSODIC", "0.28")),
    Layer.SPEAKER.value: float(os.getenv("VG_W_SPEAKER", "0.19")),
    Layer.CONTEXT.value: float(os.getenv("VG_W_CONTEXT", "0.08")),
}

#: Platt-scaling parameters (a, b) applied to the fused logit. Refitted by ``ml/train.py``.
DEFAULT_CALIBRATION = (
    float(os.getenv("VG_CALIB_A", "1.0")),
    float(os.getenv("VG_CALIB_B", "0.0")),
)


# --------------------------------------------------------------------------------------
# Risk bands and use-case profiles
# --------------------------------------------------------------------------------------

class RiskBand(str, Enum):
    LOW = "LOW"
    ELEVATED = "ELEVATED"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


RECOMMENDED_ACTIONS: Dict[str, str] = {
    RiskBand.LOW.value: "Proceed normally.",
    RiskBand.ELEVATED.value: "Ask a knowledge-based verification question before proceeding.",
    RiskBand.HIGH.value: "Do not act on this call. Call back on the registered number first.",
    RiskBand.CRITICAL.value: "Block the requested action and escalate to a supervisor / fraud desk.",
}


@dataclass
class RiskProfile:
    """Threshold set for one use case. Scores are 0-100."""

    name: str
    elevated: float = 35.0
    high: float = 60.0
    critical: float = 80.0
    description: str = ""
    #: Channels notified when the band reaches HIGH or above.
    alert_channels: List[str] = field(default_factory=lambda: ["websocket"])

    def band(self, score: float) -> RiskBand:
        if score >= self.critical:
            return RiskBand.CRITICAL
        if score >= self.high:
            return RiskBand.HIGH
        if score >= self.elevated:
            return RiskBand.ELEVATED
        return RiskBand.LOW

    def as_dict(self) -> dict:
        return {
            "name": self.name,
            "elevated": self.elevated,
            "high": self.high,
            "critical": self.critical,
            "description": self.description,
            "alert_channels": list(self.alert_channels),
        }


DEFAULT_PROFILES: Dict[str, RiskProfile] = {
    "default": RiskProfile(
        name="default",
        description="Balanced profile for general inbound calls.",
    ),
    "wire_transfer": RiskProfile(
        name="wire_transfer",
        elevated=25.0,
        high=45.0,
        critical=65.0,
        description="High-value fund movement — favours sensitivity over alert fatigue.",
        alert_channels=["websocket", "webhook", "email"],
    ),
    "contact_center": RiskProfile(
        name="contact_center",
        elevated=40.0,
        high=65.0,
        critical=85.0,
        description="Agent desktop — tuned for a low false-positive rate.",
    ),
    "consumer": RiskProfile(
        name="consumer",
        elevated=45.0,
        high=70.0,
        critical=85.0,
        description="Individual / senior-citizen app — only warn when confident.",
        alert_channels=["websocket", "sms"],
    ),
    "privileged_access": RiskProfile(
        name="privileged_access",
        elevated=20.0,
        high=40.0,
        critical=60.0,
        description="Credential reset / privileged approval — maximum sensitivity.",
        alert_channels=["websocket", "webhook", "email", "sms"],
    ),
}


# --------------------------------------------------------------------------------------
# Language / accent priors
# --------------------------------------------------------------------------------------

@dataclass(frozen=True)
class LanguageProfile:
    """Prosody priors used to normalise features per language/accent.

    These are population priors for typical adult speech in each language, used to
    z-score prosodic measurements so that, e.g., the naturally wider pitch range of
    conversational Hindi is not read as an anomaly.
    """

    code: str
    label: str
    f0_mean_hz: float
    f0_std_hz: float
    syllable_rate_hz: float
    pause_ratio: float


LANGUAGE_PROFILES: Dict[str, LanguageProfile] = {
    "auto": LanguageProfile("auto", "Auto / language-agnostic", 155.0, 45.0, 4.6, 0.22),
    "hi-IN": LanguageProfile("hi-IN", "Hindi (India)", 158.0, 48.0, 5.0, 0.21),
    "en-IN": LanguageProfile("en-IN", "English (India)", 150.0, 44.0, 4.4, 0.24),
    "ta-IN": LanguageProfile("ta-IN", "Tamil (India)", 162.0, 46.0, 5.2, 0.20),
    "bn-IN": LanguageProfile("bn-IN", "Bengali (India)", 156.0, 47.0, 4.8, 0.22),
    "mr-IN": LanguageProfile("mr-IN", "Marathi (India)", 157.0, 46.0, 4.9, 0.21),
    "te-IN": LanguageProfile("te-IN", "Telugu (India)", 160.0, 45.0, 5.1, 0.21),
}


def language_profile(code: str | None) -> LanguageProfile:
    if not code:
        return LANGUAGE_PROFILES["auto"]
    return LANGUAGE_PROFILES.get(code, LANGUAGE_PROFILES["auto"])


# --------------------------------------------------------------------------------------
# Privacy / storage
# --------------------------------------------------------------------------------------

class RetentionMode(str, Enum):
    NONE = "none"
    FEATURES_ONLY = "features_only"
    RAW_AUDIO = "raw_audio"


# --------------------------------------------------------------------------------------
# Settings object
# --------------------------------------------------------------------------------------

@dataclass
class Settings:
    """Runtime settings, resolved from the ``VG_`` environment namespace."""

    app_name: str = "VoiceGuard"
    version: str = "1.0.0"
    environment: str = os.getenv("VG_ENVIRONMENT", "development")

    # storage
    database_url: str = os.getenv("VG_DATABASE_URL", "runtime/voiceguard.sqlite3")
    model_dir: str = os.getenv("VG_MODEL_DIR", "ml/artifacts")
    model_file: str = os.getenv("VG_MODEL_FILE", "bootstrap_model.joblib")

    # privacy
    retention_mode: str = os.getenv("VG_RETENTION_MODE", RetentionMode.FEATURES_ONLY.value)
    retention_ttl_seconds: int = int(os.getenv("VG_RETENTION_TTL_SECONDS", "86400"))
    raw_audio_ttl_seconds: int = int(os.getenv("VG_RAW_AUDIO_TTL_SECONDS", "900"))
    store_pii: bool = os.getenv("VG_STORE_PII", "false").lower() == "true"
    pii_salt: str = os.getenv("VG_PII_SALT", "voiceguard-dev-salt")
    require_consent_header: bool = (
        os.getenv("VG_REQUIRE_CONSENT_HEADER", "false").lower() == "true"
    )

    # security
    api_keys: List[str] = field(
        default_factory=lambda: [
            k.strip()
            for k in os.getenv("VG_API_KEYS", "demo-key-sih26104").split(",")
            if k.strip()
        ]
    )
    jwt_secret: str = os.getenv("VG_JWT_SECRET", "voiceguard-dev-jwt-secret")
    auth_required: bool = os.getenv("VG_AUTH_REQUIRED", "false").lower() == "true"
    cors_origins: List[str] = field(
        default_factory=lambda: [
            o.strip() for o in os.getenv("VG_CORS_ORIGINS", "*").split(",") if o.strip()
        ]
    )

    # alerting
    webhook_url: str = os.getenv("VG_WEBHOOK_URL", "")
    alert_cooldown_seconds: float = float(os.getenv("VG_ALERT_COOLDOWN_SECONDS", "10"))

    # session lifecycle
    session_idle_timeout_seconds: float = float(os.getenv("VG_SESSION_IDLE_TIMEOUT", "600"))
    max_active_sessions: int = int(os.getenv("VG_MAX_ACTIVE_SESSIONS", "500"))

    default_profile: str = os.getenv("VG_DEFAULT_PROFILE", "default")

    def model_path(self) -> str:
        return os.path.join(self.model_dir, self.model_file)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


def reset_settings_cache() -> None:
    """Used by tests after mutating the environment."""
    get_settings.cache_clear()
