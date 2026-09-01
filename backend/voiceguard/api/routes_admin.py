"""Admin: health, risk profiles, fusion weights, alerts and audit inspection (FR-10)."""

from __future__ import annotations

from typing import Any, Dict, List

from fastapi import APIRouter, Depends, HTTPException, Query, status

from voiceguard.api.deps import AppState, get_state, require_api_key
from voiceguard.api.schemas import FusionWeightsModel, HealthResponse, ProfileModel
from voiceguard.config import RiskProfile

router = APIRouter(prefix="/v1", tags=["admin"])


@router.get("/health", response_model=HealthResponse, summary="Liveness and degradation state")
def health(state: AppState = Depends(get_state)):
    """Always 200 while the process is up.

    ``status`` is ``degraded`` (not an error) when a layer has fallen back — a degraded
    detector is still a working detector, and returning 503 would take a usable service
    out of a load balancer for no reason. Operators watch the ``degraded`` list.
    """
    return HealthResponse(**{k: v for k, v in state.health().items()
                             if k in HealthResponse.model_fields})


@router.get("/health/full", summary="Full diagnostic payload")
def health_full(state: AppState = Depends(get_state)) -> Dict[str, Any]:
    payload = state.health()
    payload["audit"] = state.repository.stats()
    payload["sweeper"] = state.sweeper.describe()
    payload["registry"] = state.registry.describe()
    return payload


# ------------------------------------------------------------------------- profiles

@router.get("/admin/profiles", response_model=List[ProfileModel],
            dependencies=[Depends(require_api_key)], summary="List risk profiles")
def list_profiles(state: AppState = Depends(get_state)):
    return [ProfileModel(**profile.as_dict()) for profile in state.profiles.values()]


@router.put("/admin/profiles/{name}", response_model=ProfileModel,
            dependencies=[Depends(require_api_key)],
            summary="Create or update a risk profile")
def upsert_profile(name: str, profile: ProfileModel, state: AppState = Depends(get_state)):
    """Thresholds are policy, so they are editable at runtime without a redeploy."""
    if not (profile.elevated <= profile.high <= profile.critical):
        raise HTTPException(
            status.HTTP_422_UNPROCESSABLE_ENTITY,
            "Thresholds must satisfy elevated <= high <= critical.",
        )
    updated = RiskProfile(
        name=name, elevated=profile.elevated, high=profile.high, critical=profile.critical,
        description=profile.description, alert_channels=list(profile.alert_channels),
    )
    state.profiles[name] = updated
    return ProfileModel(**updated.as_dict())


@router.delete("/admin/profiles/{name}", dependencies=[Depends(require_api_key)],
               summary="Delete a custom profile")
def delete_profile(name: str, state: AppState = Depends(get_state)):
    if name == "default":
        raise HTTPException(status.HTTP_400_BAD_REQUEST,
                            "The 'default' profile cannot be deleted.")
    if name not in state.profiles:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"No profile {name!r}.")
    del state.profiles[name]
    return {"name": name, "deleted": True}


# --------------------------------------------------------------------- fusion weights

@router.get("/admin/fusion", dependencies=[Depends(require_api_key)],
            summary="Current fusion weights and calibration")
def get_fusion(state: AppState = Depends(get_state)):
    from voiceguard.scoring.fusion import ScoreFusion

    fusion = ScoreFusion(state.registry.fusion_weights, state.registry.calibration)
    return fusion.as_dict()


@router.put("/admin/fusion", dependencies=[Depends(require_api_key)],
            summary="Override the per-layer fusion weights")
def set_fusion(weights: FusionWeightsModel, state: AppState = Depends(get_state)):
    """Deployments differ: a bank with enrolment data should weight layer 3 far higher
    than a consumer app that has none."""
    supplied = {k: v for k, v in weights.model_dump().items() if v is not None}
    if not supplied:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "Supply at least one weight.")
    if state.registry.bundle is not None:
        state.registry.bundle.fusion_weights.update(supplied)
    else:
        from voiceguard.config import DEFAULT_FUSION_WEIGHTS

        DEFAULT_FUSION_WEIGHTS.update(supplied)
    return {"updated": supplied, "note": "applies to sessions created from now on"}


@router.post("/admin/reload", dependencies=[Depends(require_api_key)],
             summary="Reload model artifacts from disk")
def reload_models(state: AppState = Depends(get_state)):
    """Pick up a freshly-trained model without restarting the service."""
    state.registry.reload()
    return {"reloaded": True, "model_loaded": state.registry.bundle is not None,
            "degraded": state.registry.degraded}


# --------------------------------------------------------------------------- alerts

@router.get("/admin/alerts", dependencies=[Depends(require_api_key)],
            summary="Recent alerts")
def recent_alerts(limit: int = Query(50, ge=1, le=200), state: AppState = Depends(get_state)):
    return {"in_memory": state.alerts.recent(limit),
            "persisted": state.repository.recent_alerts(limit)}


@router.get("/admin/audit", dependencies=[Depends(require_api_key)],
            summary="Stored session records")
def audit(limit: int = Query(50, ge=1, le=500), state: AppState = Depends(get_state)):
    return {"retention": state.retention_policy().as_dict(),
            "sessions": state.repository.list_sessions(limit)}


@router.post("/admin/retention/sweep", dependencies=[Depends(require_api_key)],
             summary="Run the retention sweeper now")
def sweep(state: AppState = Depends(get_state)):
    return {"removed": state.sweeper.sweep(),
            "policy": state.retention_policy().as_dict()}


@router.get("/admin/config", dependencies=[Depends(require_api_key)],
            summary="Effective non-secret configuration")
def config(state: AppState = Depends(get_state)):
    settings = state.settings
    return {
        "environment": settings.environment,
        "auth_required": settings.auth_required,
        "require_consent_header": settings.require_consent_header,
        "retention": state.retention_policy().as_dict(),
        "default_profile": settings.default_profile,
        "max_active_sessions": settings.max_active_sessions,
        "session_idle_timeout_seconds": settings.session_idle_timeout_seconds,
        "alert_cooldown_seconds": settings.alert_cooldown_seconds,
        "model_path": settings.model_path(),
        "sample_rate": 16000,
        "window_seconds": 1.0,
        "hop_seconds": 0.5,
    }
