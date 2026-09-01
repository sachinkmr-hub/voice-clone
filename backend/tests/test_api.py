"""API tests: REST, WebSocket streaming, admin, privacy and the bank integration."""

import base64
import json
import os

import numpy as np
import pytest
from fastapi.testclient import TestClient

from voiceguard.audio.io import float_to_pcm16_bytes, write_wav
from voiceguard.config import Settings
from voiceguard.simulation import synthesize_bonafide, synthesize_cloned

SR = 16000


@pytest.fixture()
def client(tmp_path):
    from voiceguard.api.app import create_app

    settings = Settings(
        database_url=":memory:",
        model_dir=str(tmp_path),
        model_file="absent.joblib",
        auth_required=False,
    )
    with TestClient(create_app(settings)) as test_client:
        yield test_client


@pytest.fixture()
def secured_client(tmp_path):
    from voiceguard.api.app import create_app

    settings = Settings(
        database_url=":memory:",
        model_dir=str(tmp_path),
        model_file="absent.joblib",
        auth_required=True,
        api_keys=["secret-key"],
    )
    with TestClient(create_app(settings)) as test_client:
        yield test_client


def wav_bytes(audio, path):
    write_wav(str(path), audio, SR)
    return path.read_bytes()


# ------------------------------------------------------------------------------ health

def test_health_reports_degradation_without_a_model(client):
    body = client.get("/v1/health").json()
    assert body["status"] == "degraded"           # still 200 — degraded is serving
    assert body["model_loaded"] is False
    assert any("no trained acoustic model" in note for note in body["degraded"])
    assert set(body["detectors"]) == {"acoustic", "prosodic", "speaker", "context"}


def test_health_endpoint_is_always_200(client):
    assert client.get("/v1/health").status_code == 200
    assert client.get("/v1/health/full").status_code == 200


def test_root_redirects_to_console(client):
    assert client.get("/", follow_redirects=False).status_code in (307, 302)


def test_openapi_documents_every_router(client):
    paths = client.get("/openapi.json").json()["paths"]
    for path in ("/v1/analyze", "/v1/analyze/file", "/v1/sessions",
                 "/v1/health", "/v1/integrations/bank/approval", "/v1/enrol"):
        assert path in paths


# ----------------------------------------------------------------------------- analyze

def test_analyze_file_scores_a_recording(client, tmp_path):
    raw = wav_bytes(synthesize_bonafide(4.0, seed=5), tmp_path / "a.wav")
    response = client.post(
        "/v1/analyze/file",
        files={"file": ("a.wav", raw, "audio/wav")},
        data={"language": "hi-IN", "profile": "default"},
    )
    assert response.status_code == 200
    body = response.json()
    assert 0.0 <= body["score"] <= 100.0
    assert body["band"] in ("LOW", "ELEVATED", "HIGH", "CRITICAL")
    assert body["windows_analyzed"] >= 3
    assert body["headline"]
    assert body["timeline"]
    assert body["layers"]


def test_analyze_scores_cloned_above_genuine(client, tmp_path):
    def score(audio, name):
        raw = wav_bytes(audio, tmp_path / name)
        return client.post("/v1/analyze/file",
                           files={"file": (name, raw, "audio/wav")}).json()["peak_score"]

    genuine = score(synthesize_bonafide(5.0, seed=9), "g.wav")
    cloned = score(synthesize_cloned(5.0, seed=9, method="neural"), "c.wav")
    assert cloned > genuine


def test_analyze_base64_endpoint(client):
    audio = synthesize_bonafide(3.0, seed=6)
    payload = base64.b64encode(float_to_pcm16_bytes(audio)).decode()
    response = client.post("/v1/analyze", json={
        "audio_base64": payload, "encoding": "pcm16", "sample_rate": SR,
        "language": "en-IN", "verbose": True,
    })
    assert response.status_code == 200
    body = response.json()
    assert body["report"] is not None
    assert body["report"]["trail"]


def test_analyze_rejects_bad_base64(client):
    response = client.post("/v1/analyze", json={"audio_base64": "!!!not base64!!!"})
    assert response.status_code == 400


def test_analyze_rejects_too_short_audio(client):
    payload = base64.b64encode(float_to_pcm16_bytes(np.zeros(1000, np.float32))).decode()
    response = client.post("/v1/analyze", json={
        "audio_base64": payload, "encoding": "pcm16", "sample_rate": SR})
    assert response.status_code == 422


def test_analyze_rejects_undecodable_payload(client):
    payload = base64.b64encode(b"").decode()
    assert client.post("/v1/analyze", json={"audio_base64": payload}).status_code == 400


def test_silent_recording_reports_insufficient_audio(client, tmp_path):
    raw = wav_bytes(np.zeros(4 * SR, dtype=np.float32), tmp_path / "s.wav")
    body = client.post("/v1/analyze/file",
                       files={"file": ("s.wav", raw, "audio/wav")}).json()
    assert body["verdict"] == "insufficient_audio"
    assert body["score"] == 0.0
    assert body["caveats"]


def test_analyze_accepts_call_context_form_fields(client, tmp_path):
    raw = wav_bytes(synthesize_bonafide(3.0, seed=8), tmp_path / "c.wav")
    response = client.post(
        "/v1/analyze/file",
        files={"file": ("c.wav", raw, "audio/wav")},
        data={"transaction_amount": "4200000", "known_contact": "false",
              "claimed_role": "CFO", "transcript": "urgent, transfer to 4012 3344 now"},
    )
    assert response.status_code == 200


# ---------------------------------------------------------------------------- sessions

def test_session_lifecycle(client):
    created = client.post("/v1/sessions", json={"profile": "contact_center",
                                                "language": "ta-IN"}).json()
    session_id = created["session_id"]
    assert created["open"] is True

    assert client.get(f"/v1/sessions/{session_id}").status_code == 200
    assert any(row["session_id"] == session_id for row in client.get("/v1/sessions").json())

    report = client.post(f"/v1/sessions/{session_id}/close").json()
    assert report["session_id"] == session_id
    assert report["verdict"] == "insufficient_audio"


def test_unknown_session_is_404(client):
    assert client.get("/v1/sessions/nope").status_code == 404
    assert client.get("/v1/sessions/nope/report").status_code == 404
    assert client.delete("/v1/sessions/nope").status_code == 404


def test_report_contains_the_confidence_trail(client, tmp_path):
    raw = wav_bytes(synthesize_cloned(5.0, seed=3, method="neural"), tmp_path / "r.wav")
    session_id = client.post("/v1/analyze/file",
                             files={"file": ("r.wav", raw, "audio/wav")}).json()["session_id"]

    report = client.get(f"/v1/sessions/{session_id}/report").json()
    assert report["session_id"] == session_id
    assert report["trail"]
    assert report["top_factors"]
    assert "chunk_hashes" in report          # dispute resolution without keeping audio


def test_delete_session_erases_stored_rows(client, tmp_path):
    raw = wav_bytes(synthesize_bonafide(3.0, seed=4), tmp_path / "d.wav")
    session_id = client.post("/v1/analyze/file",
                             files={"file": ("d.wav", raw, "audio/wav")}).json()["session_id"]

    body = client.delete(f"/v1/sessions/{session_id}").json()
    assert body["deleted"] is True
    assert body["removed_rows"]["assessments"] > 0
    assert client.get(f"/v1/sessions/{session_id}").status_code == 404


# --------------------------------------------------------------------------- enrolment

def test_enrol_and_list_and_delete(client):
    from voiceguard.simulation.voice import SpeakerTimbre

    timbre = SpeakerTimbre.random(np.random.default_rng(3), "cfo")
    for seed in range(3):
        audio = synthesize_bonafide(2.5, timbre=timbre, seed=seed)
        response = client.post("/v1/enrol", json={
            "identity": "cfo",
            "audio_base64": base64.b64encode(float_to_pcm16_bytes(audio)).decode(),
            "encoding": "pcm16", "sample_rate": SR,
        })
        assert response.status_code == 200
        body = response.json()

    assert body["samples"] == 3
    assert body["embedding_dim"] == 64
    assert "Enrolment active" in body["note"]

    listing = client.get("/v1/enrol").json()
    assert listing["identities"][0]["identity"] == "cfo"

    assert client.delete("/v1/enrol/cfo").json()["deleted"] is True
    assert client.delete("/v1/enrol/cfo").status_code == 404


def test_enrol_rejects_too_short_audio(client):
    payload = base64.b64encode(float_to_pcm16_bytes(np.zeros(8000, np.float32))).decode()
    response = client.post("/v1/enrol", json={
        "identity": "x", "audio_base64": payload, "encoding": "pcm16", "sample_rate": SR})
    assert response.status_code == 422


# ------------------------------------------------------------------------------- admin

def test_profiles_can_be_listed_and_updated(client):
    profiles = client.get("/v1/admin/profiles").json()
    assert {p["name"] for p in profiles} >= {"default", "wire_transfer", "consumer"}

    updated = client.put("/v1/admin/profiles/wire_transfer", json={
        "name": "wire_transfer", "elevated": 20, "high": 35, "critical": 55,
        "description": "tightened for the demo", "alert_channels": ["websocket"],
    })
    assert updated.status_code == 200
    assert updated.json()["critical"] == 55


def test_profile_thresholds_must_be_ordered(client):
    response = client.put("/v1/admin/profiles/bad", json={
        "name": "bad", "elevated": 90, "high": 50, "critical": 10})
    assert response.status_code == 422


def test_default_profile_cannot_be_deleted(client):
    assert client.delete("/v1/admin/profiles/default").status_code == 400


def test_fusion_weights_can_be_overridden(client):
    assert client.put("/v1/admin/fusion", json={"acoustic": 0.6}).status_code == 200
    assert client.put("/v1/admin/fusion", json={}).status_code == 400
    assert "weights" in client.get("/v1/admin/fusion").json()


def test_model_reload_is_safe_without_an_artifact(client):
    body = client.post("/v1/admin/reload").json()
    assert body["reloaded"] is True
    assert body["model_loaded"] is False


def test_retention_sweep_and_audit(client, tmp_path):
    raw = wav_bytes(synthesize_bonafide(3.0, seed=2), tmp_path / "au.wav")
    client.post("/v1/analyze/file", files={"file": ("au.wav", raw, "audio/wav")})

    audit = client.get("/v1/admin/audit").json()
    assert audit["retention"]["mode"] == "features_only"
    assert audit["retention"]["keeps_raw_audio"] is False
    assert audit["sessions"]

    assert "removed" in client.post("/v1/admin/retention/sweep").json()


def test_admin_config_exposes_no_secrets(client):
    config = client.get("/v1/admin/config").json()
    body = json.dumps(config).lower()
    assert "jwt_secret" not in body and "secret" not in body
    assert config["window_seconds"] == 1.0


# -------------------------------------------------------------------------------- auth

def test_auth_required_rejects_missing_key(secured_client):
    assert secured_client.get("/v1/sessions").status_code == 401
    assert secured_client.post("/v1/analyze", json={"audio_base64": ""}).status_code == 401


def test_auth_accepts_api_key_and_bearer(secured_client):
    assert secured_client.get("/v1/sessions",
                              headers={"X-API-Key": "secret-key"}).status_code == 200
    assert secured_client.get("/v1/sessions",
                              headers={"Authorization": "Bearer secret-key"}).status_code == 200
    assert secured_client.get("/v1/sessions",
                              headers={"X-API-Key": "wrong"}).status_code == 401


def test_health_stays_public_when_auth_is_on(secured_client):
    assert secured_client.get("/v1/health").status_code == 200


# ------------------------------------------------------------------------ integrations

def test_bank_approval_blocks_a_high_risk_call(client, tmp_path):
    raw = wav_bytes(synthesize_cloned(6.0, seed=7, method="neural"), tmp_path / "k.wav")
    session_id = client.post(
        "/v1/analyze/file",
        files={"file": ("k.wav", raw, "audio/wav")},
        data={"profile": "wire_transfer"},
    ).json()["session_id"]

    body = client.post("/v1/integrations/bank/approval", json={
        "session_id": session_id, "amount": 4_200_000, "profile": "wire_transfer",
        "beneficiary": "ACME", "reference": "TXN-1",
    }).json()

    assert body["decision"] in ("block", "step_up")
    assert body["reference"] == "TXN-1"
    assert body["reasons"]


def test_bank_approval_requires_a_known_session(client):
    response = client.post("/v1/integrations/bank/approval",
                           json={"session_id": "missing", "amount": 1000})
    assert response.status_code == 404


def test_bank_approval_never_silently_allows_on_thin_evidence(client):
    """A low score from two seconds of audio must not clear a transaction."""
    session_id = client.post("/v1/sessions", json={"profile": "wire_transfer"}).json()["session_id"]
    body = client.post("/v1/integrations/bank/approval", json={
        "session_id": session_id, "amount": 50_000, "profile": "wire_transfer"}).json()
    assert body["decision"] == "step_up"
    assert any("not enough evidence" in reason for reason in body["reasons"])


def test_bank_approval_steps_up_large_amounts_even_when_clean(client, tmp_path):
    raw = wav_bytes(synthesize_bonafide(8.0, seed=11), tmp_path / "ok.wav")
    session_id = client.post("/v1/analyze/file",
                             files={"file": ("ok.wav", raw, "audio/wav")}).json()["session_id"]
    body = client.post("/v1/integrations/bank/approval", json={
        "session_id": session_id, "amount": 9_000_000, "profile": "contact_center"}).json()
    assert body["decision"] in ("step_up", "block")


def test_bank_policy_is_documented(client):
    policy = client.get("/v1/integrations/bank/policy").json()
    assert policy["always_step_up_above"] > 0
    assert set(policy["decisions"]) == {"allow", "step_up", "block"}


# --------------------------------------------------------------------------- streaming

def test_websocket_stream_returns_risk_frames(client):
    audio = synthesize_cloned(5.0, seed=13, method="neural")
    with client.websocket_connect("/v1/stream") as socket:
        socket.send_text(json.dumps({"action": "start", "profile": "wire_transfer",
                                     "language": "hi-IN", "encoding": "pcm16",
                                     "sample_rate": SR}))
        started = socket.receive_json()
        assert started["type"] == "started"
        assert started["hop_seconds"] == 0.5

        # Push the whole call in 0.5 s chunks, then stop and drain. Draining after `stop`
        # rather than after every chunk is what keeps this test terminating: a chunk does
        # not necessarily complete a window, so a per-chunk read can block forever.
        step = SR // 2
        for offset in range(0, len(audio), step):
            socket.send_bytes(float_to_pcm16_bytes(audio[offset : offset + step]))
        socket.send_text(json.dumps({"action": "stop"}))

        risks, final = [], None
        while final is None:
            message = socket.receive_json()
            if message["type"] == "risk":
                risks.append(message)
            elif message["type"] == "final":
                final = message

        assert len(risks) >= 4
        assert all(0 <= r["score"] <= 100 for r in risks)
        assert risks[0]["provisional"] is True
        assert not risks[-1]["provisional"]
        assert final["report"]["stats"]["windows_analyzed"] >= 4


def test_websocket_rejects_audio_before_start(client):
    with client.websocket_connect("/v1/stream") as socket:
        socket.send_bytes(b"\x00\x01" * 100)
        message = socket.receive_json()
        assert message["type"] == "error"
        assert "start" in message["error"]


def test_websocket_handles_a_bad_control_frame(client):
    with client.websocket_connect("/v1/stream") as socket:
        socket.send_text("{not json")
        assert socket.receive_json()["type"] == "error"
        socket.send_text(json.dumps({"action": "wat"}))
        assert "unknown action" in socket.receive_json()["error"]


def test_websocket_ping_pong(client):
    with client.websocket_connect("/v1/stream") as socket:
        socket.send_text(json.dumps({"action": "ping"}))
        assert socket.receive_json()["type"] == "pong"


def test_dashboard_socket_sends_a_snapshot(client):
    with client.websocket_connect("/v1/dashboard") as socket:
        snapshot = socket.receive_json()
        assert snapshot["type"] == "snapshot"
        assert "sessions" in snapshot and "health" in snapshot


# ---------------------------------------------------------------------------- console

def test_console_and_static_assets_are_served(client):
    page = client.get("/console")
    assert page.status_code == 200
    body = page.text
    assert "VoiceGuard" in body
    assert "/static/console.css" in body and "/static/console.js" in body
    assert client.get("/static/console.css").status_code == 200
    assert client.get("/static/console.js").status_code == 200


def test_demo_sample_returns_playable_wav(client):
    response = client.get("/v1/demo/sample?kind=bonafide&seconds=3&speaker=4")
    assert response.status_code == 200
    assert response.headers["content-type"] == "audio/wav"
    assert response.headers["X-VoiceGuard-Simulated"] == "true"
    assert response.content[:4] == b"RIFF"

    import io
    import wave

    with wave.open(io.BytesIO(response.content), "rb") as handle:
        assert handle.getframerate() == SR
        assert handle.getnchannels() == 1
        assert handle.getnframes() / SR == pytest.approx(3.0, abs=0.2)


def test_demo_sample_same_speaker_is_reproducible(client):
    """The pair demo depends on 'same speaker value' meaning the same voice."""
    first = client.get("/v1/demo/sample?kind=bonafide&seconds=2&speaker=9&seed=5").content
    second = client.get("/v1/demo/sample?kind=bonafide&seconds=2&speaker=9&seed=5").content
    assert first == second


def test_demo_sample_rejects_an_unknown_method(client):
    assert client.get("/v1/demo/sample?kind=cloned&method=nope").status_code == 400
    assert client.get("/v1/demo/sample?kind=wat").status_code == 422


def test_demo_scenarios_are_labelled_as_simulated(client):
    body = client.get("/v1/demo/scenarios").json()
    assert "simulated" in body["note"].lower()
    assert len(body["scenarios"]) >= 4
    for scenario in body["scenarios"]:
        assert {"id", "title", "kind", "story", "expect"} <= set(scenario)


def test_demo_pair_separates_genuine_from_cloned(client):
    """The headline demo: same synthetic speaker, real voice vs clone of it."""
    def score(query, name):
        wav = client.get(f"/v1/demo/sample?{query}").content
        return client.post("/v1/analyze/file",
                           files={"file": (name, wav, "audio/wav")},
                           data={"language": "hi-IN"}).json()["peak_score"]

    genuine = score("kind=bonafide&seconds=7&speaker=7&language=hi-IN", "g.wav")
    cloned = score("kind=cloned&method=neural&seconds=7&speaker=7&language=hi-IN", "c.wav")
    assert cloned > genuine


def test_pitch_deck_is_served(client):
    response = client.get("/pitch")
    assert response.status_code == 200
    body = response.text
    assert body.count('<section class="slide') == 10
    # The deck must keep the honesty slide — the claim and its caveat travel together.
    assert "too good" in body
    assert "adversarial audio" in body
