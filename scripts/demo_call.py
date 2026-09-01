#!/usr/bin/env python3
"""End-to-end demo: stream a call at real-time speed and watch the score move.

    python scripts/demo_call.py                       # genuine, then cloned
    python scripts/demo_call.py --kind cloned --approval 4200000

Streams over the same WebSocket a telephony bridge would use, pacing the audio at
wall-clock speed so the printed timeline is the real latency behaviour, not a
fast-forwarded replay. Use it to rehearse the demo, or as the smallest working example of
integrating the streaming API.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from typing import List, Optional

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))

import numpy as np

from voiceguard.audio.io import float_to_pcm16_bytes, load_audio
from voiceguard.config import HOP_SECONDS, SAMPLE_RATE
from voiceguard.simulation import synthesize_bonafide, synthesize_cloned

BAR_WIDTH = 34
BAND_COLOR = {"LOW": "\033[32m", "ELEVATED": "\033[33m",
              "HIGH": "\033[38;5;208m", "CRITICAL": "\033[31m"}
RESET = "\033[0m"


def render(score: float, band: str, elapsed: float, latency: float,
           factor: str, provisional: bool) -> str:
    filled = int(BAR_WIDTH * min(100.0, max(0.0, score)) / 100.0)
    colour = BAND_COLOR.get(band, "")
    bar = f"{colour}{'█' * filled}{RESET}{'░' * (BAR_WIDTH - filled)}"
    flag = " (settling)" if provisional else ""
    return (f"  t={elapsed:5.1f}s  {bar} {score:5.1f}  "
            f"{colour}{band:<8}{RESET} {latency:4.0f}ms  {factor[:44]}{flag}")


async def stream_call(
    audio: np.ndarray,
    url: str,
    *,
    profile: str,
    language: str,
    call_context: Optional[dict],
    realtime: bool,
    api_key: Optional[str],
) -> dict:
    import websockets

    # --url is given as an HTTP base (that is what a user has to hand); the WebSocket
    # endpoint needs the ws/wss scheme.
    ws_base = url.replace("https://", "wss://", 1).replace("http://", "ws://", 1)
    endpoint = f"{ws_base}/v1/stream" + (f"?api_key={api_key}" if api_key else "")
    hop_samples = int(HOP_SECONDS * SAMPLE_RATE)
    report: dict = {}

    async with websockets.connect(endpoint, open_timeout=15) as socket:
        start = {"action": "start", "profile": profile, "language": language,
                 "encoding": "pcm16", "sample_rate": SAMPLE_RATE}
        if call_context:
            start["call_context"] = call_context
        await socket.send(json.dumps(start))

        first = json.loads(await socket.recv())
        if first.get("type") != "started":
            raise SystemExit(f"stream did not start: {first}")
        print(f"  session {first['session_id']} "
              f"(window {first['window_seconds']}s / hop {first['hop_seconds']}s)\n")

        alerts: List[dict] = []
        done = asyncio.Event()

        async def receive() -> None:
            try:
                while True:
                    message = json.loads(await socket.recv())
                    kind = message.get("type")
                    if kind == "risk":
                        factor = (message.get("factors") or [{}])[0].get("label", "—")
                        print(render(message["score"], message["band"],
                                     message["elapsed_seconds"], message["latency_ms"],
                                     factor, message.get("provisional", False)))
                    elif kind == "alert":
                        alerts.append(message)
                        print(f"  {BAND_COLOR.get(message['band'], '')}"
                              f"*** ALERT: {message['headline']}{RESET}")
                    elif kind == "final":
                        report.update(message.get("report") or {})
                        done.set()
                        return
                    elif kind == "error":
                        print(f"  error: {message['error']}")
            except Exception:
                done.set()

        task = asyncio.create_task(receive())
        started = time.time()
        for offset in range(0, len(audio), hop_samples):
            await socket.send(float_to_pcm16_bytes(audio[offset : offset + hop_samples]))
            if realtime:
                # Pace against the wall clock rather than sleeping a fixed amount, so a
                # slow machine does not silently turn this into a faster-than-real-time
                # replay and flatter the latency numbers.
                target = started + (offset + hop_samples) / SAMPLE_RATE
                await asyncio.sleep(max(0.0, target - time.time()))

        await socket.send(json.dumps({"action": "stop"}))
        try:
            await asyncio.wait_for(done.wait(), timeout=20)
        except asyncio.TimeoutError:
            pass
        task.cancel()

    report["alerts"] = alerts
    return report


def summarise(label: str, report: dict) -> None:
    stats = report.get("stats", {})
    print(f"\n  ── {label} ─────────────────────────────")
    print(f"     verdict        {report.get('verdict', '?')}")
    print(f"     final / peak   {report.get('final_score', 0):.1f} / "
          f"{report.get('peak_score', 0):.1f}")
    print(f"     windows        {stats.get('windows_analyzed', 0)} "
          f"({stats.get('mean_latency_ms', 0):.0f} ms mean)")
    print(f"     alerts         {len(report.get('alerts', []))}")
    for factor in report.get("top_factors", [])[:3]:
        print(f"       · {factor['label']} (seen in {factor['count']} windows)")


async def main_async(args: argparse.Namespace) -> int:
    scenarios = []
    if args.file:
        audio, _ = load_audio(args.file, SAMPLE_RATE)
        scenarios.append(("uploaded file", audio, None))
    elif args.kind == "both":
        scenarios.append(("GENUINE call", synthesize_bonafide(
            args.seconds, seed=args.seed, language=args.language, timbre=None), None))
        scenarios.append(("CLONED call", synthesize_cloned(
            args.seconds, seed=args.seed, language=args.language,
            method=args.method), None))
    else:
        maker = synthesize_bonafide if args.kind == "bonafide" else synthesize_cloned
        kwargs = {"seed": args.seed, "language": args.language}
        if args.kind == "cloned":
            kwargs["method"] = args.method
        scenarios.append((args.kind, maker(args.seconds, **kwargs), None))

    context = None
    if args.approval:
        context = {"known_contact": False, "caller_id_verified": False, "origin": "voip",
                   "claimed_role": "CFO", "transaction_amount": args.approval,
                   "local_hour": 22,
                   "transcript": "This is urgent, do not tell anyone, release it now."}

    last_session = None
    for label, audio, _ in scenarios:
        print(f"\n▶ {label} — {len(audio) / SAMPLE_RATE:.1f}s, profile '{args.profile}'")
        report = await stream_call(
            audio, args.url, profile=args.profile, language=args.language,
            call_context=context, realtime=not args.fast, api_key=args.api_key,
        )
        summarise(label, report)
        last_session = report.get("session_id", last_session)

    if args.approval and last_session:
        import httpx

        headers = {"X-API-Key": args.api_key} if args.api_key else {}
        response = httpx.post(
            f"{args.url}/v1/integrations/bank/approval",
            json={"session_id": last_session, "amount": args.approval,
                  "profile": "wire_transfer", "reference": "DEMO-TXN-1"},
            headers=headers, timeout=15,
        )
        decision = response.json()
        icon = {"allow": "✅", "step_up": "🔐", "block": "⛔"}.get(decision["decision"], "?")
        print(f"\n  ── Core-banking approval ────────────")
        print(f"     {icon} {decision['decision'].upper()} — {decision['message']}")
        for reason in decision["reasons"]:
            print(f"       · {reason}")

    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Stream a demo call through VoiceGuard")
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--kind", default="both",
                        choices=["both", "bonafide", "cloned"])
    parser.add_argument("--method", default="neural",
                        choices=["auto", "griffin_lim", "neural", "hybrid"])
    parser.add_argument("--file", default="", help="stream a real recording instead")
    parser.add_argument("--seconds", type=float, default=10.0)
    parser.add_argument("--language", default="hi-IN")
    parser.add_argument("--profile", default="wire_transfer")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--approval", type=float, default=0.0,
                        help="after the call, request approval for this amount")
    parser.add_argument("--fast", action="store_true",
                        help="do not pace at real time (latency figures become meaningless)")
    parser.add_argument("--api-key", default=os.getenv("VOICEGUARD_API_KEY"))
    args = parser.parse_args()

    try:
        return asyncio.run(main_async(args))
    except KeyboardInterrupt:
        return 130
    except ConnectionRefusedError:
        print(f"Could not reach {args.url}. Start the API first:  make run")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
