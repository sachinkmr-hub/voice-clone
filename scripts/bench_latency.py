#!/usr/bin/env python3
"""Latency benchmark.

    python scripts/bench_latency.py --windows 60

Measures what the latency NFR actually depends on: the cost of scoring one analysis
window, broken down by stage. The wall-clock latency a caller experiences is
``hop (0.5 s) + this``, so the number that matters is whether the per-window total stays
comfortably under the hop — if it does not, windows queue and the lag grows without bound.

Reports p50/p95/p99 rather than a mean. A detector that is fast on average and
occasionally takes two seconds is a detector that drops alerts during exactly the bursts
you care about.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from typing import Dict, List

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))

import numpy as np

from voiceguard.config import HOP_SECONDS, SAMPLE_RATE, Settings
from voiceguard.features.artifacts import artifact_feature_dict
from voiceguard.features.extractor import FeatureExtractor
from voiceguard.features.prosody import prosody_feature_dict
from voiceguard.features.spectral import spectral_feature_dict
from voiceguard.models.registry import ModelRegistry
from voiceguard.pipeline.session import CallSession
from voiceguard.simulation import synthesize_bonafide, synthesize_cloned


def percentiles(values: List[float]) -> Dict[str, float]:
    if not values:
        return {"p50": 0.0, "p95": 0.0, "p99": 0.0, "mean": 0.0, "max": 0.0}
    ordered = sorted(values)
    return {
        "p50": round(statistics.median(ordered), 2),
        "p95": round(ordered[min(len(ordered) - 1, int(0.95 * len(ordered)))], 2),
        "p99": round(ordered[min(len(ordered) - 1, int(0.99 * len(ordered)))], 2),
        "mean": round(statistics.fmean(ordered), 2),
        "max": round(ordered[-1], 2),
    }


def bench_stages(windows: List[np.ndarray]) -> Dict[str, Dict[str, float]]:
    """Per-stage cost, so an optimisation effort knows where to aim."""
    stages: Dict[str, List[float]] = {"spectral": [], "prosody": [], "artifacts": [],
                                      "embedding": []}
    from voiceguard.features.embedding import classical_embedding

    for window in windows:
        for name, function in (
            ("spectral", lambda w: spectral_feature_dict(w, SAMPLE_RATE)),
            ("prosody", lambda w: prosody_feature_dict(w, SAMPLE_RATE)),
            ("artifacts", lambda w: artifact_feature_dict(w, SAMPLE_RATE)),
            ("embedding", lambda w: classical_embedding(w, SAMPLE_RATE)),
        ):
            started = time.perf_counter()
            function(window)
            stages[name].append((time.perf_counter() - started) * 1000.0)
    return {name: percentiles(values) for name, values in stages.items()}


def main() -> int:
    parser = argparse.ArgumentParser(description="VoiceGuard latency benchmark")
    parser.add_argument("--windows", type=int, default=40, help="analysis windows to time")
    parser.add_argument("--model-dir", default="ml/artifacts")
    parser.add_argument("--json", default="", help="write results to this path")
    args = parser.parse_args()

    print("Generating test audio ...")
    audio = np.concatenate([
        synthesize_bonafide(20.0, seed=1),
        synthesize_cloned(20.0, seed=1, method="neural"),
    ])

    step = int(HOP_SECONDS * SAMPLE_RATE)
    size = SAMPLE_RATE
    windows = [audio[i : i + size] for i in range(0, len(audio) - size, step)][: args.windows]
    print(f"  {len(windows)} windows of {size / SAMPLE_RATE:.1f}s\n")

    extractor = FeatureExtractor()
    extractor.extract(windows[0])          # warm the filterbank caches

    print("Per-stage feature extraction:")
    stages = bench_stages(windows)
    for name, stats in stages.items():
        print(f"  {name:12s} p50 {stats['p50']:6.1f} ms   p95 {stats['p95']:6.1f} ms")

    print("\nFull pipeline (extract → 4 layers → fuse → risk):")
    settings = Settings(model_dir=args.model_dir, database_url=":memory:")
    registry = ModelRegistry(settings)
    if registry.is_degraded:
        for note in registry.degraded:
            print(f"  note: {note}")

    session = CallSession(registry=registry, settings=settings)
    end_to_end: List[float] = []
    for window in windows:
        started = time.perf_counter()
        session.ingest(window[: step])     # one hop of audio per iteration
        end_to_end.append((time.perf_counter() - started) * 1000.0)

    per_window = [a.latency_ms for a in session.assessments]
    full = percentiles(per_window if per_window else end_to_end)
    budget_ms = HOP_SECONDS * 1000.0

    print(f"  p50 {full['p50']:6.1f} ms")
    print(f"  p95 {full['p95']:6.1f} ms")
    print(f"  p99 {full['p99']:6.1f} ms")
    print(f"  max {full['max']:6.1f} ms")
    print(f"\n  Hop budget: {budget_ms:.0f} ms per window")

    headroom = budget_ms / max(full["p95"], 1e-6)
    verdict = "PASS" if full["p95"] < budget_ms else "FAIL"
    print(f"  p95 uses {full['p95'] / budget_ms:.0%} of the budget "
          f"({headroom:.1f}x headroom) → {verdict}")
    print(f"\n  End-to-end user-visible latency ≈ hop + p95 = "
          f"{(budget_ms + full['p95']) / 1000:.2f} s (NFR: < 5 s)")

    # Rough capacity: how many concurrent calls one core can sustain, allowing headroom.
    concurrent = int(budget_ms / max(full["p95"], 1e-6))
    print(f"  Sustainable concurrent calls per CPU core: ~{concurrent}")

    if args.json:
        payload = {
            "windows": len(windows),
            "stages": stages,
            "full_pipeline_ms": full,
            "hop_budget_ms": budget_ms,
            "within_budget": full["p95"] < budget_ms,
            "concurrent_calls_per_core": concurrent,
            "model_loaded": registry.bundle is not None,
        }
        os.makedirs(os.path.dirname(args.json) or ".", exist_ok=True)
        with open(args.json, "w") as handle:
            json.dump(payload, handle, indent=2)
        print(f"\nWrote {args.json}")

    return 0 if full["p95"] < budget_ms else 1


if __name__ == "__main__":
    raise SystemExit(main())
