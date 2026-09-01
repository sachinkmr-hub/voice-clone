"""Evaluate a saved model bundle.

    python -m ml.evaluate --corpus data/corpus --model ml/artifacts/bootstrap_model.joblib

Separate from ``ml/train.py`` so a model can be scored against a corpus it never saw —
train on the bootstrap corpus, evaluate on ASVspoof, and the gap between the two numbers
is the honest measure of what this system is worth.

Also evaluates the **full four-layer stack**, not just the classifier, because that is
what actually runs in production: the fused score with prosody and context in the mix is
the number a bank would act on, and it is not the same number as the classifier's.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Dict, List, Optional

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))

import numpy as np

from ml.datasets.loaders import load_corpus, summarise
from ml.features import aggregate_to_utterances, extract_corpus_features
from ml.metrics import detect_leakage, evaluate as compute_metrics, per_method_breakdown
from voiceguard.audio.io import load_audio
from voiceguard.config import SAMPLE_RATE, Settings
from voiceguard.models.registry import ModelRegistry
from voiceguard.pipeline.session import CallSession


def load_bundle(path: str):
    import joblib

    if not os.path.exists(path):
        raise SystemExit(f"No model at {path}. Run `make train` first.")
    return joblib.load(path)


def evaluate_classifier(bundle, matrix) -> Dict[str, object]:
    """Score the acoustic classifier alone, at window and utterance level."""
    model = bundle.model if hasattr(bundle, "model") else bundle["model"]
    scaler = getattr(bundle, "scaler", None)
    names = list(getattr(bundle, "feature_names", matrix.feature_names))

    # Re-order columns to the bundle's own schema: a feature added since training must not
    # silently shift what the model reads.
    index = {name: i for i, name in enumerate(matrix.feature_names)}
    missing = [n for n in names if n not in index]
    if missing:
        print(f"  warning: {len(missing)} feature(s) in the model are absent from this "
              f"corpus extraction and will be zero-filled: {missing[:5]}")
    columns = np.zeros((matrix.X.shape[0], len(names)), dtype=np.float64)
    for position, name in enumerate(names):
        if name in index:
            columns[:, position] = matrix.X[:, index[name]]

    if scaler is not None:
        columns = scaler.transform(columns)
    scores = (model.predict_proba(columns)[:, -1] if hasattr(model, "predict_proba")
              else 1.0 / (1.0 + np.exp(-model.decision_function(columns))))

    window_metrics = compute_metrics(matrix.y, scores)
    utterance_scores, truth, methods = aggregate_to_utterances(matrix, scores)
    utterance_metrics = compute_metrics(truth, utterance_scores)

    return {
        "window_level": window_metrics.as_dict(),
        "utterance_level": utterance_metrics.as_dict(),
        "per_method": per_method_breakdown(truth, utterance_scores, methods),
        "_window_summary": window_metrics.summary_line(),
        "_utterance_summary": utterance_metrics.summary_line(),
    }


def evaluate_full_stack(items, model_dir: str, model_file: str,
                        limit: int = 0, verbose: bool = True) -> Dict[str, object]:
    """Replay every utterance through the real streaming pipeline.

    This is slower than scoring the classifier directly and it is the number that matters:
    it includes prosody, the speaker layer's abstention, the context layer, fusion,
    calibration and the score smoother — the whole thing a caller actually faces.
    """
    settings = Settings(model_dir=model_dir, model_file=model_file, database_url=":memory:")
    registry = ModelRegistry(settings)
    if registry.is_degraded:
        for note in registry.degraded:
            print(f"  note: {note}")

    # Stratified subsample. A plain items[:limit] slice would take the corpus in write
    # order, which is all bona fide first — giving a single-class evaluation set, an AUC
    # of exactly 0.5 and a recall of 0 that look like a broken detector rather than a
    # broken sample.
    if limit and limit < len(items):
        rng = np.random.default_rng(11)
        genuine = [i for i in items if i.label == 0]
        spoof = [i for i in items if i.label == 1]
        per_class = max(1, limit // 2)
        subset = (
            [genuine[i] for i in rng.permutation(len(genuine))[:per_class]]
            + [spoof[i] for i in rng.permutation(len(spoof))[:per_class]]
        )
    else:
        subset = list(items)
    scores: List[float] = []
    truth: List[int] = []
    methods: List[str] = []
    latencies: List[float] = []
    started = time.time()

    for position, item in enumerate(subset):
        try:
            audio, _ = load_audio(item.path, SAMPLE_RATE)
        except Exception:
            continue
        if audio.size < SAMPLE_RATE:
            continue

        session = CallSession(registry=registry, language=item.language, settings=settings)
        step = SAMPLE_RATE // 2
        for offset in range(0, len(audio), step):
            session.ingest(audio[offset : offset + step])
        session.flush()

        if not session.assessments:
            continue
        scores.append(session.stats.peak_score / 100.0)
        truth.append(item.label)
        methods.append(item.method)
        latencies.append(session.stats.mean_latency_ms)

        if verbose and (position + 1) % 25 == 0:
            print(f"  {position + 1}/{len(subset)} utterances replayed")

    if not scores:
        return {"error": "no utterances could be replayed"}
    if len(set(truth)) < 2:
        return {"error": f"evaluation set has only one class ({set(truth)}) — "
                         f"increase --limit or check the corpus"}

    metrics = compute_metrics(np.array(truth), np.array(scores))
    return {
        "metrics": metrics.as_dict(),
        "per_method": per_method_breakdown(np.array(truth), np.array(scores), methods),
        "latency": {
            "mean_ms_per_window": round(float(np.mean(latencies)), 2),
            "p95_ms_per_window": round(float(np.percentile(latencies, 95)), 2),
            "hop_budget_ms": 500.0,
            "within_budget": bool(float(np.percentile(latencies, 95)) < 500.0),
        },
        "replay_seconds": round(time.time() - started, 1),
        "_summary": metrics.summary_line(),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate a VoiceGuard model")
    parser.add_argument("--corpus", default="data/corpus")
    parser.add_argument("--model", default="ml/artifacts/bootstrap_model.joblib")
    parser.add_argument("--layout", default="auto",
                        choices=["auto", "manifest", "asvspoof", "folders"])
    parser.add_argument("--out", default="ml/artifacts/evaluation.json")
    parser.add_argument("--full-stack", action="store_true",
                        help="also replay every utterance through the live pipeline")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    print(f"Loading corpus from {args.corpus} ...")
    items = load_corpus(args.corpus, args.layout)
    stats = summarise(items)
    print(f"  {stats['total']} utterances, {stats['speakers']} speakers, "
          f"methods={stats['methods']}")

    print("Extracting features ...")
    matrix = extract_corpus_features(items, verbose=True)

    leaks = detect_leakage(matrix.X, matrix.y, matrix.feature_names)
    if leaks:
        print("  !! suspected corpus leakage:")
        for row in leaks:
            print(f"     {row['feature']:26s} AUC {row['auc']:.3f}")

    print(f"Evaluating classifier from {args.model} ...")
    bundle = load_bundle(args.model)
    classifier = evaluate_classifier(bundle, matrix)
    print(f"  window level:    {classifier.pop('_window_summary')}")
    print(f"  utterance level: {classifier.pop('_utterance_summary')}")
    for name, row in classifier["per_method"].items():
        if name == "bonafide":
            print(f"    bonafide      n={row['n']:4d}  FPR {row['false_positive_rate']:.3f}")
        else:
            print(f"    {name:13s} n={row['n']:4d}  detected {row['detection_rate']:.3f}")

    report = {
        "created_at": time.time(),
        "corpus": args.corpus,
        "corpus_stats": stats,
        "model": args.model,
        "trained_on": getattr(bundle, "trained_on", ""),
        "suspected_leakage": leaks,
        "classifier": classifier,
    }

    if args.full_stack:
        print("Replaying through the full four-layer pipeline ...")
        full = evaluate_full_stack(items, os.path.dirname(args.model),
                                   os.path.basename(args.model), args.limit)
        if "error" not in full:
            print(f"  full stack:      {full.pop('_summary')}")
            print(f"  latency:         {full['latency']['mean_ms_per_window']:.0f} ms/window "
                  f"mean, {full['latency']['p95_ms_per_window']:.0f} ms p95 "
                  f"(budget {full['latency']['hop_budget_ms']:.0f} ms)")
        report["full_stack"] = full

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as handle:
        json.dump(report, handle, indent=2, default=str)
    print(f"\nSaved evaluation -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
