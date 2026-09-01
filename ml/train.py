"""Train the acoustic detector and produce the deployable bundle.

    python -m ml.train --corpus data/corpus --out ml/artifacts

What a run does, in order:

1. Load the corpus (bootstrap manifest, ASVspoof protocol, or bonafide/spoof folders).
2. Extract window-level features, cached on disk.
3. Split **by speaker** into train / calibration / test — never by file.
4. Fit a gradient-boosted classifier on the training windows.
5. Fit Platt scaling on the *calibration* split, so the probability means something.
6. Refit the heuristic rule anchors from the bona fide distribution and prune rules that
   do not separate.
7. Evaluate on the untouched test split at utterance level, plus an unseen-vocoder
   evaluation for every synthesis family present.
8. Persist the bundle and a metrics report.

The unseen-vocoder number is deliberately printed next to the headline number. A detector
that scores 0.99 on the vocoders it trained on and 0.6 on a new one is a detector that
will fail on the next TTS release, and hiding that behind an average would make this
project worse than useless in the field it targets.
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
from ml.features import (
    FeatureMatrix,
    aggregate_to_utterances,
    extract_corpus_features,
    split_by_speaker,
    split_holdout_method,
)
from ml.metrics import detect_leakage, evaluate, per_method_breakdown, platt_scaling
from voiceguard.models.acoustic import ACOUSTIC_RULES
from voiceguard.models.calibration import fit_rule_anchors, summarise_report
from voiceguard.models.registry import ARTIFACT_FORMAT_VERSION, ModelBundle


def build_model(kind: str = "gradient_boosting", seed: int = 7):
    """Create the classifier plus its scaler.

    Gradient boosting by default: it handles the wildly different scales in this feature
    vector (Hz next to dimensionless ratios), needs no feature engineering, trains in
    seconds on a laptop, and — unlike a deep model — produces something a reviewer can
    interrogate. A small MLP and a logistic baseline are available for comparison.
    """
    from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.neural_network import MLPClassifier
    from sklearn.preprocessing import StandardScaler

    if kind == "random_forest":
        return RandomForestClassifier(
            n_estimators=300, max_depth=12, min_samples_leaf=3,
            class_weight="balanced", random_state=seed, n_jobs=-1), None
    if kind == "logistic":
        return LogisticRegression(max_iter=2000, class_weight="balanced",
                                  random_state=seed), StandardScaler()
    if kind == "mlp":
        return MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=600,
                             early_stopping=True, random_state=seed), StandardScaler()
    return GradientBoostingClassifier(
        n_estimators=250, learning_rate=0.08, max_depth=3,
        subsample=0.85, random_state=seed), None


def _fit(model, scaler, X: np.ndarray, y: np.ndarray):
    if scaler is not None:
        X = scaler.fit_transform(X)
    model.fit(X, y)
    return model, scaler


def _predict(model, scaler, X: np.ndarray) -> np.ndarray:
    if scaler is not None:
        X = scaler.transform(X)
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, -1]
    scores = model.decision_function(X)
    return 1.0 / (1.0 + np.exp(-scores))


def _importances(model, names: List[str], top: int = 25) -> Dict[str, float]:
    if hasattr(model, "feature_importances_"):
        values = np.asarray(model.feature_importances_, dtype=float)
    elif hasattr(model, "coef_"):
        values = np.abs(np.asarray(model.coef_, dtype=float).ravel())
    else:
        return {}
    total = float(values.sum()) or 1.0
    ranked = sorted(zip(names, values / total), key=lambda kv: -kv[1])[:top]
    return {name: round(float(value), 5) for name, value in ranked}


def unseen_vocoder_evaluation(
    matrix: FeatureMatrix, kind: str, seed: int, verbose: bool = True
) -> Dict[str, dict]:
    """Retrain with one synthesis family removed, then test only on that family."""
    families = sorted({m for m, label in zip(matrix.methods, matrix.y)
                       if label == 1 and m != "unknown"})
    results: Dict[str, dict] = {}
    if len(families) < 2:
        return results

    for family in families:
        train_mask, test_mask = split_holdout_method(matrix, family)
        train, test = matrix.subset(train_mask), matrix.subset(test_mask)
        if train.y.sum() == 0 or test.y.sum() == 0:
            continue

        model, scaler = build_model(kind, seed)
        model, scaler = _fit(model, scaler, train.X, train.y)
        scores = _predict(model, scaler, test.X)

        utterance_scores, truth, _ = aggregate_to_utterances(test, scores)
        metrics = evaluate(truth, utterance_scores)
        results[family] = {
            "eer": round(metrics.eer, 4),
            "auc": round(metrics.auc, 4),
            "detection_rate_at_50": round(metrics.tpr_at_threshold.get("0.50", 0.0), 4),
            "false_positive_rate_at_50": round(metrics.fpr_at_threshold.get("0.50", 0.0), 4),
            "n_spoof_utterances": metrics.n_spoof,
        }
        if verbose:
            print(f"    unseen '{family}': EER {metrics.eer * 100:5.2f}%  "
                  f"AUC {metrics.auc:.3f}  detection@0.50 "
                  f"{metrics.tpr_at_threshold.get('0.50', 0.0):.3f}")
    return results


def train(
    corpus: str,
    out_dir: str,
    *,
    kind: str = "gradient_boosting",
    layout: str = "auto",
    test_fraction: float = 0.3,
    seed: int = 7,
    max_windows: int = 8,
    skip_unseen: bool = False,
) -> dict:
    started = time.time()
    os.makedirs(out_dir, exist_ok=True)

    print(f"Loading corpus from {corpus} ...")
    items = load_corpus(corpus, layout)
    stats = summarise(items)
    print(f"  {stats['total']} utterances — {stats['bonafide']} bona fide, "
          f"{stats['spoof']} spoof, {stats['speakers']} speakers")
    print(f"  methods: {stats['methods']}")

    if stats["bonafide"] == 0 or stats["spoof"] == 0:
        raise SystemExit("Corpus must contain both bona fide and spoofed utterances.")

    print("Extracting features ...")
    matrix = extract_corpus_features(items, max_windows=max_windows)
    print(f"  {matrix.summary()}")

    # Leakage check before anything is trained: a single feature that separates the
    # classes on its own almost always means the two classes took different code paths
    # through the corpus builder, and every downstream number would be fiction.
    leaks = detect_leakage(matrix.X, matrix.y, matrix.feature_names)
    if leaks:
        print("\n  !! SUSPECTED CORPUS LEAKAGE — these features separate the classes "
              "almost perfectly on their own:")
        for row in leaks:
            print(f"     {row['feature']:26s} AUC {row['auc']:.3f}  "
                  f"({row['direction']}: genuine {row['genuine_mean']:.6g} vs "
                  f"spoof {row['spoof_mean']:.6g})")
        print("     Treat the headline metrics below as unproven until each is explained.\n")
    else:
        print("  leakage check: no single feature separates the classes "
              "(all AUC < 0.90) — good\n")

    # Speaker-disjoint three-way split. Calibration gets its own speakers so Platt
    # scaling is never fitted on data the classifier has already seen.
    train_mask, holdout_mask = split_by_speaker(matrix, test_fraction + 0.15, seed)
    holdout = matrix.subset(holdout_mask)
    calibration_mask, test_mask = split_by_speaker(holdout, 0.65, seed + 1)
    train_set = matrix.subset(train_mask)
    calibration_set = holdout.subset(calibration_mask)
    test_set = holdout.subset(test_mask)

    print(f"  train {len(train_set)} windows / {len(set(train_set.speakers))} speakers")
    print(f"  calib {len(calibration_set)} windows / "
          f"{len(set(calibration_set.speakers))} speakers")
    print(f"  test  {len(test_set)} windows / {len(set(test_set.speakers))} speakers")
    overlap = set(train_set.speakers) & set(test_set.speakers)
    assert not overlap, f"speaker leakage between train and test: {overlap}"

    print(f"Training {kind} ...")
    model, scaler = build_model(kind, seed)
    model, scaler = _fit(model, scaler, train_set.X, train_set.y)

    print("Calibrating ...")
    if len(calibration_set) > 20 and 0 < calibration_set.y.sum() < len(calibration_set):
        calibration_scores = _predict(model, scaler, calibration_set.X)
        a, b = platt_scaling(calibration_scores, calibration_set.y)
    else:
        a, b = 1.0, 0.0
        print("  calibration split too small — using identity calibration")
    print(f"  Platt scaling: a={a:.3f} b={b:.3f}")

    print("Refitting heuristic rule anchors from the bona fide distribution ...")
    genuine_rows = train_set.rows_as_dicts(train_set.y == 0)
    spoof_rows = train_set.rows_as_dicts(train_set.y == 1)
    _, rule_report = fit_rule_anchors(list(ACOUSTIC_RULES), genuine_rows, spoof_rows)
    print(summarise_report(rule_report))

    print("Evaluating on the held-out speakers (utterance level) ...")
    window_scores = _predict(model, scaler, test_set.X)
    utterance_scores, truth, methods = aggregate_to_utterances(test_set, window_scores)
    metrics = evaluate(truth, utterance_scores)
    window_metrics = evaluate(test_set.y, window_scores)
    breakdown = per_method_breakdown(truth, utterance_scores, methods)

    print(f"  utterance level: {metrics.summary_line()}")
    print(f"  window level:    {window_metrics.summary_line()}")
    for name, row in breakdown.items():
        if name == "bonafide":
            print(f"    bonafide      n={row['n']:4d}  FPR {row['false_positive_rate']:.3f}")
        else:
            print(f"    {name:13s} n={row['n']:4d}  detected "
                  f"{row['detection_rate']:.3f}")

    per_condition: Dict[str, dict] = {}
    conditions = {}
    for item in items:
        conditions[os.path.abspath(item.path)] = getattr(item, "meta", {}).get("condition", "")
    utterance_conditions = []
    for utterance_id in sorted(set(test_set.utterance_ids)):
        index = [i for i, u in enumerate(test_set.utterance_ids) if u == utterance_id][0]
        path = os.path.basename(test_set.paths[index])
        utterance_conditions.append("telephony" if "telephony" in path else "raw")
    for condition in sorted(set(utterance_conditions)):
        mask = np.array([c == condition for c in utterance_conditions])
        if mask.sum() < 4 or len(set(truth[mask])) < 2:
            continue
        condition_metrics = evaluate(truth[mask], utterance_scores[mask])
        per_condition[condition] = condition_metrics.as_dict()
        print(f"    condition '{condition}': {condition_metrics.summary_line()}")

    unseen: Dict[str, dict] = {}
    if not skip_unseen:
        print("Unseen-vocoder evaluation (retraining with each family removed) ...")
        unseen = unseen_vocoder_evaluation(matrix, kind, seed)

    importances = _importances(model, matrix.feature_names)
    print("  top features:")
    for name, value in list(importances.items())[:8]:
        print(f"    {name:28s} {value:.4f}")

    bundle = ModelBundle(
        model=model,
        feature_names=list(matrix.feature_names),
        model_id=f"acoustic-{kind}-v1",
        scaler=scaler,
        calibration=(a, b),
        fusion_weights={},
        importances=importances,
        metrics=metrics.as_dict(),
        trained_on=(
            f"{corpus} ({stats['total']} utterances, {stats['speakers']} speakers, "
            f"methods={sorted(stats['methods'])})"
        ),
        format_version=ARTIFACT_FORMAT_VERSION,
    )

    import joblib

    model_path = os.path.join(out_dir, "bootstrap_model.joblib")
    joblib.dump(bundle, model_path)

    report = {
        "created_at": time.time(),
        "train_seconds": round(time.time() - started, 1),
        "model": kind,
        "model_path": model_path,
        "corpus": corpus,
        "corpus_stats": stats,
        "split": {
            "strategy": "speaker-disjoint (train / calibration / test)",
            "train_windows": len(train_set),
            "calibration_windows": len(calibration_set),
            "test_windows": len(test_set),
            "train_speakers": sorted(set(train_set.speakers)),
            "test_speakers": sorted(set(test_set.speakers)),
        },
        "calibration": {"a": a, "b": b},
        "metrics_utterance_level": metrics.as_dict(),
        "metrics_window_level": window_metrics.as_dict(),
        "per_method": breakdown,
        "per_condition": per_condition,
        "suspected_leakage": leaks,
        "unseen_vocoder": unseen,
        "rule_calibration": rule_report,
        "feature_importances": importances,
        "caveat": (
            "If --corpus was the bootstrap corpus, these numbers describe the pipeline on "
            "simulated audio and say nothing about real TTS systems. Retrain on ASVspoof "
            "or In-the-Wild before quoting them. See docs/MODEL_CARD.md."
        ),
    }
    with open(os.path.join(out_dir, "metrics.json"), "w") as handle:
        json.dump(report, handle, indent=2, default=str)

    print(f"\nSaved model  -> {model_path}")
    print(f"Saved report -> {os.path.join(out_dir, 'metrics.json')}")
    print(f"Done in {report['train_seconds']}s")
    print(f"\n  {report['caveat']}")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Train the VoiceGuard acoustic detector")
    parser.add_argument("--corpus", default="data/corpus")
    parser.add_argument("--out", default="ml/artifacts")
    parser.add_argument("--model", default="gradient_boosting",
                        choices=["gradient_boosting", "random_forest", "logistic", "mlp"])
    parser.add_argument("--layout", default="auto",
                        choices=["auto", "manifest", "asvspoof", "folders"])
    parser.add_argument("--test-fraction", type=float, default=0.3)
    parser.add_argument("--max-windows", type=int, default=8)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--skip-unseen", action="store_true",
                        help="skip the unseen-vocoder evaluation (faster, less honest)")
    args = parser.parse_args()

    train(args.corpus, args.out, kind=args.model, layout=args.layout,
          test_fraction=args.test_fraction, seed=args.seed,
          max_windows=args.max_windows, skip_unseen=args.skip_unseen)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
