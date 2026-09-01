"""Detection metrics.

Implemented directly rather than pulled from a library, because the one that matters most
here — Equal Error Rate — is routinely computed two subtly different ways, and a
spoof-detection result is only comparable if you can see exactly which one produced it.

EER is the operating point where the false-acceptance rate (spoof scored as genuine)
equals the false-rejection rate (genuine scored as spoof). We find it by sweeping every
achievable threshold and interpolating at the crossing, which is the ASVspoof convention.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


@dataclass
class DetectionMetrics:
    """Everything we report about a detector on one evaluation set."""

    accuracy: float
    auc: float
    eer: float
    eer_threshold: float
    precision: float
    recall: float
    f1: float
    fpr_at_threshold: Dict[str, float] = field(default_factory=dict)
    tpr_at_threshold: Dict[str, float] = field(default_factory=dict)
    det_curve: List[Tuple[float, float]] = field(default_factory=list)
    n_genuine: int = 0
    n_spoof: int = 0
    confusion: Dict[str, int] = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "accuracy": round(self.accuracy, 4),
            "auc": round(self.auc, 4),
            "eer": round(self.eer, 4),
            "eer_percent": round(self.eer * 100, 2),
            "eer_threshold": round(self.eer_threshold, 4),
            "precision": round(self.precision, 4),
            "recall": round(self.recall, 4),
            "f1": round(self.f1, 4),
            "fpr_at_threshold": {k: round(v, 4) for k, v in self.fpr_at_threshold.items()},
            "tpr_at_threshold": {k: round(v, 4) for k, v in self.tpr_at_threshold.items()},
            "n_genuine": self.n_genuine,
            "n_spoof": self.n_spoof,
            "confusion": dict(self.confusion),
        }

    def summary_line(self) -> str:
        return (
            f"acc {self.accuracy:.3f} | AUC {self.auc:.3f} | EER {self.eer * 100:.2f}% "
            f"| P {self.precision:.3f} R {self.recall:.3f} "
            f"| FPR@50 {self.fpr_at_threshold.get('0.50', float('nan')):.3f}"
        )


def roc_curve(y_true: np.ndarray, scores: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(fpr, tpr, thresholds), descending by threshold. Ties handled correctly."""
    y_true = np.asarray(y_true).astype(int)
    scores = np.asarray(scores, dtype=np.float64)

    order = np.argsort(-scores, kind="mergesort")
    scores_sorted = scores[order]
    labels_sorted = y_true[order]

    distinct = np.flatnonzero(np.diff(scores_sorted))
    threshold_idx = np.r_[distinct, labels_sorted.size - 1]

    tps = np.cumsum(labels_sorted)[threshold_idx]
    fps = 1 + threshold_idx - tps

    n_pos = max(int(y_true.sum()), 1)
    n_neg = max(int((y_true == 0).sum()), 1)

    tpr = np.r_[0.0, tps / n_pos]
    fpr = np.r_[0.0, fps / n_neg]
    thresholds = np.r_[np.inf, scores_sorted[threshold_idx]]
    return fpr, tpr, thresholds


def auc_score(y_true: np.ndarray, scores: np.ndarray) -> float:
    """AUC via the Mann-Whitney U statistic (exact with ties)."""
    y_true = np.asarray(y_true).astype(int)
    scores = np.asarray(scores, dtype=np.float64)
    positives, negatives = scores[y_true == 1], scores[y_true == 0]
    if positives.size == 0 or negatives.size == 0:
        return 0.5

    combined = np.concatenate([positives, negatives])
    order = np.argsort(combined, kind="mergesort")
    sorted_values = combined[order]
    ranks = np.empty(combined.size, dtype=np.float64)
    rank_values = np.empty(combined.size, dtype=np.float64)

    i = 0
    while i < sorted_values.size:
        j = i
        while j + 1 < sorted_values.size and sorted_values[j + 1] == sorted_values[i]:
            j += 1
        rank_values[i : j + 1] = 0.5 * (i + j) + 1.0
        i = j + 1
    ranks[order] = rank_values

    u = float(ranks[: positives.size].sum() - positives.size * (positives.size + 1) / 2.0)
    return float(u / (positives.size * negatives.size))


def equal_error_rate(y_true: np.ndarray, scores: np.ndarray) -> Tuple[float, float]:
    """EER and the threshold at which it occurs.

    ``scores`` are P(spoof). FRR is the fraction of genuine scored above the threshold;
    FAR is the fraction of spoof scored below it. We interpolate at the crossing rather
    than taking the nearest sweep point, so the value does not jump with sample count.
    """
    fpr, tpr, thresholds = roc_curve(y_true, scores)
    frr = fpr            # genuine incorrectly flagged as spoof
    far = 1.0 - tpr      # spoof incorrectly passed as genuine

    difference = frr - far
    crossings = np.flatnonzero(np.diff(np.sign(difference)))
    if crossings.size == 0:
        index = int(np.argmin(np.abs(difference)))
        return float((frr[index] + far[index]) / 2.0), float(thresholds[index])

    i = int(crossings[0])
    d0, d1 = difference[i], difference[i + 1]
    weight = 0.0 if d1 == d0 else float(-d0 / (d1 - d0))
    eer = float(frr[i] + weight * (frr[i + 1] - frr[i]))

    # thresholds[0] is +inf by construction (the "flag nothing" operating point). A
    # perfectly-separated detector crosses at i = 0, and interpolating from infinity
    # gives inf + (-inf) = NaN — a NaN threshold in every metrics report for exactly the
    # detectors that work best. Fall back to the first finite neighbour instead.
    low, high = float(thresholds[i]), float(thresholds[i + 1])
    if not np.isfinite(low):
        threshold = high
    elif not np.isfinite(high):
        threshold = low
    else:
        threshold = low + weight * (high - low)
    return max(0.0, min(1.0, eer)), float(threshold)


def evaluate(
    y_true: np.ndarray,
    scores: np.ndarray,
    *,
    decision_threshold: float = 0.5,
    report_thresholds: Sequence[float] = (0.35, 0.50, 0.60, 0.80),
) -> DetectionMetrics:
    """Full metric set for one evaluation run."""
    y_true = np.asarray(y_true).astype(int)
    scores = np.asarray(scores, dtype=np.float64)

    predictions = (scores >= decision_threshold).astype(int)
    tp = int(((predictions == 1) & (y_true == 1)).sum())
    tn = int(((predictions == 0) & (y_true == 0)).sum())
    fp = int(((predictions == 1) & (y_true == 0)).sum())
    fn = int(((predictions == 0) & (y_true == 1)).sum())

    accuracy = (tp + tn) / max(len(y_true), 1)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-9)

    eer, eer_threshold = equal_error_rate(y_true, scores)
    fpr, tpr, _ = roc_curve(y_true, scores)

    n_genuine = int((y_true == 0).sum())
    n_spoof = int((y_true == 1).sum())

    fpr_at: Dict[str, float] = {}
    tpr_at: Dict[str, float] = {}
    for threshold in report_thresholds:
        flagged = scores >= threshold
        fpr_at[f"{threshold:.2f}"] = float(
            (flagged & (y_true == 0)).sum() / max(n_genuine, 1))
        tpr_at[f"{threshold:.2f}"] = float(
            (flagged & (y_true == 1)).sum() / max(n_spoof, 1))

    step = max(1, len(fpr) // 60)
    det = [(float(f), float(1.0 - t)) for f, t in zip(fpr[::step], tpr[::step])]

    return DetectionMetrics(
        accuracy=float(accuracy),
        auc=auc_score(y_true, scores),
        eer=eer,
        eer_threshold=eer_threshold,
        precision=float(precision),
        recall=float(recall),
        f1=float(f1),
        fpr_at_threshold=fpr_at,
        tpr_at_threshold=tpr_at,
        det_curve=det,
        n_genuine=n_genuine,
        n_spoof=n_spoof,
        confusion={"tp": tp, "tn": tn, "fp": fp, "fn": fn},
    )


def per_method_breakdown(
    y_true: np.ndarray,
    scores: np.ndarray,
    methods: Sequence[str],
    *,
    threshold: float = 0.5,
) -> Dict[str, dict]:
    """Detection rate per synthesis family — where a single average hides everything.

    A headline 95 % can easily be 99 % on two vocoders and 60 % on the third. Judges and
    security teams both need the third number, so it is reported by default.
    """
    y_true = np.asarray(y_true).astype(int)
    scores = np.asarray(scores, dtype=np.float64)
    methods_array = np.array(methods)

    genuine_mask = y_true == 0
    genuine_fpr = float((scores[genuine_mask] >= threshold).sum() / max(genuine_mask.sum(), 1))

    breakdown: Dict[str, dict] = {
        "bonafide": {
            "n": int(genuine_mask.sum()),
            "false_positive_rate": round(genuine_fpr, 4),
            "mean_score": round(float(scores[genuine_mask].mean()), 4)
            if genuine_mask.any() else 0.0,
        }
    }

    for method in sorted(set(methods_array[y_true == 1])):
        mask = (methods_array == method) & (y_true == 1)
        if not mask.any():
            continue
        breakdown[method] = {
            "n": int(mask.sum()),
            "detection_rate": round(float((scores[mask] >= threshold).sum() / mask.sum()), 4),
            "mean_score": round(float(scores[mask].mean()), 4),
        }
    return breakdown


#: A single feature separating the classes this well is almost never a real voice
#: property. In practice it means the two classes took different code paths through the
#: corpus builder and the classifier is reading the pipeline, not the speech.
LEAKAGE_AUC = 0.90


def detect_leakage(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: Sequence[str],
    *,
    threshold: float = LEAKAGE_AUC,
) -> List[dict]:
    """Flag features that separate the classes suspiciously well on their own.

    This exists because it caught a real bug. An early build of the bootstrap corpus
    reported 100 % accuracy and 0 % EER; the cause was that bona fide utterances carried a
    DC offset the synthetic ones did not, because only the synthetic path went through an
    STFT round-trip. A perfect score from a corpus artifact is worse than a mediocre score
    from real signal, because it is confidently wrong and it is invisible in the headline
    number. So the check runs on every training run and prints loudly.

    A flag is not automatically a bug — ``digital_silence_score`` legitimately separates
    raw TTS exports from real recordings — but it always deserves an explanation.
    """
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y).astype(int)
    flagged: List[dict] = []

    for index, name in enumerate(feature_names):
        column = X[:, index]
        if not np.isfinite(column).all() or float(np.std(column)) < 1e-12:
            continue
        auc = auc_score(y, column)
        oriented = max(auc, 1.0 - auc)
        if oriented >= threshold:
            genuine = column[y == 0]
            spoof = column[y == 1]
            flagged.append({
                "feature": name,
                "auc": round(float(oriented), 4),
                "direction": "higher_in_spoof" if auc > 0.5 else "higher_in_genuine",
                "genuine_mean": round(float(genuine.mean()), 8),
                "spoof_mean": round(float(spoof.mean()), 8),
            })
    return sorted(flagged, key=lambda row: -row["auc"])


def platt_scaling(scores: np.ndarray, y_true: np.ndarray,
                  *, iterations: int = 200, learning_rate: float = 0.15) -> Tuple[float, float]:
    """Fit ``sigmoid(a * logit(score) + b)`` by gradient descent on log loss.

    Calibration is what lets the UI say "87 % likely synthetic" and mean it. Fitted on a
    held-out split, never on the training split, or it just re-learns the training fit.
    """
    scores = np.clip(np.asarray(scores, dtype=np.float64), 1e-6, 1 - 1e-6)
    y_true = np.asarray(y_true, dtype=np.float64)
    z = np.log(scores / (1.0 - scores))

    a, b = 1.0, 0.0
    n = max(len(z), 1)
    for _ in range(iterations):
        predictions = 1.0 / (1.0 + np.exp(-(a * z + b)))
        error = predictions - y_true
        a -= learning_rate * float(np.dot(error, z)) / n
        b -= learning_rate * float(error.sum()) / n
    return float(a), float(b)
