"""Data-driven calibration of the heuristic rule anchors.

The evidence rules ship with anchors taken from published ranges for human speech (jitter
0.3–1.5 %, shimmer 3–8 %, and so on). Those are defensible defaults, but two of them are
only defensible *in general*: a feature like ``f0_micro_var`` depends on our own pitch
tracker's smoothing window, so its numeric range is a property of this implementation, not
of human physiology. Hand-tuning those by eye against a demo corpus would be exactly the
overfitting we criticise elsewhere.

So instead we fit them, from the bona fide side only:

* ``low`` — where genuine speech commonly sits (a rule should almost never fire here);
* ``high`` — the far tail of genuine speech (a rule firing fully here is unusual enough
  to be worth an analyst's time).

Fitting on the **bona fide distribution alone** is deliberate. It means the anchors encode
"what does real human speech look like", which generalises to synthesis methods we have
never seen, rather than "what did this particular vocoder do", which does not.

Separation against a spoof set is then used only to *prune*: a rule that does not separate
at all is a rule that will only ever contribute noise and false positives, so its weight
goes to zero and it stops being shown to users as evidence.
"""

from __future__ import annotations

import math
from dataclasses import replace
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from voiceguard.models.base import EvidenceRule

#: Quantiles used for the two anchors, expressed as "fraction of bona fide speech that is
#: more extreme than this anchor".
DEFAULT_LOW_TAIL = 0.25
DEFAULT_HIGH_TAIL = 0.02
#: A rule needs at least this AUC (in its own direction) to stay in the ensemble.
MIN_SEPARATION_AUC = 0.55
#: Fewer samples than this and we keep the published defaults rather than fitting noise.
MIN_SAMPLES = 40


def _column(rows: Sequence[Dict[str, float]], name: str) -> np.ndarray:
    values = [float(row[name]) for row in rows if name in row and math.isfinite(float(row[name]))]
    return np.asarray(values, dtype=np.float64)


def directional_auc(genuine: np.ndarray, spoof: np.ndarray, ascending: bool) -> float:
    """AUC of this single feature, oriented so that >0.5 means "useful as written".

    Computed via the Mann-Whitney U identity, which handles ties correctly — several of
    our features (pause counts, cliff frequencies) are heavily tied at zero.
    """
    if genuine.size < 5 or spoof.size < 5:
        return 0.5
    combined = np.concatenate([spoof, genuine])
    ranks = np.empty_like(combined)
    order = np.argsort(combined, kind="mergesort")
    sorted_values = combined[order]
    i = 0
    rank = np.empty(combined.size, dtype=np.float64)
    while i < sorted_values.size:
        j = i
        while j + 1 < sorted_values.size and sorted_values[j + 1] == sorted_values[i]:
            j += 1
        rank[i : j + 1] = 0.5 * (i + j) + 1.0
        i = j + 1
    ranks[order] = rank

    spoof_ranks = ranks[: spoof.size]
    u = float(spoof_ranks.sum() - spoof.size * (spoof.size + 1) / 2.0)
    auc = u / float(spoof.size * genuine.size)
    return auc if ascending else 1.0 - auc


def fit_rule_anchors(
    rules: Sequence[EvidenceRule],
    genuine_rows: Sequence[Dict[str, float]],
    spoof_rows: Optional[Sequence[Dict[str, float]]] = None,
    *,
    low_tail: float = DEFAULT_LOW_TAIL,
    high_tail: float = DEFAULT_HIGH_TAIL,
    prune: bool = True,
) -> Tuple[List[EvidenceRule], Dict[str, dict]]:
    """Refit anchors from observed bona fide speech. Returns ``(rules, report)``."""
    fitted: List[EvidenceRule] = []
    report: Dict[str, dict] = {}

    for rule in rules:
        genuine = _column(genuine_rows, rule.feature)
        entry: Dict[str, object] = {
            "feature": rule.feature,
            "original": [rule.low, rule.high],
            "samples": int(genuine.size),
        }

        if genuine.size < MIN_SAMPLES:
            entry["action"] = "kept defaults (too few samples)"
            report[rule.feature] = entry
            fitted.append(rule)
            continue

        ascending = rule.high > rule.low
        if ascending:
            low = float(np.quantile(genuine, 1.0 - low_tail))
            high = float(np.quantile(genuine, 1.0 - high_tail))
        else:
            low = float(np.quantile(genuine, low_tail))
            high = float(np.quantile(genuine, high_tail))

        # A degenerate feature (constant on genuine speech) cannot be anchored; keep the
        # published defaults so we do not create a divide-by-nothing ramp.
        if not math.isfinite(low) or not math.isfinite(high) or abs(high - low) < 1e-9:
            entry["action"] = "kept defaults (no spread in genuine speech)"
            report[rule.feature] = entry
            fitted.append(rule)
            continue

        weight = rule.weight
        if spoof_rows is not None and prune:
            spoof = _column(spoof_rows, rule.feature)
            auc = directional_auc(genuine, spoof, ascending)
            entry["auc"] = round(float(auc), 4)
            if auc < MIN_SEPARATION_AUC:
                weight = 0.0
                entry["action"] = f"pruned (AUC {auc:.3f} < {MIN_SEPARATION_AUC})"
            else:
                # Reward genuinely discriminative rules, but keep the range narrow so one
                # feature cannot dominate the layer.
                weight = rule.weight * float(np.clip(0.6 + 1.6 * (auc - 0.5), 0.6, 1.6))
                entry["action"] = "refitted"

        entry["fitted"] = [round(low, 6), round(high, 6)]
        entry["weight"] = round(float(weight), 4)
        report[rule.feature] = entry
        fitted.append(replace(rule, low=low, high=high, weight=weight))

    return [r for r in fitted if r.weight > 0.0], report


def summarise_report(report: Dict[str, dict]) -> str:
    """A short human-readable summary, printed by ``ml/train.py``."""
    lines = []
    pruned = [k for k, v in report.items() if str(v.get("action", "")).startswith("pruned")]
    refit = [k for k, v in report.items() if v.get("action") == "refitted"]
    kept = [k for k, v in report.items() if str(v.get("action", "")).startswith("kept")]
    lines.append(f"  refitted: {len(refit)}   pruned: {len(pruned)}   defaults kept: {len(kept)}")
    if pruned:
        lines.append(f"  pruned rules: {', '.join(sorted(pruned))}")
    ranked = sorted(
        ((k, v.get("auc")) for k, v in report.items() if v.get("auc") is not None),
        key=lambda kv: -float(kv[1]),
    )
    if ranked:
        lines.append("  most discriminative single rules:")
        for name, auc in ranked[:5]:
            lines.append(f"    {name:28s} AUC {float(auc):.3f}")
    return "\n".join(lines)
