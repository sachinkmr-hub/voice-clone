"""Feature extraction over a corpus, with caching and honest splitting.

Two things here are load-bearing for the credibility of every number we report.

**Window-level rows, utterance-level truth.** The detector runs on 1 s windows, so it is
trained on 1 s windows. But an *evaluation* on windows would count a 7 s file seven times
and let one easy file inflate the score, so evaluation aggregates windows back to
utterances first. Training on windows, scoring on utterances — that is the honest pairing.

**Speaker-disjoint splits.** ``split_by_speaker`` guarantees no speaker appears in both
train and test. A random per-file split would let the model memorise a voice and report an
accuracy that does not survive contact with a new caller.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))

import numpy as np

from ml.datasets.loaders import CorpusItem
from voiceguard.audio.io import load_audio
from voiceguard.audio.stream import iter_windows
from voiceguard.config import SAMPLE_RATE
from voiceguard.features.extractor import FeatureExtractor

#: Windows per utterance. More would be more data, but adjacent windows overlap by 50 %
#: and are highly correlated, so beyond this they mostly add compute and false confidence.
MAX_WINDOWS_PER_UTTERANCE = 8
CACHE_VERSION = 2


@dataclass
class FeatureMatrix:
    """Window-level features plus the bookkeeping needed to aggregate and split."""

    X: np.ndarray                                  #: (n_windows, n_features)
    y: np.ndarray                                  #: (n_windows,) 0/1
    feature_names: List[str] = field(default_factory=list)
    utterance_ids: List[int] = field(default_factory=list)
    speakers: List[str] = field(default_factory=list)
    methods: List[str] = field(default_factory=list)
    languages: List[str] = field(default_factory=list)
    paths: List[str] = field(default_factory=list)

    def __len__(self) -> int:
        return int(self.X.shape[0])

    def subset(self, mask: np.ndarray) -> "FeatureMatrix":
        mask = np.asarray(mask, dtype=bool)
        index = np.flatnonzero(mask)
        return FeatureMatrix(
            X=self.X[mask],
            y=self.y[mask],
            feature_names=list(self.feature_names),
            utterance_ids=[self.utterance_ids[i] for i in index],
            speakers=[self.speakers[i] for i in index],
            methods=[self.methods[i] for i in index],
            languages=[self.languages[i] for i in index],
            paths=[self.paths[i] for i in index],
        )

    def rows_as_dicts(self, mask: Optional[np.ndarray] = None) -> List[Dict[str, float]]:
        """Feature dictionaries — what the rule calibrator consumes."""
        matrix = self.X if mask is None else self.X[np.asarray(mask, dtype=bool)]
        return [dict(zip(self.feature_names, row)) for row in matrix]

    def summary(self) -> Dict[str, object]:
        return {
            "windows": len(self),
            "utterances": len(set(self.utterance_ids)),
            "speakers": len(set(self.speakers)),
            "spoof_windows": int(self.y.sum()),
            "bonafide_windows": int((self.y == 0).sum()),
            "features": len(self.feature_names),
        }


def _cache_key(items: Sequence[CorpusItem], schema: Sequence[str]) -> str:
    digest = hashlib.sha256()
    digest.update(f"v{CACHE_VERSION}|{len(schema)}|".encode())
    for item in items:
        digest.update(f"{item.path}|{item.label}|".encode())
        try:
            digest.update(str(os.path.getmtime(item.path)).encode())
        except OSError:
            pass
    return digest.hexdigest()[:20]


def extract_corpus_features(
    items: Sequence[CorpusItem],
    *,
    extractor: Optional[FeatureExtractor] = None,
    max_windows: int = MAX_WINDOWS_PER_UTTERANCE,
    cache_dir: Optional[str] = "ml/artifacts/cache",
    verbose: bool = True,
) -> FeatureMatrix:
    """Extract window-level features for every utterance, with an on-disk cache."""
    extractor = extractor or FeatureExtractor()
    schema = extractor.schema()

    cache_path = None
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
        cache_path = os.path.join(cache_dir, f"features_{_cache_key(items, schema)}.npz")
        if os.path.exists(cache_path):
            try:
                blob = np.load(cache_path, allow_pickle=True)
                if verbose:
                    print(f"  loaded cached features from {cache_path}")
                return FeatureMatrix(
                    X=blob["X"], y=blob["y"],
                    feature_names=list(blob["feature_names"]),
                    utterance_ids=list(blob["utterance_ids"].tolist()),
                    speakers=list(blob["speakers"]),
                    methods=list(blob["methods"]),
                    languages=list(blob["languages"]),
                    paths=list(blob["paths"]),
                )
            except Exception:
                pass  # a stale or corrupt cache must never block a training run

    rows: List[np.ndarray] = []
    labels: List[int] = []
    utterance_ids: List[int] = []
    speakers: List[str] = []
    methods: List[str] = []
    languages: List[str] = []
    paths: List[str] = []

    started = time.time()
    skipped = 0
    for index, item in enumerate(items):
        try:
            audio, rate = load_audio(item.path, SAMPLE_RATE)
        except Exception:
            skipped += 1
            continue
        if audio.size < SAMPLE_RATE:
            skipped += 1
            continue

        windows = iter_windows(audio, rate)[:max_windows]
        kept = 0
        for window in windows:
            bundle = extractor.extract(window.samples, language=item.language)
            if not bundle.speech_detected:
                continue          # a silent window teaches the model nothing about voices
            rows.append(bundle.vector(schema))
            labels.append(item.label)
            utterance_ids.append(index)
            speakers.append(item.speaker_id)
            methods.append(item.method)
            languages.append(item.language)
            paths.append(item.path)
            kept += 1
        if kept == 0:
            skipped += 1

        if verbose and (index + 1) % 25 == 0:
            elapsed = time.time() - started
            rate_per_s = (index + 1) / max(elapsed, 1e-6)
            print(f"  {index + 1}/{len(items)} utterances "
                  f"({len(rows)} windows, {rate_per_s:.1f} utt/s)")

    if not rows:
        raise ValueError("No usable windows extracted — is the corpus silent or unreadable?")

    matrix = FeatureMatrix(
        X=np.asarray(rows, dtype=np.float32),
        y=np.asarray(labels, dtype=np.int64),
        feature_names=list(schema),
        utterance_ids=utterance_ids,
        speakers=speakers,
        methods=methods,
        languages=languages,
        paths=paths,
    )
    if verbose:
        print(f"  extracted {len(matrix)} windows from {len(items) - skipped} utterances "
              f"in {time.time() - started:.1f}s ({skipped} skipped)")

    if cache_path:
        try:
            np.savez_compressed(
                cache_path, X=matrix.X, y=matrix.y,
                feature_names=np.array(matrix.feature_names),
                utterance_ids=np.array(matrix.utterance_ids),
                speakers=np.array(matrix.speakers), methods=np.array(matrix.methods),
                languages=np.array(matrix.languages), paths=np.array(matrix.paths),
            )
        except Exception:
            pass
    return matrix


# --------------------------------------------------------------------------------------
# Splitting
# --------------------------------------------------------------------------------------

def split_by_speaker(
    matrix: FeatureMatrix,
    test_fraction: float = 0.3,
    seed: int = 7,
) -> Tuple[np.ndarray, np.ndarray]:
    """Speaker-disjoint train/test masks.

    No speaker appears on both sides. This is the difference between a number that
    describes the model and a number that describes the corpus.
    """
    speakers = sorted(set(matrix.speakers))
    rng = np.random.default_rng(seed)
    shuffled = list(speakers)
    rng.shuffle(shuffled)

    n_test = max(1, int(round(len(shuffled) * test_fraction)))
    test_speakers = set(shuffled[:n_test])

    speaker_array = np.array(matrix.speakers)
    test_mask = np.isin(speaker_array, list(test_speakers))
    return ~test_mask, test_mask


def split_holdout_method(
    matrix: FeatureMatrix,
    holdout: str,
) -> Tuple[np.ndarray, np.ndarray]:
    """Hold one synthesis family out entirely — the *unseen attack* evaluation.

    Bona fide windows stay on both sides (we are not holding out humans); only spoofed
    windows produced by ``holdout`` move to the test side. What comes back is the honest
    answer to "will this work on a vocoder we have never seen".
    """
    methods = np.array(matrix.methods)
    is_holdout = methods == holdout
    train_mask = ~is_holdout
    test_mask = is_holdout | (matrix.y == 0)
    return train_mask, test_mask


def aggregate_to_utterances(
    matrix: FeatureMatrix,
    window_scores: np.ndarray,
    *,
    quantile: float = 0.75,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Collapse window scores to one score per utterance.

    Uses a high quantile rather than the mean, matching what the live system does: a call
    is suspicious if *part* of it is suspicious, and averaging in the innocuous parts is
    precisely how a short synthetic segment inside a longer call gets missed.
    """
    scores: List[float] = []
    truths: List[int] = []
    methods: List[str] = []
    for utterance_id in sorted(set(matrix.utterance_ids)):
        index = [i for i, u in enumerate(matrix.utterance_ids) if u == utterance_id]
        scores.append(float(np.quantile(window_scores[index], quantile)))
        truths.append(int(matrix.y[index[0]]))
        methods.append(matrix.methods[index[0]])
    return np.asarray(scores), np.asarray(truths), methods
