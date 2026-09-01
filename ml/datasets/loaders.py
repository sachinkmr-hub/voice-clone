"""Corpus loaders.

Three layouts are understood, so the same ``ml/train.py`` command works whether you have
the bootstrap corpus or a real anti-spoofing dataset:

``manifest``
    A ``manifest.json`` written by ``ml/datasets/build.py``.
``asvspoof``
    ASVspoof 2019 LA / 2021 protocol files: ``speaker utt_id - attack_id label``.
``folders``
    ``<root>/bonafide/*.wav`` and ``<root>/spoof/*.wav`` — the layout In-the-Wild and most
    ad-hoc collections end up in.

Every loader returns the same :class:`CorpusItem` list, and every one populates
``speaker_id`` and ``method`` where the source provides them, because the split logic
depends on both.
"""

from __future__ import annotations

import glob
import json
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

BONAFIDE_WORDS = ("bonafide", "bona_fide", "genuine", "real", "human")
SPOOF_WORDS = ("spoof", "synthetic", "fake", "clone", "tts", "deepfake")
AUDIO_EXTENSIONS = (".wav", ".flac", ".ogg", ".mp3", ".m4a")


@dataclass
class CorpusItem:
    """One utterance, however it was catalogued."""

    path: str
    label: int                      #: 0 = bona fide, 1 = spoof
    speaker_id: str = "unknown"
    method: str = "unknown"
    language: str = "auto"
    duration: float = 0.0
    meta: Dict[str, str] = field(default_factory=dict)

    @property
    def is_spoof(self) -> bool:
        return self.label == 1


def detect_layout(root: str) -> str:
    if os.path.exists(os.path.join(root, "manifest.json")):
        return "manifest"
    protocols = glob.glob(os.path.join(root, "**", "*cm_protocol*.txt"), recursive=True)
    protocols += glob.glob(os.path.join(root, "**", "*.trn.txt"), recursive=True)
    protocols += glob.glob(os.path.join(root, "**", "*trial_metadata*.txt"), recursive=True)
    if protocols:
        return "asvspoof"
    for word in BONAFIDE_WORDS:
        if os.path.isdir(os.path.join(root, word)):
            return "folders"
    return "folders"


def load_manifest(root: str) -> List[CorpusItem]:
    with open(os.path.join(root, "manifest.json")) as handle:
        manifest = json.load(handle)
    items: List[CorpusItem] = []
    for row in manifest.get("utterances", []):
        items.append(CorpusItem(
            path=os.path.join(root, row["path"]),
            label=int(row["label"]),
            speaker_id=row.get("speaker_id", "unknown"),
            method=row.get("method", "unknown"),
            language=row.get("language", "auto"),
            duration=float(row.get("duration", 0.0)),
        ))
    return items


def load_folders(root: str) -> List[CorpusItem]:
    """``<root>/bonafide/**`` and ``<root>/spoof/**`` (any of the synonym names)."""
    items: List[CorpusItem] = []
    for label, words in ((0, BONAFIDE_WORDS), (1, SPOOF_WORDS)):
        for word in words:
            directory = os.path.join(root, word)
            if not os.path.isdir(directory):
                continue
            for extension in AUDIO_EXTENSIONS:
                for path in sorted(glob.glob(os.path.join(directory, "**", f"*{extension}"),
                                             recursive=True)):
                    relative = os.path.relpath(path, directory)
                    # A nested directory is a useful default for speaker or attack id.
                    parts = relative.split(os.sep)
                    items.append(CorpusItem(
                        path=path,
                        label=label,
                        speaker_id=parts[0] if len(parts) > 1 else "unknown",
                        method="human" if label == 0 else (parts[0] if len(parts) > 1
                                                           else "unknown"),
                    ))
    return items


def load_asvspoof(root: str, protocol: Optional[str] = None) -> List[CorpusItem]:
    """ASVspoof LA protocol files.

    Line format (2019 LA): ``LA_0079 LA_T_1138215 - A07 spoof``
    2021 trial metadata carries extra columns; we read positionally from the front and
    take the label from the last field, which is stable across both.
    """
    if protocol is None:
        candidates: List[str] = []
        for pattern in ("*cm_protocol*.txt", "*.trn.txt", "*trial_metadata*.txt", "*.trl.txt"):
            candidates += glob.glob(os.path.join(root, "**", pattern), recursive=True)
        if not candidates:
            raise FileNotFoundError(f"No ASVspoof protocol file found under {root!r}")
        protocol = sorted(candidates)[0]

    audio_index: Dict[str, str] = {}
    for extension in AUDIO_EXTENSIONS:
        for path in glob.glob(os.path.join(root, "**", f"*{extension}"), recursive=True):
            audio_index[os.path.splitext(os.path.basename(path))[0]] = path

    items: List[CorpusItem] = []
    missing = 0
    with open(protocol) as handle:
        for line in handle:
            fields = line.split()
            if len(fields) < 3:
                continue
            speaker, utterance_id = fields[0], fields[1]
            label_token = fields[-1].lower()
            if label_token not in ("bonafide", "spoof"):
                continue
            path = audio_index.get(utterance_id)
            if path is None:
                missing += 1
                continue
            attack = next((f for f in fields[2:-1] if f.startswith("A")), "unknown")
            items.append(CorpusItem(
                path=path,
                label=0 if label_token == "bonafide" else 1,
                speaker_id=speaker,
                method="human" if label_token == "bonafide" else attack,
                meta={"protocol": os.path.basename(protocol)},
            ))
    if missing:
        print(f"  note: {missing} protocol entries had no matching audio file")
    return items


def load_corpus(root: str, layout: str = "auto",
                protocol: Optional[str] = None) -> List[CorpusItem]:
    """Load any supported layout."""
    if not os.path.isdir(root):
        raise FileNotFoundError(f"Corpus directory {root!r} does not exist. "
                                f"Run `make demo-data` to build the bootstrap corpus.")
    if layout == "auto":
        layout = detect_layout(root)

    if layout == "manifest":
        items = load_manifest(root)
    elif layout == "asvspoof":
        items = load_asvspoof(root, protocol)
    else:
        items = load_folders(root)

    if not items:
        raise ValueError(
            f"No utterances found in {root!r} (layout={layout}). Expected either a "
            f"manifest.json, an ASVspoof protocol file, or bonafide/ and spoof/ folders."
        )
    return items


def summarise(items: Sequence[CorpusItem]) -> Dict[str, object]:
    methods: Dict[str, int] = {}
    languages: Dict[str, int] = {}
    for item in items:
        methods[item.method] = methods.get(item.method, 0) + 1
        languages[item.language] = languages.get(item.language, 0) + 1
    return {
        "total": len(items),
        "bonafide": sum(1 for i in items if i.label == 0),
        "spoof": sum(1 for i in items if i.label == 1),
        "speakers": len({i.speaker_id for i in items}),
        "methods": dict(sorted(methods.items(), key=lambda kv: -kv[1])),
        "languages": dict(sorted(languages.items(), key=lambda kv: -kv[1])),
    }
