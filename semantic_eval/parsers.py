"""Dataset parsers and manifest builders for semantic_eval.

Only the four ARCH datasets requested by the task are supported:
  - ravdess
  - emovo
  - audio_mnist
  - slurp
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional


RAVDESS_LABEL_MAP: Dict[str, str] = {
    "01": "neutral",
    "02": "calm",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fear",
    "07": "disgust",
    "08": "surprised",
}

EMOVO_LABEL_MAP: Dict[str, str] = {
    "dis": "disgust",
    "gio": "happy",
    "neu": "neutral",
    "pau": "fear",
    "rab": "angry",
    "sor": "surprise",
    "tri": "sad",
}


@dataclass(frozen=True)
class SemanticRecord:
    dataset: str
    split: str
    path: str
    label: str
    sample_id: str
    speaker_id: Optional[str] = None
    transcript: Optional[str] = None
    transcript_path: Optional[str] = None
    metadata: Optional[Dict[str, str]] = None


def _load_jsonl(path: Path) -> Iterable[Dict[str, object]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def parse_ravdess(dataset_root: Path) -> List[SemanticRecord]:
    base = dataset_root
    if not base.exists():
        raise FileNotFoundError(f"ravdess dataset root not found: {base}")

    records: List[SemanticRecord] = []
    for wav in sorted(base.glob("**/*.wav")):
        parts = wav.stem.split("-")
        if len(parts) != 7:
            continue
        emo_code = parts[2]
        label = RAVDESS_LABEL_MAP.get(emo_code)
        if label is None:
            continue
        speaker = parts[-1].lstrip("0") or parts[-1]
        metadata = {
            "modality": parts[0],
            "vocal_channel": parts[1],
            "intensity": parts[3],
            "statement": parts[4],
            "repetition": parts[5],
        }
        records.append(
            SemanticRecord(
                dataset="ravdess",
                split="all",
                path=str(wav.resolve()),
                label=label,
                sample_id=wav.stem,
                speaker_id=speaker,
                metadata=metadata,
            )
        )

    return records


def parse_emovo(dataset_root: Path) -> List[SemanticRecord]:
    base = dataset_root / "EMOVO"
    if not base.exists():
        base = dataset_root
    if not base.exists():
        raise FileNotFoundError(f"EMOVO folder not found: {base}")

    records: List[SemanticRecord] = []
    for wav in sorted(base.glob("**/*.wav")):
        stem = wav.stem
        m = re.match(r"^([a-z]{3})-([^\-]+)-([^\-]+)$", stem)
        if not m:
            continue
        emo_code, speaker, variant = m.group(1), m.group(2), m.group(3)
        label = EMOVO_LABEL_MAP.get(emo_code)
        if label is None:
            continue
        metadata = {
            "variant": variant,
            "prefix": emo_code,
        }
        records.append(
            SemanticRecord(
                dataset="emovo",
                split="all",
                path=str(wav.resolve()),
                label=label,
                sample_id=stem,
                speaker_id=speaker,
                metadata=metadata,
            )
        )

    return records


def parse_audio_mnist(dataset_root: Path) -> List[SemanticRecord]:
    base = dataset_root / "data"
    if not base.exists():
        base = dataset_root
    if not base.exists():
        raise FileNotFoundError(f"AudioMNIST data folder not found: {base}")

    records: List[SemanticRecord] = []
    for wav in sorted(base.glob("**/*.wav")):
        stem = wav.stem
        parts = stem.split("_")
        if len(parts) < 2:
            continue
        label = parts[0]
        if not label.isdigit():
            continue
        speaker = parts[1]
        records.append(
            SemanticRecord(
                dataset="audio_mnist",
                split="all",
                path=str(wav.resolve()),
                label=label,
                sample_id=stem,
                speaker_id=speaker,
                transcript=label,
                metadata={"digit": label, "index": parts[2] if len(parts) > 2 else ""},
            )
        )

    return records


def parse_slurp(dataset_root: Path) -> List[SemanticRecord]:
    base = dataset_root
    if not base.exists():
        raise FileNotFoundError(f"SLURP root not found: {base}")

    split_files = {
        "train": base / "train.jsonl",
        "devel": base / "devel.jsonl",
        "test": base / "test.jsonl",
    }
    records: List[SemanticRecord] = []
    for split_name, jsonl_path in split_files.items():
        if not jsonl_path.is_file():
            raise FileNotFoundError(f"Missing SLURP split file: {jsonl_path}")

        for row in _load_jsonl(jsonl_path):
            label = str(row.get("intent", "")).strip()
            if not label:
                continue
            sentence = row.get("sentence") or row.get("sentence_annotation")
            sentence = str(sentence).strip() if sentence is not None else None
            utterance_id = str(row.get("slurp_id", ""))
            recordings = row.get("recordings", [])
            if not isinstance(recordings, list):
                continue
            for rec in recordings:
                if not isinstance(rec, dict):
                    continue
                filename = str(rec.get("file", "")).strip()
                if not filename:
                    continue
                real_path = base / "slurp_real" / filename
                synth_path = base / "slurp_synth" / filename
                source_path = real_path if real_path.is_file() else synth_path
                if not source_path.is_file():
                    continue
                transcript_path = source_path.with_suffix(".trans.txt")
                transcript_path_value = (
                    str(transcript_path.resolve()) if transcript_path.is_file() else None
                )
                records.append(
                    SemanticRecord(
                        dataset="slurp",
                        split=split_name,
                        path=str(source_path.resolve()),
                        label=label,
                        sample_id=f"{utterance_id}:{filename}",
                        transcript=sentence,
                        transcript_path=transcript_path_value,
                        metadata={
                            "scenario": str(row.get("scenario", "")),
                            "action": str(row.get("action", "")),
                            "slurp_id": utterance_id,
                        },
                    )
                )

    return records


def _resolve_dataset_root(data_root: Path, dataset_name: str) -> Path:
    candidate = data_root / dataset_name
    if candidate.exists():
        return candidate
    return data_root


def parse_dataset(name: str, data_root: Path) -> List[SemanticRecord]:
    name = name.lower()
    if name == "ravdess":
        return parse_ravdess(_resolve_dataset_root(data_root, "ravdess"))
    if name == "emovo":
        return parse_emovo(_resolve_dataset_root(data_root, "emovo"))
    if name == "audio_mnist":
        return parse_audio_mnist(_resolve_dataset_root(data_root, "audio_mnist"))
    if name == "slurp":
        return parse_slurp(_resolve_dataset_root(data_root, "slurp"))
    raise ValueError(f"Unsupported dataset: {name}")


def write_jsonl(records: List[SemanticRecord], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(asdict(r), ensure_ascii=False) + "\n")


def write_label_map(records: List[SemanticRecord], out_path: Path) -> Dict[str, int]:
    labels = sorted({r.label for r in records})
    label_map = {label: idx for idx, label in enumerate(labels)}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"labels": labels, "label_to_id": label_map}
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    return label_map
