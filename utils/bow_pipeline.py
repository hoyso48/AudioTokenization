import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from tqdm import tqdm


def normalize_text(text: str) -> str:
    """Lowercase and strip non-alphabet characters (keeps apostrophes)."""
    text = text.lower()
    text = re.sub(r"[^a-z' ]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def read_filelist(path: Path, dataset_root: Optional[Path]) -> List[Path]:
    paths: List[Path] = []
    with open(path, "r") as f:
        for line in f:
            if not line.strip():
                continue
            raw = line.strip().split("\t")[0]
            p = Path(raw)
            if not p.is_absolute() and dataset_root is not None:
                p = dataset_root / p
            paths.append(p)
    return paths


def load_transcript_file(transcript_path: Path) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    with open(transcript_path, "r") as f:
        for line in f:
            if not line.strip():
                continue
            utt, text = line.strip().split(" ", 1)
            mapping[utt] = text
    return mapping


def collect_transcripts(audio_paths: Iterable[Path]) -> Dict[Path, str]:
    cache: Dict[Path, Dict[str, str]] = {}
    transcripts: Dict[Path, str] = {}
    for audio_path in tqdm(audio_paths, desc="Collect transcripts"):
        stem = audio_path.stem
        parts = stem.split("-")
        if len(parts) < 3:
            continue
        transcript_file = audio_path.with_name(f"{parts[0]}-{parts[1]}.trans.txt")
        if transcript_file not in cache:
            cache[transcript_file] = load_transcript_file(transcript_file)
        text = cache[transcript_file].get(stem)
        if text:
            transcripts[audio_path] = text
    return transcripts


def build_vocab(
    paths: List[Path],
    transcripts: Dict[Path, str],
    vocab_size: int,
    min_freq: int,
) -> Tuple[Dict[str, int], Counter]:
    counter: Counter = Counter()
    for audio_path in tqdm(paths, desc="Build vocab counts"):
        text = transcripts.get(audio_path)
        if not text:
            continue
        tokens = normalize_text(text).split()
        counter.update(tokens)

    vocab: Dict[str, int] = {}
    for word, _ in counter.most_common():
        if counter[word] < min_freq:
            continue
        if len(vocab) >= vocab_size:
            break
        vocab[word] = len(vocab)
    return vocab, counter


def encode_bow(text: str, vocab: Dict[str, int]) -> List[int]:
    tokens = normalize_text(text).split()
    token_ids = sorted({vocab[t] for t in tokens if t in vocab})
    return token_ids


def write_labels(
    split_name: str,
    filelist_paths: List[Path],
    transcripts: Dict[Path, str],
    vocab: Dict[str, int],
    output_dir: Path,
) -> Path:
    label_path = output_dir / f"{split_name}_bow.jsonl"
    with open(label_path, "w") as f:
        for audio_path in tqdm(filelist_paths, desc=f"Write labels ({split_name})"):
            text = transcripts.get(audio_path)
            if not text:
                continue
            bow_ids = encode_bow(text, vocab)
            entry = {
                "audio": str(audio_path.resolve()),
                "token_ids": bow_ids,
                "text": text,
            }
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    return label_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Bag-of-Words labels for LibriSpeech filelists.")
    parser.add_argument(
        "--filelists",
        nargs="+",
        required=True,
        help="Paths to filelist txt files (one audio path per line).",
    )
    parser.add_argument(
        "--dataset-root",
        type=str,
        default=None,
        help="Optional root to prepend when filelist entries are relative.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Where to write vocab + label files.",
    )
    parser.add_argument(
        "--vocab-filelist",
        type=str,
        default=None,
        help="Use this filelist (path) to build the vocab; default=first in --filelists.",
    )
    parser.add_argument("--vocab-size", type=int, default=512, help="Max vocab size.")
    parser.add_argument("--min-freq", type=int, default=2, help="Minimum frequency for a word to enter vocab.")
    return parser.parse_args()


def main():
    args = parse_args()
    dataset_root = Path(args.dataset_root) if args.dataset_root else None
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    filelist_map: Dict[str, List[Path]] = {}
    filelist_path_map: Dict[str, Path] = {}
    for fpath in args.filelists:
        path = Path(fpath)
        split_name = path.stem
        filelist_path_map[split_name] = path
        filelist_map[split_name] = read_filelist(path, dataset_root)

    vocab_filelist = Path(args.vocab_filelist) if args.vocab_filelist else Path(args.filelists[0])
    vocab_split = vocab_filelist.stem

    flat_paths = [p for paths in filelist_map.values() for p in paths]
    transcripts = collect_transcripts(flat_paths)

    vocab_paths = filelist_map[vocab_split]
    vocab, counts = build_vocab(vocab_paths, transcripts, args.vocab_size, args.min_freq)

    vocab_path = output_dir / "vocab.json"
    with open(vocab_path, "w") as f:
        json.dump({"vocab": vocab, "size": len(vocab)}, f, ensure_ascii=False, indent=2)

    label_paths: Dict[str, str] = {}
    for split, paths in filelist_map.items():
        label_path = write_labels(split, paths, transcripts, vocab, output_dir)
        label_paths[split] = str(label_path)

    meta = {
        "vocab_size": len(vocab),
        "min_freq": args.min_freq,
        "vocab_source_split": vocab_split,
        "filelists": {split: str(path) for split, path in filelist_path_map.items()},
        "label_paths": label_paths,
        "vocab_path": str(vocab_path),
        "text_normalization": "lowercase + strip non-alpha except apostrophes, collapse spaces",
    }
    meta_path = output_dir / "bow_meta.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"[BOW] vocab size={len(vocab)} written to {vocab_path}")
    print(f"[BOW] meta written to {meta_path}")
    for split, path in label_paths.items():
        print(f"[BOW] {split}: {path}")


if __name__ == "__main__":
    main()

