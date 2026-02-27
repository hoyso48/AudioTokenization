#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List


def _setup_path() -> None:
    root = Path(__file__).resolve().parents[1]
    src = root / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build offline text tokenizer for TTS text")
    p.add_argument("--input_jsonl", type=str, required=True, help="jsonl containing a text field")
    p.add_argument("--text_field", type=str, default="text")
    p.add_argument("--tokenizer_type", type=str, default="phoneme", choices=["phoneme", "char"])
    p.add_argument("--word_boundary_token", type=str, default="|")
    p.add_argument("--output_path", type=str, required=True)
    return p.parse_args()


def collect_texts(path: str, text_field: str) -> List[str]:
    texts: List[str] = []
    with open(path, "r", encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            obj = json.loads(ln)
            txt = str(obj.get(text_field, "")).strip()
            if txt:
                texts.append(txt)
    if not texts:
        raise RuntimeError(f"No valid text found in {path}")
    return texts


def main() -> None:
    _setup_path()
    from tts.text_tokenizer import build_tokenizer

    args = parse_args()
    texts = collect_texts(args.input_jsonl, args.text_field)
    tok = build_tokenizer(texts, tokenizer_type=args.tokenizer_type, word_boundary_token=args.word_boundary_token)
    tok.save(args.output_path)
    print(f"Saved {args.tokenizer_type} tokenizer to {args.output_path} (vocab_size={tok.vocab_size})")


if __name__ == "__main__":
    main()
