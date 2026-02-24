#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, List


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build AR-TTS training examples from utterance token jsonl")
    p.add_argument("--input_jsonl", type=str, required=True)
    p.add_argument("--output_jsonl", type=str, required=True)
    p.add_argument("--prompt_tokens", type=int, default=120)
    p.add_argument("--max_target_tokens", type=int, default=1024)
    p.add_argument("--min_target_tokens", type=int, default=16)
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--max_examples", type=int, default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(args.seed)

    records: List[Dict[str, object]] = []
    with open(args.input_jsonl, "r", encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            obj = json.loads(ln)
            if "tokens" not in obj or "text" not in obj or "speaker_id" not in obj:
                continue
            if len(obj["tokens"]) < args.min_target_tokens:
                continue
            records.append(obj)

    by_spk: Dict[str, List[Dict[str, object]]] = defaultdict(list)
    for r in records:
        by_spk[str(r["speaker_id"])].append(r)

    out_path = Path(args.output_jsonl)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with open(out_path, "w", encoding="utf-8") as f:
        for target in records:
            spk = str(target["speaker_id"])
            candidates = by_spk[spk]
            if len(candidates) < 2:
                continue

            prompt = random.choice(candidates)
            if prompt.get("utt_id") == target.get("utt_id") and len(candidates) > 1:
                for _ in range(4):
                    prompt = random.choice(candidates)
                    if prompt.get("utt_id") != target.get("utt_id"):
                        break

            prompt_tokens = [int(x) for x in prompt["tokens"][: args.prompt_tokens]]
            target_tokens = [int(x) for x in target["tokens"][: args.max_target_tokens]]
            if len(prompt_tokens) == 0 or len(target_tokens) < args.min_target_tokens:
                continue

            sample = {
                "text": str(target["text"]),
                "speaker_id": spk,
                "prompt_utt_id": prompt.get("utt_id"),
                "target_utt_id": target.get("utt_id"),
                "prompt_tokens": prompt_tokens,
                "target_tokens": target_tokens,
            }

            if "spans" in prompt and "spans" in target:
                prompt_spans = [int(x) for x in prompt["spans"][: len(prompt_tokens)]]
                target_spans = [int(x) for x in target["spans"][: len(target_tokens)]]
                if len(prompt_spans) == len(prompt_tokens) and len(target_spans) == len(target_tokens):
                    sample["prompt_spans"] = prompt_spans
                    sample["target_spans"] = target_spans

            f.write(json.dumps(sample, ensure_ascii=True) + "\n")
            written += 1
            if args.max_examples is not None and written >= args.max_examples:
                break

    print(f"Wrote {written} examples to {out_path}")


if __name__ == "__main__":
    main()
