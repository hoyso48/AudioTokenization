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
    p.add_argument("--prompt_seconds", type=float, default=3.0)
    p.add_argument("--min_prompt_tokens", type=int, default=8)
    p.add_argument("--fallback_prompt_token_rate", type=float, default=40.0)
    p.add_argument("--prompt_tokens", type=int, default=None, help="legacy fixed prompt tokens (overrides --prompt_seconds)")
    p.add_argument("--max_target_tokens", type=int, default=1024)
    p.add_argument("--min_target_tokens", type=int, default=16)
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--max_examples", type=int, default=None)
    return p.parse_args()


def infer_prompt_token_count(
    rec: Dict[str, object],
    prompt_seconds: float,
    min_prompt_tokens: int,
    fallback_prompt_token_rate: float,
    legacy_prompt_tokens: int | None,
) -> int:
    token_seq = rec.get("tokens")
    if not isinstance(token_seq, list):
        return 0
    token_len = len(token_seq)
    if token_len <= 0:
        return 0
    if legacy_prompt_tokens is not None:
        return max(1, min(token_len, int(legacy_prompt_tokens)))

    sample_rate = rec.get("sample_rate")
    num_samples = rec.get("orig_num_samples")
    token_rate = fallback_prompt_token_rate

    if isinstance(sample_rate, (int, float, str)) and isinstance(num_samples, (int, float, str)):
        sr = float(sample_rate)
        ns = float(num_samples)
        if sr > 0.0 and ns > 0.0:
            duration_sec = ns / sr
            if duration_sec > 0.0:
                token_rate = max(token_len / duration_sec, 1.0)

    n_prompt = int(round(max(prompt_seconds, 0.0) * token_rate))
    n_prompt = max(min_prompt_tokens, n_prompt)
    return max(1, min(token_len, n_prompt))


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
            if not isinstance(obj["tokens"], list) or len(obj["tokens"]) < args.min_target_tokens:
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

            n_prompt_tokens = infer_prompt_token_count(
                prompt,
                prompt_seconds=args.prompt_seconds,
                min_prompt_tokens=args.min_prompt_tokens,
                fallback_prompt_token_rate=args.fallback_prompt_token_rate,
                legacy_prompt_tokens=args.prompt_tokens,
            )
            prompt_seq = prompt.get("tokens")
            target_seq = target.get("tokens")
            if not isinstance(prompt_seq, list) or not isinstance(target_seq, list):
                continue

            prompt_tokens = [int(x) for x in prompt_seq[:n_prompt_tokens]]
            target_tokens = [int(x) for x in target_seq[: args.max_target_tokens]]
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

            if "sample_rate" in prompt and "orig_num_samples" in prompt:
                sample_rate_raw = prompt.get("sample_rate")
                num_samples_raw = prompt.get("orig_num_samples")
                if not isinstance(sample_rate_raw, (int, float, str)) or not isinstance(num_samples_raw, (int, float, str)):
                    sample_rate_raw = None
                    num_samples_raw = None

                if sample_rate_raw is not None and num_samples_raw is not None:
                    sr = float(sample_rate_raw)
                    ns = float(num_samples_raw)
                else:
                    sr = 0.0
                    ns = 0.0
                if sr > 0.0 and ns > 0.0 and len(prompt_seq) > 0:
                    prompt_duration = ns / sr
                    prompt_token_rate = len(prompt_seq) / prompt_duration
                    sample["prompt_duration_sec"] = float(len(prompt_tokens) / max(prompt_token_rate, 1e-6))

            if "spans" in prompt and "spans" in target:
                prompt_spans_src = prompt.get("spans")
                target_spans_src = target.get("spans")
                if not isinstance(prompt_spans_src, list) or not isinstance(target_spans_src, list):
                    continue

                prompt_spans = [int(x) for x in prompt_spans_src[: len(prompt_tokens)]]
                target_spans = [int(x) for x in target_spans_src[: len(target_tokens)]]
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
