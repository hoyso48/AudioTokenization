from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

import torch

from .collator import TTSCollator
from .constants import BOS_ID, EOS_ID, SEP_ID
from .modeling_ar_tts import ARTTSForConditionalGeneration
from .text_tokenizer import load_tokenizer


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="AR TTS inference (token generation)")
    p.add_argument("--model_dir", type=str, required=True)
    p.add_argument("--tokenizer_path", type=str, required=True)
    p.add_argument("--speech_vocab_size", type=int, required=True)
    p.add_argument("--text", type=str, required=True)
    p.add_argument("--prompt_tokens_json", type=str, required=True)
    p.add_argument("--prompt_spans_json", type=str, default=None)
    p.add_argument("--use_vfr", action="store_true")
    p.add_argument("--max_new_tokens", type=int, default=1024)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top_k", type=int, default=0)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--output_json", type=str, required=True)
    return p.parse_args()


def load_json_array(path: str) -> List[int]:
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    if not isinstance(obj, list):
        raise ValueError(f"Expected list json at {path}")
    return [int(x) for x in obj]


def sample_from_logits(logits: torch.Tensor, temperature: float, top_k: int) -> int:
    if temperature <= 0:
        return int(torch.argmax(logits).item())
    logits = logits / temperature
    if top_k > 0:
        k = min(top_k, logits.numel())
        vals, idxs = torch.topk(logits, k=k)
        probs = torch.softmax(vals, dim=-1)
        pick = torch.multinomial(probs, num_samples=1)
        return int(idxs[pick].item())
    probs = torch.softmax(logits, dim=-1)
    return int(torch.multinomial(probs, num_samples=1).item())


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)

    tokenizer = load_tokenizer(args.tokenizer_path)
    text_vocab_size = tokenizer.vocab_size
    collator = TTSCollator(
        text_vocab_size=text_vocab_size,
        speech_vocab_size=args.speech_vocab_size,
        use_vfr=args.use_vfr,
    )

    model = ARTTSForConditionalGeneration.from_pretrained(args.model_dir).to(device)
    model.eval()

    text_ids = tokenizer.encode(args.text)
    prompt_tokens_raw = load_json_array(args.prompt_tokens_json)
    prompt_tokens = [collator.speech_offset + x for x in prompt_tokens_raw]
    prompt_spans = None
    if args.use_vfr:
        if args.prompt_spans_json is None:
            raise ValueError("--prompt_spans_json is required when --use_vfr")
        prompt_spans = load_json_array(args.prompt_spans_json)
        if len(prompt_spans) != len(prompt_tokens):
            raise ValueError("prompt span length must match prompt token length")

    seq = [BOS_ID] + [collator.text_offset + x for x in text_ids] + [SEP_ID] + prompt_tokens + [SEP_ID]
    speech_mask = [0] * (2 + len(text_ids)) + [1] * len(prompt_tokens) + [0]
    span_ids = [1] * (2 + len(text_ids))
    if args.use_vfr:
        assert prompt_spans is not None
        span_ids += prompt_spans + [1]
    else:
        span_ids += [1] * (len(prompt_tokens) + 1)

    generated_tokens: List[int] = []
    generated_spans: List[int] = []

    for _ in range(args.max_new_tokens):
        input_ids = torch.tensor([seq], dtype=torch.long, device=device)
        attention_mask = torch.ones_like(input_ids, dtype=torch.long, device=device)
        speech_mask_t = torch.tensor([speech_mask], dtype=torch.bool, device=device)
        span_ids_t = torch.tensor([span_ids], dtype=torch.long, device=device)

        with torch.no_grad():
            out = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                speech_mask=speech_mask_t,
                span_ids=span_ids_t,
            )
        token_logits = out["logits"][0, -1].clone()

        # Restrict generation to acoustic tokens + EOS.
        allow = torch.full_like(token_logits, fill_value=float("-inf"))
        s0 = collator.speech_offset
        s1 = collator.speech_offset + args.speech_vocab_size
        allow[s0:s1] = token_logits[s0:s1]
        allow[EOS_ID] = token_logits[EOS_ID]

        next_id = sample_from_logits(allow, temperature=args.temperature, top_k=args.top_k)
        if next_id == EOS_ID:
            seq.append(EOS_ID)
            speech_mask.append(0)
            span_ids.append(1)
            break

        seq.append(next_id)
        speech_mask.append(1)
        generated_tokens.append(next_id - collator.speech_offset)

        if args.use_vfr:
            span_logits = out["span_logits"][0, -1]
            span_logits = span_logits.clone()
            span_logits[0] = float("-inf")
            next_span = sample_from_logits(span_logits, temperature=1.0, top_k=0)
            next_span = max(1, next_span)
            generated_spans.append(next_span)
            span_ids.append(next_span)
        else:
            span_ids.append(1)

    payload: Dict[str, object] = {
        "text": args.text,
        "generated_tokens": generated_tokens,
        "use_vfr": bool(args.use_vfr),
    }
    if args.use_vfr:
        payload["generated_spans"] = generated_spans

    out_path = Path(args.output_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=True, indent=2)
    print(f"Saved generation to {out_path}")


if __name__ == "__main__":
    main()
