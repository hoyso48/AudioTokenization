from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from transformers import Trainer, TrainingArguments

from .collator import TTSCollator
from .constants import BOS_ID, EOS_ID, PAD_ID
from .dataset import JsonlTTSDataset
from .modeling_ar_tts import ARTTSConfig, ARTTSForConditionalGeneration
from .text_tokenizer import CharTokenizer


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def infer_speech_vocab_size(jsonl_path: str) -> int:
    max_tok = 0
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            ex = json.loads(ln)
            for key in ("prompt_tokens", "target_tokens"):
                arr = ex.get(key, [])
                if arr:
                    max_tok = max(max_tok, int(max(arr)))
    return max_tok + 1


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train AR TTS model (FFR/VFR) with HuggingFace Trainer")
    p.add_argument("--train_jsonl", type=str, required=True)
    p.add_argument("--val_jsonl", type=str, required=True)
    p.add_argument("--tokenizer_path", type=str, required=True)
    p.add_argument("--output_dir", type=str, required=True)

    p.add_argument("--speech_vocab_size", type=int, default=None)
    p.add_argument("--use_vfr", action="store_true")
    p.add_argument("--max_span_len", type=int, default=512)
    p.add_argument("--lambda_span", type=float, default=1.0)

    # VARSTOK appendix-style defaults:
    # decoder-only transformer: 12 layers, 16 heads, dim 1024
    p.add_argument("--d_model", type=int, default=1024)
    p.add_argument("--n_head", type=int, default=16)
    p.add_argument("--n_layer", type=int, default=12)
    p.add_argument("--ffn_mult", type=int, default=4)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--max_position_embeddings", type=int, default=4096)

    p.add_argument("--max_text_tokens", type=int, default=256)
    p.add_argument("--max_prompt_tokens", type=int, default=256)
    p.add_argument("--max_target_tokens", type=int, default=1024)

    p.add_argument("--learning_rate", type=float, default=5e-2)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--num_train_epochs", type=float, default=100.0)
    p.add_argument("--max_steps", type=int, default=-1)
    p.add_argument("--warmup_ratio", type=float, default=0.0)
    p.add_argument("--lr_scheduler_type", type=str, default="cosine")
    p.add_argument("--per_device_train_batch_size", type=int, default=4)
    p.add_argument("--per_device_eval_batch_size", type=int, default=4)
    p.add_argument("--gradient_accumulation_steps", type=int, default=1)
    p.add_argument("--logging_steps", type=int, default=20)
    p.add_argument("--eval_steps", type=int, default=500)
    p.add_argument("--save_steps", type=int, default=500)
    p.add_argument("--save_total_limit", type=int, default=3)
    p.add_argument("--dataloader_num_workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--disable_bf16", action="store_true")
    p.add_argument("--disable_gradient_checkpointing", action="store_true")
    p.add_argument("--report_to", nargs="*", default=["none"])
    p.add_argument("--resume_from_checkpoint", type=str, default=None)
    return p.parse_args()


def build_trainer(args: argparse.Namespace) -> Trainer:
    tokenizer = CharTokenizer.load(args.tokenizer_path)
    text_vocab_size = tokenizer.vocab_size
    speech_vocab_size = args.speech_vocab_size or infer_speech_vocab_size(args.train_jsonl)
    full_vocab_size = 4 + text_vocab_size + speech_vocab_size

    train_ds = JsonlTTSDataset(
        jsonl_path=args.train_jsonl,
        tokenizer_path=args.tokenizer_path,
        use_vfr=args.use_vfr,
        max_text_tokens=args.max_text_tokens,
        max_prompt_tokens=args.max_prompt_tokens,
        max_target_tokens=args.max_target_tokens,
    )
    val_ds = JsonlTTSDataset(
        jsonl_path=args.val_jsonl,
        tokenizer_path=args.tokenizer_path,
        use_vfr=args.use_vfr,
        max_text_tokens=args.max_text_tokens,
        max_prompt_tokens=args.max_prompt_tokens,
        max_target_tokens=args.max_target_tokens,
    )

    collator = TTSCollator(
        text_vocab_size=text_vocab_size,
        speech_vocab_size=speech_vocab_size,
        use_vfr=args.use_vfr,
        max_span_len=args.max_span_len,
    )

    model_cfg = ARTTSConfig(
        vocab_size=full_vocab_size,
        d_model=args.d_model,
        n_head=args.n_head,
        n_layer=args.n_layer,
        ffn_mult=args.ffn_mult,
        dropout=args.dropout,
        max_position_embeddings=args.max_position_embeddings,
        use_vfr=args.use_vfr,
        max_span_len=args.max_span_len,
        lambda_span=args.lambda_span,
        pad_token_id=PAD_ID,
        bos_token_id=BOS_ID,
        eos_token_id=EOS_ID,
    )
    model = ARTTSForConditionalGeneration(model_cfg)

    bf16 = (not args.disable_bf16) and torch.cuda.is_available()
    train_args = TrainingArguments(
        output_dir=args.output_dir,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        num_train_epochs=args.num_train_epochs,
        max_steps=args.max_steps,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type=args.lr_scheduler_type,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        logging_steps=args.logging_steps,
        eval_steps=args.eval_steps,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        dataloader_num_workers=args.dataloader_num_workers,
        eval_strategy="steps",
        save_strategy="steps",
        bf16=bf16,
        report_to=args.report_to,
        gradient_checkpointing=False,
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model,
        args=train_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collator,
        tokenizer=None,
    )

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    with open(Path(args.output_dir) / "model_setup.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "text_vocab_size": text_vocab_size,
                "speech_vocab_size": speech_vocab_size,
                "full_vocab_size": full_vocab_size,
                "use_vfr": bool(args.use_vfr),
                "max_span_len": int(args.max_span_len),
            },
            f,
            ensure_ascii=True,
            indent=2,
        )

    return trainer


def main(args: Optional[argparse.Namespace] = None) -> None:
    args = args or parse_args()
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    set_seed(args.seed)
    trainer = build_trainer(args)
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    trainer.save_model(args.output_dir)
    metrics = trainer.evaluate()
    with open(Path(args.output_dir) / "eval_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=True, indent=2)
    print(json.dumps(metrics, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
