#!/usr/bin/env python3
"""
Language model fine-tuning over codec indices.

Loads the flattened corpora produced by extract_indices.py, converts them
into a contiguous token stream, and trains a Qwen2.5 0.5B causal LM to predict
the next codec symbol. Supports both vanilla codec tokens and DTP-enhanced
tokens via the trailing-zero folding rule described in the spec.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Optional

import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import (
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
)

DEFAULT_MODEL_ID = "Qwen/Qwen2.5-0.5B"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_metadata(data_dir: Path) -> Dict[str, object]:
    metadata_path = data_dir / "metadata.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(f"metadata.json not found in {data_dir}")
    with open(metadata_path, "r") as f:
        return json.load(f)


def load_npz(path: Path) -> Dict[str, np.ndarray]:
    npz = np.load(path, allow_pickle=False)
    data = {key: npz[key] for key in npz.files}
    npz.close()
    return data


def combine_tokens(tokens: np.ndarray, trailing: Optional[np.ndarray], max_trailing: int) -> np.ndarray:
    if trailing is None:
        return tokens.astype(np.int64, copy=False)
    if max_trailing < 0:
        raise ValueError("max_trailing must be >= 0 when DTP is enabled")
    span = max_trailing + 1
    clipped = np.minimum(trailing, max_trailing).astype(np.int64, copy=False)
    base = tokens.astype(np.int64, copy=False)
    return base * span + clipped


@dataclass
class PackedSequenceDataset(Dataset):
    tokens: torch.Tensor
    block_size: int
    stride: int

    def __post_init__(self) -> None:
        total = self.tokens.numel()
        if total <= self.block_size:
            raise ValueError(
                f"Token stream length ({total}) must exceed block_size ({self.block_size}) + 1"
            )
        effective = total - (self.block_size + 1)
        self.num_chunks = 1 + effective // self.stride
        if self.num_chunks <= 0:
            raise ValueError("Computed dataset has zero chunks. Adjust block size or stride.")

    def __len__(self) -> int:
        return self.num_chunks

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        start = idx * self.stride
        end = start + self.block_size + 1
        if end > self.tokens.numel():
            end = self.tokens.numel()
            start = end - (self.block_size + 1)
        chunk = self.tokens[start:end]
        return {
            "input_ids": chunk[:-1],
            "labels": chunk[1:],
        }


def build_datasets(
    train_tokens: np.ndarray,
    test_tokens: np.ndarray,
    block_size: int,
    stride: Optional[int],
    max_train_chunks: Optional[int],
    max_eval_chunks: Optional[int],
) -> Dict[str, PackedSequenceDataset]:
    stride_val = stride or block_size
    train_tensor = torch.from_numpy(train_tokens.astype(np.int64, copy=False))
    test_tensor = torch.from_numpy(test_tokens.astype(np.int64, copy=False))

    train_ds = PackedSequenceDataset(train_tensor, block_size=block_size, stride=stride_val)
    eval_ds = PackedSequenceDataset(test_tensor, block_size=block_size, stride=stride_val)

    if max_train_chunks is not None:
        train_ds.num_chunks = min(train_ds.num_chunks, max_train_chunks)
    if max_eval_chunks is not None:
        eval_ds.num_chunks = min(eval_ds.num_chunks, max_eval_chunks)
    return {"train": train_ds, "eval": eval_ds}


def reset_embeddings(model: AutoModelForCausalLM) -> None:
    input_emb = model.get_input_embeddings()
    if input_emb is not None:
        input_emb.reset_parameters()
    output_emb = model.get_output_embeddings()
    if output_emb is not None and output_emb is not input_emb:
        output_emb.reset_parameters()
    if hasattr(model, "tie_weights"):
        model.tie_weights()


def build_model(model_id: str, vocab_size: int, use_bf16: bool) -> AutoModelForCausalLM:
    torch_dtype = torch.bfloat16 if use_bf16 and torch.cuda.is_available() else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch_dtype,
    )
    model.resize_token_embeddings(vocab_size)
    reset_embeddings(model)
    model.config.vocab_size = vocab_size
    if model.config.pad_token_id is None:
        model.config.pad_token_id = 0
    model.config.use_cache = False
    return model


def build_metric_fn() -> Callable:
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        logits = torch.from_numpy(logits)
        labels = torch.from_numpy(labels)
        valid_mask = labels.ge(0)
        total = valid_mask.sum().item()
        if total == 0:
            return {"accuracy": 0.0, "top5_accuracy": 0.0}

        top1 = logits.argmax(dim=-1)
        top1_correct = (top1 == labels) & valid_mask
        acc = top1_correct.sum().item() / total

        k = min(5, logits.size(-1))
        topk = logits.topk(k, dim=-1).indices
        target = labels.unsqueeze(-1)
        topk_match = (topk == target)
        top5_correct = topk_match.any(dim=-1) & valid_mask
        top5_acc = top5_correct.sum().item() / total

        return {"accuracy": acc, "top5_accuracy": top5_acc}

    return compute_metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train an LM over codec indices.")
    parser.add_argument("--data_dir", type=str, required=True, help="Directory produced by extract_indices.py")
    parser.add_argument("--output_dir", type=str, default=None, help="Directory for LM checkpoints/results")
    parser.add_argument("--model_name", type=str, default=DEFAULT_MODEL_ID, help="HF model id to fine-tune")
    parser.add_argument("--block_size", type=int, default=400, help="Context length used for training")
    parser.add_argument("--stride", type=int, default=None, help="Stride between training windows (default=block_size)")
    parser.add_argument("--train_batch_size", type=int, default=32, help="Per-device train batch size")
    parser.add_argument("--eval_batch_size", type=int, default=32, help="Per-device eval batch size")
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--max_steps", type=int, default=10000)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--logging_steps", type=int, default=100)
    parser.add_argument("--eval_steps", type=int, default=500)
    parser.add_argument("--save_steps", type=int, default=1000)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--disable_gradient_checkpointing", action="store_true", help="Disable gradient checkpointing")
    parser.add_argument("--max_train_chunks", type=int, default=None, help="Optional limit on train windows")
    parser.add_argument("--max_eval_chunks", type=int, default=None, help="Optional limit on eval windows")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--report_to", type=str, nargs="*", default=["none"])
    parser.add_argument("--resume_from", type=str, default=None)
    parser.add_argument("--disable_bf16", action="store_true", help="Force fp32 even if bf16 is available")
    parser.add_argument("--save_total_limit", type=int, default=2)
    return parser.parse_args()


def main():
    args = parse_args()
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("medium")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    set_seed(args.seed)

    data_dir = Path(args.data_dir).resolve()
    metadata = load_metadata(data_dir)
    use_checkpointing = not args.disable_gradient_checkpointing
    use_bf16 = (not args.disable_bf16) and torch.cuda.is_available()

    train_npz = load_npz(Path(metadata["train"]["npz_path"]))
    test_npz = load_npz(Path(metadata["test"]["npz_path"]))

    use_dtp = bool(metadata["use_dtp"])
    train_max_trailing = int(metadata["train"]["max_trailing_zero"])
    codebook_size = int(metadata["codebook_size"])
    dtp_span = train_max_trailing + 1 if use_dtp else 1
    vocab_size = codebook_size * dtp_span

    train_tokens = combine_tokens(
        train_npz["tokens"],
        train_npz.get("trailing"),
        train_max_trailing if use_dtp else 0,
    )
    test_trailing = test_npz.get("trailing")
    if use_dtp and test_trailing is None:
        raise RuntimeError("DTP metadata missing trailing counts for test split.")
    if use_dtp and test_trailing is not None:
        test_trailing = np.minimum(test_trailing, train_max_trailing)
    test_tokens = combine_tokens(
        test_npz["tokens"],
        test_trailing,
        train_max_trailing if use_dtp else 0,
    )

    datasets = build_datasets(
        train_tokens,
        test_tokens,
        block_size=args.block_size,
        stride=args.stride,
        max_train_chunks=args.max_train_chunks,
        max_eval_chunks=args.max_eval_chunks,
    )

    output_dir = Path(args.output_dir).resolve() if args.output_dir else (data_dir / "lm_runs" / "qwen2p5_0p5b")
    output_dir.mkdir(parents=True, exist_ok=True)

    model = build_model(args.model_name, vocab_size, use_bf16)
    if use_checkpointing:
        model.gradient_checkpointing_enable()

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        max_steps=args.max_steps,
        warmup_ratio=args.warmup_ratio,
        logging_steps=args.logging_steps,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        bf16=use_bf16,
        report_to=args.report_to,
        dataloader_num_workers=4,
        gradient_checkpointing=use_checkpointing,
    )

    def collate(batch):
        input_ids = torch.stack([item["input_ids"] for item in batch])
        labels = torch.stack([item["labels"] for item in batch])
        return {"input_ids": input_ids, "labels": labels}

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=datasets["train"],
        eval_dataset=datasets["eval"],
        tokenizer=None,
        data_collator=collate,
        compute_metrics=build_metric_fn(),
    )

    trainer.train(resume_from_checkpoint=args.resume_from)

    eval_metrics = trainer.evaluate()
    train_metrics = trainer.evaluate(eval_dataset=datasets["train"], metric_key_prefix="train")

    def add_ppl(metrics: Dict[str, float], key: str) -> None:
        loss_key = f"{key}_loss"
        if loss_key in metrics:
            loss_val = metrics[loss_key]
            metrics[f"{key}_perplexity"] = float(math.exp(min(50.0, loss_val)))

    add_ppl(eval_metrics, "eval")
    add_ppl(train_metrics, "train")

    results = {
        "eval": eval_metrics,
        "train": train_metrics,
        "vocab_size": vocab_size,
        "dtp_span": dtp_span,
        "max_trailing_zero": train_max_trailing,
    }

    results_path = output_dir / "metrics.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()

