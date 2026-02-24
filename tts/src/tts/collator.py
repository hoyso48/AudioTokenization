from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import torch

from .constants import BOS_ID, EOS_ID, PAD_ID, SEP_ID


@dataclass
class TTSCollator:
    text_vocab_size: int
    speech_vocab_size: int
    use_vfr: bool
    max_span_len: int = 512

    @property
    def text_offset(self) -> int:
        return 4

    @property
    def speech_offset(self) -> int:
        return 4 + self.text_vocab_size

    @property
    def vocab_size(self) -> int:
        return 4 + self.text_vocab_size + self.speech_vocab_size

    def _build_one(self, ex: Dict[str, object]) -> Dict[str, torch.Tensor]:
        text_ids = [self.text_offset + int(x) for x in ex["text_ids"]]
        prompt_tokens = [self.speech_offset + int(x) for x in ex["prompt_tokens"]]
        target_tokens = [self.speech_offset + int(x) for x in ex["target_tokens"]]

        seq = [BOS_ID] + text_ids + [SEP_ID] + prompt_tokens + [SEP_ID] + target_tokens + [EOS_ID]
        attention_mask = [1] * len(seq)

        # Predict only target tokens (+ EOS), conditioning on text + prompt + separator.
        target_start = 1 + len(text_ids) + 1 + len(prompt_tokens) + 1
        label_mask = [0] * len(seq)
        for i in range(target_start, len(seq)):
            label_mask[i] = 1

        speech_mask = [0] * len(seq)
        prompt_start = 1 + len(text_ids) + 1
        for i in range(prompt_start, prompt_start + len(prompt_tokens)):
            speech_mask[i] = 1
        for i in range(target_start, target_start + len(target_tokens)):
            speech_mask[i] = 1

        out = {
            "input_ids": torch.tensor(seq, dtype=torch.long),
            "labels": torch.tensor(seq, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "label_mask": torch.tensor(label_mask, dtype=torch.bool),
            "speech_mask": torch.tensor(speech_mask, dtype=torch.bool),
        }

        if self.use_vfr:
            prompt_spans = [max(1, min(int(x), self.max_span_len)) for x in ex["prompt_spans"]]
            target_spans = [max(1, min(int(x), self.max_span_len)) for x in ex["target_spans"]]
            span_ids = [1] + [1] * len(text_ids) + [1] + prompt_spans + [1] + target_spans + [1]
            out["span_ids"] = torch.tensor(span_ids, dtype=torch.long)
            out["span_labels"] = torch.tensor(span_ids, dtype=torch.long)

        return out

    def __call__(self, batch: List[Dict[str, object]]) -> Dict[str, torch.Tensor]:
        items = [self._build_one(ex) for ex in batch]
        max_len = max(x["input_ids"].numel() for x in items)

        def pad_1d(v: torch.Tensor, pad_value: int) -> torch.Tensor:
            if v.numel() == max_len:
                return v
            pad_len = max_len - v.numel()
            pad = torch.full((pad_len,), fill_value=pad_value, dtype=v.dtype)
            return torch.cat([v, pad], dim=0)

        out: Dict[str, torch.Tensor] = {
            "input_ids": torch.stack([pad_1d(x["input_ids"], PAD_ID) for x in items], dim=0),
            "labels": torch.stack([pad_1d(x["labels"], -100) for x in items], dim=0),
            "attention_mask": torch.stack([pad_1d(x["attention_mask"], 0) for x in items], dim=0),
            "label_mask": torch.stack([pad_1d(x["label_mask"].to(torch.long), 0).to(torch.bool) for x in items], dim=0),
            "speech_mask": torch.stack([pad_1d(x["speech_mask"].to(torch.long), 0).to(torch.bool) for x in items], dim=0),
        }

        if self.use_vfr:
            out["span_ids"] = torch.stack([pad_1d(x["span_ids"], 1) for x in items], dim=0)
            out["span_labels"] = torch.stack([pad_1d(x["span_labels"], -100) for x in items], dim=0)

        return out
