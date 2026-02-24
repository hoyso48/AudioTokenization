from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

from torch.utils.data import Dataset

from .text_tokenizer import CharTokenizer


class JsonlTTSDataset(Dataset):
    def __init__(
        self,
        jsonl_path: str,
        tokenizer_path: str,
        use_vfr: bool,
        max_text_tokens: Optional[int] = None,
        max_prompt_tokens: Optional[int] = None,
        max_target_tokens: Optional[int] = None,
    ) -> None:
        self.path = Path(jsonl_path)
        if not self.path.is_file():
            raise FileNotFoundError(f"Dataset jsonl not found: {self.path}")
        self.tokenizer = CharTokenizer.load(tokenizer_path)
        self.use_vfr = bool(use_vfr)
        self.max_text_tokens = max_text_tokens
        self.max_prompt_tokens = max_prompt_tokens
        self.max_target_tokens = max_target_tokens
        self.samples = self._load_samples()

    def _load_samples(self) -> List[Dict[str, object]]:
        out: List[Dict[str, object]] = []
        with open(self.path, "r", encoding="utf-8") as f:
            for ln in f:
                ln = ln.strip()
                if not ln:
                    continue
                raw = json.loads(ln)
                text = str(raw.get("text", "")).strip()
                if not text:
                    continue

                if "text_ids" in raw:
                    text_ids = [int(x) for x in raw["text_ids"]]
                else:
                    text_ids = self.tokenizer.encode(text)
                prompt_tokens = [int(x) for x in raw["prompt_tokens"]]
                target_tokens = [int(x) for x in raw["target_tokens"]]

                if self.max_text_tokens is not None:
                    text_ids = text_ids[: self.max_text_tokens]
                if self.max_prompt_tokens is not None:
                    prompt_tokens = prompt_tokens[: self.max_prompt_tokens]
                if self.max_target_tokens is not None:
                    target_tokens = target_tokens[: self.max_target_tokens]

                if len(prompt_tokens) == 0 or len(target_tokens) == 0:
                    continue

                sample: Dict[str, object] = {
                    "text_ids": text_ids,
                    "prompt_tokens": prompt_tokens,
                    "target_tokens": target_tokens,
                    "text": text,
                }

                if self.use_vfr:
                    prompt_spans = raw.get("prompt_spans")
                    target_spans = raw.get("target_spans")
                    if prompt_spans is None or target_spans is None:
                        raise ValueError("use_vfr=True but prompt_spans/target_spans missing in dataset.")
                    prompt_spans = [int(x) for x in prompt_spans][: len(prompt_tokens)]
                    target_spans = [int(x) for x in target_spans][: len(target_tokens)]
                    if len(prompt_spans) != len(prompt_tokens) or len(target_spans) != len(target_tokens):
                        continue
                    sample["prompt_spans"] = prompt_spans
                    sample["target_spans"] = target_spans

                out.append(sample)
        if not out:
            raise RuntimeError(f"No valid samples loaded from {self.path}")
        return out

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, object]:
        return self.samples[idx]
