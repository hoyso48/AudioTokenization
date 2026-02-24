from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List


@dataclass
class CharTokenizer:
    stoi: Dict[str, int]
    unk_token: str = "<unk>"

    @property
    def itos(self) -> Dict[int, str]:
        return {idx: ch for ch, idx in self.stoi.items()}

    @property
    def vocab_size(self) -> int:
        return len(self.stoi)

    def encode(self, text: str) -> List[int]:
        unk_id = self.stoi[self.unk_token]
        return [self.stoi.get(ch, unk_id) for ch in text]

    def decode(self, ids: Iterable[int]) -> str:
        inv = self.itos
        out_chars: List[str] = []
        for idx in ids:
            out_chars.append(inv.get(int(idx), self.unk_token))
        return "".join(out_chars)

    def save(self, path: str | Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "type": "char",
            "unk_token": self.unk_token,
            "stoi": self.stoi,
        }
        with open(p, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=True, indent=2)

    @classmethod
    def load(cls, path: str | Path) -> "CharTokenizer":
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)
        if obj.get("type") != "char":
            raise ValueError(f"Unsupported tokenizer type: {obj.get('type')}")
        return cls(stoi={k: int(v) for k, v in obj["stoi"].items()}, unk_token=obj.get("unk_token", "<unk>"))

    @classmethod
    def build(cls, texts: Iterable[str]) -> "CharTokenizer":
        charset = set()
        for t in texts:
            charset.update(list(t))
        symbols = ["<unk>"] + sorted(charset)
        stoi = {sym: i for i, sym in enumerate(symbols)}
        return cls(stoi=stoi)
