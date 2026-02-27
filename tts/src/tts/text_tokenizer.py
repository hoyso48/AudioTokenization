from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Union


def _dedupe_word_boundaries(symbols: Iterable[str], word_boundary_token: str) -> List[str]:
    out: List[str] = []
    for sym in symbols:
        if sym == word_boundary_token and (not out or out[-1] == word_boundary_token):
            continue
        out.append(sym)
    if out and out[-1] == word_boundary_token:
        out = out[:-1]
    return out


_G2P_INSTANCE = None


def _get_g2p():
    global _G2P_INSTANCE
    if _G2P_INSTANCE is None:
        try:
            from g2p_en import G2p
        except Exception as exc:
            raise RuntimeError(
                "PhonemeTokenizer requires g2p_en. Install with: pip install g2p-en"
            ) from exc
        _G2P_INSTANCE = G2p()
    return _G2P_INSTANCE


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


@dataclass
class PhonemeTokenizer:
    stoi: Dict[str, int]
    unk_token: str = "<unk>"
    word_boundary_token: str = "|"

    @property
    def itos(self) -> Dict[int, str]:
        return {idx: sym for sym, idx in self.stoi.items()}

    @property
    def vocab_size(self) -> int:
        return len(self.stoi)

    def _phonemize(self, text: str) -> List[str]:
        raw = _get_g2p()(text)
        symbols: List[str] = []
        for s in raw:
            if s == " ":
                symbols.append(self.word_boundary_token)
                continue
            token = str(s).strip()
            if not token:
                continue
            if token in {",", ".", "!", "?", ":", ";", "...", "-", "--", "\"", "'", "(", ")", "[", "]", "{", "}"}:
                symbols.append(self.word_boundary_token)
                continue
            symbols.append(token)
        return _dedupe_word_boundaries(symbols, self.word_boundary_token)

    def encode(self, text: str) -> List[int]:
        unk_id = self.stoi[self.unk_token]
        symbols = self._phonemize(text)
        return [self.stoi.get(sym, unk_id) for sym in symbols]

    def decode(self, ids: Iterable[int]) -> str:
        inv = self.itos
        syms: List[str] = []
        for idx in ids:
            syms.append(inv.get(int(idx), self.unk_token))
        out: List[str] = []
        for sym in syms:
            if sym == self.word_boundary_token:
                if out and out[-1] != " ":
                    out.append(" ")
            else:
                if out and out[-1] != " ":
                    out.append(" ")
                out.append(sym)
        return "".join(out).strip()

    def save(self, path: str | Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "type": "phoneme",
            "unk_token": self.unk_token,
            "word_boundary_token": self.word_boundary_token,
            "stoi": self.stoi,
        }
        with open(p, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=True, indent=2)

    @classmethod
    def load(cls, path: str | Path) -> "PhonemeTokenizer":
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)
        if obj.get("type") != "phoneme":
            raise ValueError(f"Unsupported tokenizer type: {obj.get('type')}")
        return cls(
            stoi={k: int(v) for k, v in obj["stoi"].items()},
            unk_token=obj.get("unk_token", "<unk>"),
            word_boundary_token=obj.get("word_boundary_token", "|"),
        )

    @classmethod
    def build(cls, texts: Iterable[str], word_boundary_token: str = "|") -> "PhonemeTokenizer":
        tmp = cls(stoi={"<unk>": 0, word_boundary_token: 1}, word_boundary_token=word_boundary_token)
        vocab = {tmp.unk_token, word_boundary_token}
        for t in texts:
            vocab.update(tmp._phonemize(t))
        symbols = [tmp.unk_token] + sorted(sym for sym in vocab if sym != tmp.unk_token)
        stoi = {sym: i for i, sym in enumerate(symbols)}
        return cls(stoi=stoi, word_boundary_token=word_boundary_token)


TokenizerType = Union[CharTokenizer, PhonemeTokenizer]


def load_tokenizer(path: str | Path) -> TokenizerType:
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    ttype = obj.get("type")
    if ttype == "char":
        return CharTokenizer(stoi={k: int(v) for k, v in obj["stoi"].items()}, unk_token=obj.get("unk_token", "<unk>"))
    if ttype == "phoneme":
        return PhonemeTokenizer(
            stoi={k: int(v) for k, v in obj["stoi"].items()},
            unk_token=obj.get("unk_token", "<unk>"),
            word_boundary_token=obj.get("word_boundary_token", "|"),
        )
    raise ValueError(f"Unsupported tokenizer type: {ttype}")


def build_tokenizer(texts: Iterable[str], tokenizer_type: str = "phoneme", word_boundary_token: str = "|") -> TokenizerType:
    if tokenizer_type == "phoneme":
        return PhonemeTokenizer.build(texts, word_boundary_token=word_boundary_token)
    if tokenizer_type == "char":
        return CharTokenizer.build(texts)
    raise ValueError(f"Unsupported tokenizer_type: {tokenizer_type}")
