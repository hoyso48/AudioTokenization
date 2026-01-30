#!/usr/bin/env python3
"""
Compute "GT-only" metrics for a filelist of ground-truth audio.

Notes on definitions:
- WER: ASR(hyp) on GT audio vs GT transcript (if available).
- UTMOS: MOS prediction on GT audio.
- STOI/PESQ: computed as GT vs GT (identity) to provide an upper-bound / sanity reference.

Outputs:
- <output_dir>/per_file.jsonl : one JSON object per processed audio file
- <output_dir>/metrics.json   : aggregate (means + corpus WER)
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torchaudio
from jiwer import wer as jiwer_wer  # type: ignore
from torchmetrics.audio import (
    ShortTimeObjectiveIntelligibility,
    PerceptualEvaluationSpeechQuality,
)
from transformers import Wav2Vec2Processor, HubertForCTC

EVAL_ROOT = Path(__file__).resolve().parent
FAIRSEQ_PYTHON_ROOT = EVAL_ROOT / "fairseq"
if FAIRSEQ_PYTHON_ROOT.is_dir():
    p = str(FAIRSEQ_PYTHON_ROOT)
    if p not in sys.path:
        sys.path.insert(0, p)

from UTMOS import UTMOSScore  # local file in eval/

ALLOWED_AUDIO_EXTS = {".wav", ".flac"}


def read_lines(path: Path) -> List[str]:
    with path.open("r") as f:
        return [l.strip() for l in f if l.strip()]


def parse_input_paths(input_path: str) -> List[str]:
    p = Path(input_path)
    if p.is_dir():
        files = [str(fp.resolve()) for fp in p.rglob("*") if fp.is_file() and fp.suffix.lower() in ALLOWED_AUDIO_EXTS]
        files.sort()
        return files
    if p.is_file():
        if p.suffix.lower() == ".txt":
            paths = read_lines(p)
            return [str(Path(x).as_posix()) for x in paths]
        if p.suffix.lower() in ALLOWED_AUDIO_EXTS:
            return [str(p.resolve())]
    raise FileNotFoundError(f"Invalid --input: {input_path}. Provide a directory, a .txt filelist, or a single audio file.")


def load_audio_mono_16k(path: str) -> torch.Tensor:
    wav, sr = torchaudio.load(path)
    if sr != 16000:
        wav = torchaudio.transforms.Resample(sr, 16000)(wav)
    if wav.dim() == 2 and wav.size(0) > 1:
        wav = wav[:1, :]
    elif wav.dim() == 1:
        wav = wav.unsqueeze(0)
    return wav


def load_librispeech_transcript_for_audio(audio_path: Path) -> Tuple[Optional[str], Optional[str]]:
    """
    LibriSpeech convention: transcripts are stored at <speaker>-<chapter>.trans.txt in the same directory.
    Each line: "<utt_id> <TRANSCRIPT>"
    """
    file_id = audio_path.stem
    if "-" in file_id:
        prefix = "-".join(file_id.split("-")[:2])
    else:
        prefix = "_".join(file_id.split("_")[:2])
    trans_path = audio_path.parent / f"{prefix}.trans.txt"
    if not trans_path.is_file():
        return None, None
    with trans_path.open("r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(file_id + " "):
                return line[len(file_id) + 1 :].strip(), str(trans_path.resolve())
    return None, str(trans_path.resolve())


def seconds(wav_16k: torch.Tensor) -> float:
    # wav_16k: [1, T]
    return float(wav_16k.size(-1) / 16000.0)


def mean_ignore_none(values: List[Optional[float]]) -> Optional[float]:
    arr = [v for v in values if v is not None]
    if len(arr) == 0:
        return None
    return float(np.mean(arr))


@dataclass(frozen=True)
class PerFileResult:
    path: str
    duration_s: float
    transcript_path: Optional[str]
    gt_text: Optional[str]
    asr_text: Optional[str]
    wer_fraction: Optional[float]
    stoi: Optional[float]
    pesq_wb: Optional[float]
    pesq_nb: Optional[float]
    utmos: Optional[float]

    def to_dict(self) -> Dict[str, object]:
        return {
            "path": self.path,
            "duration_s": self.duration_s,
            "transcript_path": self.transcript_path,
            "gt_text": self.gt_text,
            "asr_text": self.asr_text,
            "wer_fraction": self.wer_fraction,
            "wer": (self.wer_fraction * 100.0) if self.wer_fraction is not None else None,
            "stoi": self.stoi,
            "pesq_wb": self.pesq_wb,
            "pesq_nb": self.pesq_nb,
            "utmos": self.utmos,
        }


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute GT-only metrics (WER/UTMOS + GT-vs-GT STOI/PESQ).")
    parser.add_argument("--input", type=str, required=True, help="Directory (recursive) or .txt filelist or single audio file")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to write metrics.json and per_file.jsonl")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--num_workers", type=int, default=0, help="Reserved (kept for symmetry); currently unused.")
    args = parser.parse_args()

    torch.set_grad_enabled(False)

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    per_file_path = output_dir / "per_file.jsonl"
    metrics_path = output_dir / "metrics.json"

    paths = parse_input_paths(args.input)
    if len(paths) == 0:
        raise RuntimeError("No input audio files found.")

    # Models / metrics
    processor = Wav2Vec2Processor.from_pretrained("facebook/hubert-large-ls960-ft")
    asr_model = HubertForCTC.from_pretrained("facebook/hubert-large-ls960-ft").to(args.device).eval()
    utmos_model = UTMOSScore(device=args.device)

    stoi_metric = ShortTimeObjectiveIntelligibility(fs=16000, extended=False)
    pesq_wb_metric = PerceptualEvaluationSpeechQuality(fs=16000, mode="wb")
    pesq_nb_metric = PerceptualEvaluationSpeechQuality(fs=8000, mode="nb")

    stoi_vals: List[Optional[float]] = []
    pesq_wb_vals: List[Optional[float]] = []
    pesq_nb_vals: List[Optional[float]] = []
    utmos_vals: List[Optional[float]] = []
    wer_frac_vals: List[Optional[float]] = []

    corpus_refs: List[str] = []
    corpus_hyps: List[str] = []

    results: List[PerFileResult] = []

    for p in paths:
        ap = Path(p)
        wav_16k = load_audio_mono_16k(str(ap))
        dur_s = seconds(wav_16k)

        gt_text, transcript_path = load_librispeech_transcript_for_audio(ap)

        # UTMOS on GT
        utmos_val: Optional[float] = None
        utmos_tensor = utmos_model.score(wav_16k.to(args.device))
        utmos_val = float(utmos_tensor.squeeze().item())

        # WER: ASR(GT audio) vs transcript (if available)
        asr_text: Optional[str] = None
        wer_frac: Optional[float] = None
        if gt_text is not None and gt_text.strip():
            inputs = processor(wav_16k.squeeze().numpy(), sampling_rate=16000, return_tensors="pt").input_values.to(args.device)
            with torch.inference_mode():
                logits = asr_model(inputs).logits
                predicted_ids = torch.argmax(logits, dim=-1)
                hyp_text = processor.decode(predicted_ids[0].detach().cpu())
            asr_text = hyp_text
            wer_frac = float(jiwer_wer(gt_text, hyp_text))
            corpus_refs.append(gt_text)
            corpus_hyps.append(hyp_text)

        # STOI / PESQ: GT vs GT (identity). Skip if too short for PESQ.
        stoi_val: Optional[float] = None
        pesq_wb_val: Optional[float] = None
        pesq_nb_val: Optional[float] = None

        if wav_16k.numel() > 0:
            stoi_metric.reset()
            stoi_metric.update(wav_16k.unsqueeze(0), wav_16k.unsqueeze(0))
            stoi_val = float(stoi_metric.compute().item())

        if dur_s >= 0.25:
            pesq_wb_metric.reset()
            pesq_wb_metric.update(wav_16k.unsqueeze(0), wav_16k.unsqueeze(0))
            pesq_wb_val = float(pesq_wb_metric.compute().item())

            wav_8k = torchaudio.transforms.Resample(16000, 8000)(wav_16k)
            pesq_nb_metric.reset()
            pesq_nb_metric.update(wav_8k.unsqueeze(0), wav_8k.unsqueeze(0))
            pesq_nb_val = float(pesq_nb_metric.compute().item())

            if pesq_wb_val is not None and (math.isnan(pesq_wb_val) or math.isinf(pesq_wb_val)):
                pesq_wb_val = None
            if pesq_nb_val is not None and (math.isnan(pesq_nb_val) or math.isinf(pesq_nb_val)):
                pesq_nb_val = None

        r = PerFileResult(
            path=str(ap.resolve()),
            duration_s=dur_s,
            transcript_path=transcript_path,
            gt_text=gt_text,
            asr_text=asr_text,
            wer_fraction=wer_frac,
            stoi=stoi_val,
            pesq_wb=pesq_wb_val,
            pesq_nb=pesq_nb_val,
            utmos=utmos_val,
        )
        results.append(r)

        stoi_vals.append(stoi_val)
        pesq_wb_vals.append(pesq_wb_val)
        pesq_nb_vals.append(pesq_nb_val)
        utmos_vals.append(utmos_val)
        wer_frac_vals.append(wer_frac)

    with per_file_path.open("w") as f:
        for r in results:
            f.write(json.dumps(r.to_dict(), ensure_ascii=False) + "\n")

    sentence_avg_wer_frac = mean_ignore_none(wer_frac_vals)
    corpus_wer_frac: Optional[float] = None
    if len(corpus_refs) > 0 and len(corpus_refs) == len(corpus_hyps):
        corpus_wer_frac = float(jiwer_wer(corpus_refs, corpus_hyps))

    metrics = {
        "input": str(args.input),
        "device": str(args.device),
        "count": int(len(results)),
        "stoi_gt_vs_gt": mean_ignore_none(stoi_vals),
        "pesq_wb_gt_vs_gt": mean_ignore_none(pesq_wb_vals),
        "pesq_nb_gt_vs_gt": mean_ignore_none(pesq_nb_vals),
        "utmos_gt": mean_ignore_none(utmos_vals),
        "wer_gt_asr_corpus": (corpus_wer_frac * 100.0) if corpus_wer_frac is not None else None,
        "wer_gt_asr_sentence_avg": (sentence_avg_wer_frac * 100.0) if sentence_avg_wer_frac is not None else None,
        "per_file_jsonl": str(per_file_path),
    }

    with metrics_path.open("w") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    print(json.dumps(metrics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()


