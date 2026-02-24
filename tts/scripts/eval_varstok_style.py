#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import torch
import torchaudio
import torch.nn.functional as F
from jiwer import wer as jiwer_wer
from tqdm import tqdm
from transformers import pipeline


def _setup_path() -> None:
    tts_root = Path(__file__).resolve().parents[1]
    project_root = tts_root.parent
    eval_root = project_root / "eval"
    speaker_verif_root = eval_root / "speaker_verification"
    fairseq_root = eval_root / "fairseq"
    for p in (tts_root / "src", tts_root, project_root, eval_root, speaker_verif_root, fairseq_root):
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))


@dataclass
class MetaItem:
    file_id: str
    prompt_audio: Path
    target_text: str
    gt_audio: Optional[Path]


def parse_meta_lst(path: str) -> List[MetaItem]:
    meta_path = Path(path).resolve()
    base = meta_path.parent
    out: List[MetaItem] = []

    with open(meta_path, "r", encoding="utf-8") as f:
        for ln in f:
            raw = ln.strip()
            if not raw:
                continue
            parts = [p.strip() for p in raw.split("|")]
            if len(parts) < 4:
                continue

            file_id = parts[0]
            prompt_audio = Path(parts[2])
            target_text = parts[3]
            gt_audio = Path(parts[4]) if len(parts) >= 5 and parts[4] else None

            if not prompt_audio.is_absolute():
                prompt_audio = (base / prompt_audio).resolve()
            if gt_audio is not None and not gt_audio.is_absolute():
                gt_audio = (base / gt_audio).resolve()

            out.append(MetaItem(file_id=file_id, prompt_audio=prompt_audio, target_text=target_text, gt_audio=gt_audio))

    return out


def normalize_text(s: str) -> str:
    s = s.lower().strip()
    s = re.sub(r"\s+", " ", s)
    return s


def load_audio_16k_mono(path: Path) -> torch.Tensor:
    wav, sr = torchaudio.load(str(path))
    if wav.dim() == 2 and wav.shape[0] > 1:
        wav = wav[:1, :]
    if sr != 16000:
        wav = torchaudio.transforms.Resample(sr, 16000)(wav)
    return wav[:1, :]


def resolve_synth_path(synth_dir: Path, file_id: str) -> Path:
    p = synth_dir / file_id
    if p.is_file():
        return p
    if not file_id.lower().endswith(".wav"):
        p2 = synth_dir / f"{file_id}.wav"
        if p2.is_file():
            return p2
    raise FileNotFoundError(f"Synthesized file not found for id={file_id} in {synth_dir}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="VARSTOK-style objective TTS evaluation")
    p.add_argument("--meta_lst", type=str, required=True)
    p.add_argument("--synth_dir", type=str, required=True)
    p.add_argument("--output_json", type=str, required=True)
    p.add_argument("--language", type=str, choices=["en", "zh"], default="en")
    p.add_argument("--asr_model", type=str, default="openai/whisper-large-v3")
    p.add_argument("--wavlm_ckpt", type=str, default="/home/hoyso/projects/AudioTokenization/eval/wavlm_large_finetune.pth")
    p.add_argument("--sim_reference", type=str, choices=["prompt", "gt"], default="prompt")
    p.add_argument("--skip_sim", action="store_true")
    p.add_argument("--skip_utmos", action="store_true")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--limit", type=int, default=None)
    return p.parse_args()


def main() -> None:
    _setup_path()
    from UTMOS import UTMOSScore
    from verification import init_model as init_spk_model

    args = parse_args()
    synth_dir = Path(args.synth_dir).resolve()
    if not synth_dir.is_dir():
        raise FileNotFoundError(f"synth_dir not found: {synth_dir}")

    items = parse_meta_lst(args.meta_lst)
    if args.limit is not None:
        items = items[: args.limit]

    device = args.device
    asr_pipe = pipeline(
        "automatic-speech-recognition",
        model=args.asr_model,
        device=0 if (device.startswith("cuda") and torch.cuda.is_available()) else -1,
    )

    spk_model = None
    if not args.skip_sim:
        if not Path(args.wavlm_ckpt).is_file():
            raise FileNotFoundError(f"WavLM speaker ckpt not found: {args.wavlm_ckpt}")
        spk_model = init_spk_model("wavlm_large", args.wavlm_ckpt).to(device).eval()

    utmos_model = None
    if not args.skip_utmos:
        utmos_model = UTMOSScore(device=device)

    refs: List[str] = []
    hyps: List[str] = []
    sims: List[float] = []
    utmos_scores: List[float] = []
    per_sample: List[Dict[str, object]] = []

    iterator = tqdm(items, desc="Evaluating")
    for item in iterator:
        try:
            synth_path = resolve_synth_path(synth_dir, item.file_id)
        except FileNotFoundError:
            continue

        ref_text = normalize_text(item.target_text)
        asr_out = asr_pipe(str(synth_path))
        hyp_text = normalize_text(str(asr_out.get("text", "")))

        refs.append(ref_text)
        hyps.append(hyp_text)

        sim_val: Optional[float] = None
        if spk_model is not None:
            ref_audio_path = item.prompt_audio
            if args.sim_reference == "gt" and item.gt_audio is not None and item.gt_audio.is_file():
                ref_audio_path = item.gt_audio

            if ref_audio_path.is_file():
                wav_ref = load_audio_16k_mono(ref_audio_path).to(device)
                wav_syn = load_audio_16k_mono(synth_path).to(device)
                with torch.inference_mode():
                    emb_ref = spk_model(wav_ref)
                    emb_syn = spk_model(wav_syn)
                sim_val = float(F.cosine_similarity(emb_ref, emb_syn).mean().item())
                sims.append(sim_val)

        utmos_val: Optional[float] = None
        if utmos_model is not None:
            wav_syn = load_audio_16k_mono(synth_path)
            with torch.inference_mode():
                score = utmos_model.score(wav_syn.to(device))
            utmos_val = float(score.squeeze().item())
            utmos_scores.append(utmos_val)

        rec = {
            "file_id": item.file_id,
            "synth_path": str(synth_path),
            "ref_text": ref_text,
            "hyp_text": hyp_text,
            "sim": sim_val,
            "utmos": utmos_val,
        }
        per_sample.append(rec)

    corpus_wer = float(jiwer_wer(refs, hyps) * 100.0) if refs else None
    metrics = {
        "count": len(per_sample),
        "language": args.language,
        "asr_model": args.asr_model,
        "wer": corpus_wer,
        "speaker_similarity": (sum(sims) / len(sims)) if sims else None,
        "utmos": (sum(utmos_scores) / len(utmos_scores)) if utmos_scores else None,
    }
    output = {
        "metrics": metrics,
        "per_sample": per_sample,
    }

    out_path = Path(args.output_json).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=True, indent=2)

    print(json.dumps(metrics, ensure_ascii=True, indent=2))
    print(f"Saved full report: {out_path}")


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
