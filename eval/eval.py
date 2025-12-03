
import os
import sys
import json
import math
import argparse
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Iterable

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DTMAE_ROOT = PROJECT_ROOT / "DTMAE"
for path in (PROJECT_ROOT, DTMAE_ROOT):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

import torch
import torchaudio
import numpy as np
from omegaconf import OmegaConf
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from tqdm import tqdm
from torchmetrics.audio import (
    ShortTimeObjectiveIntelligibility,
    PerceptualEvaluationSpeechQuality,
    ScaleInvariantSignalNoiseRatio,
    ScaleInvariantSignalDistortionRatio,
)
from pesq import NoUtterancesError

from transformers import Wav2Vec2Processor, HubertForCTC

from DTMAE.lightning_module import (
    CodecLightningModule,
    CodebookPerplexity,
    CodebookUtilization,
)

# jiwer is preferred for WER; gracefully fallback if unavailable
try:
    from jiwer import wer as jiwer_wer  # type: ignore
except Exception:
    jiwer_wer = None


ALLOWED_AUDIO_EXTS = {".wav", ".flac"}


def read_lines(path: str) -> List[str]:
    with open(path, "r") as f:
        return [l.strip() for l in f if l.strip()]


def pad_to_multiple_1d(waveform: torch.Tensor, multiple_of: int) -> Tuple[torch.Tensor, int]:
    length = waveform.shape[-1]
    if multiple_of <= 0:
        return waveform, length
    if length % multiple_of == 0:
        return waveform, length
    pad_len = multiple_of - (length % multiple_of)
    padded = F.pad(waveform, (0, pad_len))
    return padded, length


def ensure_parent_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def last_k_parts(path: Path, k: int) -> Path:
    parts = path.parts
    if len(parts) <= k:
        return Path(*parts)
    return Path(*parts[-k:])


def parse_input_paths(input_path: str) -> List[str]:
    p = Path(input_path)
    if p.is_dir():
        files = [str(fp.resolve()) for fp in p.rglob("*") if fp.is_file() and fp.suffix.lower() in ALLOWED_AUDIO_EXTS]
        files.sort()
        return files
    if p.is_file():
        if p.suffix.lower() == ".txt":
            paths = read_lines(str(p))
            return [str(Path(x).as_posix()) for x in paths]
        if p.suffix.lower() in ALLOWED_AUDIO_EXTS:
            return [str(p.resolve())]
    raise FileNotFoundError(f"Invalid --input: {input_path}. Provide a directory, a .txt filelist, or a single audio file.")


def resolve_with_dataset_roots(paths: List[str], cfg) -> List[str]:
    roots: List[Path] = []
    try:
        roots.append(Path(cfg.preprocess.datasets.LibriSpeech.root))
    except Exception:
        pass
    try:
        roots.append(Path(cfg.preprocess.datasets.LibriTTS.root))
    except Exception:
        pass
    resolved: List[str] = []
    for p in paths:
        pp = Path(p)
        if pp.is_absolute() and pp.exists():
            resolved.append(str(pp.resolve()))
            continue
        if pp.exists():
            resolved.append(str(pp.resolve()))
            continue
        found = False
        for r in roots:
            candidate = r / p
            if candidate.exists():
                resolved.append(str(candidate.resolve()))
                found = True
                break
        if not found:
            print(f"[Warning] Input path not found: {p}. Skipping.")
    return resolved


def load_transcript_for_audio(audio_path: Path) -> Optional[str]:
    file_id = audio_path.stem
    if "-" in file_id:
        prefix = "-".join(file_id.split("-")[:2])
    else:
        prefix = "_".join(file_id.split("_")[:2])
    trans_path = audio_path.parent / f"{prefix}.trans.txt"
    if not trans_path.is_file():
        return None
    with open(trans_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(file_id + " "):
                return line[len(file_id) + 1 :].strip()
    return None


def compute_wer(ref: str, hyp: str) -> float:
    ref_words = ref.strip().upper().split()
    hyp_words = hyp.strip().upper().split()
    n = len(ref_words)
    m = len(hyp_words)
    if n == 0:
        return 0.0 if m == 0 else 1.0
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if ref_words[i - 1] == hyp_words[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,
                dp[i][j - 1] + 1,
                dp[i - 1][j - 1] + cost,
            )
    return dp[n][m] / max(1, n)


# Helpers for metrics stage

def load_audio_mono_16k(path: str) -> torch.Tensor:
    wav, sr = torchaudio.load(path)
    if sr != 16000:
        wav = torchaudio.transforms.Resample(sr, 16000)(wav)
    wav = wav[:1, :]
    return wav


def compute_pair_metrics(gt_wav: torch.Tensor, pred_wav: torch.Tensor,
                         asr_model, processor,
                         spk_model, mcd_compare,
                         pred_path: str) -> Dict[str, Optional[float]]:
    T = min(gt_wav.size(1), pred_wav.size(1))
    gt = gt_wav[:, :T]
    pr = pred_wav[:, :T]

    stoi_metric = ShortTimeObjectiveIntelligibility(fs=16000, extended=False)
    pesq_wb_metric = PerceptualEvaluationSpeechQuality(fs=16000, mode='wb')
    pesq_nb_metric = PerceptualEvaluationSpeechQuality(fs=8000, mode='nb')
    si_snr_metric = ScaleInvariantSignalNoiseRatio()
    si_sdr_metric = ScaleInvariantSignalDistortionRatio()

    stoi_metric.update(pr.unsqueeze(0), gt.unsqueeze(0))
    stoi_val = float(stoi_metric.compute().item())

    try:
        pesq_wb_metric.update(pr.unsqueeze(0), gt.unsqueeze(0))
        pesq_wb_val = float(pesq_wb_metric.compute().item())
    except NoUtterancesError as e:
        print(f"[Warning] PESQ-WB utterance error for {pred_path}: {e}")
        pesq_wb_val = None
    except Exception as e:
        print(f"[Error] PESQ-WB compute failed for {pred_path}: {e}")
        pesq_wb_val = None

    try:
        pr8 = torchaudio.transforms.Resample(16000, 8000)(pr)
        gt8 = torchaudio.transforms.Resample(16000, 8000)(gt)
        pesq_nb_metric.update(pr8.unsqueeze(0), gt8.unsqueeze(0))
        pesq_nb_val = float(pesq_nb_metric.compute().item())
    except NoUtterancesError as e:
        print(f"[Warning] PESQ-NB utterance error for {pred_path}: {e}")
        pesq_nb_val = None
    except Exception as e:
        print(f"[Error] PESQ-NB compute failed for {pred_path}: {e}")
        pesq_nb_val = None

    si_snr_metric.update(pr.unsqueeze(0), gt.unsqueeze(0))
    si_snr_val = float(si_snr_metric.compute().item())

    si_sdr_metric.update(pr.unsqueeze(0), gt.unsqueeze(0))
    si_sdr_val = float(si_sdr_metric.compute().item())

    spk_sim_val = None
    if spk_model is not None:
        try:
            with torch.inference_mode():
                emb_ref = spk_model(gt.to('cuda'))
                emb_rec = spk_model(pr.to('cuda'))
            spk_sim_val = float(F.cosine_similarity(emb_ref, emb_rec).mean().item())
        except Exception as e:
            print(f"[Error] Speaker similarity failed for {pred_path}: {e}")

    return {
        "stoi": stoi_val,
        "pesq_wb": pesq_wb_val,
        "pesq_nb": pesq_nb_val,
        "si_snr": si_snr_val,
        "si_sdr": si_sdr_val,
        "speaker_similarity": spk_sim_val,
    }


class AudioDataset(Dataset):
    def __init__(self, paths: List[str], target_sr: int, multiple_of: int, length_mode: str):
        assert length_mode in ("pad", "truncate"), "length_mode must be 'pad' or 'truncate'"
        self.paths = [str(Path(p)) for p in paths]
        self.target_sr = int(target_sr)
        self.multiple_of = int(multiple_of) if multiple_of is not None else 1
        self.length_mode = length_mode

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> Dict[str, object]:
        path = self.paths[idx]
        wav, sr = torchaudio.load(path)
        if wav.dim() == 2 and wav.size(0) > 1:
            wav = wav[:1, :]
        elif wav.dim() == 1:
            wav = wav.unsqueeze(0)
        if sr != self.target_sr:
            wav = torchaudio.transforms.Resample(sr, self.target_sr)(wav)
        wav_1d = wav[0]
        orig_len = int(wav_1d.shape[-1])

        if self.length_mode == "pad":
            wav_proc, _ = pad_to_multiple_1d(wav_1d, self.multiple_of)
            proc_len = int(wav_proc.shape[-1])
        else:  # truncate
            proc_len = (orig_len // self.multiple_of) * self.multiple_of if self.multiple_of > 0 else orig_len
            wav_proc = wav_1d[:proc_len]
        return {"wav": wav_proc, "path": path, "orig_length": orig_len, "proc_length": proc_len}

    @staticmethod
    def collate_fn(batch: List[Dict[str, object]]) -> Dict[str, object]:
        assert len(batch) == 1, "Batch size must be 1 for this model."
        b = batch[0]
        return {
            "wav": b["wav"].unsqueeze(0),
            "paths": [b["path"]],
            "orig_lengths": torch.tensor([b["orig_length"]], dtype=torch.long),
            "proc_lengths": torch.tensor([b["proc_length"]], dtype=torch.long),
        }


def run_save_stage(args, cfg, model, input_paths: List[str], eval_dir: Path, gt_out_dir: Optional[Path], pred_out_dir: Path, manifest_path: Path) -> Dict[str, Optional[float]]:
    target_sr = int(cfg.dataset.sample_rate)
    multiple_of = int(cfg.dataset.multiple_of)

    ds = AudioDataset(input_paths, target_sr=target_sr, multiple_of=multiple_of, length_mode=args.length_mode)
    print('total to be evaluated:', len(ds))
    dl = DataLoader(ds, batch_size=1, shuffle=False, num_workers=args.num_workers, pin_memory=True, collate_fn=AudioDataset.collate_fn)

    codebook_perplexity = CodebookPerplexity(codebook_size=cfg.model.codec_decoder.codebook_size)
    codebook_utilization = CodebookUtilization(codebook_size=cfg.model.codec_decoder.codebook_size)

    avg_sims: List[float] = []

    if gt_out_dir is not None:
        gt_out_dir.mkdir(parents=True, exist_ok=True)
    pred_out_dir.mkdir(parents=True, exist_ok=True)
    ensure_parent_dir(manifest_path)

    device_type = 'cuda' if ('cuda' in str(args.device)) else 'cpu'
    from contextlib import nullcontext

    with open(manifest_path, "w") as mf:
        for batch in tqdm(dl, total=len(ds)):
            wav = batch["wav"].to(args.device)
            paths = batch["paths"]
            orig_lengths = batch["orig_lengths"].tolist()
            proc_lengths = batch["proc_lengths"].tolist()
            assert wav.size(0) == 1, "Batch size must be 1."

            if args.length_mode == "truncate" and proc_lengths[0] == 0:
                print(f"[Warning] Skipping {paths[0]}: length < multiple_of ({multiple_of}), proc_length=0 in truncate mode.")
                continue

            with torch.inference_mode():
                ac = torch.autocast(device_type=device_type, dtype=torch.bfloat16) if device_type == 'cuda' else nullcontext()
                with ac:
                    out = model({"wav": wav})

            y_ref = out['gt_wav']
            y_rec = out['gen_wav']
            vq_code = out.get('vq_code', None)
            avg_sim_val = out.get('avg_sim', None)

            cut_len = int(proc_lengths[0]) if args.length_mode == "truncate" else int(orig_lengths[0])
            y_ref = y_ref[:, :, :cut_len]
            y_rec = y_rec[:, :, :cut_len]

            y_ref_16 = torchaudio.functional.resample(y_ref.detach().to(torch.float32).cpu(), target_sr, 16000)
            y_rec_16 = torchaudio.functional.resample(y_rec.detach().to(torch.float32).cpu(), target_sr, 16000)

            src_path = Path(paths[0])
            rel = last_k_parts(src_path.parent, 2) / (src_path.stem + ".wav")

            gt_16k_path = None
            if gt_out_dir is not None:
                gt_16k_path = gt_out_dir / rel
                ensure_parent_dir(gt_16k_path)
                torchaudio.save(str(gt_16k_path), y_ref_16[0].to(torch.float32).detach().cpu(), sample_rate=16000)

            pred_16k_path = pred_out_dir / rel
            ensure_parent_dir(pred_16k_path)
            torchaudio.save(str(pred_16k_path), y_rec_16[0].to(torch.float32).detach().cpu(), sample_rate=16000)

            if vq_code is not None:
                try:
                    codebook_perplexity.update(vq_code.detach().cpu())
                    codebook_utilization.update(vq_code.detach().cpu())
                except Exception as e:
                    print(f"[Error] Codebook metrics update failed for {src_path}: {e}")

            if avg_sim_val is not None:
                try:
                    avg_sims.append(float(avg_sim_val.detach().mean().item()))
                except Exception as e:
                    print(f"[Error] avg_sim extraction failed for {src_path}: {e}")

            transcript_text = load_transcript_for_audio(src_path)
            transcript_path = None
            if transcript_text is not None:
                if "-" in src_path.stem:
                    prefix = "-".join(src_path.stem.split("-")[:2])
                else:
                    prefix = "_".join(src_path.stem.split("_")[:2])
                tp = src_path.parent / f"{prefix}.trans.txt"
                if tp.is_file():
                    transcript_path = str(tp.resolve())

            record = {
                "orig_path": str(src_path.resolve()),
                "gt_16k_path": str(gt_16k_path) if gt_16k_path is not None else None,
                "pred_16k_path": str(pred_16k_path.resolve()),
                "transcript_path": transcript_path,
                "avg_sim": (float(avg_sims[-1]) if len(avg_sims) > 0 else None),
            }
            mf.write(json.dumps(record) + "\n")

    stats = {
        "codebook_perplexity": None,
        "codebook_utilization": None,
        "avg_sim_mean": float(np.mean(avg_sims)) if len(avg_sims) > 0 else None,
    }
    try:
        stats["codebook_perplexity"] = float(codebook_perplexity.compute().item())
    except Exception as e:
        print(f"[Error] codebook_perplexity compute failed: {e}")
    try:
        stats["codebook_utilization"] = float(codebook_utilization.compute().item())
    except Exception as e:
        print(f"[Error] codebook_utilization compute failed: {e}")

    with open(eval_dir / "save_stage_stats.json", "w") as f:
        json.dump(stats, f, indent=2)

    return stats


def run_metrics_stage(args, manifest_path: Path, eval_dir: Path) -> Dict[str, Optional[float]]:
    processor = Wav2Vec2Processor.from_pretrained("facebook/hubert-large-ls960-ft")
    asr_model = HubertForCTC.from_pretrained("facebook/hubert-large-ls960-ft").to(args.device).eval()

    sys.path.append(str((Path(__file__).resolve().parent / "speaker_verification")))
    try:
        from verification import init_model as init_spk_model
        spk_ckpt = Path(__file__).resolve().parent / "wavlm_large_finetune.pth"
        spk_model = init_spk_model('wavlm_large', str(spk_ckpt))
        spk_model = spk_model.to(args.device).eval()
    except Exception as e:
        print(f"[Warning] Speaker model init failed: {e}")
        spk_model = None

    try:
        from mel_cepstral_distance import compare_audio_files as mcd_compare
    except Exception as e:
        print(f"[Warning] mel_cepstral_distance unavailable: {e}")
        mcd_compare = None

    # Initialize UTMOS model (optional)
    try:
        # Import from local UTMOS implementation
        from UTMOS import UTMOSScore  # type: ignore
        utmos_model = UTMOSScore(device=args.device)
    except Exception as e:
        print(f"[Warning] UTMOS init failed: {e}")
        utmos_model = None

    stoi_vals: List[float] = []
    pesq_wb_vals: List[Optional[float]] = []
    pesq_nb_vals: List[Optional[float]] = []
    si_snr_vals: List[float] = []
    si_sdr_vals: List[float] = []
    spk_sim_vals: List[Optional[float]] = []
    mcd_vals: List[Optional[float]] = []
    wer_frac_vals: List[Optional[float]] = []
    utmos_vals: List[Optional[float]] = []

    # For corpus-level WER (not mean of per-sample)
    corpus_refs: List[str] = []
    corpus_hyps: List[str] = []

    lines = open(manifest_path, "r").read().splitlines()
    updated_records: List[Dict[str, object]] = []

    for line in tqdm(lines, total=len(lines)):
        rec = json.loads(line)
        gt_path = rec.get("gt_16k_path")
        pred_path = rec.get("pred_16k_path")
        transcript_path = rec.get("transcript_path")

        if gt_path is None:
            orig_path = rec.get("orig_path")
            if orig_path is None:
                raise RuntimeError("Manifest entry missing orig_path.")
            info = torchaudio.info(orig_path)
            if Path(orig_path).suffix.lower() != ".wav" or info.sample_rate != 16000:
                raise RuntimeError("GT is not saved and original is not 16k WAV. Re-run with --stage save and provide --gt_out_dir to save GT at 16k WAV.")
            gt_path = orig_path

        if not (gt_path and pred_path):
            print(f"[Warning] Skipping entry due to missing paths: {rec}")
            continue

        try:
            gt_wav = load_audio_mono_16k(gt_path)
            pred_wav = load_audio_mono_16k(pred_path)
        except Exception as e:
            print(f"[Error] Failed to load audio (gt={gt_path}, pred={pred_path}): {e}")
            continue

        pair = compute_pair_metrics(gt_wav, pred_wav, asr_model, processor, spk_model, mcd_compare, pred_path)

        mcd_val = None
        if mcd_compare is not None:
            try:
                mcd, _ = mcd_compare(gt_path, pred_path)
                mcd_val = float(mcd) if mcd is not None and not math.isnan(mcd) else None
            except Exception as e:
                print(f"[Warning] MCD failed for {pred_path}: {e}")

        # UTMOS predicted MOS on reconstructed audio
        utmos_val = None
        if utmos_model is not None:
            try:
                # pred_wav is 1xT at 16k; send to the same device as the model
                utmos_tensor = utmos_model.score(pred_wav.to(args.device))
                # Score returns a tensor of shape [B]; take scalar
                utmos_val = float(utmos_tensor.squeeze().item())
            except Exception as e:
                print(f"[Warning] UTMOS failed for {pred_path}: {e}")

        wer_frac = None
        gt_text: Optional[str] = None
        asr_text: Optional[str] = None
        if transcript_path and Path(transcript_path).is_file():
            try:
                with open(transcript_path, "r") as tf:
                    ref_line = None
                    file_id = Path(gt_path).stem
                    for tline in tf:
                        if tline.startswith(file_id + " "):
                            ref_line = tline[len(file_id) + 1 :].strip()
                            break
                if ref_line is not None:
                    gt_text = ref_line
                    inputs = processor(pred_wav.squeeze().numpy(), sampling_rate=16000, return_tensors="pt").input_values.to(args.device)
                    with torch.inference_mode():
                        logits = asr_model(inputs).logits
                        predicted_ids = torch.argmax(logits, dim=-1)
                        hyp_text = processor.decode(predicted_ids[0].detach().cpu())
                    asr_text = hyp_text
                    # Per-sample WER using jiwer if available
                    if jiwer_wer is not None:
                        try:
                            wer_frac = float(jiwer_wer(ref_line, hyp_text))
                        except Exception as e2:
                            print(f"[Warning] jiwer WER failed for {pred_path}: {e2}; falling back to internal compute_wer")
                            wer_frac = float(compute_wer(ref_line, hyp_text))
                    else:
                        wer_frac = float(compute_wer(ref_line, hyp_text))
                    # Accumulate for corpus-level WER
                    corpus_refs.append(ref_line)
                    corpus_hyps.append(hyp_text)
            except Exception as e:
                print(f"[Warning] WER failed for {pred_path}: {e}")

        rec.update({
            "stoi": pair["stoi"],
            "pesq_wb": pair["pesq_wb"],
            "pesq_nb": pair["pesq_nb"],
            "si_snr": pair["si_snr"],
            "si_sdr": pair["si_sdr"],
            "speaker_similarity": pair["speaker_similarity"],
            "mcd": mcd_val,
            "wer_fraction": wer_frac,
            "wer": (wer_frac * 100.0) if wer_frac is not None else None,
            "utmos": utmos_val,
            "gt_text": gt_text,
            "asr_text": asr_text,
        })
        updated_records.append(rec)

        if pair["stoi"] is not None: stoi_vals.append(pair["stoi"]) 
        pesq_wb_vals.append(pair["pesq_wb"]) 
        pesq_nb_vals.append(pair["pesq_nb"]) 
        if pair["si_snr"] is not None: si_snr_vals.append(pair["si_snr"]) 
        if pair["si_sdr"] is not None: si_sdr_vals.append(pair["si_sdr"]) 
        spk_sim_vals.append(pair["speaker_similarity"]) 
        mcd_vals.append(mcd_val)
        wer_frac_vals.append(wer_frac)
        utmos_vals.append(utmos_val)

    with open(manifest_path, "w") as mf:
        for rec in updated_records:
            mf.write(json.dumps(rec) + "\n")

    def mean_ignore_none(values: List[Optional[float]]) -> Optional[float]:
        arr = [v for v in values if v is not None]
        return float(np.mean(arr)) if len(arr) > 0 else None

    sentence_avg_wer_frac = mean_ignore_none(wer_frac_vals)
    corpus_wer_frac: Optional[float] = None
    if len(corpus_refs) > 0 and len(corpus_refs) == len(corpus_hyps):
        try:
            corpus_wer_frac = float(jiwer_wer(corpus_refs, corpus_hyps))
        except Exception as e:
            print(f"[Warning] Corpus WER failed: {e}")

    metrics = {
        "count": int(len(updated_records)),
        "stoi": mean_ignore_none(stoi_vals),
        "si_snr": mean_ignore_none(si_snr_vals),
        "si_sdr": mean_ignore_none(si_sdr_vals),
        "pesq_wb": mean_ignore_none(pesq_wb_vals),
        "pesq_nb": mean_ignore_none(pesq_nb_vals),
        "speaker_similarity": mean_ignore_none(spk_sim_vals),
        "mcd": mean_ignore_none(mcd_vals),
        # 'wer' is corpus-level WER (not an average of per-sample)
        "wer": (corpus_wer_frac * 100.0) if corpus_wer_frac is not None else None,
        # Provide the mean sentence-level WER as a separate metric
        "wer_sentence_avg": (sentence_avg_wer_frac * 100.0) if sentence_avg_wer_frac is not None else None,
        "utmos": mean_ignore_none(utmos_vals),
    }

    with open(eval_dir / "audio_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    return metrics


def load_save_stage_stats_if_any(eval_dir: Path) -> Dict[str, Optional[float]]:
    stats_path = eval_dir / "save_stage_stats.json"
    if not stats_path.is_file():
        return {}
    try:
        with open(stats_path, "r") as f:
            return json.load(f)
    except Exception as e:
        print(f"[Warning] Failed to load save_stage_stats.json: {e}")
        return {}


def main():
    parser = argparse.ArgumentParser(description="Evaluate BigCodec SSL with two-stage pipeline (save -> metrics). Batch size is always 1.")
    parser.add_argument("--input", type=str, required=True, help="Directory (recursive) or .txt filelist or single audio file for GT inputs")
    parser.add_argument("--run_dir", type=str, required=True, help="Run directory containing hydra/config.yaml and pl_log/last.ckpt")
    parser.add_argument("--stage", type=str, choices=["save", "metrics", "all"], default="all")
    parser.add_argument("--gt_out_dir", type=str, default=None, help="Where to save 16k WAV GTs during save stage")
    parser.add_argument("--pred_out_dir", type=str, default=None, help="Where to save 16k WAV predictions (default: run_dir/eval/pred_16k)")
    parser.add_argument("--manifest", type=str, default=None, help="Path to manifest.jsonl (default: run_dir/eval/manifest.jsonl)")
    parser.add_argument("--length_mode", type=str, choices=["pad", "truncate"], default="pad", help="How to make length a multiple of cfg.dataset.multiple_of")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    torch.set_grad_enabled(False)
    torch.backends.cudnn.benchmark = True

    run_dir = Path(args.run_dir).resolve()
    cfg_path = run_dir / "hydra" / "config.yaml"
    ckpt_path = run_dir / "pl_log" / "last.ckpt"
    if not cfg_path.is_file():
        raise FileNotFoundError(f"Config not found at {cfg_path}")
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found at {ckpt_path}")

    cfg = OmegaConf.load(str(cfg_path))

    eval_dir = run_dir / "eval"
    eval_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = Path(args.manifest) if args.manifest else (eval_dir / "manifest.jsonl")
    pred_out_dir = Path(args.pred_out_dir) if args.pred_out_dir else (eval_dir / "pred_16k")
    gt_out_dir = Path(args.gt_out_dir) if args.gt_out_dir else (eval_dir / "gt_16k" if args.stage in ("save", "all") else None)

    raw_paths = parse_input_paths(args.input)
    input_paths = resolve_with_dataset_roots(raw_paths, cfg)

    if args.stage in ("save", "all"):
        model = CodecLightningModule(cfg=cfg).to(args.device).eval()
        state = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
        state_dict = state.get("state_dict", state)
        missing, unexpected = model.load_state_dict(state_dict, strict=True)
        if len(missing) or len(unexpected):
            print(f"[Warning] Missing keys: {len(missing)}, Unexpected keys: {len(unexpected)}")

        save_stats = run_save_stage(args, cfg, model, input_paths, eval_dir, gt_out_dir, pred_out_dir, manifest_path)
    else:
        save_stats = load_save_stage_stats_if_any(eval_dir)

    if args.stage in ("metrics", "all"):
        if not manifest_path.is_file():
            raise FileNotFoundError(f"Manifest not found at {manifest_path}. Run with --stage save first to generate 16k GT/PRED and a manifest, optionally providing --gt_out_dir.")
        audio_metrics = run_metrics_stage(args, manifest_path, eval_dir)
    else:
        audio_metrics = {}

    final_metrics = {}
    final_metrics.update(audio_metrics)
    final_metrics.update({
        "codebook_perplexity": save_stats.get("codebook_perplexity"),
        "codebook_utilization": save_stats.get("codebook_utilization"),
        "avg_sim": save_stats.get("avg_sim_mean"),
    })

    with open(eval_dir / "metrics.json", "w") as f:
        json.dump(final_metrics, f, indent=2)

    print(json.dumps(final_metrics, indent=2))


if __name__ == "__main__":
    main()