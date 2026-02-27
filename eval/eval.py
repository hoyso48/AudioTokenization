
import os
import sys
import json
import math
import argparse
import shutil
import time
import inspect
from collections import OrderedDict
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Iterable, Set, cast
from contextlib import nullcontext

EVAL_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = EVAL_ROOT.parent
DTMAE_ROOT = PROJECT_ROOT / "DTMAE"
SPEAKER_VERIFICATION_ROOT = EVAL_ROOT / "speaker_verification"
FAIRSEQ_PYTHON_ROOT = EVAL_ROOT / "fairseq"
S3PRL_ROOT = EVAL_ROOT / "s3prl"

for path in (FAIRSEQ_PYTHON_ROOT, S3PRL_ROOT, SPEAKER_VERIFICATION_ROOT, EVAL_ROOT, PROJECT_ROOT, DTMAE_ROOT):
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
from transformers import Wav2Vec2Processor, HubertForCTC

from DTMAE.lightning_module import (
    CodecLightningModule,
    CodebookPerplexity,
    CodebookUtilization,
)

from jiwer import wer as jiwer_wer  # type: ignore
from verification import init_model as init_spk_model
from UTMOS import UTMOSScore  # type: ignore
from mel_cepstral_distance import compare_audio_files as mcd_compare

try:
    import utmosv2  # type: ignore
except Exception:
    utmosv2 = None


ALLOWED_AUDIO_EXTS = {".wav", ".flac"}

DEFAULT_METRICS = [
    "stoi",
    "pesq_wb",
    "pesq_nb",
    "si_snr",
    "si_sdr",
    "speaker_similarity",
    "mcd",
    "wer",
    "utmos",
    "utmos_v2",
]

SUPPORTED_METRICS = set(DEFAULT_METRICS + ["wer_sentence_avg"])

def infer_codebook_size_from_cfg(cfg) -> int:
    """
    New configs store quantizer info in cfg.model.quantizer.params.
    Legacy configs stored codebook_size in cfg.model.codec_decoder.codebook_size.
    """
    # Legacy fallback
    try:
        legacy = cfg.model.codec_decoder.codebook_size
        if legacy is not None:
            return int(legacy)
    except Exception:
        pass

    if not hasattr(cfg.model, "quantizer"):
        raise RuntimeError("Config missing cfg.model.quantizer (run utils/update_legacy_config.py).")
    qparams = cfg.model.quantizer.params

    # Mirror DTMAE.lightning_module.CodecLightningModule.construct_metrics logic.
    if "codebook_size" in qparams:
        return int(qparams.codebook_size)
    if "inference_levels" in qparams:
        inf_levels = qparams.inference_levels
        if isinstance(inf_levels, (list, tuple)) or (hasattr(inf_levels, "__iter__") and not isinstance(inf_levels, (int, str))):
            size = 1
            for L in inf_levels:
                size *= int(L)
            return int(size)
        return int(inf_levels) ** int(qparams.codebook_dim)
    if "train_levels" in qparams and "codebook_dim" in qparams:
        return int(max(qparams.train_levels)) ** int(qparams.codebook_dim)
    if "levels" in qparams:
        size = 1
        for L in qparams.levels:
            size *= int(L)
        return int(size)

    # Conservative fallback: match previous default.
    return 16384



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


def _is_within_dir(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except Exception:
        return False


def maybe_cleanup_audio_dirs(
    *,
    eval_dir: Path,
    gt_out_dir: Optional[Path],
    pred_out_dir: Optional[Path],
    keep_audio: bool,
    gt_was_default: bool,
    pred_was_default: bool,
) -> None:
    """
    Delete generated audio directories after evaluation to save disk.

    Safety rules:
    - Only delete if keep_audio=False
    - Only delete directories that are inside eval_dir
    - Only delete directories that were created by default paths (unless user explicitly wants to keep audio)
    """
    if keep_audio:
        return

    to_delete: List[Path] = []
    if gt_out_dir is not None and gt_was_default and _is_within_dir(gt_out_dir, eval_dir):
        to_delete.append(gt_out_dir)
    if pred_out_dir is not None and pred_was_default and _is_within_dir(pred_out_dir, eval_dir):
        to_delete.append(pred_out_dir)

    for p in to_delete:
        if p.exists() and p.is_dir():
            shutil.rmtree(p)
            print(f"[Cleanup] Deleted audio directory: {p}")


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


def apply_cfg_overrides(cfg, overrides: Optional[List[str]]):
    if not overrides:
        return cfg
    override_conf = OmegaConf.from_dotlist(overrides)
    return OmegaConf.merge(cfg, override_conf)


def patch_legacy_dtp_state_dict(state_dict: Dict[str, torch.Tensor]) -> None:
    legacy_keys = ["dtp.log_tau", "dtp.r_ema", "dtp.steps"]
    if not all(k in state_dict for k in legacy_keys):
        return
    log_tau = state_dict.pop("dtp.log_tau")
    tau = torch.exp(log_tau)
    state_dict["dtp.tau_train"] = tau.clone()
    state_dict["dtp.tau_eval"] = tau.clone()

    r_ema = state_dict.pop("dtp.r_ema")
    state_dict["dtp.r_ema_train"] = r_ema.clone()
    state_dict["dtp.r_ema_eval"] = r_ema.clone()

    steps = state_dict.pop("dtp.steps")
    state_dict["dtp.steps_train"] = steps.clone()
    state_dict["dtp.steps_eval"] = steps.clone()


def patch_legacy_norm_state_dict(
    state_dict: Dict[str, torch.Tensor],
    model_state_dict: Dict[str, torch.Tensor],
) -> Dict[str, int]:
    """
    Compat patch for older checkpoints where RMSNorm parameters were saved as
    `<module>.weight` (no bias), while current code expects
    `<module>.norm.weight` / `<module>.norm.bias`.
    """
    remapped_norm_weights = 0
    added_norm_biases = 0
    added_optional_defaults = 0

    for old_key in list(state_dict.keys()):
        if not old_key.endswith(".weight"):
            continue

        stem = old_key[: -len(".weight")]
        new_weight_key = f"{stem}.norm.weight"
        if new_weight_key not in model_state_dict or new_weight_key in state_dict:
            continue

        state_dict[new_weight_key] = state_dict.pop(old_key)
        remapped_norm_weights += 1

        new_bias_key = f"{stem}.norm.bias"
        if new_bias_key in model_state_dict and new_bias_key not in state_dict:
            state_dict[new_bias_key] = torch.zeros_like(model_state_dict[new_bias_key])
            added_norm_biases += 1

    # Older checkpoints may not contain this currently-unused projection.
    for key in ("encoder.proj.weight", "encoder.proj.bias"):
        if key in model_state_dict and key not in state_dict:
            state_dict[key] = model_state_dict[key].clone()
            added_optional_defaults += 1

    return {
        "remapped_norm_weights": remapped_norm_weights,
        "added_norm_biases": added_norm_biases,
        "added_optional_defaults": added_optional_defaults,
    }


def resolve_with_dataset_roots(paths: List[str], cfg) -> List[str]:
    roots: List[Path] = []
    datasets_cfg = cfg.preprocess.datasets
    if hasattr(datasets_cfg, "LibriSpeech"):
        roots.append(Path(datasets_cfg.LibriSpeech.root))
    if hasattr(datasets_cfg, "LibriTTS"):
        roots.append(Path(datasets_cfg.LibriTTS.root))
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


def parse_metrics_arg(metrics_arg: Optional[str]) -> List[str]:
    if metrics_arg is None:
        return list(DEFAULT_METRICS)

    raw = metrics_arg.strip()
    if not raw:
        return list(DEFAULT_METRICS)
    if raw.lower() == "all":
        return list(DEFAULT_METRICS)

    aliases = {
        "spk": "speaker_similarity",
        "spk_sim": "speaker_similarity",
        "speaker": "speaker_similarity",
        "utmosv2": "utmos_v2",
    }

    parsed: List[str] = []
    for token in raw.split(","):
        key = token.strip().lower()
        if not key:
            continue
        key = aliases.get(key, key)
        if key not in SUPPORTED_METRICS:
            supported = ", ".join(sorted(SUPPORTED_METRICS))
            raise ValueError(f"Unknown metric '{token}'. Supported metrics: {supported}")
        if key not in parsed:
            parsed.append(key)

    if not parsed:
        return list(DEFAULT_METRICS)

    # 'wer_sentence_avg' is derived from the same ASR pass as 'wer'.
    if "wer_sentence_avg" in parsed and "wer" not in parsed:
        parsed.append("wer")

    return parsed


class UTMOSv2Score:
    def __init__(self, device: str):
        if utmosv2 is None:
            raise RuntimeError(
                "utmosv2 is not installed. Install with either: "
                "(1) bash setup_conda_envs_minimal.sh --skip_train, or "
                "(2) pip install utmosv2"
            )
        self.device = device
        self.model = utmosv2.create_model(pretrained=True, device=device)
        sig = inspect.signature(self.model.predict)
        params = set(sig.parameters.keys())
        self._predict_supports_input_path = "input_path" in params
        self._predict_supports_data = "data" in params

    def score(self, pred_path: str, wavs: Optional[torch.Tensor] = None) -> torch.Tensor:
        if self._predict_supports_input_path:
            pred = self.model.predict(
                input_path=pred_path,
                device=self.device,
                num_workers=0,
                batch_size=1,
                verbose=False,
            )
        elif self._predict_supports_data:
            if wavs is None:
                raise RuntimeError("UTMOSv2 predict(data=...) path requires waveform tensor fallback.")
            if wavs.dim() == 1:
                payload = wavs.unsqueeze(0)
            elif wavs.dim() == 2:
                payload = wavs
            elif wavs.dim() == 3:
                if wavs.size(1) != 1:
                    raise ValueError("UTMOSv2Score expects mono audio for 3D input ([B,1,T]).")
                payload = wavs.squeeze(1)
            else:
                raise ValueError("UTMOSv2Score expects input tensor with <= 3 dimensions.")

            payload = payload.detach().to(torch.float32).cpu()
            pred = self.model.predict(
                data=payload,
                sr=16000,
                device=self.device,
                num_workers=0,
                batch_size=max(1, int(payload.size(0))),
                verbose=False,
            )
        else:
            sig = str(inspect.signature(self.model.predict))
            raise RuntimeError(f"Unsupported UTMOSv2 predict signature: {sig}")

        if torch.is_tensor(pred):
            out = pred.detach().cpu().to(torch.float32)
        elif isinstance(pred, np.ndarray):
            out = torch.from_numpy(pred).to(torch.float32)
        elif isinstance(pred, list):
            vals: List[float] = []
            for item in pred:
                if isinstance(item, dict) and "score" in item:
                    vals.append(float(item["score"]))
                elif isinstance(item, (int, float, np.floating, np.integer)):
                    vals.append(float(item))
                else:
                    raise RuntimeError(f"Unsupported UTMOSv2 list item type: {type(item)!r}")
            out = torch.tensor(vals, dtype=torch.float32)
        else:
            out = torch.tensor([float(pred)], dtype=torch.float32)
        return out


# Helpers for metrics stage

def load_audio_mono_16k(path: str) -> torch.Tensor:
    wav, sr = torchaudio.load(path)
    if sr != 16000:
        wav = torchaudio.transforms.Resample(sr, 16000)(wav)
    wav = wav[:1, :]
    return wav


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
        wav = cast(torch.Tensor, b["wav"])
        return {
            "wav": wav.unsqueeze(0),
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

    codebook_size = infer_codebook_size_from_cfg(cfg)
    codebook_perplexity = CodebookPerplexity(codebook_size=codebook_size)
    codebook_utilization = CodebookUtilization(codebook_size=codebook_size)

    prediction_total_samples = 0
    prediction_total_seconds = 0.0
    prediction_num_items = 0
    prediction_total_samples_raw = 0
    prediction_total_seconds_raw = 0.0
    prediction_num_items_raw = 0
    dtp_avg_r_vals: List[float] = []
    dtp_tau_used_vals: List[float] = []

    throughput_warmup_items = max(0, int(getattr(args, "throughput_warmup_items", 0)))
    use_cuda_timing = bool(torch.cuda.is_available() and str(args.device).startswith("cuda"))
    sync_device = torch.device(str(args.device)) if use_cuda_timing else None

    if gt_out_dir is not None:
        gt_out_dir.mkdir(parents=True, exist_ok=True)
    pred_out_dir.mkdir(parents=True, exist_ok=True)
    ensure_parent_dir(manifest_path)

    device_type = 'cuda' if ('cuda' in str(args.device)) else 'cpu'
    from contextlib import nullcontext

    with open(manifest_path, "w") as mf:
        for batch in tqdm(dl, total=len(ds)):
            if use_cuda_timing and sync_device is not None:
                torch.cuda.synchronize(sync_device)
            pred_t0 = time.perf_counter()
            wav = batch["wav"]
            assert isinstance(wav, torch.Tensor)
            wav = wav.to(args.device)
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

            y_ref = cast(torch.Tensor, out['gt_wav'])
            y_rec = cast(torch.Tensor, out['gen_wav'])
            vq_code = out.get('vq_code', None)
            avg_r_val = out.get('avg_r', None)
            tau_used_val = out.get('tau_used', None)

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

            if torch.is_tensor(vq_code):
                vq_code_t = cast(torch.Tensor, vq_code)
                codebook_perplexity.update(vq_code_t.detach().cpu())
                codebook_utilization.update(vq_code_t.detach().cpu())

            if use_cuda_timing and sync_device is not None:
                torch.cuda.synchronize(sync_device)
            pred_dt = max(0.0, time.perf_counter() - pred_t0)
            prediction_num_items_raw += 1
            prediction_total_samples_raw += int(cut_len)
            prediction_total_seconds_raw += float(pred_dt)

            throughput_excluded = prediction_num_items_raw <= throughput_warmup_items
            if not throughput_excluded:
                prediction_num_items += 1
                prediction_total_samples += int(cut_len)
                prediction_total_seconds += float(pred_dt)

            prediction_sps = (float(cut_len) / pred_dt) if pred_dt > 0.0 else None
            prediction_ips = (1.0 / pred_dt) if pred_dt > 0.0 else None

            dtp_avg_r = None
            if avg_r_val is not None:
                if torch.is_tensor(avg_r_val):
                    avg_r_t = cast(torch.Tensor, avg_r_val)
                    dtp_avg_r = float(avg_r_t.detach().cpu().item())
                elif isinstance(avg_r_val, (int, float, np.floating, np.integer)):
                    dtp_avg_r = float(avg_r_val)
                else:
                    dtp_avg_r = None
                if dtp_avg_r is not None:
                    dtp_avg_r_vals.append(dtp_avg_r)

            dtp_tau_used = None
            if tau_used_val is not None:
                if torch.is_tensor(tau_used_val):
                    tau_used_t = cast(torch.Tensor, tau_used_val)
                    dtp_tau_used = float(tau_used_t.detach().cpu().item())
                elif isinstance(tau_used_val, (int, float, np.floating, np.integer)):
                    dtp_tau_used = float(tau_used_val)
                else:
                    dtp_tau_used = None
                if dtp_tau_used is not None:
                    dtp_tau_used_vals.append(dtp_tau_used)

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
                "prediction_elapsed_sec": float(pred_dt),
                "prediction_samples": int(cut_len),
                "prediction_samples_per_sec": prediction_sps,
                "prediction_items_per_sec": prediction_ips,
                "throughput_warmup_excluded": bool(throughput_excluded),
                "dtp_avg_r": dtp_avg_r,
                "dtp_tau_used": dtp_tau_used,
            }
            mf.write(json.dumps(record) + "\n")

    stats = {
        "codebook_perplexity": None,
        "codebook_utilization": None,
        "throughput_warmup_items": int(throughput_warmup_items),
        "prediction_num_items_raw": int(prediction_num_items_raw),
        "prediction_total_samples_raw": int(prediction_total_samples_raw),
        "prediction_total_seconds_raw": float(prediction_total_seconds_raw),
        "prediction_num_items": int(prediction_num_items),
        "prediction_total_samples": int(prediction_total_samples),
        "prediction_total_seconds": float(prediction_total_seconds),
        "prediction_samples_per_sec": (
            float(prediction_total_samples / prediction_total_seconds)
            if prediction_total_seconds > 0.0
            else None
        ),
        "prediction_items_per_sec": (
            float(prediction_num_items / prediction_total_seconds)
            if prediction_total_seconds > 0.0
            else None
        ),
        "dtp_avg_r_mean": float(np.mean(dtp_avg_r_vals)) if len(dtp_avg_r_vals) > 0 else None,
        "dtp_avg_r_std": float(np.std(dtp_avg_r_vals)) if len(dtp_avg_r_vals) > 0 else None,
        "dtp_tau_used_mean": float(np.mean(dtp_tau_used_vals)) if len(dtp_tau_used_vals) > 0 else None,
        "dtp_tau_used_std": float(np.std(dtp_tau_used_vals)) if len(dtp_tau_used_vals) > 0 else None,
    }
    stats["codebook_perplexity"] = float(codebook_perplexity.compute().item())
    stats["codebook_utilization"] = float(codebook_utilization.compute().item())

    with open(eval_dir / "save_stage_stats.json", "w") as f:
        json.dump(stats, f, indent=2)

    return stats


def run_metrics_stage(
    args,
    manifest_path: Path,
    eval_dir: Path,
    selected_metrics: Set[str],
) -> Dict[str, Optional[float]]:
    selected = set(selected_metrics)
    want_stoi = "stoi" in selected
    want_pesq_wb = "pesq_wb" in selected
    want_pesq_nb = "pesq_nb" in selected
    want_si_snr = "si_snr" in selected
    want_si_sdr = "si_sdr" in selected
    want_spk = "speaker_similarity" in selected
    want_mcd = "mcd" in selected
    want_wer = "wer" in selected
    want_utmos = "utmos" in selected
    want_utmos_v2 = "utmos_v2" in selected

    processor = None
    asr_model = None
    if want_wer:
        processor = Wav2Vec2Processor.from_pretrained("facebook/hubert-large-ls960-ft")
        asr_model = HubertForCTC.from_pretrained("facebook/hubert-large-ls960-ft").to(args.device).eval()

    spk_model = None
    if want_spk:
        spk_ckpt = EVAL_ROOT / "wavlm_large_finetune.pth"
        spk_model = init_spk_model("wavlm_large", str(spk_ckpt))
        spk_model = spk_model.to(args.device).eval()

    utmos_model = UTMOSScore(device=args.device) if want_utmos else None
    utmos_v2_model = UTMOSv2Score(device=args.device) if want_utmos_v2 else None

    stoi_metric = ShortTimeObjectiveIntelligibility(fs=16000, extended=False) if want_stoi else None
    pesq_wb_metric = PerceptualEvaluationSpeechQuality(fs=16000, mode="wb") if want_pesq_wb else None
    pesq_nb_metric = PerceptualEvaluationSpeechQuality(fs=8000, mode="nb") if want_pesq_nb else None
    si_snr_metric = ScaleInvariantSignalNoiseRatio() if want_si_snr else None
    si_sdr_metric = ScaleInvariantSignalDistortionRatio() if want_si_sdr else None
    to_8k = torchaudio.transforms.Resample(16000, 8000) if want_pesq_nb else None

    stoi_vals: List[Optional[float]] = []
    pesq_wb_vals: List[Optional[float]] = []
    pesq_nb_vals: List[Optional[float]] = []
    si_snr_vals: List[Optional[float]] = []
    si_sdr_vals: List[Optional[float]] = []
    spk_sim_vals: List[Optional[float]] = []
    mcd_vals: List[Optional[float]] = []
    wer_frac_vals: List[Optional[float]] = []
    utmos_vals: List[Optional[float]] = []
    utmos_v2_vals: List[Optional[float]] = []

    # For corpus-level WER (not mean of per-sample)
    corpus_refs: List[str] = []
    corpus_hyps: List[str] = []

    with open(manifest_path, "r") as f:
        lines = f.read().splitlines()
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
                raise RuntimeError(
                    "GT is not saved and original is not 16k WAV. "
                    "Re-run with --stage save and provide --gt_out_dir to save GT at 16k WAV."
                )
            gt_path = orig_path

        if not (gt_path and pred_path):
            print(f"[Warning] Skipping entry due to missing paths: {rec}")
            continue

        gt_wav = load_audio_mono_16k(gt_path)
        pred_wav = load_audio_mono_16k(pred_path)
        T = min(gt_wav.size(1), pred_wav.size(1))
        gt = gt_wav[:, :T]
        pr = pred_wav[:, :T]

        if want_stoi and stoi_metric is not None:
            stoi_metric.reset()
            stoi_metric.update(pr.unsqueeze(0), gt.unsqueeze(0))
            stoi_val = float(stoi_metric.compute().item())
            rec["stoi"] = stoi_val
            stoi_vals.append(stoi_val)

        if want_pesq_wb and pesq_wb_metric is not None:
            pesq_wb_metric.reset()
            pesq_wb_metric.update(pr.unsqueeze(0), gt.unsqueeze(0))
            pesq_wb_val = float(pesq_wb_metric.compute().item())
            rec["pesq_wb"] = pesq_wb_val
            pesq_wb_vals.append(pesq_wb_val)

        if want_pesq_nb and pesq_nb_metric is not None and to_8k is not None:
            pr8 = to_8k(pr)
            gt8 = to_8k(gt)
            pesq_nb_metric.reset()
            pesq_nb_metric.update(pr8.unsqueeze(0), gt8.unsqueeze(0))
            pesq_nb_val = float(pesq_nb_metric.compute().item())
            rec["pesq_nb"] = pesq_nb_val
            pesq_nb_vals.append(pesq_nb_val)

        if want_si_snr and si_snr_metric is not None:
            si_snr_metric.reset()
            si_snr_metric.update(pr.unsqueeze(0), gt.unsqueeze(0))
            si_snr_val = float(si_snr_metric.compute().item())
            rec["si_snr"] = si_snr_val
            si_snr_vals.append(si_snr_val)

        if want_si_sdr and si_sdr_metric is not None:
            si_sdr_metric.reset()
            si_sdr_metric.update(pr.unsqueeze(0), gt.unsqueeze(0))
            si_sdr_val = float(si_sdr_metric.compute().item())
            rec["si_sdr"] = si_sdr_val
            si_sdr_vals.append(si_sdr_val)

        if want_spk and spk_model is not None:
            with torch.inference_mode():
                emb_ref = spk_model(gt.to(args.device))
                emb_rec = spk_model(pr.to(args.device))
            spk_sim_val = float(F.cosine_similarity(emb_ref, emb_rec).mean().item())
            rec["speaker_similarity"] = spk_sim_val
            spk_sim_vals.append(spk_sim_val)

        if want_mcd and mcd_compare is not None:
            mcd, _ = mcd_compare(gt_path, pred_path)
            mcd_val = float(mcd) if mcd is not None and not math.isnan(mcd) else None
            rec["mcd"] = mcd_val
            mcd_vals.append(mcd_val)

        if want_utmos and utmos_model is not None:
            utmos_tensor = utmos_model.score(pr.to(args.device))
            utmos_val = float(utmos_tensor.squeeze().item())
            rec["utmos"] = utmos_val
            utmos_vals.append(utmos_val)

        if want_utmos_v2 and utmos_v2_model is not None:
            utmos_v2_tensor = utmos_v2_model.score(pred_path=pred_path, wavs=pr)
            utmos_v2_val = float(utmos_v2_tensor.squeeze().item())
            rec["utmos_v2"] = utmos_v2_val
            utmos_v2_vals.append(utmos_v2_val)

        if want_wer:
            wer_frac = None
            gt_text: Optional[str] = None
            asr_text: Optional[str] = None
            if transcript_path and Path(transcript_path).is_file() and processor is not None and asr_model is not None:
                with open(transcript_path, "r") as tf:
                    ref_line = None
                    file_id = Path(gt_path).stem
                    for tline in tf:
                        if tline.startswith(file_id + " "):
                            ref_line = tline[len(file_id) + 1 :].strip()
                            break
                if ref_line is not None:
                    gt_text = ref_line
                    inputs = processor(
                        pr.squeeze().numpy(),
                        sampling_rate=16000,
                        return_tensors="pt",
                    ).input_values.to(args.device)
                    with torch.inference_mode():
                        logits = asr_model(inputs).logits
                        predicted_ids = torch.argmax(logits, dim=-1)
                        hyp_text = processor.decode(predicted_ids[0].detach().cpu())
                    asr_text = hyp_text
                    wer_frac = float(jiwer_wer(ref_line, hyp_text))
                    corpus_refs.append(ref_line)
                    corpus_hyps.append(hyp_text)

            rec["wer_fraction"] = wer_frac
            rec["wer"] = (wer_frac * 100.0) if wer_frac is not None else None
            rec["gt_text"] = gt_text
            rec["asr_text"] = asr_text
            wer_frac_vals.append(wer_frac)

        updated_records.append(rec)

    with open(manifest_path, "w") as mf:
        for rec in updated_records:
            mf.write(json.dumps(rec) + "\n")

    def mean_ignore_none(values: Iterable[Optional[float]]) -> Optional[float]:
        arr = [v for v in values if v is not None]
        return float(np.mean(arr)) if len(arr) > 0 else None

    audio_metrics_path = eval_dir / "audio_metrics.json"
    metrics: Dict[str, Optional[float]] = {}
    if audio_metrics_path.is_file():
        try:
            with open(audio_metrics_path, "r") as f:
                prev_metrics = json.load(f)
            if isinstance(prev_metrics, dict):
                metrics.update(prev_metrics)
        except Exception:
            pass

    metrics["count"] = int(len(updated_records))

    if want_stoi:
        metrics["stoi"] = mean_ignore_none(stoi_vals)
    if want_si_snr:
        metrics["si_snr"] = mean_ignore_none(si_snr_vals)
    if want_si_sdr:
        metrics["si_sdr"] = mean_ignore_none(si_sdr_vals)
    if want_pesq_wb:
        metrics["pesq_wb"] = mean_ignore_none(pesq_wb_vals)
    if want_pesq_nb:
        metrics["pesq_nb"] = mean_ignore_none(pesq_nb_vals)
    if want_spk:
        metrics["speaker_similarity"] = mean_ignore_none(spk_sim_vals)
    if want_mcd:
        metrics["mcd"] = mean_ignore_none(mcd_vals)
    if want_utmos:
        metrics["utmos"] = mean_ignore_none(utmos_vals)
    if want_utmos_v2:
        metrics["utmos_v2"] = mean_ignore_none(utmos_v2_vals)
    if want_wer:
        sentence_avg_wer_frac = mean_ignore_none(wer_frac_vals)
        corpus_wer_frac: Optional[float] = None
        if len(corpus_refs) > 0 and len(corpus_refs) == len(corpus_hyps):
            corpus_wer_frac = float(jiwer_wer(corpus_refs, corpus_hyps))
        metrics["wer"] = (corpus_wer_frac * 100.0) if corpus_wer_frac is not None else None
        metrics["wer_sentence_avg"] = (
            (sentence_avg_wer_frac * 100.0) if sentence_avg_wer_frac is not None else None
        )

    with open(audio_metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    return metrics


def load_save_stage_stats_if_any(eval_dir: Path) -> Dict[str, Optional[float]]:
    stats_path = eval_dir / "save_stage_stats.json"
    if not stats_path.is_file():
        return {}
    with open(stats_path, "r") as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser(description="Evaluate BigCodec SSL with two-stage pipeline (save -> metrics). Batch size is always 1.")
    parser.add_argument("--input", type=str, required=True, help="Directory (recursive) or .txt filelist or single audio file for GT inputs")
    parser.add_argument("--run_dir", type=str, required=True, help="Run directory containing hydra/config.yaml and pl_log/last.ckpt")
    parser.add_argument("--output_dir", type=str, default=None, help="Directory to store eval artifacts (default: <run_dir>/eval)")
    parser.add_argument("--stage", type=str, choices=["save", "metrics", "all"], default="all")
    parser.add_argument("--gt_out_dir", type=str, default=None, help="Where to save 16k WAV GTs during save stage")
    parser.add_argument("--pred_out_dir", type=str, default=None, help="Where to save 16k WAV predictions (default: run_dir/eval/pred_16k)")
    parser.add_argument("--manifest", type=str, default=None, help="Path to manifest.jsonl (default: run_dir/eval/manifest.jsonl)")
    parser.add_argument("--length_mode", type=str, choices=["pad", "truncate"], default="pad", help="How to make length a multiple of cfg.dataset.multiple_of")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--throughput_warmup_items",
        type=int,
        default=5,
        help="Exclude first N save-stage iterations from throughput aggregation (default: 5).",
    )
    parser.add_argument(
        "--metrics",
        type=str,
        default="all",
        help=(
            "Comma-separated metric names to compute during metrics stage. "
            "Use 'all' for the default metric set. "
            "Supported: stoi,pesq_wb,pesq_nb,si_snr,si_sdr,speaker_similarity,mcd,wer,wer_sentence_avg,utmos,utmos_v2"
        ),
    )
    parser.add_argument(
        "--cfg_override",
        action="append",
        default=None,
        help="Hydra-style dotlist override applied after loading hydra/config.yaml "
        "(e.g., --cfg_override model.resampler.dtp_params.r=0.4). "
        "Use multiple flags for multiple overrides.",
    )
    parser.add_argument(
        "--keep_audio",
        action="store_true",
        help="Keep generated gt_16k/pred_16k audio directories. Default behavior is to delete them after metrics are computed.",
    )
    args = parser.parse_args()
    selected_metrics = parse_metrics_arg(args.metrics)

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
    cfg = apply_cfg_overrides(cfg, args.cfg_override)

    eval_dir = Path(args.output_dir).resolve() if args.output_dir else (run_dir / "eval")
    eval_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = Path(args.manifest) if args.manifest else (eval_dir / "manifest.jsonl")
    pred_was_default = args.pred_out_dir is None
    gt_was_default = args.gt_out_dir is None
    pred_out_dir = Path(args.pred_out_dir).resolve() if args.pred_out_dir else (eval_dir / "pred_16k")
    gt_out_dir = (
        Path(args.gt_out_dir).resolve()
        if args.gt_out_dir
        else (eval_dir / "gt_16k" if args.stage in ("save", "all") else None)
    )
    save_stage_stats_path = eval_dir / "save_stage_stats.json"
    audio_metrics_path = eval_dir / "audio_metrics.json"
    final_metrics_path = eval_dir / "metrics.json"

    raw_paths = parse_input_paths(args.input)
    input_paths = resolve_with_dataset_roots(raw_paths, cfg)
    resolved_input_count = len(input_paths)

    metadata = OrderedDict()
    metadata["run_dir"] = str(run_dir)
    metadata["stage"] = args.stage
    metadata["device"] = str(args.device)
    metadata["input"] = str(args.input)
    metadata["resolved_input_count"] = resolved_input_count
    metadata["length_mode"] = args.length_mode
    metadata["throughput_warmup_items"] = int(args.throughput_warmup_items)
    metadata["manifest_path"] = str(manifest_path)
    metadata["pred_out_dir"] = str(pred_out_dir)
    metadata["gt_out_dir"] = str(gt_out_dir) if gt_out_dir is not None else None
    metadata["cfg_overrides"] = list(args.cfg_override) if args.cfg_override else []
    metadata["selected_metrics"] = list(selected_metrics)
    metadata["save_stage_stats_path"] = str(save_stage_stats_path)
    metadata["audio_metrics_path"] = str(audio_metrics_path)
    metadata["final_metrics_path"] = str(final_metrics_path)

    if args.stage in ("save", "all"):
        model = CodecLightningModule(cfg=cfg).to(args.device).eval()
        state = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
        state_dict = state.get("state_dict", state)
        patch_legacy_dtp_state_dict(state_dict)
        compat_stats = patch_legacy_norm_state_dict(state_dict, model.state_dict())
        if any(compat_stats.values()):
            print(
                "[Compat] Applied legacy checkpoint patch: "
                f"remapped_norm_weights={compat_stats['remapped_norm_weights']}, "
                f"added_norm_biases={compat_stats['added_norm_biases']}, "
                f"added_optional_defaults={compat_stats['added_optional_defaults']}"
            )

        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if len(missing) or len(unexpected):
            print(f"[Warning] Missing keys: {len(missing)}, Unexpected keys: {len(unexpected)}")
            if missing:
                print(f"[Warning] Missing examples: {missing[:8]}")
            if unexpected:
                print(f"[Warning] Unexpected examples: {unexpected[:8]}")

        save_stats = run_save_stage(args, cfg, model, input_paths, eval_dir, gt_out_dir, pred_out_dir, manifest_path)
        save_stats_source = "computed"
    else:
        save_stats = load_save_stage_stats_if_any(eval_dir)
        save_stats_source = "loaded" if save_stats else "missing"

    metadata["save_stage_stats_source"] = save_stats_source

    if args.stage in ("metrics", "all"):
        if not manifest_path.is_file():
            raise FileNotFoundError(f"Manifest not found at {manifest_path}. Run with --stage save first to generate 16k GT/PRED and a manifest, optionally providing --gt_out_dir.")
        audio_metrics = run_metrics_stage(args, manifest_path, eval_dir, set(selected_metrics))
        metrics_stage_source = "computed"
    else:
        audio_metrics = {}
        metrics_stage_source = "skipped"

    metadata["audio_metrics_source"] = metrics_stage_source

    final_metrics = OrderedDict(metadata)
    final_metrics.update(audio_metrics)
    final_metrics["codebook_perplexity"] = save_stats.get("codebook_perplexity")
    final_metrics["codebook_utilization"] = save_stats.get("codebook_utilization")
    final_metrics["prediction_samples_per_sec"] = save_stats.get("prediction_samples_per_sec")
    final_metrics["prediction_items_per_sec"] = save_stats.get("prediction_items_per_sec")
    final_metrics["prediction_total_samples"] = save_stats.get("prediction_total_samples")
    final_metrics["prediction_total_seconds"] = save_stats.get("prediction_total_seconds")
    final_metrics["throughput_warmup_items"] = save_stats.get("throughput_warmup_items")
    final_metrics["dtp_avg_r_mean"] = save_stats.get("dtp_avg_r_mean")
    final_metrics["dtp_avg_r_std"] = save_stats.get("dtp_avg_r_std")
    final_metrics["dtp_tau_used_mean"] = save_stats.get("dtp_tau_used_mean")
    final_metrics["dtp_tau_used_std"] = save_stats.get("dtp_tau_used_std")

    with open(final_metrics_path, "w") as f:
        json.dump(final_metrics, f, indent=2)

    print(json.dumps(final_metrics, indent=2))

    # Default: remove generated audio after metrics are computed (disk-saving).
    # Only applies when metrics are computed in this invocation (stage=metrics/all).
    if args.stage in ("metrics", "all"):
        maybe_cleanup_audio_dirs(
            eval_dir=eval_dir,
            gt_out_dir=gt_out_dir,
            pred_out_dir=pred_out_dir,
            keep_audio=bool(args.keep_audio),
            gt_was_default=bool(gt_was_default),
            pred_was_default=bool(pred_was_default),
        )


if __name__ == "__main__":
    main()
