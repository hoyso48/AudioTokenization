#!/usr/bin/env python3

"""ARCH-compatible semantic evaluation runner for codec checkpoints.

Design goal: keep evaluation behavior as close as possible to upstream ARCH,
while plugging in our codec model as the embedding extractor.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf


THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parent
DTMAE_ROOT = PROJECT_ROOT / "DTMAE"

for p in (PROJECT_ROOT, THIS_DIR, DTMAE_ROOT):
    p_str = str(p)
    if p_str not in sys.path:
        sys.path.insert(0, p_str)

try:
    from semantic_eval.parsers import SemanticRecord, parse_dataset, write_jsonl, write_label_map
except ModuleNotFoundError:
    from parsers import SemanticRecord, parse_dataset, write_jsonl, write_label_map

from DTMAE.lightning_module import CodecLightningModule  # noqa: E402


SUPPORTED_DATASETS = ("ravdess", "emovo", "audio_mnist", "slurp")


def _ensure_arch_import_path(arch_repo: Path) -> None:
    if not arch_repo.is_dir():
        raise FileNotFoundError(f"ARCH repo directory not found: {arch_repo}")
    p_str = str(arch_repo)
    if p_str not in sys.path:
        sys.path.insert(0, p_str)


def _parse_arch_commit(arch_repo: Path) -> str:
    try:
        out = subprocess.check_output(
            ["git", "-C", str(arch_repo), "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        return out
    except Exception:
        return "unknown"


def _resolve_dataset_root(data_root: Path, dataset_name: str) -> Path:
    candidate = data_root / dataset_name
    if candidate.exists():
        return candidate
    return data_root


def _build_manifests(datasets: Sequence[str], data_root: Path, output_dir: Path) -> Dict[str, Dict[str, int]]:
    manifests_root = output_dir / "manifests"
    label_maps_root = output_dir / "label_maps"
    summary: Dict[str, Dict[str, int]] = {}

    for dataset in datasets:
        records = parse_dataset(dataset, data_root)
        if dataset == "slurp":
            real_records = [r for r in records if "/slurp_real/" in r.path]
            if real_records:
                records = real_records

        out_dir = manifests_root / dataset
        write_jsonl(records, out_dir / "all.jsonl")
        write_label_map(records, label_maps_root / f"{dataset}.json")

        split_to_records: Dict[str, List[SemanticRecord]] = {"train": [], "devel": [], "test": []}
        for rec in records:
            if rec.split in split_to_records:
                split_to_records[rec.split].append(rec)

        for split in ("train", "devel", "test"):
            if len(split_to_records[split]) > 0:
                write_jsonl(split_to_records[split], out_dir / f"{split}.jsonl")

        summary[dataset] = {
            "num_samples": len(records),
            "num_train": len(split_to_records["train"]),
            "num_devel": len(split_to_records["devel"]),
            "num_test": len(split_to_records["test"]),
        }

    with (output_dir / "build_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return summary


def _patch_checkpoint_dtp_compat(state_dict: Dict[str, torch.Tensor]) -> None:
    compat_keys = ["dtp.log_tau", "dtp.r_ema", "dtp.steps"]
    if not all(k in state_dict for k in compat_keys):
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


def _patch_checkpoint_norm_compat(
    state_dict: Dict[str, torch.Tensor],
    model_state_dict: Dict[str, torch.Tensor],
) -> Dict[str, int]:
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

    for key in ("encoder.proj.weight", "encoder.proj.bias"):
        if key in model_state_dict and key not in state_dict:
            state_dict[key] = model_state_dict[key].clone()
            added_optional_defaults += 1

    return {
        "remapped_norm_weights": remapped_norm_weights,
        "added_norm_biases": added_norm_biases,
        "added_optional_defaults": added_optional_defaults,
    }


def _load_codec_model(run_dir: Path, device: str) -> Tuple[CodecLightningModule, object]:
    cfg_path = run_dir / "hydra" / "config.yaml"
    ckpt_path = run_dir / "pl_log" / "last.ckpt"
    if not cfg_path.is_file():
        raise FileNotFoundError(f"Config not found: {cfg_path}")
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    cfg = OmegaConf.load(str(cfg_path))
    model = CodecLightningModule(cfg=cfg).to(device).eval()

    state = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    state_dict = state.get("state_dict", state)
    _patch_checkpoint_dtp_compat(state_dict)
    compat = _patch_checkpoint_norm_compat(state_dict, model.state_dict())
    if any(compat.values()):
        print(
            "[Compat] Applied checkpoint compatibility patch: "
            f"remapped_norm_weights={compat['remapped_norm_weights']}, "
            f"added_norm_biases={compat['added_norm_biases']}, "
            f"added_optional_defaults={compat['added_optional_defaults']}"
        )

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if len(missing) > 0:
        print(f"[Warning] Missing keys while loading checkpoint: {len(missing)}")
    if len(unexpected) > 0:
        print(f"[Warning] Unexpected keys while loading checkpoint: {len(unexpected)}")

    return model, cfg


def _parse_downsample_output(out):
    if isinstance(out, tuple):
        if len(out) == 5:
            vq_emb, position_ids, cu_seqlens, max_seqlen, _mask = out
            return vq_emb, position_ids, cu_seqlens, max_seqlen
        if len(out) == 4:
            vq_emb, position_ids, cu_seqlens, max_seqlen = out
            return vq_emb, position_ids, cu_seqlens, max_seqlen
        if len(out) > 0:
            return out[0], None, None, None
        raise ValueError("downsampler returned an empty tuple")
    return out, None, None, None


def _ensure_1d_tensor(audio) -> torch.Tensor:
    if isinstance(audio, np.ndarray):
        wav = torch.from_numpy(audio)
    elif isinstance(audio, torch.Tensor):
        wav = audio
    else:
        raise TypeError(f"Unsupported audio type: {type(audio)}")

    wav = wav.detach().to(torch.float32)
    if wav.dim() == 2:
        wav = wav.mean(dim=0)
    if wav.dim() != 1:
        raise ValueError(f"Expected 1D waveform, got shape={tuple(wav.shape)}")
    return wav


def _pad_to_multiple_1d(waveform: torch.Tensor, multiple_of: int) -> torch.Tensor:
    if multiple_of <= 0:
        return waveform
    length = int(waveform.shape[-1])
    rem = length % multiple_of
    if rem == 0:
        return waveform
    return F.pad(waveform, (0, multiple_of - rem))


def _build_arch_wrapper_class():
    from arch_eval import Model as ArchModel

    class CodecARCHWrapper(ArchModel):
        def __init__(
            self,
            codec_model,
            cfg,
            device: str,
            use_amp: bool = True,
            feature_source: str = "post_vq",
        ):
            super().__init__(codec_model)
            self.model = codec_model
            self.model.eval()
            self.cfg = cfg
            self.device = device
            self.use_amp = bool(use_amp)
            self.feature_source = str(feature_source)
            self.sample_rate = int(cfg.dataset.sample_rate)
            self.multiple_of = int(cfg.dataset.multiple_of)
            self._embedding_size = self._infer_embedding_size()

        def _extract_token_embeddings(self, audio_1d: torch.Tensor) -> torch.Tensor:
            wav = _pad_to_multiple_1d(audio_1d, self.multiple_of)
            wav = wav.unsqueeze(0).to(self.device)
            device_type = "cuda" if "cuda" in str(self.device) else "cpu"

            with torch.inference_mode():
                if self.use_amp and device_type == "cuda":
                    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                        return self._forward_tokens(wav)
                return self._forward_tokens(wav)

        def _forward_tokens(self, wav_b1t: torch.Tensor) -> torch.Tensor:
            vq_emb = self.model.encoder(wav_b1t.unsqueeze(1), level=1)

            if self.model.use_dtp:
                dtp_out = self.model.dtp(vq_emb)
                if isinstance(dtp_out, tuple) and len(dtp_out) == 4:
                    mask, _avg_r, _tau_used, _ = dtp_out
                else:
                    mask, _avg_r, _tau_used = dtp_out
                downsample_out = self.model.downsampler(vq_emb, mask)
            else:
                downsample_out = self.model.downsampler(vq_emb)

            vq_emb, position_ids, cu_seqlens, max_seqlen = _parse_downsample_output(downsample_out)
            vq_emb = self.model.encoder(
                vq_emb,
                position_ids=position_ids,
                cu_seqlens=cu_seqlens,
                max_seqlen=max_seqlen,
                level=2,
            )

            if self.feature_source == "pre_vq":
                features = vq_emb
            elif self.feature_source == "post_vq":
                features, _vq_code, _vq_loss = self.model.decoder(vq_emb, vq=True)
            else:
                raise ValueError(f"Unsupported feature_source: {self.feature_source}")

            return features.squeeze(0).to(torch.float32).detach().cpu()

        def _infer_embedding_size(self) -> int:
            dummy_len = max(self.multiple_of, self.sample_rate)
            dummy = torch.zeros(dummy_len, dtype=torch.float32)
            token_emb = self._extract_token_embeddings(dummy)
            if token_emb.dim() == 1:
                return int(token_emb.shape[0])
            return int(token_emb.shape[-1])

        def get_embeddings(self, audio, **kwargs):
            wav = _ensure_1d_tensor(audio)
            token_emb = self._extract_token_embeddings(wav)
            if token_emb.dim() == 1:
                return token_emb
            return token_emb.mean(dim=0)

        def get_sequence_embeddings(self, audio, **kwargs):
            wav = _ensure_1d_tensor(audio)
            return self._extract_token_embeddings(wav)

        def get_classification_embedding_size(self):
            return self._embedding_size

        def get_token_embedding_size(self):
            return self._embedding_size

        def get_sampling_rate(self):
            return self.sample_rate

    return CodecARCHWrapper


def _dataset_path_for_arch(data_root: Path, dataset: str) -> str:
    root = _resolve_dataset_root(data_root, dataset).resolve()
    path = str(root)
    if dataset == "emovo" and not path.endswith("/"):
        path = path + "/"
    return path


def _mean_std_dict(dicts: List[Dict[str, float]]) -> Tuple[Dict[str, float], Dict[str, float]]:
    keys = sorted(dicts[0].keys())
    mean_d: Dict[str, float] = {}
    std_d: Dict[str, float] = {}
    for k in keys:
        vals = [float(d[k]) for d in dicts]
        mean_d[k] = float(np.mean(vals))
        std_d[k] = float(np.std(vals))
    return mean_d, std_d


def _paths_match(a: str, b: str) -> bool:
    return str(Path(a).resolve()) == str(Path(b).resolve())


def _can_reuse_dataset_result(existing: Dict[str, object], dataset: str, dataset_path: str, args) -> bool:
    if str(existing.get("dataset", "")).strip().lower() != dataset:
        return False

    existing_path = str(existing.get("path", ""))
    if existing_path and not _paths_match(existing_path, dataset_path):
        return False

    if str(existing.get("mode", "")) != str(args.mode):
        return False

    if str(existing.get("feature_source", "post_vq")) != str(args.feature_source):
        return False

    try:
        if int(existing.get("n_iters", -1)) != int(args.n_iters):
            return False
    except Exception:
        return False

    metrics_mean = existing.get("metrics_mean")
    if not isinstance(metrics_mean, dict) or len(metrics_mean) == 0:
        return False

    return True


def _run_arch_evaluation(args, datasets: Sequence[str], data_root: Path, output_dir: Path) -> Dict[str, object]:
    _ensure_arch_import_path(Path(args.arch_repo).resolve())

    try:
        from arch_eval import AudioMNIST, EMOVO, RAVDESS, SLURP
    except ModuleNotFoundError as exc:
        name = getattr(exc, "name", "")
        arch_import_deps = {
            "pyannote.core",
            "pyannote.metrics",
            "xmltodict",
            "joblib",
            "yt_dlp",
        }
        if name in arch_import_deps or (name and name.startswith("pyannote")):
            raise ModuleNotFoundError(
                "ARCH import failed because ARCH runtime dependencies are missing. "
                "Install `pyannote.core`, `pyannote.metrics`, `xmltodict`, `joblib`, and `yt-dlp` "
                "in the current Python environment, "
                "or run bootstrap without --skip_env_setup to auto-install semantic_eval requirements."
            ) from exc
        raise

    dataset_classes = {
        "ravdess": RAVDESS,
        "emovo": EMOVO,
        "audio_mnist": AudioMNIST,
        "slurp": SLURP,
    }

    run_dir = Path(args.run_dir).resolve()
    codec_model, cfg = _load_codec_model(run_dir, device=args.device)
    wrapper_cls = _build_arch_wrapper_class()

    results_dir = output_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    all_dataset_results: Dict[str, object] = {}
    reused_datasets: List[str] = []
    recomputed_datasets: List[str] = []
    for dataset in datasets:
        ds_path = _dataset_path_for_arch(data_root, dataset)

        dataset_result_path = results_dir / f"{dataset}.json"
        if (not args.force_recompute_existing) and dataset_result_path.is_file():
            try:
                with dataset_result_path.open("r", encoding="utf-8") as f:
                    existing_result = json.load(f)
                if _can_reuse_dataset_result(existing_result, dataset, ds_path, args):
                    print(f"[Resume] Reusing existing result for dataset={dataset}: {dataset_result_path}")
                    all_dataset_results[dataset] = existing_result
                    reused_datasets.append(dataset)
                    continue
                print(
                    f"[Resume] Existing result for dataset={dataset} is not compatible with current args; recomputing."
                )
            except Exception as exc:
                print(
                    f"[Resume] Failed to read existing result for dataset={dataset} ({dataset_result_path}): {exc}; recomputing."
                )

        evaluator = dataset_classes[dataset](
            ds_path,
            verbose=bool(args.verbose),
            precompute_embeddings=bool(args.precompute_embeddings),
        )

        iter_metrics: List[Dict[str, float]] = []
        for _ in range(args.n_iters):
            wrapper = wrapper_cls(
                codec_model,
                cfg,
                device=args.device,
                use_amp=not args.no_amp,
                feature_source=args.feature_source,
            )
            metrics = evaluator.evaluate(
                wrapper,
                mode=args.mode,
                device=args.device,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                max_num_epochs=args.max_epochs,
            )
            iter_metrics.append({k: float(v) for k, v in metrics.items()})

        mean_metrics, std_metrics = _mean_std_dict(iter_metrics)
        dataset_result = {
            "dataset": dataset,
            "path": ds_path,
            "n_iters": int(args.n_iters),
            "mode": args.mode,
            "feature_source": args.feature_source,
            "metrics_mean": mean_metrics,
            "metrics_std": std_metrics,
            "iter_metrics": iter_metrics,
        }
        all_dataset_results[dataset] = dataset_result
        recomputed_datasets.append(dataset)

        with dataset_result_path.open("w", encoding="utf-8") as f:
            json.dump(dataset_result, f, indent=2)

    accs = [all_dataset_results[d]["metrics_mean"].get("accuracy") for d in datasets]
    f1s = [all_dataset_results[d]["metrics_mean"].get("f1") for d in datasets]
    accs = [float(x) for x in accs if x is not None]
    f1s = [float(x) for x in f1s if x is not None]

    summary = {
        "arch_repo": str(Path(args.arch_repo).resolve()),
        "arch_commit": _parse_arch_commit(Path(args.arch_repo).resolve()),
        "run_dir": str(run_dir),
        "datasets": list(datasets),
        "mode": args.mode,
        "feature_source": args.feature_source,
        "max_epochs": int(args.max_epochs),
        "batch_size": int(args.batch_size),
        "num_workers": int(args.num_workers),
        "n_iters": int(args.n_iters),
        "device": str(args.device),
        "force_recompute_existing": bool(args.force_recompute_existing),
        "reused_datasets": reused_datasets,
        "recomputed_datasets": recomputed_datasets,
        "arch_1": float(np.mean(accs)) if len(accs) > 0 else None,
        "arch_2": float(np.mean(f1s)) if len(f1s) > 0 else None,
        "dataset_results": all_dataset_results,
    }

    with (results_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    return summary


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ARCH-compatible semantic eval runner")
    parser.add_argument("--stage", choices=["build", "eval", "all"], default="all")
    parser.add_argument("--datasets", type=str, default=",".join(SUPPORTED_DATASETS))
    parser.add_argument("--data_root", type=str, default="/home/hoyso/projects/datasets")
    parser.add_argument("--output_dir", type=str, default=str(THIS_DIR / "outputs" / "arch_speech"))

    parser.add_argument("--run_dir", type=str, default=None)
    parser.add_argument("--arch_repo", type=str, default=str(THIS_DIR / "third_party" / "ARCH"))
    parser.add_argument("--mode", choices=["linear", "non-linear", "attention-pooling"], default="linear")
    parser.add_argument("--feature_source", choices=["post_vq", "pre_vq"], default="post_vq")
    parser.add_argument("--max_epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--n_iters", type=int, default=1)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--precompute_embeddings", action="store_true")
    parser.add_argument("--no_amp", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--force_recompute_existing",
        action="store_true",
        help="Ignore existing results/{dataset}.json and recompute all selected datasets.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    datasets = [x.strip().lower() for x in args.datasets.split(",") if x.strip()]
    if len(datasets) == 0:
        raise ValueError("No datasets selected")
    bad = [d for d in datasets if d not in SUPPORTED_DATASETS]
    if len(bad) > 0:
        raise ValueError(f"Unsupported datasets: {bad}")

    if args.seed is not None:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)

    data_root = Path(args.data_root).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.stage in ("build", "all"):
        build_summary = _build_manifests(datasets, data_root=data_root, output_dir=output_dir)
        print(json.dumps({"stage": "build", "summary": build_summary}, indent=2))

    if args.stage in ("eval", "all"):
        if args.run_dir is None:
            raise ValueError("--run_dir is required for stage eval/all")
        summary = _run_arch_evaluation(args, datasets=datasets, data_root=data_root, output_dir=output_dir)
        print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
