#!/usr/bin/env python3
"""
Update a legacy DTMAE YAML config to the newer "quantizer" config style.

Legacy style (example):
- Quantizer parameters live inside: model.codec_decoder.{vq_*, fsq, fsq_levels, codebook_*}

New style (example):
- Quantizer is separated into: model.quantizer:
    cls: <ResidualVQ|DitheredFSQ|FSQ|SimVQ>
    params: {...}

This script:
- Detects legacy quantizer fields.
- Writes a new config with a proper quantizer section.
- Removes legacy quantizer fields from codec_decoder.
- Renames the original file to a backup name (default suffix: "_legacy").

It supports:
- A single YAML file (e.g., .../hydra/config.yaml)
- A directory containing YAMLs (recursively converts matching files)

Notes:
- We intentionally do NOT preserve YAML comments/formatting. The output is a clean YAML.
- Conversion is "best-effort" with explicit rules. If something is ambiguous, we fail with a clear error.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from omegaconf import OmegaConf


LEGACY_DECODER_KEYS = (
    "vq_num_quantizers",
    "vq_commit_weight",
    "vq_weight_init",
    "vq_full_commit_loss",
    "codebook_size",
    "codebook_dim",
    "fsq",
    "fsq_levels",
)


@dataclass(frozen=True)
class ConvertedFile:
    source: Path
    backup: Path
    output: Path
    quantizer_cls: str


def _is_yaml(p: Path) -> bool:
    return p.is_file() and p.suffix.lower() in (".yaml", ".yml")


def _load_yaml(path: Path) -> Dict[str, Any]:
    cfg = OmegaConf.load(str(path))
    data = OmegaConf.to_container(cfg, resolve=False)
    if not isinstance(data, dict):
        raise ValueError(f"Expected a YAML mapping at root: {path}")
    return data


def _save_yaml(path: Path, data: Dict[str, Any]) -> None:
    cfg = OmegaConf.create(data)
    path.parent.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, str(path))


def _get_model_node(root: Dict[str, Any]) -> Tuple[Dict[str, Any], str]:
    """
    Returns (model_node, placement), where placement is:
    - "nested" if model_node is root['model']
    - "flat" if model_node is root (Hydra model group file like DTMAE/config/model/base.yaml)
    """
    if "model" in root and isinstance(root["model"], dict):
        return root["model"], "nested"
    return root, "flat"


def _is_legacy_quantizer(model_node: Dict[str, Any]) -> bool:
    codec_decoder = model_node.get("codec_decoder")
    if not isinstance(codec_decoder, dict):
        return False
    has_legacy = any(k in codec_decoder for k in LEGACY_DECODER_KEYS)
    has_new = isinstance(model_node.get("quantizer"), dict) and "cls" in model_node.get("quantizer", {})
    return bool(has_legacy and not has_new)


def _pop_optional(d: Dict[str, Any], key: str, default: Any = None) -> Any:
    if key in d:
        return d.pop(key)
    return default


def _remove_legacy_keys(codec_decoder: Dict[str, Any]) -> Dict[str, Any]:
    removed: Dict[str, Any] = {}
    for k in LEGACY_DECODER_KEYS:
        if k in codec_decoder:
            removed[k] = codec_decoder.pop(k)
    return removed


def _infer_quantizer_from_legacy(
    codec_decoder: Dict[str, Any],
) -> Tuple[str, Dict[str, Any]]:
    """
    Convert legacy codec_decoder quantizer fields to (quantizer_cls, quantizer_params).
    """
    in_channels = codec_decoder.get("in_channels")
    if in_channels is None:
        raise ValueError("Cannot infer quantizer params: model.codec_decoder.in_channels is missing.")

    # Read legacy values without mutating first (we mutate later after deciding).
    fsq_enabled = bool(codec_decoder.get("fsq", False))
    fsq_levels = codec_decoder.get("fsq_levels")
    vq_num_quantizers = codec_decoder.get("vq_num_quantizers", 1)
    vq_commit_weight = codec_decoder.get("vq_commit_weight", 0.5)
    vq_weight_init = codec_decoder.get("vq_weight_init", False)
    vq_full_commit_loss = codec_decoder.get("vq_full_commit_loss", False)
    codebook_size = codec_decoder.get("codebook_size", 16384)
    codebook_dim = codec_decoder.get("codebook_dim", 8)

    # Rule 1: fsq=True => map to DitheredFSQ (closest supported match in this repo).
    if fsq_enabled:
        if fsq_levels is None:
            raise ValueError("Legacy fsq=True but fsq_levels is missing; cannot build DitheredFSQ params.")
        if not isinstance(fsq_levels, list) or len(fsq_levels) == 0 or not all(isinstance(x, int) for x in fsq_levels):
            raise ValueError("Legacy fsq_levels must be a non-empty list of ints.")

        # Legacy fsq_levels is most consistent with DitheredFSQ's inference_levels_list.
        # For training, keep behavior deterministic by setting train_levels=[max_level] and noise_dropout=0.0.
        max_level = int(max(fsq_levels))
        params = {
            "dim": int(in_channels),
            "codebook_dim": int(len(fsq_levels)),
            "train_levels": [max_level],
            "train_num_residuals": 1,
            "inference_levels": list(fsq_levels),
            "inference_num_residuals": int(vq_num_quantizers),
            "num_codebooks": 1,
            "noise_dropout": 0.0,
            "scale": 1.0,
        }
        return "DitheredFSQ", params

    # Rule 2: default => ResidualVQ (legacy vq_* and codebook_* map directly).
    params = {
        "dim": int(in_channels),
        "codebook_size": int(codebook_size),
        "codebook_dim": int(codebook_dim),
        "num_quantizers": int(vq_num_quantizers),
        "commitment": float(vq_commit_weight),
        # These extra fields are present in the new template configs; implementations may ignore them.
        "weight_init": bool(vq_weight_init),
        "full_commit_loss": bool(vq_full_commit_loss),
        "threshold_ema_dead_code": 2,
    }
    return "ResidualVQ", params


def convert_config_dict(root: Dict[str, Any]) -> Tuple[Dict[str, Any], str]:
    """
    Returns (new_root, quantizer_cls). Raises on invalid/ambiguous configs.
    """
    model_node, _ = _get_model_node(root)
    codec_decoder = model_node.get("codec_decoder")
    if not isinstance(codec_decoder, dict):
        raise ValueError("Missing model.codec_decoder section; cannot convert.")

    # If already new-style, leave as-is.
    if isinstance(model_node.get("quantizer"), dict) and "cls" in model_node.get("quantizer", {}):
        qcls = str(model_node["quantizer"]["cls"])
        return root, qcls

    # Legacy -> new.
    qcls, qparams = _infer_quantizer_from_legacy(codec_decoder)
    model_node["quantizer"] = {"cls": qcls, "params": qparams}

    # Remove legacy decoder keys to avoid confusion.
    _remove_legacy_keys(codec_decoder)
    return root, qcls


def _backup_path_for(source: Path, backup_suffix: str) -> Path:
    if source.name == "config.yaml":
        return source.with_name(f"config{backup_suffix}.yaml")
    if source.suffix.lower() in (".yaml", ".yml"):
        return source.with_name(f"{source.stem}{backup_suffix}{source.suffix}")
    return source.with_name(f"{source.name}{backup_suffix}")


def convert_file_inplace(
    path: Path,
    *,
    backup_suffix: str,
    dry_run: bool,
) -> Optional[ConvertedFile]:
    if not _is_yaml(path):
        return None

    root = _load_yaml(path)
    model_node, _ = _get_model_node(root)
    if not _is_legacy_quantizer(model_node):
        return None

    new_root, qcls = convert_config_dict(root)
    backup = _backup_path_for(path, backup_suffix)
    output = path

    if dry_run:
        return ConvertedFile(source=path, backup=backup, output=output, quantizer_cls=qcls)

    if backup.exists():
        raise FileExistsError(f"Backup file already exists: {backup}")

    shutil.move(str(path), str(backup))
    _save_yaml(output, new_root)
    return ConvertedFile(source=path, backup=backup, output=output, quantizer_cls=qcls)


def iter_yaml_files(target: Path) -> Iterable[Path]:
    if target.is_file():
        yield target
        return
    if not target.is_dir():
        return
    for p in sorted(target.rglob("*.yaml")):
        yield p
    for p in sorted(target.rglob("*.yml")):
        yield p


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Convert legacy DTMAE configs to new quantizer style.")
    p.add_argument(
        "--path",
        type=str,
        required=True,
        help="A YAML file or a directory (recursively converts matching YAMLs).",
    )
    p.add_argument(
        "--backup_suffix",
        type=str,
        default="_legacy",
        help="Backup suffix for the original file (default: _legacy).",
    )
    p.add_argument("--dry_run", action="store_true", help="Print what would change without writing files.")
    return p


def main() -> None:
    args = build_parser().parse_args()
    target = Path(args.path).expanduser().resolve()
    if not target.exists():
        raise FileNotFoundError(f"Path not found: {target}")

    changed: List[ConvertedFile] = []
    for p in iter_yaml_files(target):
        out = convert_file_inplace(p, backup_suffix=str(args.backup_suffix), dry_run=bool(args.dry_run))
        if out is not None:
            changed.append(out)

    if not changed:
        print("No legacy configs found (nothing to convert).")
        return

    for rec in changed:
        if args.dry_run:
            print(f"[DRY RUN] Would convert: {rec.source} -> {rec.output} (backup: {rec.backup}), quantizer={rec.quantizer_cls}")
        else:
            print(f"Converted: {rec.output} (backup: {rec.backup}), quantizer={rec.quantizer_cls}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"[Error] {exc}", file=sys.stderr)
        raise


