#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

try:
    import yaml  # type: ignore
except ImportError as e:  # pragma: no cover
    raise SystemExit(
        "Missing dependency: pyyaml\n"
        "Install with: python -m pip install pyyaml"
    ) from e


CKPT_STEP_RE = re.compile(r"step=(\d+)")


@dataclass
class CheckpointInfo:
    path: Path
    step: int
    size_bytes: int
    mtime: float


@dataclass
class MatchInfo:
    run_dir: Path
    config_path: Path
    config_size_bytes: int
    config_mtime: float
    ckpt_count: int
    best_ckpt: CheckpointInfo | None


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Search DTMAE outputs/*/*/hydra/config.yaml by top-level `name`, "
            "then choose one run (highest checkpoint step) per name."
        )
    )
    parser.add_argument(
        "names",
        nargs="+",
        help="One or more target names to search (exact match).",
    )
    parser.add_argument(
        "--outputs-dir",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "outputs",
        help="Outputs root directory (default: <repo>/outputs)",
    )
    parser.add_argument(
        "--contains",
        action="store_true",
        help="Use substring match instead of exact name match.",
    )
    return parser.parse_args()


def _fmt_size(num_bytes: int) -> str:
    units = ["B", "KB", "MB", "GB", "TB"]
    size = float(num_bytes)
    for unit in units:
        if size < 1024.0 or unit == units[-1]:
            if unit == "B":
                return f"{int(size)} {unit}"
            return f"{size:.2f} {unit}"
        size /= 1024.0
    return f"{num_bytes} B"


def _fmt_time(ts: float) -> str:
    return datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S")


def _extract_step(path: Path) -> int:
    match = CKPT_STEP_RE.search(path.name)
    if not match:
        return -1
    return int(match.group(1))


def _safe_yaml_load(path: Path) -> dict[str, Any] | None:
    try:
        with path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
    except Exception:
        return None
    if not isinstance(data, dict):
        return None
    return data


def _find_best_ckpt(run_dir: Path) -> tuple[CheckpointInfo | None, int]:
    candidates = list(run_dir.rglob("*.ckpt"))
    if not candidates:
        return None, 0

    ckpts: list[CheckpointInfo] = []
    for ckpt in candidates:
        try:
            st = ckpt.stat()
        except OSError:
            continue
        ckpts.append(
            CheckpointInfo(
                path=ckpt,
                step=_extract_step(ckpt),
                size_bytes=st.st_size,
                mtime=st.st_mtime,
            )
        )

    if not ckpts:
        return None, 0

    best = max(ckpts, key=lambda c: (c.step, c.mtime))
    return best, len(ckpts)


def _collect_matches(outputs_dir: Path, target_name: str, contains: bool) -> list[MatchInfo]:
    config_paths = sorted(outputs_dir.glob("*/*/hydra/config.yaml"))
    matches: list[MatchInfo] = []

    for cfg in config_paths:
        data = _safe_yaml_load(cfg)
        if data is None:
            continue
        name_val = data.get("name")
        if not isinstance(name_val, str):
            continue

        ok = target_name in name_val if contains else (target_name == name_val)
        if not ok:
            continue

        try:
            cfg_st = cfg.stat()
        except OSError:
            continue

        run_dir = cfg.parent.parent
        best_ckpt, ckpt_count = _find_best_ckpt(run_dir)
        matches.append(
            MatchInfo(
                run_dir=run_dir,
                config_path=cfg,
                config_size_bytes=cfg_st.st_size,
                config_mtime=cfg_st.st_mtime,
                ckpt_count=ckpt_count,
                best_ckpt=best_ckpt,
            )
        )

    return matches


def _select_one(matches: list[MatchInfo]) -> MatchInfo:
    return max(
        matches,
        key=lambda m: (
            m.best_ckpt.step if m.best_ckpt is not None else -1,
            m.best_ckpt.mtime if m.best_ckpt is not None else m.config_mtime,
        ),
    )


def _print_match(idx: int, m: MatchInfo) -> None:
    print(f"  [{idx}] run_dir        : {m.run_dir}")
    print(f"      config         : {m.config_path}")
    print(
        f"      config_info    : size={_fmt_size(m.config_size_bytes)}, "
        f"mtime={_fmt_time(m.config_mtime)}"
    )
    print(f"      ckpt_count     : {m.ckpt_count}")
    if m.best_ckpt is None:
        print("      best_ckpt      : (none)")
    else:
        print(f"      best_ckpt      : {m.best_ckpt.path}")
        print(
            f"      best_ckpt_info : step={m.best_ckpt.step}, "
            f"size={_fmt_size(m.best_ckpt.size_bytes)}, "
            f"mtime={_fmt_time(m.best_ckpt.mtime)}"
        )


def main() -> None:
    args = _parse_args()
    outputs_dir = args.outputs_dir.resolve()

    if not outputs_dir.exists() or not outputs_dir.is_dir():
        raise SystemExit(f"outputs directory not found: {outputs_dir}")

    print(f"[info] outputs_dir: {outputs_dir}")
    print(f"[info] match_mode : {'contains' if args.contains else 'exact'}")

    any_found = False
    for name in args.names:
        print("\n" + "=" * 90)
        print(f"[query] name: {name}")
        matches = _collect_matches(outputs_dir, name, args.contains)
        if not matches:
            print("[result] no matching config.yaml found")
            continue

        any_found = True
        print(f"[result] found {len(matches)} matching run(s)")

        for i, m in enumerate(matches, start=1):
            _print_match(i, m)

        selected = _select_one(matches)
        print("\n[selected] choose one run (highest checkpoint step)")
        _print_match(1, selected)

        if len(matches) > 1:
            print("[note] multiple matches existed; full list shown above.")

    if not any_found:
        sys.exit(1)


if __name__ == "__main__":
    main()
