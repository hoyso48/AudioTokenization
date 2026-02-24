#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import torchaudio


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Download or verify LibriTTS subsets")
    p.add_argument("--root", type=str, required=True, help="Dataset root directory")
    p.add_argument(
        "--subsets",
        nargs="+",
        default=["train-clean-100", "train-clean-360", "train-other-500", "dev-clean", "test-clean"],
        help="LibriTTS subsets",
    )
    p.add_argument("--folder_in_archive", type=str, default="LibriTTS")
    p.add_argument("--download", action="store_true", help="Actually download missing subsets")
    return p.parse_args()


def subset_exists(root: Path, folder_in_archive: str, subset: str) -> bool:
    return (root / folder_in_archive / subset).is_dir()


def main() -> None:
    args = parse_args()
    root = Path(args.root).resolve()
    root.mkdir(parents=True, exist_ok=True)

    done: List[str] = []
    missing: List[str] = []
    for subset in args.subsets:
        if subset_exists(root, args.folder_in_archive, subset):
            done.append(subset)
        else:
            missing.append(subset)

    if missing and args.download:
        for subset in missing:
            print(f"Downloading {subset} ...")
            torchaudio.datasets.LIBRITTS(
                str(root),
                url=subset,
                folder_in_archive=args.folder_in_archive,
                download=True,
            )
            print(f"Done {subset}")

    print("Present subsets:", done + ([m for m in missing] if args.download else []))
    if missing and not args.download:
        print("Missing subsets:", missing)
        print("Run again with --download to fetch them.")


if __name__ == "__main__":
    main()
