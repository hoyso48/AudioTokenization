#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path


def _setup_path() -> None:
    root = Path(__file__).resolve().parents[1]
    src = root / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))


def main() -> None:
    _setup_path()
    from tts.infer import main as infer_main

    infer_main()


if __name__ == "__main__":
    main()
