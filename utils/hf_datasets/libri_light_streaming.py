# coding=utf-8
"""
Libri-Light (streaming-friendly) dataset script.

This is adapted from:
  https://huggingface.co/datasets/HugoLaurencon/libri_light/blob/main/libri_light.py

Why this exists:
- The original script uses `download_and_extract(...)` and then `glob(...)`.
- In HF streaming mode, TAR extraction is not supported; you must iterate archives via
  `dl_manager.iter_archive(url)`.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Dict, Iterable, Iterator, Optional, Tuple

import datasets

_CITATION = """\
@INPROCEEDINGS{librilight,
author={J. Kahn and M. Rivière and W. Zheng and E. Kharitonov and Q. Xu and P. E. Mazaré and J. Karadayi and V. Liptchinsky and R. Collobert and C. Fuegen and T. Likhomanenko and G. Synnaeve and A. Joulin and A. Mohamed and E. Dupoux},
booktitle={ICASSP 2020 - 2020 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
title={Libri-Light: A Benchmark for ASR with Limited or No Supervision},
year={2020},
pages={7669-7673},
}
"""

_DESCRIPTION = """\
Libri-Light is a large dataset of ~60K hours of unlabelled speech from audiobooks in English.
"""

_HOMEPAGE = "https://ai.facebook.com/tools/libri-light/"

_LICENSE = "MIT"

_DL_URL = "https://dl.fbaipublicfiles.com/librilight/data/{name}.tar"


class LibriLightConfig(datasets.BuilderConfig):
    """BuilderConfig for Libri-Light."""

    def __init__(self, **kwargs):
        super().__init__(version=datasets.Version("2.1.0", ""), **kwargs)


@dataclass(frozen=True)
class _Pending:
    audio_bytes: Optional[bytes] = None
    json_bytes: Optional[bytes] = None


class LibriLight(datasets.GeneratorBasedBuilder):
    """Libri-Light dataset that works in HF streaming mode."""

    BUILDER_CONFIGS = [
        LibriLightConfig(name="small", description="1000 hours, ~61 GB."),
        LibriLightConfig(name="medium", description="5193 hours, ~321 GB."),
        LibriLightConfig(name="large", description="51934 hours, ~3.05 TB."),
    ]

    def _info(self) -> datasets.DatasetInfo:
        features = datasets.Features(
            {
                "id": datasets.Value("string"),
                "file": datasets.Value("string"),
                "audio": datasets.Audio(sampling_rate=16_000),
                "speaker_id": datasets.Value("int64"),
                # Keep metadata as a JSON string to avoid having to hardcode the full schema.
                "metadata": datasets.Value("string"),
            }
        )
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager: datasets.DownloadManager):
        url = _DL_URL.format(name=self.config.name)
        url = dl_manager.download(url)
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={"tar_iter": dl_manager.iter_archive(url)},
            )
        ]

    def _generate_examples(
        self, tar_iter: Iterable[Tuple[str, object]]
    ) -> Iterator[Tuple[int, Dict]]:
        """
        Stream over the TAR file and pair each .flac with its .json metadata.

        The archive structure (from the original script):
          {config}/**/**/*.flac  and corresponding .json next to it.
        """

        pending: Dict[str, _Pending] = {}
        key = 0

        for filename, fileobj in tar_iter:
            # HF returns POSIX paths inside archives.
            if not (filename.endswith(".flac") or filename.endswith(".json")):
                continue

            base = filename[:-5]  # remove ".flac" or ".json" (same length)
            ex_id = os.path.basename(base)

            # speaker id: same logic as upstream script (path_split[-3])
            parts = filename.split("/")
            if len(parts) < 3:
                continue
            speaker_part = parts[-3]
            try:
                speaker_id = int(speaker_part)
            except ValueError:
                continue

            if ex_id not in pending:
                pending[ex_id] = _Pending()

            # Read bytes (streaming archive => must read sequentially).
            data = fileobj.read()
            cur = pending[ex_id]

            if filename.endswith(".flac"):
                pending[ex_id] = _Pending(audio_bytes=data, json_bytes=cur.json_bytes)
            else:
                pending[ex_id] = _Pending(audio_bytes=cur.audio_bytes, json_bytes=data)

            cur = pending[ex_id]
            if cur.audio_bytes is None or cur.json_bytes is None:
                continue

            # Emit example once both are available.
            metadata_str = cur.json_bytes.decode("utf-8", errors="strict")

            # Optional validation: ensure it's valid JSON (kept explicit, no silent fallback).
            json.loads(metadata_str)

            yield key, {
                "id": ex_id,
                "file": filename,
                "audio": {"path": filename, "bytes": cur.audio_bytes},
                "speaker_id": speaker_id,
                "metadata": metadata_str,
            }
            key += 1
            del pending[ex_id]


