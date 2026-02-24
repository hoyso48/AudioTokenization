from __future__ import annotations

import torch

from tts.span_utils import mask_to_span_lengths, mask_to_trailing_zeros, span_lengths_to_mask


def test_mask_to_trailing_and_span_roundtrip() -> None:
    mask = torch.tensor([1, 0, 0, 1, 0, 1], dtype=torch.bool)
    trailing = mask_to_trailing_zeros(mask)
    assert trailing.tolist() == [2, 1, 0]

    spans = mask_to_span_lengths(mask, max_span_len=512)
    assert spans.tolist() == [3, 2, 1]

    recon = span_lengths_to_mask(spans.tolist())
    assert recon.tolist() == mask.tolist()


def test_span_clipping() -> None:
    mask = torch.tensor([1] + [0] * 1000 + [1], dtype=torch.bool)
    spans = mask_to_span_lengths(mask, max_span_len=512)
    assert spans[0].item() == 512
    assert spans[1].item() == 1
