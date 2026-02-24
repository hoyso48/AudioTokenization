from __future__ import annotations

import torch

from tts.modeling_ar_tts import ARTTSConfig, ARTTSForConditionalGeneration


def test_forward_vfr_shapes_and_loss() -> None:
    cfg = ARTTSConfig(
        vocab_size=128,
        d_model=64,
        n_head=4,
        n_layer=2,
        ffn_mult=2,
        max_position_embeddings=128,
        use_vfr=True,
        max_span_len=32,
        lambda_span=1.0,
    )
    model = ARTTSForConditionalGeneration(cfg)

    bsz, seq = 2, 20
    input_ids = torch.randint(0, cfg.vocab_size, (bsz, seq))
    labels = input_ids.clone()
    attention_mask = torch.ones(bsz, seq, dtype=torch.long)
    label_mask = torch.zeros(bsz, seq, dtype=torch.bool)
    label_mask[:, 10:] = True
    speech_mask = torch.zeros(bsz, seq, dtype=torch.bool)
    speech_mask[:, 5:18] = True
    span_ids = torch.randint(1, cfg.max_span_len + 1, (bsz, seq))
    span_labels = span_ids.clone()

    out = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=labels,
        label_mask=label_mask,
        speech_mask=speech_mask,
        span_ids=span_ids,
        span_labels=span_labels,
    )

    assert out["logits"].shape == (bsz, seq, cfg.vocab_size)
    assert out["span_logits"] is not None
    assert out["span_logits"].shape == (bsz, seq, cfg.max_span_len + 1)
    assert out["loss"] is not None
    assert torch.isfinite(out["loss"])
