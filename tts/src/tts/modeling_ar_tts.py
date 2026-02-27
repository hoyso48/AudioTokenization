from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PreTrainedModel, PretrainedConfig


class ARTTSConfig(PretrainedConfig):
    model_type = "ar_tts"

    def __init__(
        self,
        vocab_size: int = 32000,
        d_model: int = 768,
        n_head: int = 12,
        n_layer: int = 12,
        ffn_mult: int = 4,
        dropout: float = 0.1,
        max_position_embeddings: int = 4096,
        use_vfr: bool = False,
        max_span_len: int = 512,
        lambda_span: float = 1.0,
        pad_token_id: int = 0,
        bos_token_id: int = 1,
        eos_token_id: int = 2,
        **kwargs,
    ):
        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            **kwargs,
        )
        self.vocab_size = int(vocab_size)
        self.d_model = int(d_model)
        self.n_head = int(n_head)
        self.n_layer = int(n_layer)
        self.ffn_mult = int(ffn_mult)
        self.dropout = float(dropout)
        self.max_position_embeddings = int(max_position_embeddings)
        self.use_vfr = bool(use_vfr)
        self.max_span_len = int(max_span_len)
        self.lambda_span = float(lambda_span)


class ARTTSForConditionalGeneration(PreTrainedModel):
    config_class = ARTTSConfig

    def __init__(self, config: ARTTSConfig):
        super().__init__(config)
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.position_embedding = nn.Embedding(config.max_position_embeddings, config.d_model)
        self.embed_dropout = nn.Dropout(config.dropout)

        if config.use_vfr:
            self.span_embedding = nn.Embedding(config.max_span_len + 1, config.d_model)
            self.span_head = nn.Linear(config.d_model, config.max_span_len + 1)
        else:
            self.span_embedding = None
            self.span_head = None

        dim_ff = config.d_model * config.ffn_mult
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.n_head,
            dim_feedforward=dim_ff,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer=encoder_layer, num_layers=config.n_layer)
        self.final_norm = nn.LayerNorm(config.d_model)
        self.token_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        self.post_init()

    def _causal_mask(self, seq_len: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        mask = torch.full((seq_len, seq_len), fill_value=float("-inf"), device=device, dtype=dtype)
        return torch.triu(mask, diagonal=1)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        label_mask: Optional[torch.Tensor] = None,
        speech_mask: Optional[torch.Tensor] = None,
        span_ids: Optional[torch.Tensor] = None,
        span_labels: Optional[torch.Tensor] = None,
        **_: torch.Tensor,
    ) -> dict:
        bsz, seq_len = input_ids.shape
        if seq_len > self.config.max_position_embeddings:
            raise ValueError(
                f"input length {seq_len} exceeds max_position_embeddings {self.config.max_position_embeddings}"
            )

        pos = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(bsz, seq_len)
        hidden = self.token_embedding(input_ids) + self.position_embedding(pos)

        if self.config.use_vfr:
            if span_ids is None:
                raise ValueError("span_ids must be provided when use_vfr=True")
            span_ids = torch.clamp(span_ids.to(torch.long), min=1, max=self.config.max_span_len)
            span_emb = self.span_embedding(span_ids)
            if speech_mask is not None:
                span_emb = span_emb * speech_mask.unsqueeze(-1).to(span_emb.dtype)
            hidden = hidden + span_emb

        hidden = self.embed_dropout(hidden)

        key_padding_mask = None
        if attention_mask is not None:
            key_padding_mask = ~attention_mask.to(torch.bool)

        h = self.transformer(
            hidden,
            mask=self._causal_mask(seq_len, hidden.device, hidden.dtype),
            src_key_padding_mask=key_padding_mask,
        )
        h = self.final_norm(h)

        token_logits = self.token_head(h)
        span_logits = self.span_head(h) if self.config.use_vfr else None

        total_loss = None
        token_loss = None
        span_loss = None

        if labels is not None:
            labels_for_loss = labels.clone()
            if label_mask is not None:
                labels_for_loss = labels_for_loss.masked_fill(~label_mask.to(torch.bool), -100)

            shift_token_logits = token_logits[:, :-1, :].contiguous()
            shift_labels = labels_for_loss[:, 1:].contiguous()
            token_loss = F.cross_entropy(
                shift_token_logits.view(-1, shift_token_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
            )
            total_loss = token_loss

            if self.config.use_vfr and span_logits is not None and span_labels is not None:
                span_labels_for_loss = span_labels.clone()
                span_labels_for_loss = torch.clamp(span_labels_for_loss, min=-100, max=self.config.max_span_len)

                valid_mask = torch.ones_like(span_labels_for_loss, dtype=torch.bool)
                if label_mask is not None:
                    valid_mask = valid_mask & label_mask.to(torch.bool)
                if speech_mask is not None:
                    valid_mask = valid_mask & speech_mask.to(torch.bool)
                span_labels_for_loss = span_labels_for_loss.masked_fill(~valid_mask, -100)

                shift_span_logits = span_logits[:, :-1, :].contiguous()
                shift_span_labels = span_labels_for_loss[:, 1:].contiguous()
                span_loss = F.cross_entropy(
                    shift_span_logits.view(-1, shift_span_logits.size(-1)),
                    shift_span_labels.view(-1),
                    ignore_index=-100,
                )
                total_loss = total_loss + self.config.lambda_span * span_loss

        return {
            "loss": total_loss,
            "logits": token_logits,
            "token_loss": token_loss,
            "span_loss": span_loss,
            "span_logits": span_logits,
        }
