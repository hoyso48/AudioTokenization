# Offline LibriTTS AR-TTS (FFR/VFR)

This directory contains a self-contained starter pipeline for LibriTTS-based TTS modeling with:

- AR transformer (decoder-only)
- HuggingFace `Trainer`
- Offline codebook-token workflow
- Two variants:
  - `FFR`: token-only AR modeling
  - `VFR`: `use_dtp=True` and non-fixed selector with span modeling

## VFR rule implemented

- Convert mask to span length via:
  - `span_len = (# trailing zeros) + 1`
- Clip by `max_span_len` (default `512`)
- Embedding:
  - `h = token_emb(token_id) + span_emb(span_len)` (speech positions only)
- Heads:
  - token vocab head
  - independent span length head

## Layout

- `scripts/download_libritts.py`: download/verify LibriTTS subsets
- `scripts/extract_codec_tokens.py`: extract offline utterance-level codec tokens (+ VFR spans)
- `scripts/build_tts_examples.py`: build train/val examples with prompt-target pairing
- `scripts/prepare_text_tokenizer.py`: build offline char tokenizer
- `scripts/train_ar_tts.py`: train FFR/VFR AR model with HF Trainer
- `scripts/infer_ar_tts.py`: token generation inference
- `scripts/synthesize_from_meta.py`: synthesize wavs from benchmark meta list
- `scripts/eval_varstok_style.py`: VARSTOK-style objective eval (WER/SIM/UTMOS)
- `scripts/run_seed_tts_eval.sh`: wrapper for `seed-tts-eval` WER/SIM
- `scripts/run_tts_modeling_and_eval.sh`: full modeling + eval pipeline
- `src/tts/*`: model, dataset, collator, utils

## 0) Environment

```bash
bash scripts/setup_env_tts.sh audiotok_tts
conda activate audiotok_tts
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
```

## 1) Dataset download (optional)

```bash
python scripts/download_libritts.py --root /path/to/datasets --download
```

## 2) Offline utterance token extraction

```bash
python scripts/extract_codec_tokens.py \
  --run_dir /path/to/tokenizer_run \
  --input /path/to/libritts/train-clean-100 \
  --output_jsonl ./data/utt_tokens_train.jsonl \
  --max_span_len 512
```

Run similarly for validation split.

## 3) Build prompt-target examples

```bash
python scripts/build_tts_examples.py \
  --input_jsonl ./data/utt_tokens_train.jsonl \
  --output_jsonl ./data/examples_train.jsonl
```

## 4) Build text tokenizer

```bash
python scripts/prepare_text_tokenizer.py \
  --input_jsonl ./data/examples_train.jsonl \
  --output_path ./artifacts/text_tokenizer.json
```

## 5) Train (FFR)

```bash
python scripts/train_ar_tts.py \
  --train_jsonl ./data/examples_train.jsonl \
  --val_jsonl ./data/examples_val.jsonl \
  --tokenizer_path ./artifacts/text_tokenizer.json \
  --output_dir ./runs/ffr \
  --speech_vocab_size 16384
```

## 6) Train (VFR)

```bash
python scripts/train_ar_tts.py \
  --train_jsonl ./data/examples_train.jsonl \
  --val_jsonl ./data/examples_val.jsonl \
  --tokenizer_path ./artifacts/text_tokenizer.json \
  --output_dir ./runs/vfr \
  --speech_vocab_size 16384 \
  --use_vfr \
  --max_span_len 512 \
  --lambda_span 1.0
```

## 7) Inference (token generation)

```bash
python scripts/infer_ar_tts.py \
  --model_dir ./runs/vfr \
  --tokenizer_path ./artifacts/text_tokenizer.json \
  --speech_vocab_size 16384 \
  --text "hello world" \
  --prompt_tokens_json ./sample_prompt_tokens.json \
  --prompt_spans_json ./sample_prompt_spans.json \
  --use_vfr \
  --output_json ./gen/output_tokens.json
```

The current inference script generates token sequences. Waveform decoding from generated tokens is intentionally separated and can be attached to your existing codec decoder runtime.

## 8) Evaluation (SEED-TTS + VARSTOK-style)

Evaluation guidance is documented in:

- `AudioTokenization/tts/EVALUATION.md`

Recommended policy:

- Use **SEED-TTS benchmark** as the primary external objective benchmark.
- Keep **VARSTOK-style protocol** for direct comparability with existing experiments.
- Report FFR vs VFR at **matched total bitrate**.

### End-to-end command (modeling + both eval tracks)

```bash
bash scripts/run_tts_modeling_and_eval.sh \
  --codec-run-dir /path/to/codec_run \
  --variant vfr \
  --train-input /path/to/LibriTTS/train-clean-100 \
  --val-input /path/to/LibriTTS/dev-clean \
  --work-dir ./experiments/my_vfr \
  --seed-eval-repo /path/to/seed-tts-eval \
  --seed-meta-en /path/to/seed_tts/en/meta.lst \
  --seed-meta-zh /path/to/seed_tts/zh/meta.lst \
  --varstok-meta /path/to/varstok_eval/meta.lst
```

### Results0117 prefilled command template

```bash
bash scripts/run_results0117_tts_modeling_eval.sh
```

Edit variable paths at the top of `scripts/run_results0117_tts_modeling_eval.sh` before running.
