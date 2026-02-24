# TTS Evaluation Protocol

This project should use two complementary evaluation tracks:

- **Track A (primary, external benchmark):** `SEED-TTS` objective benchmark
- **Track B (paper comparability):** VARSTOK-style protocol used in your current study

Using both avoids blind spots: SEED-TTS gives standardized OOD objective comparison, while VARSTOK-style keeps direct comparability with existing tables.

## Track A: SEED-TTS benchmark (recommended as main external result)

Reference repository:

- `https://github.com/BytedanceSpeech/seed-tts-eval`

From their released protocol:

- Objective metrics: **WER** and **SIM**
- ASR:
  - EN: Whisper-large-v3
  - ZH: Paraformer-zh
- Speaker similarity:
  - WavLM-large speaker verification checkpoint (cosine similarity)

### How to run in practice

1. Generate TTS wav files for each line in the benchmark meta list.
2. Save synthesized wav names to match benchmark filenames.
3. Run their scripts directly:
   - `bash cal_wer.sh <meta.lst> <synth_dir> <en|zh>`
   - `bash cal_sim.sh <meta.lst> <synth_dir> <wavlm_ckpt>`
4. Report EN and ZH separately, plus overall weighted average.

Practical wrapper in this repo:

- `AudioTokenization/tts/scripts/run_seed_tts_eval.sh`

## Track B: VARSTOK-style protocol (for in-paper consistency)

Use the same setup as your existing TTS section for FFR vs VFR fairness:

- Prompt condition: fixed 3-second speech prompt
- Dataset split/protocol: same as prior VARSTOK-style comparison set
- Metrics:
  - **WER** (Whisper-large-v3)
  - **SIM** (WavLM speaker cosine)
  - **UTMOS** (optional but strongly recommended)
  - Subjective: MOS/SMOS if available

## Reporting rules (important)

- Always report **FFR vs VFR at matched total bitrate**.
- For VFR, keep duration/span side-information accounting explicit.
- Report decoding config with each table:
  - temperature
  - top-k / top-p
  - max generation length
  - prompt duration

## Minimal table template

- Model: FFR / VFR
- Tokenizer: checkpoint name
- Total bitrate (matched)
- WER (SEED-EN, SEED-ZH, VARSTOK-protocol)
- SIM (SEED-EN, SEED-ZH, VARSTOK-protocol)
- UTMOS (VARSTOK-protocol)

Practical script in this repo:

- `AudioTokenization/tts/scripts/eval_varstok_style.py`

## Recommended release order

1. Internal smoke eval (small sample)
2. Full VARSTOK-style table
3. Full SEED-TTS benchmark
4. Optional subjective MOS/SMOS

## Single-command full pipeline

- `AudioTokenization/tts/scripts/run_tts_modeling_and_eval.sh`

This pipeline includes modeling and both evaluation tracks (SEED-TTS + VARSTOK-style).
