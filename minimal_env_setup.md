# Minimal Env Setup (atk + speech_eval)

This repo now includes a bootstrap script for minimal train/eval environments:

- `setup_conda_envs_minimal.sh`

It targets this flow:
- train in `atk`
- eval in `speech_eval`
- robust execution in non-interactive shells via `conda run`

## Quick start

```bash
bash setup_conda_envs_minimal.sh \
  --train_env atk \
  --eval_env speech_eval \
  --python_version 3.10 \
  --recreate_on_python_mismatch
```

Then run:

```bash
bash run_train_then_eval.sh
```

If you already have run directories and want eval only:

```bash
bash run_eval_only.sh outputs/exp1
```

With eval config override:

```bash
bash run_eval_only.sh outputs/exp1 \
  --cfg_override model.resampler.dtp_params.r=0.4
```

With DTP target_r matching (if run uses `use_dtp=true` and supports `fixed_tau`):

```bash
bash run_eval_only.sh outputs/exp1
```

If you want to override config target explicitly:

```bash
bash run_eval_only.sh outputs/exp1 --target_avg_r 0.5
```

## What it installs

### Train env (`atk`)
- Installs from `requirements.txt`.
- Installs `ffmpeg` via conda (default).
- Installs `flash-attn==2.8.3` (required).
- Reinstalls `nvidia-nccl-cu12>=2.26.2` (default).
- Verifies core imports (`torch`, `torchaudio`, `hydra`, `pytorch_lightning`).

### Eval env (`speech_eval`)
- Ensures torch stack exists (or installs pinned CUDA wheels).
- Installs minimal eval runtime deps used by `eval/eval.py` and local speaker/UTMOS stack.
- Installs `flash-attn==2.8.3` (required).
- Verifies key imports including local modules (`fairseq`, `s3prl`, `verification`, `UTMOS`).

## Useful options

```bash
# Only setup eval env
bash setup_conda_envs_minimal.sh --skip_train

# Reinstall eval torch stack explicitly
bash setup_conda_envs_minimal.sh --skip_train --force_reinstall_eval_torch
```

## Notes
- This repo now vendors `eval/fairseq` directly (no submodule required).
- If you are on a legacy clone and `eval/fairseq` is broken, repair once:

```bash
git pull
git submodule deinit -f eval/fairseq || true
rm -rf .git/modules/eval/fairseq eval/fairseq
git restore --source=HEAD --staged --worktree eval/fairseq
```

- Default train env is `atk`. If you use a different local env name, pass `--train_env <name>`.
- `eval/wavlm_large_finetune.pth` is required for speaker similarity in eval.
  - The setup script warns if missing.
  - This file can be tracked with Git LFS.
- `flash-attn` is mandatory for both envs in this setup. If install fails on your node,
  prebuild/cache the wheel in your base image or use a compatible CUDA/PyTorch toolchain.
