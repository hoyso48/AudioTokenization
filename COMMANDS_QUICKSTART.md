# Quick Commands

## 핵심만 기억
- training 포함 실행에서 가장 중요한 값: `config_path`
- eval 포함 실행에서 가장 중요한 값: `run_dir`

## 0) 경로를 먼저 변수로 잡기 (유동 경로 대응)

```bash
REPO_DIR="${REPO_DIR:-$PWD}"                    # AudioTokenization 디렉토리 경로
BASE_DIR="${BASE_DIR:-$(dirname "$REPO_DIR")}"  # 자동 감지
DATA_DIR="${DATA_DIR:-$BASE_DIR/datasets}"
```

repo 루트가 아니면 `REPO_DIR`만 정확히 주면 나머지는 자동으로 따라감.

환경 이름 기본값:
- train env: `atk`
- eval env: `speech_eval`
- 필요하면 `--train_env hoyso_ml`처럼 명시 override 가능

## 1) 1회 권한 설정

```bash
cd "$REPO_DIR"

chmod +x setup_conda_envs_minimal.sh run_train_then_eval.sh run_eval_only.sh
chmod +x slurm/*.sh slurm/*.sbatch
```

레거시 클론(예전 submodule 상태) 복구가 필요하면 1회:

```bash
cd "$REPO_DIR"
git pull
git submodule deinit -f eval/fairseq || true
rm -rf .git/modules/eval/fairseq eval/fairseq
git restore --source=HEAD --staged --worktree eval/fairseq
```

## 2) W&B 토큰 파일(권장)

```bash
mkdir -p ~/.secrets
chmod 700 ~/.secrets
printf '%s\n' 'YOUR_WANDB_API_KEY' > ~/.secrets/wandb_api_key.txt
chmod 600 ~/.secrets/wandb_api_key.txt
```

## 3) 로컬

```bash
cd "$REPO_DIR"
bash setup_conda_envs_minimal.sh \
  --train_env atk \
  --eval_env speech_eval \
  --python_version 3.10 \
  --recreate_on_python_mismatch
```

검증:

```bash
conda run -n speech_eval python -c "import sys; print(sys.version)"
# 3.10.x 출력이면 정상
```

training + eval (`config_path` 쉽게 변경):

```bash
# 방법 A: positional config_path
bash run_train_then_eval.sh config_default2 --run_dir outputs/exp_local_001 --train_cuda_visible_devices 0

# 방법 B: 명시 옵션
bash run_train_then_eval.sh --config_path config_default2 --run_dir outputs/exp_local_001 --train_cuda_visible_devices 0
```

eval-only (`run_dir`만 바꾸면 됨):

```bash
bash run_eval_only.sh outputs/exp_local_001
```

`target_avg_r` 기본 동작:
- `--target_avg_r` 미지정: run config의 `model.resampler.dtp_params.r`
- 지정: 입력값 사용

## 4) SLURM setup (1회 권장)

환경 재사용하려면 `--conda_envs_host`, `--conda_pkgs_host`를 setup/run 둘 다 동일하게 전달.

```bash
cd "$REPO_DIR"
bash slurm/submit_setup_env.sh \
  --wandb-api-key-file ~/.secrets/wandb_api_key.txt \
  -- \
  --host_base_dir "$BASE_DIR" \
  --conda_envs_host "$BASE_DIR/.conda_envs" \
  --conda_pkgs_host "$BASE_DIR/.conda_pkgs" \
  --train_env atk --eval_env speech_eval
```

## 5) SLURM training + eval

1 GPU preset:

```bash
cd "$REPO_DIR"
bash slurm/submit_train_eval.sh \
  --job-name dtmae-1g \
  --gpus 1 --cpus 16 --mem 28G --time 336:00:00 \
  --output dtmae_1g_%j.out \
  --wandb-api-key-file ~/.secrets/wandb_api_key.txt \
  -- \
  --host_base_dir "$BASE_DIR" \
  --conda_envs_host "$BASE_DIR/.conda_envs" \
  --conda_pkgs_host "$BASE_DIR/.conda_pkgs" \
  --train_env atk --eval_env speech_eval \
  --train_config_path config_default2 \
  --train_arg train.trainer.devices=1
```

2 GPU preset:

```bash
cd "$REPO_DIR"
bash slurm/submit_train_eval.sh \
  --job-name dtmae-2g \
  --gpus 2 --cpus 32 --mem 56G --time 336:00:00 \
  --output dtmae_2g_%j.out \
  --wandb-api-key-file ~/.secrets/wandb_api_key.txt \
  -- \
  --host_base_dir "$BASE_DIR" \
  --conda_envs_host "$BASE_DIR/.conda_envs" \
  --conda_pkgs_host "$BASE_DIR/.conda_pkgs" \
  --train_env atk --eval_env speech_eval \
  --train_config_path config_default2 \
  --train_arg train.trainer.devices=2
```

## 6) SLURM eval-only (기존 run_dir)

```bash
cd "$REPO_DIR"
bash slurm/submit_train_eval.sh \
  --job-name dtmae-eval \
  --gpus 1 --cpus 16 --mem 28G --time 24:00:00 \
  --output dtmae_eval_%j.out \
  --wandb-mode offline \
  -- \
  --host_base_dir "$BASE_DIR" \
  --conda_envs_host "$BASE_DIR/.conda_envs" \
  --conda_pkgs_host "$BASE_DIR/.conda_pkgs" \
  --train_env atk --eval_env speech_eval \
  --skip_train \
  --run_dir_in_container /workspace/AudioTokenization/outputs/your_existing_run
```

## 7) 상태 확인

```bash
squeue -u "$USER"
sacct -u "$USER"
scancel <JOB_ID>
```
