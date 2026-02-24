# LibriTTS AR-TTS Implementation Plan (FFR/VFR)

## Goal

`/home/hoyso/projects/AudioTokenization/tts` 에 다음 조건을 만족하는 TTS 파이프라인을 구현한다.

- LibriTTS 기반
- AR Transformer (VALL-E 유사)
- HuggingFace Trainer 기반 학습
- Offline-first (데이터/코드북/토크나이저 사전 준비)
- FFR/VFR 두 가지 variant

## Core Variant Definitions

### 1) FFR setup

- 단순 AR Transformer baseline
- 음성 codebook token만 예측
- loss: token CE only

### 2) VFR setup (critical)

아래 조건에서만 활성화:

- `use_dtp=True`
- selector가 fixed pattern이 아님

핵심 동작:

1. mask를 kept token 기준 span length로 변환
   - `span_len = (# trailing zeros) + 1`
2. `span_len`을 최대값으로 제한
   - `max_span_len` (기본값 `512`)
3. 임베딩에서 span 정보를 더함
   - `h = token_emb(token_id) + span_emb(span_len)`
4. 출력 head를 분리
   - token vocab head
   - span length head (독립 예측)
5. 학습 loss
   - `L = L_token + lambda_span * L_span`

## TODO Checklist

### A. Project scaffold

- [ ] `configs/`, `scripts/`, `src/tts/`, `tests/` 생성
- [ ] 실행/재현 중심 `README.md` 작성

### B. Self-contained environment

- [ ] 환경 설치 스크립트 작성 (`setup_env_tts.sh`)
- [ ] `requirements_tts.txt` 작성
- [ ] offline 실행 옵션 문서화
  - `HF_DATASETS_OFFLINE=1`
  - `TRANSFORMERS_OFFLINE=1`

### C. Dataset prep (LibriTTS)

- [ ] LibriTTS 다운로드/검증 스크립트 작성
- [ ] train/val/test manifest 생성
- [ ] 오디오/텍스트 정합성 검증

### D. Offline codebook extraction

- [ ] pretrained codec ckpt 기반 codebook token 추출 스크립트 작성
- [ ] utterance별 token 저장
- [ ] VFR용 trailing/span 라벨 저장
- [ ] `max_span_len` clipping 적용
- [ ] metadata 저장

### E. Training data pipeline

- [ ] Dataset: text + prompt + target speech tokens
- [ ] Data collator: dynamic padding + attention mask + labels
- [ ] FFR/VFR 라벨 포맷 동시 지원

### F. AR model implementation

- [ ] decoder-only AR transformer 구현
- [ ] FFR forward: token logits/loss
- [ ] VFR forward: token logits + span logits + joint loss
- [ ] causal mask 및 label mask 정합성 보장

### G. HuggingFace Trainer integration

- [ ] `train.py` 작성 (`transformers.Trainer` + `TrainingArguments`)
- [ ] config 기반 실행 파이프라인 정리
- [ ] resume/checkpoint/eval/logging 동작 보장

### H. Inference + decode

- [ ] `infer.py` 작성
- [ ] FFR decode 경로
- [ ] VFR decode 경로 (token+span -> mask 복원 -> waveform)

### I. Tests / sanity

- [ ] unit test: mask -> trailing zeros -> span 변환
- [ ] unit test: span clipping (`max_span_len`)
- [ ] unit test: output/loss shape 검증
- [ ] smoke test: 소규모 학습 + 샘플 합성

## Acceptance Criteria

- clean 환경에서 다음 순서가 end-to-end 동작
  1. env setup
  2. LibriTTS 준비
  3. offline codebook 추출
  4. HF Trainer 학습 (FFR 또는 VFR)
  5. 추론 및 waveform 생성
- VFR 구현에서 아래 4가지가 모두 충족
  - mask-to-span 변환
  - span cap (기본 512)
  - span embedding 추가
  - span head 독립 예측
