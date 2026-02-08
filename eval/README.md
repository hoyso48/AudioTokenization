# DTP Tau-L Analysis Notes

이 문서는 `eval/dtp_tau_l_curve.py`로 진행한 분석(체크포인트 기반 `L` 분포 + `tau -> avg_r`)과,
최근 반영된 보정 로직/보조 곡선 의미를 정리한다.

## 1) 무엇을 계산하나

- 입력 오디오(예: LibriSpeech clean filelist)를 모델 encoder(level=1)까지 통과
- 시퀀스별 누적 similarity path length `L` 계산
  - `d_t = 1 - cos(x_t, x_{t-1})`
  - `D_j = sum_{t<=j} d_t`, `L = D_{N-1}`
- 고정 `tau`를 sweep하며 PLE frontier를 재현하고 `avg_r` 계산
  - 파란선: `tau`에 대한 token-weighted `avg_r`
  - 파란 음영: per-seq `r`의 q10~q90

## 2) 작은 tau 구간 clamp 일관성 보정

`DTMAE/dtp/ops.py`의 `PLEBatchTopK`, `PLEBatchTopKJitter`에 아래 최소 보정을 반영함.

- 조건: `m_raw = floor(L/tau) > N-1` (tiny-tau saturation)
- 처리: 해당 시퀀스는 경계 토큰을 모두 keep (`mask[:,1:]=True`)

이 보정으로 매우 작은 `tau`에서의 병리적 중복 경계 collapse를 피하고,
`tau -> avg_r` 곡선 해석이 안정적이게 된다.

## 3) 보조 곡선(주황) 정의

요청된 역수형 정책만 사용:

- `tau_b = (L_b / N_b) / (1 - target_r)`

이때 `target_r`를 여러 값으로 sweep(기본 `0.1~0.9`, step `0.1`)해서,
각 target에 대해 실제로 나온 `avg_r`를 계산하고 `tau_b_mean` 위치에 주황 점선/마커로 오버레이한다.

즉, 주황은 "고정 tau sweep"이 아니라 "공식 기반 per-seq tau 정책 sweep" 결과다.

## 4) 실행 방법

```bash
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate hoyso_ml

python eval/dtp_tau_l_curve.py \
  --run_dir /path/to/run_dir \
  --input DTMAE/filelists/librispeech_test_clean.txt \
  --output_dir /path/to/output_dir \
  --tau_min 0.001 --tau_max 1.0 --tau_step 0.01 \
  --target_avg_r 0.5 \
  --formula_r_min 0.1 --formula_r_max 0.9 --formula_r_step 0.1 \
  --length_mode pad --num_workers 4 --save_l_npy
```

## 5) 주요 출력 파일

- `l_distribution_and_tau_avg_r.png`: 좌측 `L` 히스토그램, 우측 파란/주황 곡선
- `tau_avg_r_curve.csv`: 파란선 데이터(`fixed tau` sweep)
- `formula_r_sweep.csv`: 주황선 데이터(공식 기반 sweep)
- `l_per_sequence.jsonl`: 시퀀스별 `L`, `N` 메타
- `summary.json`: 요약 통계(closest tau, formula sweep 결과 포함)

## 6) 현재 관찰 포인트

- 본 실험 설정에서는 주황(공식 sweep)과 파란(고정 tau sweep)이 중고 `r` 구간에서 잘 맞고,
  낮은 `r` 구간에서 차이가 상대적으로 더 커질 수 있다.
- `tau` 단일값 튜닝은 파란선(`tau_avg_r_curve.csv`)을 기준으로,
  공식 초기값/보조 해석은 주황선(`formula_r_sweep.csv`)을 기준으로 보면 된다.
