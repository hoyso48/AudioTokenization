# semantic_eval Agent Readme

이 문서는 `semantic_eval/` 작업용 최신 기준 문서다.
이전 handoff 메모(장황한 탐색 기록, `eval.py` 재사용 전제, 범위 밖 자료)는 제거했다.

## 목적

- `ravdess`, `emovo`, `audio_mnist`, `slurp` 4개 데이터셋만 지원한다.
- `semantic_eval`은 `eval/eval.py`와 분리된 독립 파이프라인으로 유지한다.
- 평가 로직은 "원본 ARCH 방식"을 유지한다. 즉, split/fold/classifier 학습 로직은 ARCH upstream(`arch_eval`) 코드를 직접 호출하고, 우리 코덱은 wrapper로만 주입한다.
- 데이터셋 다운로드/환경 설정/평가 실행을 self-contained 스크립트로 제공한다.

## 범위(고정)

- 포함: 데이터셋 파서, manifest/label map 생성, ARCH upstream 평가 실행, 결과 집계 JSON.
- 제외: ARCH 전체 데이터셋 확장, 불필요한 baseline 실험 코드, 과도한 의존성 추가.

## 현재 파일 역할

- `semantic_eval/download_arch_datasets.sh`
  - 4개 데이터셋 다운로드 보조 스크립트.
- `semantic_eval/parsers.py`
  - 4개 데이터셋 파싱 + manifest 유틸.
- `semantic_eval/run.py`
  - ARCH upstream evaluator(`arch_eval`)를 직접 호출해 speech 4개 데이터셋 평가를 수행.
  - 우리 codec은 `arch_eval.Model` 호환 wrapper로만 주입하며, classifier/split/fold 로직은 ARCH 원본 코드를 그대로 사용.
  - `build` stage는 manifest/label map 기록용, `eval` stage는 원본 ARCH 방식 실행용, `all`은 둘 다 수행.
  - `--feature_source {post_vq,pre_vq}` 인자로 post-VQ(default) 또는 pre-VQ(vq 직전) feature를 선택할 수 있다.
  - 기본 동작으로 `results/{dataset}.json`이 이미 있고 현재 인자와 호환되면 해당 dataset 평가는 skip(reuse)한다.
  - `--force_recompute_existing`를 주면 기존 dataset 결과를 무시하고 처음부터 재평가한다.
- `semantic_eval/bootstrap_arch_semantic_eval.sh`
  - 환경 구성(venv + pip), ARCH repo clone/update, 데이터셋 다운로드, `run.py` 실행까지 한 번에 수행하는 self-contained 엔트리.
  - 각 단계는 완료 여부를 자동 확인해 skip한다. (`env` 메타 해시, ARCH repo 존재, 데이터셋 준비 상태, build/eval 산출물)
  - `--stage all`은 항상 `run.py --stage all`을 실행하고, `--stage auto`는 산출물 상태에 따라 `build|eval|all`을 자동 선택한다.
  - `--stage auto`에서 `-- --force_recompute_existing`를 전달하면 eval 완료 산출물이 있어도 다시 eval을 수행한다.
- `semantic_eval/requirements.txt`
  - semantic_eval 전용 의존성 묶음(`../requirements.txt` + `scikit-learn`, `gdown`, `pandas`, `tqdm`).

## 데이터셋 경로 규약

`parse_dataset(name, data_root)`는 다음 두 입력 형태를 모두 허용해야 한다.

1. 공통 루트 입력: `<root>/{ravdess,emovo,audio_mnist,slurp}`
2. 데이터셋 루트 직접 입력: `<root>`가 곧 해당 데이터셋 디렉토리

## 출력 규약(최소)

- manifest: JSONL (`dataset`, `split`, `path`, `label`, `sample_id`, optional fields)
- label map: `label_to_id`를 포함한 JSON
- 결과 JSON:
  - dataset별: `outputs/arch_speech/results/{dataset}.json`
  - 종합: `outputs/arch_speech/results/summary.json` (`arch_1`, `arch_2` 포함)

## 구현 원칙

- 데이터 원본을 수정하지 않는다. (파서에서 `.trans.txt` 생성 금지)
- dead code/중복 코드는 유지하지 않는다.
- 기본 동작은 재현 가능해야 하며(`seed`), 실패 메시지는 경로 기준으로 명확해야 한다.
- ARCH 원본과의 정합성이 최우선이다. split/fold/probe 학습 루프를 `run.py`에서 재구현하지 않고, upstream evaluator를 호출한다.

## 다음 작업

1. `README.md` 실행 예시를 ARCH 원본-호환 모드 기준으로 갱신
2. 결과 표(`Table 4`) 직결용 export 포맷(csv/latex) 추가
3. upstream ARCH 버전(커밋) 고정/검증 옵션 강화
