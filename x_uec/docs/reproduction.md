# 기존 AGNews 실험 코드에서 이어서 실행하기

[실행 안내](../README.md) · [입력 파일](data.md) · [실험 결과](results.md)

## 기존 코드에서 재사용하는 것

이전 AGNews 실험과 MS MARCO/SNLI 전체 실행에서 사용한 native 평균·분산 계산 경로를 그대로 호출합니다. 새 불확실성 계산식을 구현하거나 원래 UEC-LA를 대신 구현한 경로가 아닙니다.

| 단계 | 호출하는 기존 코드 |
|---|---|
| 실제 선택·분할 | `reference_experiment/e4_data.py`의 `prepare_data(..., mode='full')` |
| 모델별 문장 평균·출력 분산 | 기존 작업 폴더 `caia_svd_pilot_v3/uncertainty.py`의 `capture`, `native_forward`, `diagonal` |
| 기존 피팅 Gaussian | 원래 v3 실행의 `gaussian/{qwen,harrier,gte}/` |
| 배열·출처 JSON 조립 | `reference_experiment/e4_data.py`의 `assemble_cache` |
| 전체 정렬·진단 | `reference_experiment/full_e4.py` |

`prepare_alignment.py`는 위 단계들을 순서대로 실행하는 진입점입니다.
새 `reference_experiment/prepare_cache.py`는 실제 사용했던 `prepare_caia_e4_v3_full_cache.py`에서 작업 폴더·import 위치를 인자로 받을 수 있게 연결한 파일입니다. 모델별 분산 계산식은 변경하지 않았습니다. 실제 실행 소스, 원본 모델·피팅 Gaussian의 hash와 runtime version을 검사합니다.

## 필요한 외부 작업 폴더

공개 Git 저장소에는 아래 데이터·모델·환경을 포함하지 않았습니다. 현재 실험 서버의 원래 폴더는 `/data2/claire/UEC`입니다. 새 clone의 경로를 `--workspace`로 넣는 것이 아니라, **아래 기존 자산이 있는 폴더**를 지정합니다.

```text
기존실험폴더/
├── .venv/bin/python
├── .venv-reproduction/bin/python
├── caia_svd_pilot_v3/
├── caia_svd_pilot_v2/
├── caia_pair_pilot/
├── gte_la/
├── uec_experiment/
├── alignment_fma/
├── gte_frozen_uncertainty/
└── runs/
    ├── gte_la_validation_20261124/v1/data/
    ├── caia_svd_frozen_triple_20261124/v3/
    │   ├── gaussian/{qwen,harrier,gte}/
    │   ├── covariance/{qwen,harrier,gte}/summary.json
    │   ├── manifest.json
    │   └── input_hashes.json
    ├── qwen_reproduction_20261110/author_protocol/reference/qwen_author_model.py
    └── cross_predictive_partial_pilot6_20261119/data/
```

작업 폴더의 부모 디렉터리에는 기존 `home-offload/storage-env.sh`도 필요합니다.
그 파일의 `/data1`·`/data2` 캐시 설정을 다시 읽어, 현재 셸에 남은 `/home` 경로를 덮어씁니다.

이 코드가 참조하는 원래 source 의존성과 이미 저장한 Hugging Face checkpoint도 있어야 합니다. source와 데이터 경로는 원래 seal에 묶여 있습니다. 작업 폴더를 다른 서버로 단순 복사하면 원래 절대 경로와 seal까지 자동으로 이식되는 것은 아닙니다. 이 경로는 현재 기존 실험 작업 폴더에서 실제 실행 흐름을 재사용하는 용도입니다.

새 clone만 있고 위 자산이 없다면 이 준비 명령을 그대로 실행할 수 없습니다. 그런 환경에서는 실제 모델·불확실성 모델·선택 자료를 어떤 형식으로 이전할지 별도 준비가 필요합니다. 없는 자산을 새 UEC posterior나 임의 선택으로 자동 대체하지 않습니다.

## 실제 실행 명령

새 clone의 `x_uec/`에서 실행합니다. 필요한 모델·불확실성 피팅 결과는 재사용하고, 선택한 문장들의 평균·분산은 새로 계산합니다.

```bash
python prepare_alignment.py \
  --workspace /data2/claire/UEC \
  --output /data2/claire/xuec_inputs_new \
  --gpu 1

OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
  python run_alignment.py \
  --inputs /data2/claire/xuec_inputs_new \
  --output /data2/claire/xuec_alignment_new
```

`--gpu`는 비어 있는 물리 GPU 번호입니다. 세 모델은 같은 GPU를 순서대로 사용합니다. 기존 작업을 종료하거나 GPU를 강제로 비우지 않습니다. 새 출력 폴더를 사용하며, 이전 실패/완료 폴더를 덮어쓰지 않습니다.

준비 중 새 encoder 추론과 출력 분산 계산이 있습니다. 기존 Gaussian을 다시 피팅하거나 encoder를 업데이트하지 않습니다. 이전 AGNews 문장별 분산과 정렬 행렬을 새 자료의 결과로 복사하지 않습니다.

## 이미 준비한 입력의 연결 확인

모델 추론이나 정렬을 다시 하지 않고, 준비된 전체 입력·선택·원본 출처와 자동 설정 생성을 확인할 수 있습니다.

```bash
python run_alignment.py \
  --inputs /data2/claire/UEC/runs/caia_e4_v3/inputs_20261127_full_v1 \
  --output /data2/claire/unused_check_result \
  --check-only
```

성공 시 `prepared_inputs_verified`, 전체 관측 수와 자동 config 경로를 출력합니다.
과거 입력에서 시작 시 runtime 기록이 없는 경우에는 원래 `runtime_policy_verification.json`을 검증해 전달합니다.
이 기록은 후향 검증이며, 실행 전 정책 캡처로 표현하지 않습니다. `--check-only`는 새 평균·분산 계산이나 재정렬을 수행한 것으로 기록하지 않습니다. 생성 config는 `--output` 경로 옆의 `.config.json` 파일에 저장합니다. 기존 입력 폴더는 수정하지 않습니다.

## 종료 상태

| 상태 | 뜻 |
|---|---|
| `stationary_tolerance` | 해당 solver가 residual/gap 기준을 만족 |
| `budget_exhausted` | 반복 예산 종료; 수렴으로 표시하지 않음 |
| `stalled_not_stationary` | loss 변화는 작지만 stationarity 기준 미달 |
| `review_required` | 전체 진단 후에도 미수렴 결과 검토가 필요 |
| `alignment_ready_for_handoff` | 해당 입력 계약에서 수렴 및 필수 진단 통과 |
| `numerical_failure` | 입력·계산·진단 검증 실패; 승인 map 없음 |

full runner는 통과한 map을 `maps/`, 수치적으로 유효하지만 미수렴한 map을 `diagnostic_only/`에 보관합니다. 최종 상태를 마지막에 기록하므로 계산 중간의 파일만 보고 성공으로 판단하지 않습니다.

## 검증 범위

기존 전체 실행은 12,417개 정렬용/3,099개 진단용 관측으로 수행했고 원본 수치는 [`artifacts/full_run_20261127/`](../artifacts/full_run_20261127/)에 있습니다. 이번 진입점 추가 검증은 실제 원문 준비 단계와 기존 전체 입력의 출처 확인, 실행 순서·환경·자동 config 테스트입니다. 이 추가 검증을 새 GPU 전체 추론 결과라고 표현하지 않습니다.

개발자의 합성 예제·단위 테스트는 [선택적 점검](checks.md)에 있습니다. `reference_experiment/`의 기존 회귀 테스트 69개도 모델 추론 없이 실행할 수 있습니다.
