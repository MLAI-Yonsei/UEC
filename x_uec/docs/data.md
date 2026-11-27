# 입력 파일은 자동으로 만들어집니다

[실행 안내](../README.md) · [기존 실험 재현 조건](reproduction.md)

실제 실행은 `prepare_alignment.py` → `run_alignment.py` 순서입니다.
`alignment_fit_provenance.json`을 손으로 작성하는 단계는 이 경로에 없습니다.
기존 실행 코드가 사용한 **실제 파일 이름**은 다음과 같습니다.

## 준비 전: 기존 실험 폴더에서 읽는 것

| 기존 경로 | 쓰임 |
|---|---|
| `runs/gte_la_validation_20261124/v1/data/triplets.jsonl` | 이미 선택한 MS MARCO/SNLI 쌍과 원문·원래 관계 |
| 같은 폴더의 `selection.json`, `dataset_sources.json`, `manifest.json` | 선택 설정·데이터 버전·checksum |
| `runs/caia_svd_frozen_triple_20261124/v3/gaussian/{모델}/` | 기존에 피팅한 모델별 Gaussian과 원래 모델 정책 |
| 같은 v3 폴더의 `manifest.json`, `input_hashes.json` | 부모 실행의 데이터·모델·소스가 같은지 확인하는 기록 |
| `runs/cross_predictive_partial_pilot6_20261119/data/` | 기존 평가6과제의 원문 중복 검사 |

원래 공개 UEC의 `fit_la.py`가 이 경로의 파일들을 자동으로 만든다는 뜻은 아닙니다.
이들은 **이전에 AGNews 및 MS MARCO/SNLI 실험을 수행한 작업 폴더의 자산**입니다.
그 작업 폴더를 재사용하는 실행 경로를 제공한 것입니다. 없으면 먼저 자산을 준비해야 합니다.

## 준비 후: 새 출력 폴더에 생기는 것

| 자동 생성 파일 | 내용 |
|---|---|
| `observations_all.jsonl` | 선택한 모든 문장과 실제 역할·원래 ID |
| `observations.jsonl` | 중복 검사 후 사용하는 관측 |
| `split.json` | 정렬용·진단용 문장 분할 |
| `source_manifest.json` | 읽은 데이터와 부모 실행 출처 |
| `{모델}/mean.npy`, `variance.npy` | 이번 문장에서 새로 계산한 평균·출력 분산 |
| `{모델}/mean_ids.npy`, `variance_ids.npy` | 계산한 관측의 ID |
| `{모델}/summary.json` | 모델 정책·원본 encoder hash·출력 checksum·계산 시간 |
| `uec_alignment_inputs.npz` | 세 모델의 평균·분산·ID를 묶은 전체 입력 |
| `uec_input_provenance.json` | 위 입력과 실제 계산 출처를 자동으로 묶은 JSON |

`uec_input_provenance.json`은 사용자가 양식을 채우는 파일이 아닙니다.
`reference_experiment/prepare_cache.py`가 모델별 계산을 마친 뒤 `assemble` 단계에서 만듭니다.
현재 작업 경로의 관측 파일, 원래 source 파일, 모델 기록과 실제 출력 checksum에서 값을 읽습니다.

## NPZ 안의 배열 이름

각 모델에는 아래 네 배열이 있습니다. `__`는 밑줄 두 개입니다.

| 모델 | 평균 | 분산 | ID 두 개 |
|---|---|---|---|
| Qwen | `mean__qwen` | `variance__qwen` | `mean_ids__qwen`, `variance_ids__qwen` |
| Harrier | `mean__harrier` | `variance__harrier` | `mean_ids__harrier`, `variance_ids__harrier` |
| GTE | `mean__gte` | `variance__gte` | `mean_ids__gte`, `variance_ids__gte` |

이 NPZ에는 정렬용과 진단용 관측이 모두 들어갑니다. `split.json`이 구분합니다.
전체 runner는 그중 정렬용 관측만으로 γ와 행렬을 구하고 진단용 관측을 따로 사용합니다.
이전 개발 예제 `example.py`는 정렬용 관측만 받는 다른 입력 경로이므로 두 형식을 섞지 않습니다.

평균·분산의 열 개수는 Qwen1024/Harrier640/GTE768입니다. 모델 사이의 같은 행은
같은 문장·같은 역할을 뜻합니다. query와 positive passage, premise와 hypothesis는 별도 관측입니다.
정렬 입력은 같은 최종 출력 단계의 평균과 native 좌표별 분산이며, 파라미터 분산을 그대로 넣지 않습니다.

## 실행 설정도 자동으로 만듭니다

`run_alignment.py --inputs 준비폴더 --output 결과폴더`는
`uec_input_provenance.json`과 독립된 모델별 계산 기록을 읽고,
결과 폴더 옆에 `결과폴더명.config.json`을 자동 생성합니다. 입력 폴더는 수정하지 않습니다.
사용자가 model revision이나 backend checksum을 다시 옮겨 적지 않습니다.

준비 파일·원래 source·실행 코드가 계산 중 바뀌면 검증에서 거부합니다.
준비 단계의 과거 `evaluation_scope` 기록 오류는 원본을 변경하지 않고
full runner가 만드는 `verified_full_audit.json`에서 독립 재검사합니다.

`input_provenance.template.json`은 작은 개발 예제용으로 남긴 양식이며,
이 실제 준비·전체 정렬 흐름에서는 사용하지 않습니다.
