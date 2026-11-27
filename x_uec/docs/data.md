# 실제 UEC 입력 준비

[처음으로](../README.md) · [계산·진단 연결](integration.md)

## 어떤 데이터를 사용하나요?

수신자가 기존 UEC uncertainty fitting에 사용한 **동일한 선택 기록과 posterior**를 출발점으로 사용합니다. 이번 로컬 실험은 MS MARCO 3,983쌍과 SNLI 3,775쌍을 재사용했습니다. 이 개수만으로 원저자의 정확한 선택 ID를 복원했다고 볼 수는 없습니다.

CAIA에서 한 행은 모델 간에 대응하는 **같은 텍스트·같은 실제 역할의 관측**입니다.

| 데이터 | 별도 관측으로 만드는 역할 | 같은 행에서 모델끼리 비교하는 대상 |
|---|---|---|
| MS MARCO | query, positive passage | 같은 query끼리 또는 같은 positive passage끼리 |
| SNLI | premise, hypothesis | 같은 premise끼리 또는 같은 hypothesis끼리 |

query와 positive passage를 같은 문장으로 취급하지 않습니다. 정답·negative 목록·likelihood·logical batch·posterior 설정은 원래 정의를 유지합니다. 실제 UEC fitting에 negative 관측이 사용되었다면 그 원래 선택과 역할도 기록합니다. 이번 로컬 정렬 입력은 query/positive와 premise/hypothesis였고, negative 관계는 원본 triplet에 보존했습니다.

## 모델과 출력 단계

현재 수신용 예제는 아래 세 모델에 맞춰 입력을 검사합니다.

| key | 모델 | native 차원 | 공통 차원 |
|---|---|---:|---:|
| `qwen` | `Qwen/Qwen3-Embedding-0.6B` | 1024 | 1024 |
| `harrier` | `microsoft/harrier-oss-v1-270m` | 640 | 1024 |
| `gte` | `thenlper/gte-base` | 768 | 1024 |

평균과 분산은 **같은 최종 L2 출력 단계**의 값이어야 합니다. 분산은 좌표별 출력 분산이며, 표준편차·parameter variance·hidden-state variance가 아닙니다. 이미 최종 출력 단계라면 L2 Jacobian을 다시 적용하지 않습니다.

모델별 revision, prompt, pooling, 길이 제한, padding, dtype와 posterior 출처를 기록합니다. 모델마다 같은 prompt를 강제하지 않습니다. 같은 문장의 평균을 다른 역할의 분산과 결합하지 않습니다.

## NPZ 배열

모델 `m`마다 네 배열, 총 12개 key가 필요합니다. `N`은 감사·분리한 **alignment-fit 관측 수**입니다.

| key | shape | dtype·의미 |
|---|---|---|
| `mean__m` | `(N, d_m)` | float32/64, 최종 정규화된 native 평균 |
| `variance__m` | `(N, d_m)` | float32/64, 같은 단계의 비음수 출력 분산 |
| `mean_ids__m` | `(N,)` | 평균 producer의 Unicode ID |
| `variance_ids__m` | `(N,)` | 분산 producer의 독립 Unicode ID |

모든 배열은 finite여야 합니다. 모든 모델의 ID와 각 평균/분산 ID가 같은 순서인지 검사합니다. ID에는 역할과 occurrence를 포함합니다. 예: `pair17:query:0`, `pair17:positive_passage:0`.

아래 코드는 **이미 준비된 실제 배열**을 저장하는 형식 예시입니다.

```python
import numpy as np

payload = {}
for m in ("qwen", "harrier", "gte"):
    payload["mean__" + m] = output_means[m]
    payload["variance__" + m] = output_variances[m]
    payload["mean_ids__" + m] = mean_producer_ids[m]
    payload["variance_ids__" + m] = variance_producer_ids[m]
np.savez("alignment_fit.npz", **payload)
```

평균 ID를 복사해 분산 ID를 새로 만드는 대신 실제 producer 기록을 사용합니다. 불일치한 행을 자동 교집합이나 재정렬로 숨기지 않습니다.

## 먼저 감사와 분할을 고정하세요

1. 원문을 NFC 및 CRLF/CR→LF만 적용해 canonical hash로 그룹화합니다. 이 처리는 hash 계산용이며 인코더 입력을 바꾸지 않습니다.
2. 선언한 평가 범위의 query/corpus/support/test 원문과 중복을 확인하고 정렬 입력에서 제외합니다.
3. 고유 그룹을 정렬한 뒤 seed 42로 섞어 약 80/20 alignment-fit/diagnostic으로 나눕니다. 같은 원문은 역할이 달라도 한 부분에만 둡니다. occurrence는 유지합니다.
4. γ와 정렬 행렬은 fit에서만 계산하고 진단 관측을 따로 보관합니다. posterior가 이미 해당 자료를 봤다면 **정렬에만 held-out**이라고 표시합니다.

이번 로컬 실행에서는 fit 12,417개, diagnostic 3,099개였고 선언한 평가 6과제의 10개 원문 파일과 중복은 0개였습니다.

## 출처 JSON

[`input_provenance.template.json`](../input_provenance.template.json)을 복사해 실제 선택 hash, revision, posterior, 역할·출력 정책과 감사 범위를 채웁니다. placeholder나 미완료 감사는 입력 검사에서 거부합니다.

`example.py`는 이 선언과 배열·ID를 검사하지만, 원문 데이터에 접근해 감사가 사실인지 재수행하지는 않습니다. 감사 파일과 diagnostic 선택을 함께 보존하세요.
