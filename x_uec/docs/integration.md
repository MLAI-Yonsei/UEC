# 정렬 결과를 기존 UEC에 연결하기

[처음으로](../README.md) · [데이터 준비](data.md) · [수식](math.md)

## 저장하는 것은 무엇인가요?

GTE의 native 차원이 768, 공통 차원이 1024인 경우 문장마다 다음을 사용합니다.

| 정보 | shape | 보관 단위 |
|---|---|---|
| 정렬된 평균 `mu @ R` | `(1024,)` | 문장별 |
| native 출력 분산 `v` | `(768,)` | 문장별 |
| 정렬 행렬 `R` | `(768, 1024)` | 모델별 한 번 |

개념적인 정렬 공분산은 `R.T @ diag(v) @ R`입니다. `covariance_ops.py`는 필요한 계산을 이 분산 벡터와 공유 행렬로 수행합니다. **정렬 공분산의 대각만 남기는 방식이 아니며**, 회전으로 생긴 비대각 성분의 영향도 계산에 포함합니다.

이 API는 native 대각 surrogate를 받습니다. native full covariance의 비대각을 버려도 같은 계산이라는 뜻은 아닙니다.

## 모델별 native 출력을 함께 전달합니다

`x_uec/`를 작업 디렉터리 또는 import 경로에 둡니다. 아래 변수는 기존 UEC에서 생성한 실제 배열입니다. query/document를 각각 준비하고, coefficient도 그쪽의 실제 UEC 정책으로 계산합니다.

```python
import numpy as np
from covariance_ops import (
    prepare_raw_bundle, build_overlap_blocks, pair_moments_implicit,
)

# map·평균·분산·coefficient 모두 같은 dtype으로 준비합니다.
# solver가 저장한 float64 map을 float32 출력과 섞지 않습니다.
dtype = np.float32
with np.load("outputs/my_alignment/caia_maps.npz") as z:
    maps = {m: z[m].astype(dtype) for m in ("qwen", "harrier", "gte")}
overlap = build_overlap_blocks(maps)

query = prepare_raw_bundle(
    native_means={m: np.asarray(a, dtype=dtype) for m, a in query_means.items()},
    native_vars={m: np.asarray(a, dtype=dtype) for m, a in query_variances.items()},
    maps=maps,
    mean_ids_by_model=query_mean_ids,
    variance_ids_by_model=query_variance_ids,
    alpha=np.asarray(query_coefficients, dtype=dtype),
    alpha_ids=query_coefficient_ids,
    role="query",
)
document = prepare_raw_bundle(
    native_means={m: np.asarray(a, dtype=dtype) for m, a in document_means.items()},
    native_vars={m: np.asarray(a, dtype=dtype) for m, a in document_variances.items()},
    maps=maps,
    mean_ids_by_model=document_mean_ids,
    variance_ids_by_model=document_variance_ids,
    alpha=np.asarray(document_coefficients, dtype=dtype),
    alpha_ids=document_coefficient_ids,
    role="document",
)
moments = pair_moments_implicit(query, document, overlap)
M = moments["dot_mean"]
V = moments["dot_variance"]
```

`alpha`는 모델 순서 `qwen, harrier, gte`의 `(관측 수, 3)` 배열이며 finite·비음수·행 합 1이어야 합니다. 계산에 넣을 때는 관측별로 결정된 값이어야 합니다. 모델 간 및 query/document 확률변수의 독립성을 가정합니다.

각 쪽의 coefficient ID는 비어 있지 않은 Unicode 배열이어야 합니다. 모든 모델의 독립 mean/variance producer ID와 **같은 값·같은 순서**인지 검사합니다. query와 document는 각각 자기 관측의 ID를 사용합니다. float64를 사용할 때도 위 `dtype` 하나를 바꿔 모든 수치 입력을 함께 맞춥니다.

`prepare_raw_bundle`은 이미 정렬된 평균만을 입력받는 함수가 아닙니다. **모델별 native 평균·분산과 map**을 입력받아 내부에서 정렬·결합에 필요한 정보를 준비합니다. 입력을 재정규화하거나 분산을 임의 재조정하지 않습니다.

## 반환 값과 UEC 최종 점수

| key | 의미 |
|---|---|
| `dot_mean` | raw 내적 평균 `mu_q.T @ mu_d` |
| `variance_from_document` | `mu_q.T @ Sigma_d @ mu_q` |
| `variance_from_query` | `mu_d.T @ Sigma_q @ mu_d` |
| `variance_cross` | `tr(Sigma_q @ Sigma_d)` |
| `dot_variance` | 위 세 분산 항의 합 |

이 함수가 반환하는 것은 **정규화 전 raw 모멘트**입니다. 기존 [`uec.py`](../../uec.py)의 `GaussianConvolution.cosine_similarity`는 평균 내적을 `sqrt(1 + (pi/8) * beta * variance)`로 조정합니다. 이 기존 정의에 연결하려면 그 직전까지의 coefficient·평균/분산 stage·추가 정규화가 위 M/V와 일치하는지 먼저 대조해야 합니다. 행렬 이름만 맞춰 자동 대체하지 않습니다.

기존 UEC의 diagonal 분산 API에 정렬 후 대각만 넘기면 비대각 기여를 잃습니다. 최종 점수까지의 연결은 자신의 실제 UEC 출력과 작은 dense 기준에서 검산하세요. 이 브랜치는 기존 UEC score 경로를 자동 교체하지 않습니다.

## 작은 검산부터 진행하세요

1. 새 map의 row-isometry `R @ R.T = I`와 같은 모델 안 cosine 보존을 확인합니다.
2. 별도 diagnostic의 서로 다른 실제 query-side 2개/document-side 4개로 작은 dense 기준을 만듭니다.
3. 균등·비균등·입력별 coefficient와 모델별 one-hot에서 결합 평균 및 5개 모멘트를 비교합니다. float64와 실제 사용 dtype을 모두 확인합니다.
4. 같은 입력·가중치·정규화·β를 사용해 **최종 UEC 점수도** dense 기준과 비교합니다.
5. 이후 고정한 평가에서 GPA+UEC와 CAIA+UEC를 비교합니다. 같은 posterior와 coefficient 정책을 사용합니다.

`example_implicit.py`는 작은 합성 입력에서 dense/implicit·블록·문서 chunk 처리의 예시를 제공합니다. 실제 posterior와 최종 UEC 점수 검증을 대신하지는 않습니다.

## 많은 문서를 처리할 때

문서 chunk를 읽고 `prepare_raw_bundle`로 준비한 뒤 필요한 query/document 블록을 계산하고 버립니다. `iter_pair_moments`로 결과를 블록별 소비할 수 있습니다. 전체 Q×D 결과를 모으지 않아야 출력 쪽 메모리도 제한됩니다.

`transform_gaussian()`과 dense reference는 작은 검산·호환용입니다. 대량 문서에 문장별 1024×1024 공분산을 만들어 저장하는 기본 경로로 사용하지 않습니다.
