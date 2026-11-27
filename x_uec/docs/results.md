> 이 페이지의 기존 전체 결과는 이전 로컬 선택 자료를 사용했습니다. UEC main 기본 선택과 개수는 같아도 문장 목록은 다릅니다. 새 main 선택 실행의 결과로 인용하지 않습니다.

# 전체 MS MARCO/SNLI 실행 결과

[처음으로](../README.md) · [실행·재현 방법](reproduction.md)

2026-11-27에 검증한 CAIA SVD v3로 전체 데이터 정렬과 진단을 수행했습니다. 실행 ID는 `full_20261127_02`이며 결과는 **`alignment_ready_for_handoff=true`**입니다. 이 상태는 아래 로컬 출력 계약에서의 수렴·수치 검사를 뜻합니다.

## 사용한 데이터

기존 로컬 MS MARCO **3,983쌍**, SNLI **3,775쌍**의 선택 기록을 재사용했습니다. query/positive passage 및 premise/hypothesis를 각각 관측으로 펼쳐 총 **15,516개 occurrence**를 새로 인코딩했습니다. AG News나 예전 smoke 배열을 재사용하지 않았습니다. 원저자의 정확한 선택 ID를 복원했다고 주장하지 않습니다.

| 분할 | 관측 수 | 고유 원문 그룹 |
|---|---:|---:|
| alignment-fit | 12,417 | 12,318 |
| diagnostic | 3,099 | 3,080 |
| 평가 중복으로 제외 | 0 | 0 |

원문 그룹을 seed 42로 80/20 분리했고 같은 원문이 두 부분에 걸치지 않도록 했습니다. 기존 평가 6과제의 10개 query/corpus/support/test/text 파일과 중복을 독립 재감사했습니다. 이 diagnostic은 **정렬에만 held-out**이며, 원래 posterior는 해당 자료를 사용했습니다.

모델은 Qwen/Harrier/GTE, native 차원은 1024/640/768이며 공통 차원은 1024입니다. MS MARCO와 SNLI를 합쳐 모델마다 하나의 정렬 행렬을 fitting했습니다. 불확실성은 기존 **frozen-teacher v3 출력 대각 surrogate**입니다. 실제 수신자의 UEC-LA posterior, 최종 score 또는 downstream 성능과 동일하다는 검증은 아닙니다.

## 수렴 결과

fit에서만 정한 γ는 `0.086535880588821612`입니다. R0 최대 100회, GPA/CAIA 각각 최대 300회 조건으로 실행했으며 실제 종료는 다음과 같습니다.

| arm | 종료 상태 | sweeps | 상대 residual | 상대 gap |
|---|---|---:|---:|---:|
| r0 | `stationary_tolerance` | 18 | 7.298e-08 | 1.18516e-11 |
| gpa | `stationary_tolerance` | 0 | 7.298e-08 | 1.18516e-11 |
| caia | `stationary_tolerance` | 232 | 9.98984e-08 | 8.956e-11 |

GPA는 같은 새 R0에서 다시 검사했습니다. R0가 이미 평균 목적의 종료 기준을 만족해 **추가 반복이 0회**였습니다. GPA와 R0를 임의로 같은 것으로 취급한 것이 아닙니다. CAIA는 같은 R0에서 시작해 232회 갱신했습니다.

전체 CPU 수치 실행은 **296.123초**였습니다. 이 시간에는 입력 검증·통계·정렬·진단·출력 기록이 포함되며, 앞선 GPU 캐시 준비는 포함하지 않습니다. 공분산 guidance는 활성 상태였습니다.

## 같은 통계에서 다시 계산한 정렬 손실

GPA와 CAIA를 같은 matched-K와 같은 fit γ로 비교했습니다. GPA solver의 mean-only trace에 나타나는 covariance=0을 공분산 진단 값으로 쓰지 않았습니다.

| split | arm | mean | covariance | mean + γ covariance |
|---|---|---:|---:|---:|
| fit | gpa | 0.0674460812 | 0.0779400183 | 0.0741906893 |
| fit | caia | 0.0674741952 | 0.0771071709 | 0.0741467321 |
| diagnostic | gpa | 0.0723142659 | 0.0780073007 | 0.0790646964 |
| diagnostic | caia | 0.0723317862 | 0.0771777021 | 0.0790104266 |

CAIA는 이 실행에서 평균 손실이 약간 커지는 대신 공분산 손실과 총목적값이 감소했습니다. diagnostic에서도 같은 경향을 보였습니다. 이는 **정렬 목적에 대한 결과**이며 검색·STS 성능 향상률이 아닙니다.

## 큰 공분산 없는 계산 검산

작은 실제 diagnostic query-side 2개/document-side 4개에서 dense 기준과 implicit 계산을 비교했습니다. 균등·고정 비균등·입력별 Dirichlet·9개 one-hot을 사용했고, 양쪽 결합 평균·5개 raw 내적 모멘트와 같은 모델 one-hot native 불변성을 확인했습니다.

| dtype | 통과 비교 수 | 최대 절대 오차 | 허용차 atol / rtol |
|---|---:|---:|---|
| float64 | 198 | 1.22125e-15 | 1e-10 / 1e-8 |
| float32 | 198 | 1.19209e-07 | 1e-6 / 1e-4 |

총 **396개 비교가 모두 통과**했습니다. GPA/CAIA×3모델의 geometry 검사도 통과했고, 고정한 서로 다른 원문 1,000쌍에서 cosine 최대 절대 오차는 `7.77e-16`이었습니다. dense covariance는 작은 6개 관측의 기준 계산에만 사용했습니다.

## 대각만 남겼을 때의 차이

실제 query/document 역할의 서로 다른 원문 1,000쌍을 미리 고정하고 비대각을 보존하는 implicit 계산과 정렬 후 대각 절단을 비교했습니다. 2 arms×4 weight 조건×5 terms, 총 40개 요약을 기록했습니다. 아래는 raw 내적 분산의 `|diagonal−full|/full`입니다.

| arm | weight | 상대 차이 중앙값 | p95 |
|---|---|---:|---:|
| gpa | uniform | 0.0689054 | 0.100788 |
| gpa | fixed_0.6_0.2_0.2 | 0.00925045 | 0.0145916 |
| gpa | input_dirichlet | 0.0917931 | 0.286493 |
| gpa | onehot_qwen_qwen | 1.32331e-16 | 3.06061e-16 |
| caia | uniform | 0.0752201 | 0.108977 |
| caia | fixed_0.6_0.2_0.2 | 0.0100514 | 0.0157876 |
| caia | input_dirichlet | 0.0995267 | 0.309087 |
| caia | onehot_qwen_qwen | 1.32331e-16 | 3.06061e-16 |

평균 및 Qwen identity 대조도 통과했습니다. 이 차이는 **raw 내적 분산에서 비대각을 버린 영향**이며 최종 UEC score의 오차율이 아닙니다.

## 원본 기록과 재현 범위

수렴 상태, solver trace, 24개 모델 쌍별 loss 행, geometry, 396개 비교 및 40개 대각절단 요약은 [`artifacts/full_run_20261127/`](../artifacts/full_run_20261127/)에 있습니다. 출처 기록에는 검증한 소스·원래 입력 캐시·로컬 map의 SHA-256을 포함합니다. 바이너리 map과 원문·모델·posterior·대형 캐시는 Git에서 제외했습니다.

준비 기록의 `evaluation_scope` 필드에 선택 객체가 들어간 오류가 있어, 모델이 pin한 원본은 보존한 채 10개 평가 파일에서 독립적으로 감사 결과를 복원했습니다. 이 재계산을 최종 판단에 사용했습니다. 모델 policy는 상속 기록이며, 현재 interpreter package 버전과 Qwen author 소스는 추론 후에 원래 policy와 재대조했습니다. 실행 전 캡처라고 표현하지 않습니다.

첫 실행 `full_20261127_01`은 새 감사 parser가 문장 안 Unicode 구분 문자를 JSONL 경계로 오인해 fitting 전에 종료했습니다. 회귀 테스트로 고친 뒤 새 디렉터리 `full_20261127_02`에서 실행했으며, 위 결과는 모두 두 번째 실행입니다.

full runner 회귀 테스트 69개와 원래 수신용 테스트 29개가 통과했습니다. 이 브랜치는 수식 검산3개와 기존 준비 코드 연결·출처 검증 테스트11개도 포함하므로 현재 기본 테스트는43개입니다. 실행 시작 코드 hash와 export 전 hash를 대조했고, 필수 진단 coverage와 stationarity를 통과한 뒤 최종 상태를 기록했습니다.

남은 한계는 극단적인 원문 중복 입력에서 rejection sampler가 오래 걸릴 수 있다는 점입니다. 메모리는 관측 수와 샘플 수에 비례하며, 이번 15,516 occurrence/15,398 groups에서는 문제가 나타나지 않았습니다.

실제 UEC의 posterior·역할별 출력·가중치·정규화·최종 점수를 유지한 dense 기준 검산과 GPA+UEC/CAIA+UEC downstream 비교는 별도 연결 단계입니다. 수치 동치성, 성능 향상, 이론적 확장 증명은 구분합니다.
