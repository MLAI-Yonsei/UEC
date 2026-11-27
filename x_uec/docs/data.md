# 어떤 파일을 준비해야 하나요?

[실행 안내](../README.md) · [정렬 후 계산 연결](integration.md)

현재 `example.py`가 정렬을 시작할 때 직접 읽는 파일은 **두 개**입니다.
아래 이름은 실행 안내와 맞춘 권장 이름입니다. 파일명은 바꿀 수 있지만,
명령의 `--input`, `--provenance` 경로도 같이 바꿔야 합니다.

```text
x_uec/
└── local_data/
    ├── alignment_fit.npz
    └── alignment_fit_provenance.json
```

이 파일들은 저장소에 완성된 실험 입력으로 포함되어 있지 않습니다.
원래 UEC에서 사용한 실제 문장과 해당 모델의 출력으로 준비해야 합니다.

## 1. alignment_fit.npz: 실제 평균·분산·문장 ID

NPZ는 여러 숫자 배열을 이름을 붙여 한 파일에 담는 NumPy 형식입니다.
이 파일 안에 **아래 12개 이름**으로 배열이 있어야 합니다. 이름의 `__`는 밑줄 두 개입니다.
`N`은 별도 진단용 문장을 제외한, 이번 정렬에 사용할 문장 관측 수입니다.

| 파일 안의 정확한 배열 이름 | 크기 | 들어갈 값 |
|---|---|---|
| `mean__qwen` | `(N, 1024)` | Qwen의 문장 임베딩 평균 |
| `variance__qwen` | `(N, 1024)` | 같은 문장의 Qwen 출력 분산 |
| `mean_ids__qwen` | `(N,)` | Qwen 평균을 계산한 문장의 ID |
| `variance_ids__qwen` | `(N,)` | Qwen 분산을 계산한 문장의 ID |
| `mean__harrier` | `(N, 640)` | Harrier의 문장 임베딩 평균 |
| `variance__harrier` | `(N, 640)` | 같은 문장의 Harrier 출력 분산 |
| `mean_ids__harrier` | `(N,)` | Harrier 평균을 계산한 문장의 ID |
| `variance_ids__harrier` | `(N,)` | Harrier 분산을 계산한 문장의 ID |
| `mean__gte` | `(N, 768)` | GTE의 문장 임베딩 평균 |
| `variance__gte` | `(N, 768)` | 같은 문장의 GTE 출력 분산 |
| `mean_ids__gte` | `(N,)` | GTE 평균을 계산한 문장의 ID |
| `variance_ids__gte` | `(N,)` | GTE 분산을 계산한 문장의 ID |

평균·분산은 float32 또는 float64의 유한한 숫자이고 분산은 비음수여야 합니다.
평균은 이미 길이 1로 정규화한 출력이며 분산도 **그와 같은 출력 단계**의 값이어야 합니다.
분산에 표준편차나 파라미터 분산을 넣지 않습니다. 정규화 전 단계의 분산을
파일 이름만 바꿔 넣어서도 안 됩니다.

현재 이 입력 프로그램은 다음 세 모델을 검사합니다.

| 이름 | 모델 |
|---|---|
| `qwen` | `Qwen/Qwen3-Embedding-0.6B` |
| `harrier` | `microsoft/harrier-oss-v1-270m` |
| `gte` | `thenlper/gte-base` |

다른 모델의 배열을 위 이름으로 바꾸면 같은 모델이 되는 것은 아닙니다.
모델 구성이 다르면 먼저 해당 모델에 맞는 코드·출력 계약을 정해야 합니다.

### 한 행이 어떤 문장인지도 같아야 합니다

예를 들어 첫 행이 MS MARCO의 어느 query라면, 세 모델 모두 첫 행에 그 query의
평균과 분산이 들어가야 합니다. 해당 query의 positive passage는 **다른 행**입니다.

```text
행0: ms_marco:123:query:0
행1: ms_marco:123:positive_passage:0
행2: snli:456:premise:0
행3: snli:456:hypothesis:0
...
```

위는 ID 형식의 예시이며 실제 데이터의 행 번호가 아닙니다. 각 ID 배열은
비어 있지 않은 Unicode 문자열 배열입니다. 모든 모델의 평균·분산 ID가
서로 같은 순서인지 검사합니다. 평균의 ID를 복사해서 분산의 실제 출처 확인을
대신하지 않습니다.

## 2. alignment_fit_provenance.json: 이 숫자가 어디서 나왔는지

JSON은 텍스트로 읽을 수 있는 설명 파일입니다. 이미 제공한 양식을 복사해
실제 정보로 채웁니다. 아래 명령은 파일을 복사할 뿐, 실제 정보를 자동으로 채우지 않습니다.

```bash
# x_uec/에서 실행
mkdir -p local_data
cp input_provenance.template.json local_data/alignment_fit_provenance.json
```

[`input_provenance.template.json`](../input_provenance.template.json)에 있는 항목의 뜻은 다음과 같습니다.

| 항목 | 기록할 내용 |
|---|---|
| `dataset` | 사용한 MS MARCO/SNLI의 버전·분할과 실제 선택 목록 설명 |
| `split` | 이 파일에는 정렬용 관측만 있으므로 `alignment_fit` |
| `selection_manifest_sha256` | 실제 선택 기록 파일 내용의 SHA-256 값 |
| `models` | 세 모델의 정확한 이름, revision, 차원 |
| `uncertainty_source` | 분산을 만든 기존 UEC 모델·posterior·설정·출처 |
| `output_stage` | 실제 평균·분산이 최종 정규화 출력이면 `l2_normalized` |
| `covariance_representation` | 현재 입력은 `native_output_diagonal_surrogate` |
| `role_policy` | 모델별 query/passage 등의 역할, prompt, pooling, 길이 제한, dtype |
| `alignment_diagnostic_separated` | 정렬용/진단용 문장을 실제로 분리했으면 `true` |
| `evaluation_overlap_audit` | 어느 평가 원문과 중복을 검사했는지, 그 결과 |

SHA-256은 파일 내용이 같은지 확인하는 64자리 값입니다. 선택 기록이 예를 들어
`selection_records.jsonl`이라면 `sha256sum selection_records.jsonl`로 구할 수 있습니다.
프로그램은 JSON에 적힌 값의 형식을 검사합니다. 선택 기록 파일을 직접 열어
문장 목록을 대조하는 것은 아니므로, 실제 기록을 함께 보관해야 합니다.

양식의 `REPLACE_WITH_...`는 실제 값으로 교체해야 합니다. 중복 검사를 완료하기 전에
`status=completed`, 정렬용/진단용을 분리하기 전에 `true`로 표시하지 않습니다.

## 3. “기존 UEC의 선택 기록”은 어느 파일인가요?

**원래 공개 UEC 코드에는 이 이름으로 저장되는 선택 기록 파일이 없습니다.**
[`exp_mteb/fit_la.py`](../../exp_mteb/fit_la.py)는 MS MARCO/SNLI에서 일부를 뽑아
사용하지만, 선택한 원문·원본 행 번호 목록을 별도 파일로 저장하지 않습니다.
따라서 “선택 기록을 재사용한다”는 말은 **당시 그 목록을 따로 보관했다면 그것을
쓴다**는 뜻입니다. 그런 파일이 이미 생겨 있다고 가정하면 안 됩니다.

담당자가 보관한 원문 목록·인덱스·저장된 데이터가 있다면 그것을 기준으로 다음을 확인합니다.
파일명은 정해져 있지 않으며, 여기서는 보관 이름의 예로 `selection_records.jsonl`을 사용합니다.
이 파일은 `example.py`의 세 번째 입력이 아니라 **앞의 두 파일을 준비할 때 쓰는 근거**입니다.

| 데이터 | 확인해야 하는 정보 |
|---|---|
| MS MARCO | 데이터 버전·split, 원본 행 번호, 실제 query, 선택한 positive passage와 원래 선택 관계 |
| SNLI | 데이터 버전·split, 원본 행 번호, 실제 premise·hypothesis, 원래 label |
| 공통 | 실제 선택 순서, 반복 관측, 기존 negative·정답·피팅 설정을 보존한 기록 |

이번 로컬 실험의 3,983/3,775쌍이라는 개수만 맞춰 새로 뽑으면 당시와 같은
선택 기록이 되지 않습니다. 원래 목록이 없으면 당시 데이터 버전과 실행 설정에서
선택을 재구성해 확인하거나, 새 선택을 사용한 실험으로 명확히 구분해야 합니다.

## 4. 기존 UEC의 la.pt가 있으면 바로 실행되나요?

원래 [`fit_la.py`](../../exp_mteb/fit_la.py)가 저장하는 기본 파일은 모델별로
`la.pt`와 `la_posterior_variance.pt`입니다. `la.pt` 이름은 실행 옵션으로 바꿀 수 있습니다.
이 파일들은 Laplace 모델과 파라미터 쪽 불확실성 자료이며, 위 NPZ의 문장별 평균·분산 파일과 다릅니다.

**현재 `example.py`는 이 PT 파일들을 직접 읽지 않습니다.** 원문 목록과 기존 UEC
출력 계산 경로를 사용해 문장별 평균·분산을 만든 뒤, 12개 배열로 묶어야 합니다.
원래 `exp_mteb/cache.py`의 MTEB 평가 캐시도 이 NPZ 형식으로 자동 변환되지 않습니다.
새 posterior fitting을 하거나 AG News의 분산으로 채우는 단계가 아닙니다.

이 문장별 출력 생성과 NPZ 내보내기를 한 번에 수행하는 범용 준비 명령은
현재 브랜치에 연결되어 있지 않습니다. 이미 출력이 있는 경우에는 그것을 이용하고,
없다면 사용하는 모델·UEC 출력 파일 형식에 맞는 준비 단계가 추가로 필요합니다.

## 5. 정렬용과 진단용 문장을 먼저 나눕니다

원문을 NFC와 CRLF/CR→LF만 적용한 hash로 그룹화합니다. 이 처리는 중복 확인용이며
인코더에 넣는 원문을 바꾸지 않습니다. 선언한 평가 원문과 중복을 확인한 뒤,
고유 그룹을 seed42로 약80/20 분리합니다. 같은 원문은 역할이 달라도 한쪽에만 둡니다.
기존 원문 occurrence는 유지합니다.

`alignment_fit.npz`에는 **정렬용 관측만** 넣습니다. 별도 진단 관측은 필요하면
`alignment_diagnostic.npz`라는 이름으로 보관할 수 있지만, `example.py`는 이 파일을
자동으로 읽거나 진단하지 않습니다. 진단 방법은 [연결 가이드](integration.md)에 있습니다.

이번 로컬 실행은 총15,516개 관측에서 정렬용12,417개/진단용3,099개였고 평가 중복은0개였습니다.
이 분리는 정렬에만 해당하며, 원래 UEC posterior가 해당 자료를 이미 보았다면 그 사실도 기록합니다.
