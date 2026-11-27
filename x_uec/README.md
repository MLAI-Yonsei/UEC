# X-UEC: CAIA 정렬과 공분산 계산

서로 다른 임베딩 모델의 출력을 공통 공간으로 옮기면서 **평균 위치와 불확실성 구조를 함께 맞추는** 코드입니다. 정렬에는 검증한 CAIA SVD v3를 사용합니다. 정렬 뒤에는 문장마다 큰 공분산 행렬을 저장하지 않고, native 분산 벡터와 공유 정렬 행렬로 필요한 계산을 수행합니다.

이 디렉터리는 기존 UEC의 데이터 선택, 정답, posterior, 가중치와 최종 점수 정의에 연결하는 정렬·계산 모듈입니다. 기존 `uec.py`와 평가 스크립트는 그대로 있습니다.

## 1. 설치

Python 3.10 이상과 NumPy가 필요합니다. 이미 사용할 Python 환경에 NumPy가 설치되어 있다면 가상환경 생성·설치는 건너뛸 수 있습니다.

```bash
git clone --branch X-UEC https://github.com/MLAI-Yonsei/UEC.git
cd UEC

# 충분한 공간이 있는 데이터 디스크에서 실행합니다.
python -m venv .venv-xuec
source .venv-xuec/bin/activate
python -m pip install -r x_uec/requirements.txt
cd x_uec
```

설치 후에는 아래 **실제 데이터 정렬**로 진행합니다. 합성 데모나 단위 테스트를 먼저 실행해야 할 의존성은 없습니다.

## 2. 실제 MS MARCO/SNLI 자료로 정렬하기

1. **기존 UEC가 uncertainty fitting에 사용한 선택 기록**을 재사용합니다. 개수만 맞춰 다시 추출하지 않습니다.
2. 같은 문장·같은 실제 역할에 대해 각 모델의 최종 출력 평균, 좌표별 출력 분산, 독립 ID를 준비합니다.
3. 원문 중복 그룹과 평가 원문 중복을 감사한 뒤 alignment-fit/diagnostic을 고정합니다. posterior fitting 자체는 바꾸지 않습니다.
4. 피팅 관측만 `example.py`에 넣고, 별도 진단 관측에서 geometry와 작은 dense/implicit 동치성을 확인합니다.

입력 배열과 metadata를 준비한 뒤, 위와 같이 `x_uec/`에서 실행합니다.

```bash
OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 \
  python example.py \
  --input /data/uec/alignment_fit.npz \
  --provenance /data/uec/alignment_fit_provenance.json \
  --r0-sweeps 100 --max-sweeps 300 \
  --output outputs/my_alignment
```

`example.py`는 이미 감사·분리한 캐시를 받는 예제입니다. 원문 감사와 분할을 자동 수행하지 않습니다. 출력의 `summary.json`에서 GPA/CAIA 종료 상태를 확인하세요. 이 예제는 후보 map을 저장하며 독립 진단 전에는 `alignment_ready_for_handoff=false`를 유지합니다.

구체적인 배열 생성법은 [데이터 준비](docs/data.md), 진단과 기존 UEC 점수 연결은 [연결 가이드](docs/integration.md)에 있습니다.

## 3. 선택 사항: 설치 후 작동 점검

아래 명령은 코드가 새 환경에서 실행되는지 확인하고 싶을 때만 사용합니다. **실제 데이터 정렬에 필요한 캐시나 행렬을 준비하는 단계가 아니므로 생략해도 됩니다.**

| 명령 | 확인하는 것 | 실제 실험에서의 사용 |
|---|---|---|
| `python -m unittest -v` | 기존 수식·입력·API 검산 테스트 | 선택 사항 |
| `python example_implicit.py ...` | 작은 합성 데이터에서 implicit/dense 계산의 동치성 | 선택 사항; 실제 실험용 입력을 생성하지 않음 |
| `python example.py --demo ...` | 작은 합성 데이터에서 정렬 API 연결 | 선택 사항; 생성한 demo map을 실제 데이터에 사용하지 않음 |
| `python example.py --input ...` | 준비한 실제 UEC 출력으로 GPA/CAIA fitting | 위 섹션2의 실제 정렬 명령 |

```bash
# 선택: 검산 테스트
OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 python -m unittest -v

# 선택: 작은 합성 데이터에서 공분산 계산 확인
OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 \
  python example_implicit.py --output outputs/demo_implicit

# 선택: 작은 합성 데이터에서 정렬 API 확인
OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 \
  python example.py --demo --r0-sweeps 3 --max-sweeps 3 \
  --output outputs/demo_fit
```

`OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4`는 CPU 계산에 사용할 스레드를 제한합니다. `\`는 다음 줄까지 같은 명령이라는 뜻입니다. 출력 폴더가 이미 있으면 새 이름을 사용하세요.

## 4. 전체 데이터에서 확인한 결과

기존 로컬 MS MARCO **3,983쌍**과 SNLI **3,775쌍**의 전체 **15,516개 관측**을 사용했습니다.

| 항목 | 실제 결과 |
|---|---|
| alignment-fit / diagnostic | 12,417 / 3,099개 관측 |
| 평균 정렬 R0 / CAIA | 각각 18 / 232회에서 수렴 |
| GPA 추가 반복 | 0회: 이미 수렴한 R0에서 시작 |
| dense/implicit 수치 비교 | float64/float32 총 396개 통과 |
| full 실행 상태 | `alignment_ready_for_handoff=true` |

이 결과의 불확실성은 **기존 frozen-teacher v3 출력 근사**입니다. 수신자의 실제 UEC-LA posterior와 같은지, 검색·STS 성능이 좋아지는지는 별도 검증 범위입니다. 전체 결과와 원본 수치는 [실험 결과](docs/results.md)에서 확인할 수 있습니다.

## 5. 필요한 문서 찾기

| 문서·파일 | 설명 |
|---|---|
| [docs/data.md](docs/data.md) | 모델·역할·배열·ID·출력 단계 계약 |
| [docs/integration.md](docs/integration.md) | 큰 공분산 없는 계산과 UEC 연결·진단 절차 |
| [docs/math.md](docs/math.md) | PDF 목적함수와 구현의 동치, 실제 SVD-MM 갱신식 |
| [docs/results.md](docs/results.md) | 이번 전체 실행의 수렴·손실·수치 검산 |
| [docs/reproduction.md](docs/reproduction.md) | 독립 예제와 과거 전체 실험 재현의 차이 |
| `caia.py` | 평균/공분산 정렬 및 SVD-MM |
| `covariance_ops.py` | 비대각 공분산을 보존하는 raw 내적 모멘트 계산 |
| `artifacts/full_run_20261127/` | 전체 실행의 수치 원본과 출처 checksum |
| `reference_experiment/` | 전체 로컬 실행에 사용한 원래 runner와 회귀 테스트 |

모델·posterior·원문 데이터·대형 캐시·정렬 행렬 바이너리는 Git에 넣지 않았습니다. 자신의 출력 계약에 맞춰 새로 fitting하고, 문장별 분산과 모델별 공유 행렬을 함께 사용합니다.
