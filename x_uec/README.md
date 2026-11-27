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

## 2. 먼저 준비할 파일

현재 정렬 프로그램이 직접 읽는 파일은 **두 개**입니다. 아래는 `x_uec/` 안에 두는 권장 이름입니다.

```text
x_uec/
└── local_data/
    ├── alignment_fit.npz               # 정렬에 사용할 실제 평균·분산·문장 ID
    └── alignment_fit_provenance.json   # 어느 데이터·모델·불확실성 계산에서 나온 값인지
```

이 파일들은 `git clone`으로 생기지 않습니다. 기존 UEC에서 계산한 실제 값으로 준비해야 합니다. 파일명은 바꿀 수 있지만, NPZ 안의 배열 이름은 코드가 정한 이름이어야 합니다. **[정확한 배열 이름과 파일별 내용](docs/data.md)**을 확인하세요.

“기존 UEC의 선택 기록”은 당시 사용한 MS MARCO의 query–passage 목록과 SNLI의 premise–hypothesis 목록을 뜻합니다. 원래 `exp_mteb/fit_la.py`는 이 목록을 별도 파일로 저장하지 않습니다. 따라서 담당자가 당시 사용한 원문 목록·행 번호를 보관했는지 먼저 확인해야 합니다. 개수만 같게 다시 뽑아 같은 자료라고 부르지 않습니다.

현재 `example.py`는 위 두 파일을 받아 정렬하는 프로그램입니다. **원문 목록이나 `la.pt`만 넣어서 평균·분산까지 자동 생성하는 준비 단계는 아직 연결되어 있지 않습니다.** 출력 평균·분산이 없다면 기존 UEC의 해당 계산 경로에서 먼저 만들어야 합니다.

## 3. 준비한 파일로 실제 정렬 실행

설치 단계 마지막의 `cd x_uec`까지 진행한 위치에서 실행합니다. 합성 데모를 먼저 실행할 필요는 없습니다.

```bash
OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 \
  python example.py \
  --input local_data/alignment_fit.npz \
  --provenance local_data/alignment_fit_provenance.json \
  --r0-sweeps 100 --max-sweeps 300 \
  --output outputs/my_alignment
```

- `--input`: 실제 평균·분산·ID 파일입니다.
- `--provenance`: 그 값의 데이터·모델·계산 방법을 적은 JSON 파일입니다.
- `--output`: 결과를 저장할 새 폴더입니다. 같은 이름의 폴더가 이미 있으면 새 이름을 사용합니다.

완료되면 `outputs/my_alignment/`에 `gpa_maps.npz`, `caia_maps.npz`, `fit_ids.npz`, `summary.json`이 생깁니다. 앞의 두 파일이 정렬 행렬입니다. `summary.json`에서 실제 종료 상태를 확인합니다.

입력은 정렬용/진단용 문장을 미리 분리한 뒤 **정렬용 문장만** 포함해야 합니다. 이 명령은 원문 중복 검사와 분할을 자동 수행하지 않습니다. 생성한 행렬은 별도 문장에서 계산을 검산하기 전까지 후보 상태이며 `alignment_ready_for_handoff=false`입니다. 진단과 기존 UEC 점수 연결은 [연결 가이드](docs/integration.md)에 있습니다.

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
| [docs/reproduction.md](docs/reproduction.md) | 과거 전체 실험의 재현 조건 |
| [docs/checks.md](docs/checks.md) | 개발자가 필요할 때 실행하는 선택적 검산 |
| `caia.py` | 평균/공분산 정렬 및 SVD-MM |
| `covariance_ops.py` | 비대각 공분산을 보존하는 raw 내적 모멘트 계산 |
| `artifacts/full_run_20261127/` | 전체 실행의 수치 원본과 출처 checksum |
| `reference_experiment/` | 전체 로컬 실행에 사용한 원래 runner와 회귀 테스트 |

모델·posterior·원문 데이터·대형 캐시·정렬 행렬 바이너리는 Git에 넣지 않았습니다. 자신의 출력 계약에 맞춰 새로 fitting하고, 문장별 분산과 모델별 공유 행렬을 함께 사용합니다.
