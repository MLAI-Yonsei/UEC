# X-UEC: 기존 실험 코드로 MS MARCO/SNLI 정렬하기

이 브랜치는 **기존 AGNews 실험에서 사용한 평균·불확실성 계산과 CAIA SVD v3 정렬**을 연결합니다. 정렬할 문장만 MS MARCO/SNLI로 바꾸어 새 평균·분산을 계산합니다. 출처 JSON을 사용자가 손으로 작성할 필요는 없습니다.

기존에 피팅한 불확실성 모델은 재사용하고, 선택한 문장들의 평균·출력 분산과 정렬 행렬을 새로 계산합니다. 이 경로의 불확실성은 기존 frozen-teacher v3 근사입니다. 원래 UEC-LA를 새로 구현하거나 피팅하는 경로로 바꾼 것은 아닙니다.

## 1. 필요한 기존 실험 폴더

전체 준비에는 **기존 실험 코드·모델·피팅된 불확실성 모델·선택한 원문**이 있는 작업 폴더가 필요합니다. 현재 실험 서버에서는 `/data2/claire/UEC`입니다. `--workspace`에 이 경로를 넣습니다.

새 Git clone만으로 모델 추론까지 실행되는 것은 아닙니다. 다른 서버라면 기존 자산과 실행 환경도 있어야 합니다. 필요한 폴더 목록은 [기존 실험 재현 조건](docs/reproduction.md)에 적었습니다. 자산이 없으면 프로그램이 누락된 경로를 알려줍니다.

```bash
git clone --branch X-UEC https://github.com/MLAI-Yonsei/UEC.git UEC-X-UEC
cd UEC-X-UEC/x_uec
```

## 2. 선택 문장의 평균·분산과 입력 파일 만들기

기존 실험 폴더가 있는 서버에서 실행합니다. `--output`은 새 폴더, `--gpu`는 비어 있는 물리 GPU 번호입니다.

```bash
python prepare_alignment.py \
  --workspace /data2/claire/UEC \
  --output /data2/claire/xuec_inputs_new \
  --gpu 1
```

이 명령이 순서대로 하는 일은 다음과 같습니다.

1. 기존 MS MARCO 3,983쌍/SNLI 3,775쌍을 읽어 문장 목록을 만들고, 중복 검사와 정렬용/진단용 분리를 수행합니다.
2. Qwen → Harrier → GTE 순서로 **새 문장 평균·분산을 실제로 계산**합니다. 기존 AGNews 문장별 배열을 복사하지 않습니다.
3. 결과를 `uec_alignment_inputs.npz`에 묶고 `uec_input_provenance.json`을 자동으로 만듭니다. 원문 선택, 역할, 모델 정책, 계산 출처와 checksum이 기록됩니다.

```text
xuec_inputs_new/
├── observations.jsonl             # 사용한 문장·역할·ID
├── split.json                    # 정렬용/진단용 분리 기록
├── source_manifest.json          # 원래 데이터 출처
├── qwen/                        # 실제 계산한 평균·분산·ID·계산 기록
├── harrier/
├── gte/
├── uec_alignment_inputs.npz       # 정렬 프로그램에 넣을 숫자 배열
└── uec_input_provenance.json      # 자동 생성한 출처 기록
```

모델별로 원래 사용한 `.venv`/`.venv-reproduction` 환경을 사용합니다. 새로 모델을 다운로드하거나 encoder·불확실성 모델을 업데이트하지 않습니다. 기존과 같은 46,000MiB 이상 GPU를 검사하고 UUID로 고정합니다. GPU가 사용 중이면 기존 작업을 건드리지 않고 중단합니다. 출력 폴더는 덮어쓰지 않습니다.

## 3. 만들어진 입력으로 GPA/CAIA 정렬·진단하기

```bash
OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
  python run_alignment.py \
  --inputs /data2/claire/xuec_inputs_new \
  --output /data2/claire/xuec_alignment_new
```

`--inputs`에 **앞 단계의 폴더만 지정**합니다. 프로그램이 자동 생성된 출처 JSON에서 실행 설정을 만들고 전체 정렬·진단을 진행합니다. 별도의 JSON 양식을 채우지 않습니다.

```text
xuec_alignment_new/
├── status.json                  # 수렴·검산을 포함한 최종 상태
├── solver_summary.json          # GPA/CAIA 반복과 수렴 기록
├── alignment_losses.csv         # 정렬용/진단용 손실 비교
├── moment_equivalence.csv       # 작은 dense 기준과 계산 결과 비교
├── diagonal_summary.csv         # 비대각 성분을 버렸을 때의 차이
└── maps/                        # 수렴과 필수 검사를 통과한 경우
    ├── gpa_maps.npz
    └── caia_maps.npz
```

미수렴 행렬은 `diagnostic_only/`에 보관하고 수치 실패에는 승인 행렬을 만들지 않습니다. `status.json`의 `alignment_ready_for_handoff`를 확인합니다. 이는 해당 출력 계약의 정렬·수치 검증이며 UEC 최종 점수나 검색·STS 성능 향상 검증은 아닙니다.

이미 준비한 실제 전체 입력의 출처만 검사하려면 `run_alignment.py` 명령에 `--check-only`를 추가할 수 있습니다. 이 옵션은 모델 추론이나 정렬을 다시 실행하지 않습니다.

## 4. 이미 전체 실행에서 확인한 값

| 항목 | 실제 결과 |
|---|---|
| 사용한 원문 | MS MARCO 3,983쌍 + SNLI 3,775쌍 |
| 전체 문장 관측 | 15,516개 |
| 정렬용 / 진단용 | 12,417 / 3,099개 |
| R0 / CAIA 수렴 | 18 / 232회 |
| GPA 추가 반복 | 이미 수렴한 R0에서 시작해 0회 |
| float64/float32 계산 비교 | 396개 통과 |

기존 전체 실행의 원본 수치와 범위는 [실험 결과](docs/results.md)에 있습니다. 새 준비 명령의 추가 검증과 과거 전체 실행의 결과를 구분해 기록합니다.

## 5. 필요한 문서

| 문서 | 설명 |
|---|---|
| [입력 파일](docs/data.md) | 자동으로 만들어지는 파일 이름과 내용 |
| [기존 실험 재현 조건](docs/reproduction.md) | 기존 코드·모델·데이터·환경의 위치와 의존성 |
| [수식](docs/math.md) | 목적함수와 실제 SVD-MM 구현의 관계 |
| [UEC 연결](docs/integration.md) | 정렬 뒤 큰 공분산 없이 계산하고 최종 점수와 대조하는 방법 |
| [실험 결과](docs/results.md) | 전체 실행에서 확인한 수렴·손실·검산 |
| [개발자 점검](docs/checks.md) | 필요할 때만 실행하는 합성 예제·테스트 |

`example.py`, `example_implicit.py`, `input_provenance.template.json`은 별도 개발 예제입니다. 위 실제 실험 경로에서는 실행하거나 채울 필요가 없습니다. 기존 UEC의 원문·정답·negative 관계와 피팅 설정은 바꾸지 않습니다.
