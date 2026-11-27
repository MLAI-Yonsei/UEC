# X-UEC: MS MARCO/SNLI에서 CAIA v3 정렬하기

**이 GitHub 브랜치의 코드만 받아 실행할 수 있습니다.** 필요한 공개 데이터와 모델은 프로그램이 내려받습니다. 이전 실험 폴더, AGNews 배열, 이미 피팅된 Gaussian 파일을 준비할 필요가 없습니다.

실행은 두 단계입니다. 먼저 **선택한 UEC 자료로 불확실성을 새로 피팅하고 문장별 평균·분산을 계산**합니다. 다음으로 그 숫자들로 GPA/CAIA 정렬 행렬을 피팅하고 진단합니다. 출처 JSON도 자동 생성하므로 양식을 직접 채우지 않습니다.

## 설치

Linux, Python 3.11과 CUDA가 동작하는 PyTorch 환경을 사용합니다. 현재 GPU 실행은 여유 메모리 46,000MiB 이상인 GPU 한 장을 기준으로 확인합니다. 세 모델은 순서대로 실행됩니다. 데이터·환경·캐시·결과는 충분한 공간이 있는 데이터 디스크에 두세요.

```bash
git clone --branch X-UEC https://github.com/MLAI-Yonsei/UEC.git UEC-X-UEC
cd UEC-X-UEC
python3.11 -m venv .venv-xuec
source .venv-xuec/bin/activate
PIP_CACHE_DIR=/data2/xuec-pip-cache \
  python -m pip install -r x_uec/requirements-full.txt
cd x_uec
```

저장소 자체도 `/data1` 또는 `/data2` 아래에 clone하세요. 위 `/data2/xuec-pip-cache`와 아래 작업 경로는 **자신이 쓸 수 있는 경로**로 바꾸세요. CUDA 드라이버에 맞는 PyTorch 설치가 필요한 환경은 [환경·재현 안내](docs/reproduction.md)를 참고하세요. 기존 UEC용 루트 `requirements.txt`와 섞지 않고 이 환경을 사용합니다.

## 1. 실제 자료로 입력 만들기

```bash
python prepare_alignment.py \
  --work-dir /data2/xuec-full \
  --gpu 0
```

`--gpu 0`은 비어 있는 물리 GPU 번호로 바꿉니다. 사용 중인 GPU에서는 중단하며 다른 작업을 종료하지 않습니다. 기본 설정이 전체 실행입니다.

이 명령이 다음 일을 순서대로 합니다.

1. 고정된 버전의 공개 MS MARCO와 SNLI를 내려받고, UEC `main`의 기본 선택을 seed 2025로 재현해 **MS MARCO 3,983쌍 + SNLI 3,775쌍**을 얻습니다. 저장소의 행 번호·문장 해시와도 대조합니다.
2. Qwen → Harrier → GTE에서 각 모델의 불확실성을 새로 피팅합니다. 이전 AGNews 실험의 frozen-teacher v3 계산 방식을 사용하며 encoder 가중치는 고정합니다. 기본 logical batch는 512, 곡률 추정 반복은 32입니다.
3. 같은 문장들의 평균과 native 좌표별 출력 분산을 새로 계산합니다. query/positive passage, premise/hypothesis 관계를 유지합니다.
4. `inputs/alignment_fit.npz`와 `inputs/uec_input_provenance.json`을 자동으로 만듭니다.

**어떤 문장을 선택하나요?** UEC `main`처럼 같은 Python 난수 상태에서 MS MARCO train의 5%를 먼저 뽑고 첫 positive passage가 있는 행만 남긴 다음, SNLI train의 2%를 뽑고 entailment만 남깁니다. 유효한 문장부터 개수를 맞춰 다시 뽑는 방식이 아닙니다. 원본 `main` 전처리 함수를 직접 실행해 문장과 순서가 모두 같음을 확인했습니다. [선택 코드·기록](docs/data.md)을 참고하세요.

선택 데이터는 `main`과 맞추되, 불확실성 계산은 기존 AGNews 실험에서 검증한 frozen-teacher v3 방식을 이어갑니다. 원래 UEC-LA와 같은 posterior·최종 점수라고 주장하지 않습니다.

처음에는 공개 자료 다운로드가 필요합니다. 기본 캐시는 작업 폴더의 `cache/`에 생깁니다. 이미 공개 모델을 내려받은 표준 Hugging Face hub 캐시가 있다면 `--cache-dir /data2/사용자/huggingface/hub`를 추가할 수 있습니다. 모든 필요한 공개 파일을 받은 뒤에는 `--offline`으로 네트워크 없이 실행할 수 있습니다.

중단되면 **같은 명령**으로 다시 실행하세요. 설정·입력·코드가 같으면 완료한 모델을 확인해 재사용하고, 진행 중 Gaussian 피팅은 마지막 완료 logical batch에서 이어갑니다. 완료 전 문장별 평균·분산 내보내기는 해당 모델에서 다시 시작합니다. 설정이나 코드를 바꾸려면 새 작업 폴더를 사용하세요.

## 2. GPA/CAIA 행렬 피팅과 진단

```bash
OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
  python run_alignment.py \
  --inputs /data2/xuec-full/inputs \
  --output /data2/xuec-full/alignment
```

앞 명령의 작업 폴더 아래 `inputs`를 넣으면 됩니다. 설정 JSON은 프로그램이 만듭니다. **같은 UEC 선택 데이터 안에서** 문장 해시로 정렬용 약 80%와 진단용 약 20%를 나눕니다. 기본 전체 관측 15,516개 중 12,414개로만 γ와 행렬을 구하고, 나머지 3,102개에서 손실·거리 보존·공분산 계산을 진단합니다. 같은 문장은 양쪽에 겹치지 않습니다. 불확실성 피팅에는 진단 문장도 포함되므로 이 진단은 **정렬에 대해서만 보류한 자료**입니다.

결과에서 먼저 `alignment/status.json`을 보세요.

| 결과 | 의미 |
|---|---|
| `alignment_ready_for_handoff: true` | 전체 정렬의 수렴과 필수 수치 검사를 통과 |
| `maps/caia_maps.npz` | 승인된 CAIA 정렬 행렬 |
| `diagnostic_only/` | 작은 점검 또는 미수렴 결과의 행렬; 승인용 아님 |
| `moment_equivalence.csv` | 큰 공분산을 만들지 않는 계산과 작은 dense 기준의 비교 |
| `diagonal_summary.csv` | 비대각 성분을 버렸을 때 생기는 차이 |

`--output`에는 새 결과 폴더를 지정합니다. 입력만 검증하려면 위 명령에 `--check-only`를 추가합니다. UEC 최종 점수와 검색·분류·STS 성능 검증은 별도입니다.

## 선택: 먼저 작게 실행하기

전체 계산 전에 실제 공개 모델과 자료로 연결만 확인하려면 첫 명령에 `--profile smoke`를 넣고 별도 폴더를 쓰세요.

```bash
python prepare_alignment.py --work-dir /data2/xuec-smoke --profile smoke --gpu 0
OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
  python run_alignment.py --inputs /data2/xuec-smoke/inputs --output /data2/xuec-smoke/alignment
```

이 경우 각 데이터셋의 처음 32개 선택 쌍을 사용하고 정렬 반복을 3회로 줄입니다. 불확실성 계산 방식과 곡률 반복 32회는 유지합니다. 성공 상태는 `smoke_pass`이며 `alignment_ready_for_handoff`는 항상 `false`입니다. 실제 전체 실험에는 위의 기본 전체 명령을 사용하세요.

`example.py`, `example_implicit.py`, `input_provenance.template.json`은 별도 개발 예제입니다. 위 실험을 위해 실행하거나 작성할 필요가 없습니다.

## 설명 문서

| 문서 | 내용 |
|---|---|
| [입력 파일](docs/data.md) | GitHub에 있는 선택 기록과 자동 생성 파일 |
| [환경·재현 안내](docs/reproduction.md) | 공개 모델 버전, 계산 방식, 재개와 검증 범위 |
| [수식](docs/math.md) | 사용자 목적함수와 실제 SVD-MM 갱신 |
| [UEC 연결](docs/integration.md) | 정렬 후 비대각 공분산을 보존하는 계산 |
| [UEC main 전체 결과](docs/uec_main_results.md) | 이번 main 선택 데이터의 새 피팅·정렬·진단 |
| [기존 전체 결과](docs/results.md) | 이전 로컬 선택 데이터의 전체 결과; 현재 main 선택 결과와 구분 |
| [개발자 점검](docs/checks.md) | 선택적인 합성 예제와 테스트 |

기존 전체 결과와 이번 독립 실행 검증은 별도 기록입니다. 실제 검증 범위는 `VERIFICATION.json`에 남깁니다.
