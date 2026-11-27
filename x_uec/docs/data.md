# 어떤 파일을 준비해야 하나요?

[실행 안내](../README.md) · [환경·재현 안내](reproduction.md)

**사용자가 기존 실험 파일을 따로 준비할 필요는 없습니다.** GitHub에서 코드를 받은 뒤 `tools/prepare_alignment.py --work-dir 작업폴더 --gpu 번호`를 실행하면 공개 자료를 받고 모든 입력을 생성합니다. `alignment_fit_provenance.json`이나 JSON 양식을 직접 만드는 단계가 없습니다.

## GitHub에 포함한 것

| 경로 | 내용 |
|---|---|
| `standalone/reference/selection_ids.jsonl` | UEC main에서 선택되는 MS MARCO 3,983쌍/SNLI 3,775쌍의 원본 행 번호·ID·query/positive 문장 해시 |
| `standalone/reference/selection_protocol.json` | seed 2025, 샘플 비율·순서, 필터 전 추출 행, 원본 main 코드 SHA |
| `standalone/reference/dataset_sources.json` | 공개 dataset 저장소, 고정 revision, 파일 이름·크기·SHA-256 |
| `standalone/reference/evaluation_hashes.json` | 기존 평가 6과제/10파일의 32,646개 문장 해시와 출처 기록 |
| `standalone/reference/manifest.json` | 위 기록의 무결성 확인 값 |
| `standalone/`의 Python 파일 | 공개 모델 로딩, 새 불확실성 피팅, 평균·분산 계산, 자동 입력 생성 |

원문 데이터·모델 가중치는 Git에 넣지 않고 공개 저장소에서 다운로드합니다. 선택 기록은 **공개 UEC main의 기본 전처리 코드를 재현한 결과**입니다. 이전 로컬 실험에서 유효한 쌍 3,983개/3,775개를 직접 뽑았던 선택과는 다릅니다.

고정된 공개 자료에서 행을 읽은 뒤 문장 해시를 대조합니다. 원래 선택 기록의 해시는 NFC 정규화 후 연속 공백을 합칩니다. 정렬·평가 분할의 해시는 NFC와 줄바꿈만 정규화합니다. 두 기준을 구분하며 실제 모델에 넣는 원문은 변경하지 않습니다. MS MARCO에서는 실제 selected positive passage, SNLI에서는 entailment hypothesis를 유지합니다. UEC main은 별도 random-negative triplet을 저장하지 않고 positive 쌍과 배치 안의 다른 문장을 사용합니다. 이번 기록에도 임의 negative를 추가하지 않습니다. frozen-teacher 곡률 계산은 query/positive의 배치 내 점수를 사용합니다.

기본 실행은 main이 선택한 쌍을 모두 유지해 그 안에서 정렬/진단을 나눕니다. 보관한 과거 평가 6과제의 해시는 겹침을 보고하는 참고 자료이며 기본 선택을 바꾸지 않습니다. 실제 별도 평가 자료와의 중복을 제외하려면 준비 명령에 `--evaluation-texts 평가문장.jsonl`을 추가하세요. 각 줄은 `{"text": "평가 문장"}` 형식이고 옵션을 여러 번 줄 수 있습니다. 정렬 입력과 Gaussian 피팅 쌍 모두에서 해당 평가 문장을 제외하며 대체 문장을 다시 추출하지 않습니다. 이 옵션을 쓰면 main 기본 선택 전부를 사용한 실행과 관측 수가 달라집니다. 기본 실행은 외부 MMTEB와 중복이 없다는 검증이 아닙니다.

## 작업 폴더에 자동 생성되는 것

```text
작업폴더/
├── preparation_config.json         # 이번 설정·실행 소스 해시
├── cache/                          # 기본 공개 자료·모델 캐시
├── data/
│   ├── pairs.jsonl                 # 복원한 원래 쌍과 관계
│   ├── curvature_pairs.jsonl       # 평가 중복을 제외한 불확실성 피팅 쌍
│   ├── observations_all.jsonl      # query/positive를 별도 관측으로 펼친 목록
│   ├── observations.jsonl          # 평가 중복 제외 후 사용할 문장
│   ├── split.json                  # 정렬용/진단용 분할
│   └── manifest.json               # 데이터 파일별 해시
├── models/{qwen,harrier,gte}/
│   ├── gaussian/                   # 이번 자료로 새로 피팅한 파라미터 Gaussian
│   ├── mean.npy                    # 문장별 평균 임베딩
│   ├── variance.npy                # 문장별 native 출력 분산 벡터
│   ├── mean_ids.npy                # 평균 관측 ID
│   ├── variance_ids.npy            # 분산 관측 ID
│   ├── summary.json                # 실제 모델·계산 정책과 검산
│   └── manifest.json               # 출력 해시
└── inputs/
    ├── alignment_fit.npz           # 정렬 프로그램에 넣을 숫자 배열
    ├── uec_input_provenance.json    # 위 파일과 계산 출처를 묶은 자동 생성 JSON
    └── manifest.json
```

`uec_input_provenance.json`은 `standalone/export.py`의 `assemble()`이 만듭니다. 실제 선택 자료·모델별 계산 기록·파일 해시에서 내용을 읽으며, 같은 폴더 구조를 유지하면 입력 검증에 이전 서버의 절대 경로가 필요하지 않습니다. 옮길 때는 `inputs/`만 보내지 말고 **data/, models/, preparation_config.json을 포함한 작업 폴더**를 함께 옮기세요. 캐시는 정렬 단계에는 필요하지 않습니다.

## NPZ 안의 배열

| 모델 | 평균 | native 분산 | 관측 ID 두 개 |
|---|---|---|---|
| Qwen | `mean__qwen` | `variance__qwen` | `mean_ids__qwen`, `variance_ids__qwen` |
| Harrier | `mean__harrier` | `variance__harrier` | `mean_ids__harrier`, `variance_ids__harrier` |
| GTE | `mean__gte` | `variance__gte` | `mean_ids__gte`, `variance_ids__gte` |

native는 각 모델의 원래 임베딩 좌표계를 뜻합니다. 평균·분산의 열은 각각 1024/640/768개입니다. 같은 행은 세 모델에서 같은 문장·같은 역할을 뜻합니다. 분산은 파라미터 분산 자체가 아니라 **최종 L2 정규화까지 전파한 native 출력 분산**입니다. 문장마다 1024×1024 행렬을 저장하지 않습니다.

이 입력에는 정렬용과 진단용 관측이 모두 들어갑니다. `split.json`이 구분합니다. 같은 문장 해시는 한쪽에만 속합니다. 기본 main 선택은 15,516개 관측을 12,414개 정렬용/3,102개 진단용으로 나눕니다. 추가 평가 문장을 제외하면 개수는 달라질 수 있습니다.

이전 설명의 “기존 실험 폴더에서 읽는 것”은 더 이상 기본 실행의 전제 조건이 아닙니다. `--workspace`와 `reference_experiment/prepare_cache.py`는 과거 서버의 보관 자료를 확인하는 구형 경로로만 남겨 두었습니다.

## main 기본값으로 선택되는 개수

| 자료 | 원본 train | 필터 전에 무작위 추출 | 필터 후 사용하는 쌍 |
|---|---:|---:|---:|
| MS MARCO | 82,326행 | 5% = 4,116행 | positive passage가 있는 3,983쌍 |
| SNLI | 550,152행 | 2% = 11,003행 | entailment인 3,775쌍 |

두 추출 사이에서 Python 난수 seed를 다시 설정하지 않습니다. 이 순서까지 원본과 맞춥니다.

## main 선택을 확인한 근거

- [기본 seed·샘플 비율](https://github.com/MLAI-Yonsei/UEC/blob/1235269c727832a5d17bb1cefba80ce125eb2912/exp_mteb/fit_la.py#L170)
- [MS MARCO: 먼저 5% 추출한 뒤 첫 positive 유지](https://github.com/MLAI-Yonsei/UEC/blob/1235269c727832a5d17bb1cefba80ce125eb2912/utils/data_utils/ms_marco.py#L25)
- [SNLI: 이어서 2% 추출한 뒤 entailment 유지](https://github.com/MLAI-Yonsei/UEC/blob/1235269c727832a5d17bb1cefba80ce125eb2912/utils/data_utils/snli.py#L29)

원본 함수를 수정하지 않고 recording tokenizer로 실행해 반환되는 문장 목록과 순서를 모두 대조했습니다. 따라서 개수만 같은 별도 표본을 사용하는 것이 아닙니다.

성능 평가 입력도 자동으로 준비합니다. 고정된 공개6태스크의 원문 ID·해시·정답·분류 support 추출은 `tools/benchmark_reference/`에 포함했습니다. 평가용 평균은 `작업폴더/benchmarks/`에 생성하고, 피팅 문장과 겹치는 외부 평가 항목은 제외합니다. 실제 수치는 [성능 결과](performance.md)에 있습니다.
