# 공개 자료에서 새로 계산하는 경로

[실행 안내](../README.md) · [입력 파일](data.md)

기본 경로는 `prepare_alignment.py --work-dir ...`입니다. 필요한 계산 모듈, main 기본 선택의 재현 코드와 선택 기록은 이 브랜치에 포함했고, 공개 원문과 모델은 고정된 revision으로 다운로드합니다. 기존 `/data2/claire/UEC` 작업 폴더나 미리 피팅한 Gaussian은 필요하지 않습니다.

## 모델과 환경

Python 3.11과 [requirements-full.txt](../requirements-full.txt)를 사용합니다. torch 2.6.0, transformers 4.51.3, sentence-transformers 5.2.2로 세 모델을 같은 환경에서 실행합니다. Linux CUDA GPU 한 장을 사용하며 현재 실행 전 검사는 총 메모리 46,000MiB 이상, 사용량 1,024MiB 이하, 사용 중인 계산 프로세스 없음입니다. 기존 프로세스를 종료하지 않습니다. FP64 미분·분산 계산이 포함되어 GPU 종류에 따라 시간이 크게 달라집니다.

PyTorch 기본 설치가 서버 CUDA 드라이버와 맞지 않으면 [PyTorch 공식 설치 안내](https://pytorch.org/get-started/locally/)에 따라 **2.6.0에 맞는 CUDA wheel**을 같은 환경에 설치하세요. 다른 패키지를 임의 업그레이드하기보다 새 환경에서 고정 버전을 쓰는 것을 권장합니다.

| 모델 | 고정 공개 revision | native 차원 | 최대 토큰 |
|---|---|---|---|
| Qwen/Qwen3-Embedding-0.6B | `b22da495047858cce924d27d76261e96be6febc0` | 1024 | 8192 |
| microsoft/harrier-oss-v1-270m | `31de22b673913c7d658c0f03f792d77c2dcf8ebd` | 640 | 32768 |
| thenlper/gte-base | `c078288308d8dee004ab72c6191778064285ec0c` | 768 | 512 |

Qwen은 CUDA FP16, Harrier/GTE는 FP32로 encoder를 실행하고 최종 경로의 평균·미분·분산 계산은 FP64로 합니다. Qwen/Harrier는 기존 prompt와 마지막 유효 토큰, GTE는 masked mean pooling을 유지합니다. 실제 dtype, backend, padding, prompt, package version, checkpoint 해시는 출력에 기록합니다.

기본 attention은 PyTorch `sdpa`입니다. 별도 FlashAttention 설치는 필요하지 않습니다. 이전 Qwen 실행의 FlashAttention과 bit 단위 재현을 주장하지 않습니다. 필요한 환경에서는 `--qwen-attention flash_attention_2`를 선택할 수 있지만 해당 패키지는 별도 설치해야 합니다.

## 이전 계산에서 이어받은 부분

| 부분 | 독립 실행의 코드 |
|---|---|
| Qwen 마지막 블록의 forward·미분·출력 분산 | `standalone/math_qwen.py` |
| Harrier forward·미분·출력 분산, GTE 출력 분산 | `standalone/math_paths.py` |
| GTE 마지막 LN/평균 풀링 경로와 shared-weight 미분 | `standalone/math_gte.py` |
| 공개 모델 로딩·feature 추출 | `standalone/models.py` |
| frozen-teacher categorical GGN과 Gaussian 피팅 | `standalone/curvature.py` |
| 문장 평균·분산과 JSON 자동 내보내기 | `standalone/export.py` |
| GPA/CAIA 정렬 | 기존 그대로인 `caia.py` |
| 비대각 공분산을 보존하는 계산 | 기존 그대로인 `covariance_ops.py` |

이전 순수 수치 모듈의 출처와 SHA는 `standalone/NUMERICAL_SOURCES.json`에 기록했습니다. GTE의 새 역전파 계산은 같은 가중치가 모든 토큰에서 공유된다는 점을 유지하고 작은 직접 미분으로 검산합니다.

## 불확실성 피팅은 무엇을 하나요?

Encoder를 학습하거나 이동시키지 않고 마지막 projection 가중치의 현재 값을 Gaussian 평균으로 고정합니다. 선택한 query와 positive의 raw 출력 내적에서 teacher 확률을 만들고, 그 확률을 고정한 categorical GGN 대각을 계산합니다. query와 positive가 공유하는 가중치에 대한 두 미분을 **먼저 더한 뒤 제곱**합니다. GTE의 토큰 사이 공유 효과도 유지합니다.

기본은 논리 배치 512, 곡률 probe 32, SUM 정규화입니다. 순서 seed는 2025, 모델별 probe seed는 코드에 고정되어 있습니다. 16개 가중치 좌표에서 직접 계산한 GGN과 비교하고 teacher 중심 KL도 확인합니다. prior precision은 고정 가중치 제곱 평균의 역수이며 teacher variance는 `1 / (prior_precision + GGN_diagonal)`입니다.

`--physical-batch`는 encoder 추론을 나누는 크기이며 기본 8입니다. 메모리를 줄이려면 이 값만 줄일 수 있습니다. 이는 logical batch 512를 바꾸지 않습니다. 한 logical batch의 feature만 CPU에 두고 GPU에는 작은 묶음으로 보냅니다. 저장된 문장 분산도 native 벡터이며 문장별 1024×1024 공분산을 생성하지 않습니다. 길이가 매우 긴 자료는 물리 배치를 더 줄여야 할 수 있습니다.

이 경로는 이전 AGNews 실험의 frozen-teacher v3 계산을 공개 자료에서 다시 수행합니다. 원래 UEC-LA와 같은 posterior·최종 점수라고 확인한 것은 아닙니다. 불확실성 피팅에는 정렬 진단용 문장도 들어가므로 `posterior_unseen=false`로 기록합니다.

## 재개와 파일 검증

완료 단계는 상대 파일 경로와 SHA로 봉인합니다. 같은 명령을 다시 실행하면 설정·소스·완료 파일을 확인합니다. 진행 중 Gaussian은 원자적으로 기록한 마지막 완료 logical batch까지 이어서 계산하고, 완료 전 문장별 내보내기는 그 모델에서 다시 합니다. 설정·code·데이터·실제 tokenizer 파일·실행 라이브러리가 바뀌면 재사용을 거부하므로 새 작업 폴더가 필요합니다. 같은 작업 폴더에서 준비 명령 두 개를 동시에 실행하는 것도 거부합니다. 모델 전환 직후 사용률 표본만 남아 있으면 최대 5초 기다리며 실제 프로세스가 있으면 즉시 중단합니다. 정렬 결과도 새 `--output`을 사용합니다.

공개 모델·데이터 캐시는 네트워크 없이 다시 사용할 수 있습니다. `--cache-dir`는 표준 Hugging Face **hub 캐시 폴더**를 가리킵니다. 최초 다운로드와 재실행에서 같은 옵션을 사용하세요. `--offline`도 실행 설정의 일부이므로 바꾸려면 새 작업 폴더를 사용합니다.

## 수렴과 검증 범위

| 상태 | 의미 |
|---|---|
| `smoke_pass` | 실제 모델·소수 문장 경로와 필수 수치 진단 통과; 전달 승인 없음 |
| `stationary_tolerance` | 개별 solver의 residual/gap 기준 만족 |
| `review_required` | 전체 진단 후에도 solver 수렴 기준 미달 |
| `alignment_ready_for_handoff` | 전체 profile에서 수렴과 필수 진단 통과 |
| `numerical_failure` | 입력·계산·진단 실패; 승인 행렬 없음 |

작은 점검은 정렬 3회, 전체는 R0 최대 100회 및 각 arm 최대 300회입니다. 검산은 동일한 `full_e4.py`에서 GPA/CAIA를 비교하고, float64/float32의 implicit/dense 동치성 396개 항목, 거리 보존과 대각 근사 차이를 확인합니다.

기존 전체 실행의 수치는 [기존 결과](results.md)에 보관합니다. 그것은 이전 로컬 선택 자료와 이미 피팅한 Gaussian에서 시작한 실행이므로, main 기본 선택 자료에서 다시 실행한 결과와 구분합니다. 이번 독립 실행의 새 Gaussian에서 같은 γ·수렴 횟수가 나오리라고 가정하지 않습니다. 새 실행의 실제 결과는 해당 작업 폴더의 `status.json`과 `solver_summary.json`을 사용하세요. 독립 실행 점검 증거는 `VERIFICATION.json`에 별도로 기록합니다.
