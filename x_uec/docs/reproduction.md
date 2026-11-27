# 실행과 재현 범위

[처음으로](../README.md) · [실험 결과](results.md)

## 1. 새 clone에서 바로 실행할 수 있는 것

`x_uec/`의 `caia.py`, `covariance_ops.py`, 두 예제와 기본 테스트는 NumPy만 사용합니다. README의 합성 예제를 실행하면 데이터 다운로드 없이 정렬 호출, raw 모멘트 계산, 작은 dense 기준 및 문서 chunk 처리를 확인할 수 있습니다.

실제 native 출력 캐시가 있다면 [데이터 계약](data.md)에 맞춰 `example.py --input ... --provenance ...`로 GPA/CAIA 후보 행렬을 새로 구할 수 있습니다. 새 posterior fitting이나 모델 추론은 이 예제가 수행하지 않습니다.

기본 테스트는 `x_uec/`에서 실행합니다.

```bash
OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 python -m unittest -v
```

## 2. 과거 전체 실행의 원래 코드

[`reference_experiment/`](../reference_experiment/)는 `full_20261127_02`에 사용한 원래 full runner와 회귀 테스트입니다. 이 복사본의 핵심 Python 파일은 실제 실행본과 같습니다. 다음 테스트는 작은 fixture를 사용하므로 원래 모델·데이터 없이 실행할 수 있습니다.

```bash
cd reference_experiment   # x_uec/에서 시작
OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 python -m unittest -v
```

전체 69개 테스트가 포함됩니다. 회귀 테스트와 과거의 전체 데이터 실행은 다른 검증입니다.

## 3. 과거 전체 데이터 실행까지 재현하려면

참조 full runner는 **로컬 frozen-teacher 실험의 출처를 검증하는 전용 경로**입니다. 다음 외부 자원이 필요합니다.

- 원래 MS MARCO/SNLI 선택·triplet·dataset revision 기록.
- 원래 frozen-teacher Gaussian과 출력 전파 소스의 seal/checksum.
- 전체 native 출력 캐시, 독립 mean/variance ID sidecar, 관측 역할과 split.
- 선언한 평가 6과제의 원문 10개 파일 및 checksum.
- 입력 파일·소스·모델 정책을 묶은 `uec_input_provenance.json`.

이 자원은 공개 저장소에 포함하지 않았습니다. 새 clone의 참조 runner가 모델 추론과 입력 준비를 자동으로 수행한다고 해석하지 마세요. 위 입력이 있는 경우에는 자신의 실제 provenance 경로에서 config를 만듭니다.

```python
# reference_experiment/에서 실행하는 config 생성 예시
import json
from pathlib import Path

provenance_path = Path("/data/uec/full_inputs/uec_input_provenance.json")
p = json.loads(provenance_path.read_text())
config = {
    "mode": "full",
    "delivery_version": "v3",
    "provenance_path": str(provenance_path.resolve()),
    "backend_files": p["backend_files"],
    "models": {m: item["policy"] for m, item in p["models"].items()},
}
Path("local_full_config.json").write_text(json.dumps(config, indent=2))
```

```bash
OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
  python full_e4.py --config local_full_config.json --output outputs/full_new
```

provenance가 가리키는 원문·producer·sidecar·모델 출력 등 checksum 대상 파일이 실제로 존재해야 합니다. smoke 캐시는 `alignment_full`이 아니므로 거부합니다. 이 runner는 원래 3,983/3,775쌍의 로컬 선택을 독립 재검사하며, 다른 UEC posterior를 이름만 바꿔 넣는 범용 runner는 아닙니다.

실행한 소스는 시작 시 hash를 고정하고 export 전에 다시 검사합니다. 검증이 끝날 때까지 소스나 입력을 수정하지 않습니다. 준비 기록의 잘못된 감사 필드는 원본을 바꾸지 않고 `verified_full_audit.json`으로 독립 재검산합니다.

## 4. 종료 상태 읽기

| 상태 | 의미 |
|---|---|
| `stationary_tolerance` | 해당 solver arm이 residual/gap 기준을 만족 |
| `budget_exhausted` | 반복 예산 종료; 수렴으로 표시하지 않음 |
| `stalled_not_stationary` | loss 변화는 작지만 stationarity 기준 미달 |
| `review_required` | full 진단 이후에도 solver 종료 조건 검토가 필요 |
| `alignment_ready_for_handoff` | 해당 full 입력 계약에서 수렴 및 필수 진단을 통과 |
| `numerical_failure` | 수치·입력·진단 검증 실패; 승인 map 없음 |

`alignment_ready_for_handoff`는 최종 UEC 점수나 downstream 성능 승인과 다릅니다. 참조 full runner는 통과한 map을 `maps/`, 수치적으로 유효하지만 미수렴한 map을 `diagnostic_only/`에 분리합니다.

## 5. 공개한 검증 자료

[`artifacts/full_run_20261127/`](../artifacts/full_run_20261127/)에는 상태, solver trace, 정렬 손실, geometry·moment·대각절단 요약과 출처 checksum이 있습니다. 원래 giant cache, posterior·모델·바이너리 map은 제외했습니다. 자신이 사용하는 posterior와 출력 계약이 다르면 그 입력으로 다시 정렬하고 진단합니다.
