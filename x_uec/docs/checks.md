# 개발자를 위한 선택적 작동 점검

[실제 데이터 실행 안내](../README.md)

이 페이지의 명령은 프로그램을 수정하거나 새 환경에서 점검할 때 사용합니다.
**실제 UEC 자료로 정렬하기 위해 먼저 실행할 필요는 없습니다.**
합성 예제가 만든 파일은 실제 실험의 입력으로 사용하지 않습니다.

README의 `requirements-full.txt`를 설치한 환경에서, `x_uec/`로 이동해 실행합니다. 전체 테스트에는 실제 모델을 내려받지 않는 작은 PyTorch 미분 검산도 포함됩니다. NumPy만 있는 환경에서는 `python -m unittest test_caia test_covariance_ops test_examples test_objective_contract`로 기존 정렬·공분산 점검만 실행할 수 있습니다.

```bash
# 수식과 입력 검사를 포함한 테스트
OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 python -m unittest -v

# 작은 가짜 데이터로 두 계산 방식이 같은지 확인
OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 \
  python example_implicit.py --output outputs/demo_implicit

# 작은 가짜 데이터로 정렬 프로그램을 3회 실행
OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 \
  python example.py --demo --r0-sweeps 3 --max-sweeps 3 \
  --output outputs/demo_fit
```

실제 실험 명령은 README의 `prepare_alignment.py`, `tools/run_alignment.py`입니다.
`example.py --input`은 별도 개발 예제용 입력을 사용하는 간단한 경로입니다.
`--demo`는 프로그램이 자체 생성한 가짜 데이터를 쓰는 옵션입니다.

공개 데이터 선택 자체를 원본 UEC main 함수와 대조하는 추가 점검도 있습니다. 공개 MS MARCO/SNLI를 받은 표준 hub 캐시 경로를 지정합니다. 모델은 실행하지 않습니다.

```bash
XUEC_PUBLIC_CACHE=/data2/사용자/xuec-full/cache/huggingface/hub \
  python -m unittest -v test_main_selection_oracle
```

환경 변수를 주지 않은 일반 테스트에서는 이 대용량 원문 검사를 건너뜁니다. 실제 실험의 준비 명령도 고정 선택 ID와 해시를 자동으로 대조합니다.
