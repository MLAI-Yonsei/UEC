# 개발자를 위한 선택적 작동 점검

[실제 데이터 실행 안내](../README.md)

이 페이지의 명령은 프로그램을 수정하거나 새 환경에서 점검할 때 사용합니다.
**실제 UEC 자료로 정렬하기 위해 먼저 실행할 필요는 없습니다.**
합성 예제가 만든 파일은 실제 실험의 입력으로 사용하지 않습니다.

`x_uec/`에서 실행합니다.

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

실제 평균·분산 파일을 사용하는 정렬 명령은 README의 `--input` 명령입니다.
`--demo`는 프로그램이 자체 생성한 가짜 데이터를 쓰는 옵션입니다.
