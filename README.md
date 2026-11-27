# UEC · X-UEC branch

이 브랜치는 기존 UEC에 **CAIA SVD v3 정렬과 비대각 공분산을 보존하는 계산 모듈**을 추가합니다.
같은 문장을 서로 다른 임베딩 모델로 표현한 뒤, 평균과 불확실성 구조를 함께 맞추는 실험입니다.

- 처음 실행하기: [X-UEC README](x_uec/README.md)
- 실제 UEC 데이터 준비: [입력 데이터 계약](x_uec/docs/data.md)
- 기존 UEC와 연결하기: [연결 가이드](x_uec/docs/integration.md)
- 목적함수와 SVD 갱신식: [수식 설명](x_uec/docs/math.md)
- 전체 MS MARCO/SNLI 실행: [실험 결과](x_uec/docs/results.md)

`x_uec/`의 기본 예제는 NumPy만 필요합니다. 아래 기존 UEC 설치는 원래 모델·평가 경로를 실행할 때 사용합니다.

---

# Uncertainty-driven Embedding Convolution

This repository contains PyTorch implemenations of "Uncertainty-driven Embedding Convolution" <!--[paper](.)-->.

## Introduction
Uncertainty-driven Embedding Convolution (UEC) is a framework for combining embeddings in principled, uncertainty-aware manner. UEC consists of three key components.

<img width="9643" height="4862" alt="Image" src="https://github.com/user-attachments/assets/142f35f2-a464-4d94-bccc-1f8875ac993a" />

1. post-hoc conversion of deterministic embeddings into probabilistic ones via
Laplace approximation
2. Gaussian convolution with uncertainty-aware weights
3. uncertainty-aware similarity scoring 


## 🔧 Installation

To install the required dependencies, please follow the steps in order:

```bash
# 1. Install the base dependencies
pip install -r requirements.txt

# 2. Manually install flash-attn (version must match)
pip install flash-attn==2.7.4.post1
```

## ▶️ Running Scripts

Below are example commands to execute the different Python scripts in this project.

### 1. Multilingual Evaluation

For details on how to run the multilingual evaluation, please refer to the [Multilingual Evaluation README](exp_miracls/README.md).



### 2. MTEB Experiments

For details on how to run the MTEB experiments, please refer to the [MTEB Experiments README](exp_mteb/README.md).
