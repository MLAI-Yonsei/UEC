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
