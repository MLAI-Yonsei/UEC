# PDF 목적함수와 실제 v3 구현의 대응

[처음으로](../README.md) · [실험 결과](results.md)

대조 자료: 사용자 제공 `CAIA_정리.pdf` 1–3페이지. 이미지로 들어 있던 식도
렌더링해 확인했다. 아래는 모델 3개인 실제 구현의 식이다.

## 1. 목적함수는 PDF와 동치다

$`X_m\in\mathbb R^{N\times d_m}`$, $`R_m\in\mathbb R^{d_m\times D}`$,
$`R_mR_m^T=I_{d_m}`$, $`Z_m=X_mR_m`$로 둔다. 이 구현에서는
$`(d_q,d_h,d_g)=(1024,640,768)`$, $`D=1024`$이며 Qwen map은 identity다.
이는 공통 직교변환의 기준을 고정한다. native 방향을 줄이는 PCA/MRL이 아니다.

PDF의 목표는 다음과 같다.

```math
F(R,U)=\sum_m\|Z_m-U\|_F^2
 +\gamma\sum_{m<n}\sum_{i=1}^{N}\|P_{m,i}-P_{n,i}\|_F^2.
```

고정 R에서 최적 중심은 $`U^*=\frac13\sum_mZ_m`$이고,

```math
\sum_m\|Z_m-U^*\|_F^2=\frac13\sum_{m<n}\|Z_m-Z_n\|_F^2.
```

실제 `caia.objective()`가 계산하는 것은

```math
L_\mu=\frac1{9N}\sum_{m<n}\|Z_m-Z_n\|_F^2,
\qquad
L_C=\frac1{3N}\sum_{m<n,i}\|P_{m,i}-P_{n,i}\|_F^2,
```

```math
L_{code}=L_\mu+\gamma L_C=\frac1{3N}F(R,U^*).
```

따라서 **전체 목적에 공통인 양의 상수 1/(3N)만 다르며 최소화하는 R과
γ의 상대적 의미는 같다.** 평균 항과 공분산 항에 서로 다른 임의 배수를
추가한 것이 아니다. X는 기존 입력 검사 후 fitting에 사용하는 작업 복사본이다.

## 2. 공분산 제곱근도 PDF의 식이다

v는 최종 native 출력 좌표의 분산이며 표준편차나 parameter variance가 아니다.
$`D_{m,i}=\operatorname{diag}(v_{m,i})`$인 대각 surrogate에 대해

```math
\widetilde\Sigma_{m,i}=R_m^TD_{m,i}R_m,\qquad
P_{m,i}=R_m^T\operatorname{diag}(\sqrt{v_{m,i}})R_m.
```

P는 대칭 PSD이고, row-isometry 때문에

```math
P_{m,i}^2=R_m^T\sqrt{D_{m,i}}(R_mR_m^T)\sqrt{D_{m,i}}R_m
 =R_m^TD_{m,i}R_m=\widetilde\Sigma_{m,i}.
```

따라서 직사각형 R에서도 P는 정렬된 covariance의 **PSD principal square root**다.
확장된 공간의 영 고유값도 포함된다. native full covariance의 비대각까지
이 표현이 보존한다는 주장은 아니며, 출발점은 명시한 native 대각 surrogate다.
이 Frobenius 제곱근 거리는 일반적인 Bures/Wasserstein 거리와 구분한다.

## 3. 실제 갱신은 중심 소거 후의 순차 SVD-MM이다

PDF의 GPA 설명처럼 U를 고정한 Procrustes 부분문제에서는
$`X_m^TU=A\operatorname{diag}(s)B^T`$의 thin SVD로 $`R_m=AB^T`$를 얻는다.
이 설명은 평균만 정렬하는 GPA의 설명으로 맞다.

현재 v3는 U를 명시적으로 고정/교대하는 루프 대신 위 동치 목적에서 U를
소거하고 Harrier→GTE 블록을 순차 갱신한다. CAIA의 공분산 항을 포함한
각 갱신은 **MM 선형화 문제의 SVD 해**이며, 전체 비선형 문제를 반복해 푼다.
고정-U GPA와 같은 반복 궤적 또는 단 한 번의 SVD로 전체 전역 최적해를
구하는 알고리즘이라고 설명하면 정확하지 않다.

통계는 실제 코드와 같이

```math
S_{mn}=X_m^TX_n/N,\qquad
K_{mn}=\sqrt{V_m}^{\,T}\sqrt{V_n}/N
```

로 둔다. 평균/분산의 같은 행은 같은 텍스트·같은 실제 역할의 관측이다.
다른 모델을 고정한 한 블록에서 최소화는 다음을 최대화하는 것과 같다.

```math
g_m(R)=\operatorname{tr}\left(R^T\sum_{n\ne m}S_{mn}R_n\right)
 +3\gamma\sum_{n\ne m}\langle K_{mn},(RR_n^T)^{\odot2}\rangle.
```

K의 성분은 비음이므로 두 번째 항은 ambient R에 대한 볼록 이차식이다.
이를 이전 R에서 접하는 선형 하한으로 바꾸면 SVD에 넣는 행렬은

```math
M_m^{(t)}=\sum_{n\ne m}S_{mn}R_n
 +6\gamma\sum_{n\ne m}
 \left[K_{mn}\odot(R_m^{(t)}R_n^T)\right]R_n.
```

`caia.block_matrix()`가 이 식을 계산한다. 갱신은
$`M_m^{(t)}=A\operatorname{diag}(s)B^T`$, $`R_m^{new}=AB^T`$다.
GTE 갱신에는 같은 sweep에서 새로 얻은 Harrier map을 사용한다.

여기서 **6γ는 중심을 소거한 목적함수의 상대 계수와 이차항 미분에서 나온다.**
선형화는 이전 점에서 접하고 SVD는 그 선형 문제를 최대화하므로 MM 감소
성질을 갖는다. 실제 구현은 loss 비증가·row-isometry도 검사한다. 이는 전역
최적성이나 downstream 개선 보장이 아니며 residual/gap 종료 상태를 따로 본다.

## 4. 실제 코드에 대한 확인

`test_objective_contract.py`는 기존 solver를 바꾸지 않고 다음을 독립 계산한다.

1. 작은 직사각형 R에서 P의 PSD 및 P²=정렬 covariance.
2. PDF의 중심 포함 목적을 직접 계산해 코드 값×3N과 비교.
3. 6γ SVD 입력 행렬, ambient finite-difference 미분 및 한 MM 갱신의 loss 감소.

세 테스트 모두 통과했다. README의 정확한 표현은 다음과 같다.

> GPA와 동치인 평균 정렬 목적에 공분산 principal-square-root 정렬 항을
> 추가한다. 구현은 중심을 소거한 목적함수의 순차 SVD-MM이며, SVD에
> 놓는 행렬에 공분산 정렬 선형화 항을 더한다.
