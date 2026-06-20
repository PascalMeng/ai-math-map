---
title: 线性方程组
description: 面向 AI 算法工程师的线性方程组课程：从 Ax=b、高斯消元、LU/QR/Cholesky 分解，到最小二乘、线性回归、岭回归、伪逆和条件数。
---

# 线性方程组

> 求解 `Ax = b` 是数学中最古老的问题之一，但它今天仍然运行在你的神经网络和机器学习系统里。

**课程类型：** 实现 / 应用  
**所属模块：** 线性代数 / 数值线性代数 / 机器学习基础  
**前置知识：** Phase 1 Lesson 01-03：线性代数直觉、向量与矩阵、矩阵变换  
**预计时间：** 约 120 分钟  
**使用语言：** Python / NumPy / scikit-learn  

---

## 1. 提出问题：为什么 AI 算法工程师必须理解线性方程组

每次你训练一个线性回归模型，本质上都在解一个线性方程组。  
每次你做最小二乘拟合，本质上都在解一个线性方程组。  
每次神经网络线性层计算：

```python
y = W @ x + b
```

它都在执行一个线性变换的一侧。

在机器学习中，`Ax = b` 到处都是：

```txt
线性回归
岭回归
最小二乘
高斯过程
Mahalanobis distance
协方差矩阵求逆
kernel matrix 分解
数值优化中的牛顿法
大规模稀疏系统求解
```

其中：

```txt
A 是已知矩阵。
b 是已知输出向量。
x 是未知参数向量。
```

在线性回归中：

```txt
A 是数据矩阵。
b 是目标值。
x 是要学习的权重。
```

问题是：

```txt
如何稳定、快速、可靠地求解 x？
```

不同场景下答案不同：

```txt
方阵且可逆：可以直接求解。
过定系统：没有精确解，需要最小二乘。
矩阵病态：需要正则化。
大规模稀疏系统：需要迭代方法。
协方差矩阵：适合 Cholesky。
秩亏矩阵：适合 SVD / 伪逆。
```

本课要把这些方法串起来，让你知道每种方法为什么存在、适合什么场景，以及如何从零实现。

---

### 1.1 数学概念历史出现缘由

线性方程组是线性代数最核心的问题之一。  
它最早来自方程求解、几何交点、工程计算和科学建模。  
在现代 AI 中，它又成为回归、矩阵分解、协方差建模和数值优化的底层工具。

| 数学 / 工程概念 | 出现缘由 |
|---|---|
| 线性方程组 | 为了同时满足多个线性约束 |
| `Ax = b` | 为了用矩阵形式统一表示多个线性方程 |
| 行视角 | 为了把每个方程理解成一个超平面 |
| 列视角 | 为了把求解理解成列向量的线性组合 |
| 高斯消元 | 为了系统地把方程组化为上三角形式 |
| 回代 | 为了从上三角系统中从后往前求解 |
| 部分主元 | 为了避免除以很小的数，提高数值稳定性 |
| LU 分解 | 为了把一次消元结果保存下来，多次求解不同 b |
| QR 分解 | 为了用正交矩阵稳定地求解最小二乘问题 |
| Cholesky 分解 | 为了高效分解对称正定矩阵 |
| 最小二乘 | 为了在无精确解时找到误差最小的近似解 |
| 正规方程 | 为了通过求导得到最小二乘闭式解 |
| 线性回归 | 为了用线性模型拟合数据 |
| 岭回归 | 为了通过正则化改善病态矩阵和过拟合 |
| 伪逆 | 为了推广矩阵求逆到非方阵和奇异矩阵 |
| 条件数 | 为了衡量解对输入扰动的敏感程度 |
| 共轭梯度 | 为了求解大规模稀疏对称正定系统 |

可以这样理解：

```txt
现实问题：多个线性方程同时满足，未知数是多少？
数学抽象：Ax = b

现实问题：精确解不存在，怎么找到最接近的解？
数学方法：最小二乘

现实问题：同一个 A，要解很多不同 b，怎么省计算？
矩阵分解：LU

现实问题：矩阵对称正定，如何更快求解？
矩阵分解：Cholesky

现实问题：矩阵病态，解非常不稳定怎么办？
数值诊断：条件数
工程方法：正则化

现实问题：矩阵不是方阵或秩亏，怎么求广义解？
数学工具：SVD 伪逆
```

---

### 1.2 原始问题是什么

本课要解决的原始问题是：

```txt
Ax = b 几何上表示什么？
什么时候线性方程组有唯一解、无解、无穷多解？
行视角和列视角分别如何理解？
高斯消元如何把方程组变成上三角系统？
为什么部分主元对数值稳定性重要？
LU 分解为什么适合多次求解同一个 A？
QR 分解为什么适合最小二乘？
Cholesky 分解为什么适合对称正定矩阵？
过定系统为什么没有精确解？
最小二乘如何导出正规方程？
线性回归和正规方程是什么关系？
岭回归为什么是给 `X^T X` 加 `lambda I`？
伪逆如何处理非方阵、奇异矩阵和无穷多解？
条件数为什么决定解是否可信？
共轭梯度为什么适合大规模稀疏系统？
```

换成 AI 语言，就是：

```txt
LinearRegression.fit() 背后在解什么？
Ridge regression 为什么能稳定病态矩阵？
为什么不要轻易显式求 inverse？
为什么 np.linalg.lstsq 通常比 normal equations 更稳？
为什么 Gaussian Process 要用 Cholesky？
为什么特征共线性会让回归权重不稳定？
为什么正则化能改善 condition number？
```

---

### 1.3 AI 中的现代问题

| 线性系统知识点 | AI 中的现代问题 |
|---|---|
| `Ax = b` | 线性回归、线性层、最小二乘 |
| 高斯消元 | 基础直接求解方法 |
| Partial pivoting | 数值稳定性 |
| LU | 多个右端项快速求解 |
| QR | 稳定最小二乘 |
| Cholesky | Gaussian Process、协方差矩阵、ridge |
| Least squares | 回归、拟合、参数估计 |
| Normal equations | 线性回归闭式解 |
| Ridge regression | 正则化、共线性、稳定性 |
| Pseudoinverse | 非方阵和秩亏系统 |
| SVD | 稳定求解、最小范数解 |
| Condition number | 病态矩阵、特征共线性 |
| Conjugate gradient | 大规模稀疏系统 |
| Regularization | 改善条件数、防止过拟合 |
| Forward / Back substitution | 三角系统求解 |
| Kernel matrix solve | GP、kernel methods |
| Orthogonal initialization | QR 在神经网络初始化中的应用 |

---

### 1.4 本课要解决的核心问题

学完本课，你应该能够回答：

1. `Ax = b` 在几何上是什么意思？
2. 线性系统什么时候有唯一解、无解、无穷多解？
3. 行视角和列视角分别是什么？
4. Gaussian elimination 如何工作？
5. Partial pivoting 为什么能提升稳定性？
6. LU 分解如何复用一次消元结果？
7. QR 分解为什么适合 least squares？
8. Cholesky 分解为什么要求 symmetric positive definite？
9. 最小二乘如何从 `min ||Ax-b||^2` 推导出正规方程？
10. 线性回归闭式解和正规方程有什么关系？
11. Ridge regression 如何修改线性系统？
12. Moore-Penrose pseudoinverse 如何求最小范数最小二乘解？
13. Condition number 如何判断解是否可信？
14. Regularization 为什么能改善病态系统？
15. Conjugate gradient 适合什么大规模问题？
16. 每种线性系统求解方法应该在什么场景使用？

---

## 2. 概念定位：这节课学什么

---

### 2.1 所属模块

```txt
所属模块：线性代数 / 数值线性代数 / 机器学习基础
前置知识：矩阵乘法、矩阵变换、秩、特征值、SVD、范数
后续连接：线性回归、岭回归、Gaussian Process、Kernel methods、Newton-CG、数值优化
```

前面的课程已经解决了：

```txt
矩阵如何变换空间？
SVD 如何分析矩阵结构？
范数和距离如何衡量误差？
```

本课进一步解决：

```txt
如何求解由矩阵定义的线性约束系统，以及如何判断求解结果是否可靠？
```

---

### 2.2 本课学习目标

完成本课后，你应该能够：

- 用几何视角解释 `Ax = b`
- 区分 row picture 和 column picture
- 从零实现 Gaussian elimination with partial pivoting
- 从零实现 back substitution
- 实现 LU decomposition 和 LU solve
- 理解 QR decomposition 及其在 least squares 中的作用
- 实现 Cholesky decomposition
- 推导 least squares 的 normal equations
- 实现 linear regression 和 ridge regression 的闭式求解
- 使用 SVD 理解 Moore-Penrose pseudoinverse
- 计算 condition number 并诊断 ill-conditioned systems
- 理解 conjugate gradient 适合的大规模稀疏场景
- 根据矩阵性质选择求解方法

---

### 2.3 本课在整体路线中的位置

```txt
矩阵变换
    ↓
范数与距离
    ↓
SVD
    ↓
线性方程组
    ↓
最小二乘 / 线性回归 / 岭回归
    ↓
Gaussian Process / Kernel methods / 大规模优化
```

本课的核心能力是：

```txt
看到一个机器学习问题里的线性系统时，能判断它是方阵求解、最小二乘、正则化系统，还是需要稳定分解或迭代求解。
```

---

## 3. 理解概念：直觉、公式、几何解释、AI 连接

---

### 3.1 `Ax = b` 几何上是什么意思

**一句话直觉：**  
每个线性方程是一条线或一个超平面，求解线性系统就是找这些超平面的交点。

例子：

```text
2x + y = 5
x - y = 1
```

这两条直线交于：

```text
x = 2, y = 1
```

对应矩阵形式：

```text
A x = b
```

其中：

```text
A = [[2, 1],
     [1,-1]]

x = [x, y]

b = [5, 1]
```

三种情况：

| 情况 | 几何解释 | 线性代数解释 |
|---|---|---|
| 唯一解 | 所有超平面交于一点 | A 可逆，满秩 |
| 无解 | 超平面没有共同交点 | 系统不一致 |
| 无穷多解 | 超平面重合或交成子空间 | A 有 null space |

**AI 连接：**

多数机器学习回归问题不是“精确解”问题，而是：

```txt
数据点多于参数，系统过定，没有精确解。
```

因此需要 least squares。

---

### 3.2 Row picture vs Column picture

**一句话直觉：**  
行视角看每个方程；列视角看能否用 A 的列向量线性组合出 b。

#### Row picture

每一行是一个方程：

```text
2x + y = 5
x - y = 1
```

解是所有方程同时成立的位置。

#### Column picture

把 A 看成列向量：

```text
A = [[2, 1],
     [1,-1]]
```

问题变成：

```text
x1 * [2, 1] + x2 * [1, -1] = [5, 1]
```

如果：

```text
x1 = 2, x2 = 1
```

则：

```text
2 * [2, 1] + 1 * [1, -1] = [5, 1]
```

**关键直觉：**

```txt
如果 b 在 A 的 column space 中，系统有解。
如果 b 不在 column space 中，就找 column space 中离 b 最近的点。
这就是 least squares。
```

---

### 3.3 Gaussian elimination：把系统变成上三角

**一句话直觉：**  
高斯消元通过行操作消掉主元下方的元素，把 `Ax=b` 变成容易回代的上三角系统。

目标：

```text
Ax = b
```

变成：

```text
Ux = c
```

其中 `U` 是上三角矩阵。

算法：

```txt
1. 对每一列 k：
   a. 选择 pivot row。
   b. 将 pivot row 换到第 k 行。
   c. 对 k 下方每一行 i：
      multiplier = A[i,k] / A[k,k]
      row_i = row_i - multiplier * row_k
2. 得到上三角矩阵。
3. 从最后一行开始 back substitution。
```

复杂度：

```text
O(n^3)
```

**AI 连接：**

高斯消元是很多直接求解器的基础。  
但真实工程中通常不手写，而是调用：

```python
np.linalg.solve
scipy.linalg.solve
```

---

### 3.4 Back substitution：从下往上解

**一句话直觉：**  
上三角系统最后一个方程只含最后一个未知数，解出后逐步往上代回。

例如：

```text
2x1 + x2 + x3 = 8
     x2 + x3 = 4
         -2x3 = -4
```

先解：

```text
x3 = 2
```

再解：

```text
x2 + 2 = 4 => x2 = 2
```

再解：

```text
2x1 + 2 + 2 = 8 => x1 = 2
```

复杂度：

```text
O(n^2)
```

---

### 3.5 Partial pivoting：为什么要换行

**一句话直觉：**  
部分主元选择当前列中绝对值最大的可用元素作为 pivot，避免除以很小的数导致误差放大。

如果 pivot 很小：

```txt
multiplier 会很大。
浮点误差会被放大。
结果可能完全不可信。
```

partial pivoting：

```txt
在第 k 列，从第 k 行到最后一行中选绝对值最大的元素。
把该行交换到第 k 行。
```

**AI 连接：**

数值线性代数中，稳定性和复杂度同样重要。  
不要显式求 inverse 再乘，而应使用稳定 solver。

---

### 3.6 LU decomposition：一次分解，多次求解

**一句话直觉：**  
LU 把矩阵 A 分解成下三角 L 和上三角 U，让同一个 A 可以高效求解多个不同 b。

分解：

```text
A = L U
```

其中：

| 矩阵 | 含义 |
|---|---|
| `L` | lower triangular，存储消元乘子 |
| `U` | upper triangular，消元后的矩阵 |

如果有 pivoting：

```text
P A = L U
```

求解：

```text
A x = b
```

转为：

```text
L U x = b
```

令：

```text
U x = y
```

则：

```text
L y = b   forward substitution
U x = y   back substitution
```

成本：

```txt
分解一次：O(n^3)
每次求解：O(n^2)
```

**AI 连接：**

当同一个矩阵 A 对多个 b 求解时，LU 很有用。

---

### 3.7 QR decomposition：正交分解，更稳定的最小二乘

**一句话直觉：**  
QR 把 A 分解成正交矩阵 Q 和上三角矩阵 R，适合稳定求解 least squares。

分解：

```text
A = Q R
```

其中：

| 矩阵 | 含义 |
|---|---|
| `Q` | 列正交，`Q^T Q = I` |
| `R` | 上三角 |

求解：

```text
A x = b
```

变成：

```text
Q R x = b
```

两边乘 `Q^T`：

```text
R x = Q^T b
```

然后 back substitution。

**为什么稳定？**

```txt
正交矩阵不会放大向量长度。
QR 避免显式形成 A^T A，因此比 normal equations 更稳定。
```

**AI 连接：**

线性回归和最小二乘中，QR 通常比直接解 normal equations 更可靠。

---

### 3.8 Cholesky decomposition：对称正定矩阵的快速分解

**一句话直觉：**  
如果矩阵 A 是对称正定的，可以写成 `A = L L^T`，这比 LU 更快、更省内存。

要求：

```txt
A = A^T
所有特征值 > 0
```

分解：

```text
A = L L^T
```

其中 `L` 是下三角矩阵。

优点：

```txt
比 LU 约快 2 倍。
存储更少。
数值稳定。
```

常见 symmetric positive definite 矩阵：

- 协方差矩阵加正则化
- Kernel matrix 加 jitter
- Ridge regression 中的 `X^T X + lambda I`
- Gaussian Process 的 kernel matrix
- 凸优化中某些 Hessian

**AI 连接：**

Gaussian Process 中常见：

```text
K alpha = y
```

通常用 Cholesky 解。

log determinant：

```text
log det(K) = 2 * sum(log(diag(L)))
```

这在 GP marginal likelihood 中非常重要。

---

### 3.9 Least squares：无精确解时找最接近的解

**一句话直觉：**  
当方程多于未知数、无法全部满足时，least squares 找到让残差平方和最小的 x。

过定系统：

```text
A: m x n, m > n
```

通常没有精确解。  
目标改成：

```text
minimize ||A x - b||^2
```

残差：

```text
r = A x - b
```

目标：

```text
sum_i r_i^2
```

**AI 连接：**

这就是线性回归的核心。

---

### 3.10 Normal equations：最小二乘的闭式条件

**一句话直觉：**  
对 least squares 目标求导并令梯度为 0，会得到正规方程。

目标：

```text
minimize ||A x - b||^2
```

展开：

```text
(Ax - b)^T (Ax - b)
= x^T A^T A x - 2 x^T A^T b + b^T b
```

求梯度：

```text
2 A^T A x - 2 A^T b
```

令梯度为 0：

```text
A^T A x = A^T b
```

这就是 normal equations。

如果 `A^T A` 可逆：

```text
x = (A^T A)^(-1) A^T b
```

**注意：**

```txt
不要在代码中显式求 inverse。
应使用 solver、QR 或 SVD。
```

---

### 3.11 Normal equations = 线性回归闭式解

**一句话直觉：**  
线性回归就是在解一个 least squares 问题。

数据矩阵：

```text
X: n_samples x n_features
```

目标：

```text
y: n_samples
```

权重：

```text
w: n_features
```

线性回归目标：

```text
minimize ||X w - y||^2
```

正规方程：

```text
X^T X w = X^T y
```

闭式解：

```text
w = (X^T X)^(-1) X^T y
```

**AI 连接：**

这就是最基础的监督学习模型之一。  
`sklearn.linear_model.LinearRegression` 内部会用稳定数值方法计算等价解。

---

### 3.12 Ridge regression：正则化后的线性系统

**一句话直觉：**  
Ridge regression 在 least squares 上加 L2 正则化，改善病态矩阵并防止过拟合。

目标：

```text
minimize ||X w - y||^2 + lambda ||w||^2
```

求导得：

```text
(X^T X + lambda I) w = X^T y
```

闭式解：

```text
w = (X^T X + lambda I)^(-1) X^T y
```

为什么有用？

```txt
lambda I 提高最小特征值。
改善 condition number。
让解对噪声不那么敏感。
把权重往 0 收缩，降低过拟合。
```

当 `lambda > 0` 时：

```txt
X^T X + lambda I 通常是 symmetric positive definite。
适合用 Cholesky 求解。
```

---

### 3.13 Moore-Penrose Pseudoinverse：非方阵和奇异矩阵的广义逆

**一句话直觉：**  
伪逆把矩阵求逆推广到非方阵、奇异矩阵和无精确解系统。

如果：

```text
A = U Sigma V^T
```

则：

```text
A^+ = V Sigma^+ U^T
```

其中：

```txt
Sigma^+ 把非零奇异值取倒数，并转置形状。
零奇异值保持为 0。
```

解：

```text
x = A^+ b
```

意义：

| 系统类型 | `A^+ b` 给出 |
|---|---|
| 唯一解 | 唯一解 |
| 无解 | least-squares 解 |
| 无穷多解 | 最小范数解 |
| 秩亏 | 稳定的广义解 |

**AI 连接：**

NumPy 的：

```python
np.linalg.lstsq
np.linalg.pinv
```

通常都依赖 SVD 或稳定变体。

---

### 3.14 Condition number：你的解是否可信

**一句话直觉：**  
条件数衡量线性系统对输入扰动有多敏感。

定义：

```text
kappa(A) = ||A|| ||A^{-1}||
```

用 SVD 表示：

```text
kappa(A) = sigma_max / sigma_min
```

解释：

| condition number | 含义 |
|---:|---|
| 接近 1 | 很稳定 |
| `< 100` | 通常安全 |
| `10^k` | 大约损失 k 位有效数字 |
| `10^16` | float64 下基本不可信 |

病态系统例子：

```text
A = [[1, 1],
     [1, 1 + 1e-15]]
```

两列几乎一样，系统对微小扰动极度敏感。

**AI 连接：**

| 场景 | 影响 |
|---|---|
| 特征共线性 | 回归权重不稳定 |
| 协方差矩阵求逆 | Mahalanobis distance 不稳定 |
| Gaussian Process | kernel matrix 需要 jitter |
| 正则化 | 改善 condition number |
| 特征标准化 | 降低尺度导致的病态 |

---

### 3.15 Regularization 如何改善条件数

**一句话直觉：**  
给矩阵加 `lambda I` 会抬高小特征值，使系统更稳定。

如果原奇异值或特征值为：

```text
sigma_max, sigma_min
```

正则化后近似：

```text
sigma_max + lambda
sigma_min + lambda
```

条件数改善为：

```text
(sigma_max + lambda) / (sigma_min + lambda)
```

当 `sigma_min` 很小时，加 `lambda` 可以显著提升稳定性。

**AI 连接：**

Ridge regression 不是只为了防止过拟合，也是在做数值稳定化。

---

### 3.16 Conjugate Gradient：大规模稀疏系统的迭代求解

**一句话直觉：**  
共轭梯度不做矩阵分解，而是从一个初始猜测开始，迭代逼近解。

适用条件：

```txt
A 是 symmetric positive definite。
A 很大、很稀疏。
直接分解太贵。
```

基本思想：

```txt
不断沿着互相共轭的方向搜索。
理论上最多 n 步收敛。
实际中通常更快，尤其当条件数较好。
```

核心变量：

| 变量 | 含义 |
|---|---|
| `x` | 当前解 |
| `r = b - Ax` | residual |
| `p` | search direction |
| `alpha` | 当前步长 |
| `beta` | 更新搜索方向 |

**AI 连接：**

| 场景 | CG 用途 |
|---|---|
| Newton-CG | 大规模优化 |
| Kernel methods | kernel matrix 太大 |
| PDE / scientific ML | 稀疏系统 |
| Preconditioning | 加速迭代收敛 |
| 大规模 least squares | 不能直接分解时 |

---

### 3.17 不同求解方法如何选择

| 方法 | 要求 | 成本 | 适用场景 |
|---|---|---|---|
| Gaussian elimination | 方阵、非奇异 | `O(n^3)` | 一次性小中型方阵求解 |
| LU | 方阵、非奇异 | factor `O(n^3)`，solve `O(n^2)` | 同一 A 多个 b |
| QR | `m >= n` | `O(mn^2)` | 稳定 least squares |
| Cholesky | 对称正定 | 约 `O(n^3/3)` | 协方差、kernel、ridge |
| Normal equations | 过定系统 | `O(mn^2+n^3)` | 小规模线性回归 |
| SVD / pseudoinverse | 任意 A | 较慢但稳定 | 秩亏、最小范数解 |
| Conjugate gradient | 稀疏 SPD | 每步依赖 nnz | 大规模稀疏系统 |

经验法则：

```txt
不要显式求 inverse。
小规模方阵：np.linalg.solve。
最小二乘：np.linalg.lstsq 或 QR/SVD。
对称正定：Cholesky。
病态或秩亏：SVD / pseudoinverse / regularization。
大规模稀疏 SPD：conjugate gradient。
```

---

### 3.18 本课核心概念总表

| 概念 | 一句话直觉 | AI 中的对应位置 |
|---|---|---|
| `Ax=b` | 线性系统 | 回归、矩阵求解 |
| Row picture | 每行一个约束 | 方程几何 |
| Column picture | 列向量线性组合 | column space |
| Gaussian elimination | 消元成上三角 | 直接求解 |
| Partial pivoting | 选大主元 | 数值稳定 |
| Back substitution | 从下往上解 | 上三角系统 |
| LU | 下三角 × 上三角 | 多次求解 |
| QR | 正交 × 上三角 | 稳定最小二乘 |
| Cholesky | `L L^T` | SPD 系统 |
| Least squares | 最小残差平方和 | 线性回归 |
| Normal equations | `A^T A x = A^T b` | 闭式回归 |
| Ridge | 加 `lambda I` | 正则化 |
| Pseudoinverse | 广义逆 | 非方阵 / 秩亏 |
| Condition number | 解敏感性 | 病态诊断 |
| Conjugate gradient | 迭代解 SPD | 大规模稀疏 |

---

## 4. 手写实现：从零实现线性系统求解

这一节使用 NumPy 做基础数组操作，但核心算法自己实现。

---

### 4.1 Gaussian elimination with partial pivoting

```python
import numpy as np


def gaussian_elimination(A, b):
    n = len(b)
    Ab = np.hstack([
        A.astype(float),
        b.reshape(-1, 1).astype(float)
    ])

    for k in range(n):
        max_row = k + np.argmax(np.abs(Ab[k:, k]))
        Ab[[k, max_row]] = Ab[[max_row, k]]

        if abs(Ab[k, k]) < 1e-12:
            raise ValueError(
                f"Matrix is singular or nearly singular at pivot {k}"
            )

        for i in range(k + 1, n):
            m = Ab[i, k] / Ab[k, k]
            Ab[i, k:] -= m * Ab[k, k:]

    x = np.zeros(n)

    for i in range(n - 1, -1, -1):
        x[i] = (
            Ab[i, -1] - Ab[i, i + 1:n] @ x[i + 1:n]
        ) / Ab[i, i]

    return x
```

---

### 4.2 LU decomposition

```python
def lu_decompose(A):
    n = A.shape[0]

    L = np.eye(n)
    U = A.astype(float).copy()
    P = np.eye(n)

    for k in range(n):
        max_row = k + np.argmax(np.abs(U[k:, k]))

        if max_row != k:
            U[[k, max_row]] = U[[max_row, k]]
            P[[k, max_row]] = P[[max_row, k]]

            if k > 0:
                L[[k, max_row], :k] = L[[max_row, k], :k]

        for i in range(k + 1, n):
            L[i, k] = U[i, k] / U[k, k]
            U[i, k:] -= L[i, k] * U[k, k:]

    return P, L, U
```

---

### 4.3 LU solve

```python
def lu_solve(P, L, U, b):
    n = len(b)

    Pb = P @ b.astype(float)

    y = np.zeros(n)

    for i in range(n):
        y[i] = Pb[i] - L[i, :i] @ y[:i]

    x = np.zeros(n)

    for i in range(n - 1, -1, -1):
        x[i] = (
            y[i] - U[i, i + 1:] @ x[i + 1:]
        ) / U[i, i]

    return x
```

---

### 4.4 Cholesky decomposition

```python
def cholesky(A):
    n = A.shape[0]
    L = np.zeros_like(A, dtype=float)

    for i in range(n):
        for j in range(i + 1):
            s = A[i, j] - L[i, :j] @ L[j, :j]

            if i == j:
                if s <= 0:
                    raise ValueError("Matrix is not positive definite")

                L[i, j] = np.sqrt(s)
            else:
                L[i, j] = s / L[j, j]

    return L
```

---

### 4.5 Least squares via normal equations

```python
def least_squares_normal(A, b):
    AtA = A.T @ A
    Atb = A.T @ b

    return gaussian_elimination(AtA, Atb)
```

---

### 4.6 Ridge regression with Cholesky

```python
def ridge_regression(A, b, lam):
    n = A.shape[1]

    AtA = A.T @ A + lam * np.eye(n)
    Atb = A.T @ b

    L = cholesky(AtA)

    # Solve L y = Atb
    y = np.zeros(n)

    for i in range(n):
        y[i] = (Atb[i] - L[i, :i] @ y[:i]) / L[i, i]

    # Solve L.T x = y
    x = np.zeros(n)

    LT = L.T

    for i in range(n - 1, -1, -1):
        x[i] = (
            y[i] - LT[i, i + 1:] @ x[i + 1:]
        ) / LT[i, i]

    return x
```

---

### 4.7 Condition number

```python
def condition_number(A):
    _, S, _ = np.linalg.svd(A)

    return S[0] / S[-1]
```

---

### 4.8 Conjugate Gradient

```python
def conjugate_gradient(A, b, x0=None, tol=1e-8, max_iter=None):
    n = len(b)

    if x0 is None:
        x = np.zeros(n)
    else:
        x = x0.astype(float).copy()

    if max_iter is None:
        max_iter = n

    r = b - A @ x
    p = r.copy()
    rs_old = r @ r

    for _ in range(max_iter):
        Ap = A @ p
        alpha = rs_old / (p @ Ap)

        x = x + alpha * p
        r = r - alpha * Ap

        rs_new = r @ r

        if np.sqrt(rs_new) < tol:
            break

        beta = rs_new / rs_old
        p = r + beta * p
        rs_old = rs_new

    return x
```

---

## 5. 生产使用：NumPy / scikit-learn 中的线性系统

---

### 5.1 NumPy 直接求解

```python
import numpy as np

A = np.array([
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 10],
], dtype=float)

b = np.array([6, 15, 27], dtype=float)

x = np.linalg.solve(A, b)

print(x)
```

---

### 5.2 NumPy least squares

```python
X = np.random.randn(100, 3)
w_true = np.array([2.0, -1.0, 0.5])
y = X @ w_true + np.random.randn(100) * 0.1

X_with_bias = np.column_stack([np.ones(100), X])

w_lstsq = np.linalg.lstsq(X_with_bias, y, rcond=None)[0]

print(w_lstsq)
```

---

### 5.3 NumPy QR / SVD

```python
Q, R = np.linalg.qr(X_with_bias)

w_qr = np.linalg.solve(R, Q.T @ y)

U, S, Vt = np.linalg.svd(X_with_bias, full_matrices=False)

w_svd = Vt.T @ np.diag(1 / S) @ U.T @ y

print(w_qr)
print(w_svd)
```

---

### 5.4 scikit-learn Ridge

```python
from sklearn.linear_model import Ridge

ridge = Ridge(alpha=1.0, fit_intercept=False)
ridge.fit(X_with_bias, y)

print(ridge.coef_)
```

---

### 5.5 Gaussian Process 中的 Cholesky

```python
K = X @ X.T
jitter = 1e-6

K_stable = K + jitter * np.eye(K.shape[0])

L = np.linalg.cholesky(K_stable)

alpha = np.linalg.solve(L.T, np.linalg.solve(L, y))

logdet = 2 * np.sum(np.log(np.diag(L)))

print(alpha.shape)
print(logdet)
```

---

### 5.6 手写实现 vs 生产库

| 对比项 | 手写实现 | NumPy / SciPy / sklearn |
|---|---|---|
| 目的 | 理解算法原理 | 工程实践 |
| Gaussian elimination | 教学可用 | `np.linalg.solve` |
| LU | 手写分解 | `scipy.linalg.lu_factor` |
| QR | 可手写 Gram-Schmidt | `np.linalg.qr` |
| Cholesky | 手写可学原理 | `np.linalg.cholesky` |
| Least squares | Normal equations | `np.linalg.lstsq` |
| Pseudoinverse | SVD 公式 | `np.linalg.pinv` |
| Ridge | 手写 Cholesky | `sklearn.linear_model.Ridge` |
| Condition number | SVD 比值 | `np.linalg.cond` |
| Conjugate gradient | 手写迭代 | `scipy.sparse.linalg.cg` |

---

## 6. 交付沉淀：产出物、连接关系、练习、术语表

---

### 6.1 本课产出

```txt
code/linear_systems.py
code/use_numpy.py
code/use_sklearn.py
outputs/prompt-linear-system-solver.md
outputs/exercises-solutions.md
```

| 文件 | 作用 |
|---|---|
| `code/linear_systems.py` | 从零实现 Gaussian elimination、LU、Cholesky、least squares、ridge、CG |
| `code/use_numpy.py` | NumPy solve、lstsq、QR、SVD、Cholesky、condition number 示例 |
| `code/use_sklearn.py` | LinearRegression、Ridge 对照 |
| `outputs/prompt-linear-system-solver.md` | 用于判断线性系统求解方法的 prompt |
| `outputs/exercises-solutions.md` | 练习题与参考答案 |

---

### 6.2 知识连接关系

| 本课知识点 | 后续连接 |
|---|---|
| `Ax=b` | 线性回归、线性层 |
| Gaussian elimination | 直接求解器 |
| LU | 多右端项求解 |
| QR | 稳定 least squares |
| Cholesky | GP、ridge、covariance |
| Least squares | 回归、拟合 |
| Normal equations | 线性回归闭式解 |
| Ridge | 正则化、病态矩阵 |
| Pseudoinverse | SVD、最小范数解 |
| Condition number | 数值稳定性 |
| Conjugate gradient | 大规模稀疏优化 |
| Regularization | 泛化和稳定性 |

---

### 6.3 AI 应用连接

| 线性系统知识点 | AI 应用 |
|---|---|
| Normal equations | Linear regression |
| Ridge system | Ridge regression、weight decay 直觉 |
| Cholesky | Gaussian Process、kernel methods |
| QR | 稳定 least squares |
| SVD pseudoinverse | rank-deficient regression |
| Condition number | 特征共线性诊断 |
| Regularization | 改善条件数、防止过拟合 |
| Conjugate gradient | large-scale optimization |
| Orthogonal Q | neural network initialization |
| Covariance inverse | Mahalanobis distance |

---

### 6.4 练习

1. **比较三种求解器。**  
   求解：

   ```text
   [[1,2,3],
    [4,5,6],
    [7,8,10]] x = [6, 15, 27]
   ```

   分别使用你的 Gaussian elimination、LU solver 和 `np.linalg.solve`。验证三者结果在浮点误差范围内一致。

2. **Least squares 多方法对比。**  
   生成一个 `50 x 5` 随机矩阵 `X` 和目标：

   ```text
   y = X @ w_true + noise
   ```

   分别使用 normal equations、QR、SVD 和 `np.linalg.lstsq` 求解。比较解和 residual。

3. **构造病态矩阵。**  
   让第二列几乎等于第一列：

   ```text
   col2 = col1 + 1e-10 * noise
   ```

   计算 condition number。  
   比较无正则化和加 `0.01 I` 正则化后的解和 residual。

4. **实现 Conjugate Gradient。**  
   生成一个 `100 x 100` 随机 symmetric positive definite 矩阵。  
   用 CG 求解到 tolerance `1e-8`，记录迭代次数，并和理论最多 `n` 次比较。

5. **Cholesky vs LU 计时。**  
   在 size 为 `10, 50, 200, 500` 的 SPD 矩阵上比较 Cholesky、LU、`np.linalg.solve` 的时间。验证 Cholesky 通常更快。

6. **Ridge regression 实验。**  
   构造共线性强的数据，比较 OLS 和 Ridge 权重稳定性。观察 `lambda` 增大时权重如何收缩。

---

### 6.5 关键术语

| 术语 | 常见说法 | 真正含义 |
|---|---|---|
| Linear system | 解方程组 | 一组线性方程 `Ax=b` |
| Gaussian elimination | 行消元 | 用行操作把矩阵变成上三角 |
| Partial pivoting | 选主元换行 | 选择最大可用 pivot，减少数值误差 |
| Back substitution | 回代 | 解上三角系统，从最后一行往上 |
| Forward substitution | 前代 | 解下三角系统，从第一行往下 |
| LU decomposition | 三角分解 | `A=LU` 或 `PA=LU` |
| QR decomposition | 正交分解 | `A=QR`，适合稳定最小二乘 |
| Cholesky decomposition | 矩阵平方根 | SPD 矩阵 `A=LL^T` |
| Least squares | 最小二乘 | 最小化 `||Ax-b||^2` |
| Normal equations | 正规方程 | `A^T A x = A^T b` |
| Pseudoinverse | 伪逆 | SVD 定义的广义逆 |
| Condition number | 条件数 | `sigma_max/sigma_min`，衡量解敏感性 |
| Ridge regression | 岭回归 | `(X^T X + lambda I)w = X^T y` |
| Overdetermined system | 过定系统 | 方程数多于未知数 |
| Null space | 零空间 | 被 A 映射到 0 的输入方向 |
| Column space | 列空间 | A 的列向量所有线性组合 |
| Conjugate gradient | 共轭梯度 | 大规模 SPD 系统的迭代求解器 |

---

### 6.6 自检问题

学完本课后，你应该能回答：

- `Ax=b` 为什么是线性代数和机器学习的核心问题？
- 行视角和列视角分别如何理解？
- 什么时候系统有唯一解、无解、无穷多解？
- Gaussian elimination 的步骤是什么？
- Partial pivoting 为什么重要？
- Back substitution 如何工作？
- LU 分解为什么适合多个 b？
- QR 为什么比 normal equations 更稳定？
- Cholesky 为什么要求 symmetric positive definite？
- 最小二乘如何推导出 normal equations？
- Linear regression 和 `X^T X w = X^T y` 是什么关系？
- Ridge regression 为什么加 `lambda I`？
- Pseudoinverse 如何处理非方阵和奇异矩阵？
- Condition number 大意味着什么？
- Regularization 如何改善 condition number？
- Conjugate gradient 适合什么问题？
- 真实工程中为什么不推荐显式求 inverse？

---

## 附录：本课最小代码文件建议

如果要拆成代码文件，可以这样组织：

```txt
code/
├── linear_systems.py
├── use_numpy.py
└── use_sklearn.py
```

### `linear_systems.py`

包含：

```txt
gaussian_elimination
lu_decompose
lu_solve
cholesky
least_squares_normal
ridge_regression
condition_number
conjugate_gradient
```

### `use_numpy.py`

包含：

```txt
np.linalg.solve
np.linalg.lstsq
np.linalg.qr
np.linalg.svd
np.linalg.pinv
np.linalg.cholesky
np.linalg.cond
```

### `use_sklearn.py`

包含：

```txt
LinearRegression
Ridge
fit_intercept
coef_
regularization experiments
```
