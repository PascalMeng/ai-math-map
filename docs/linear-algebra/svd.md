---
title: 奇异值分解 SVD
description: 面向 AI 算法工程师的线性代数课程：从 SVD 的几何直觉、奇异值、低秩近似、图像压缩，到伪逆、推荐系统、LSA 和 PCA。
---

# 奇异值分解 SVD

> SVD 是线性代数里的瑞士军刀。任何矩阵都有 SVD，每个数据科学家都应该掌握它。

**课程类型：** 实现 / 应用  
**所属模块：** 线性代数 / 矩阵分解 / 降维  
**前置知识：** Phase 1 Lesson 01-03：线性代数直觉、向量与矩阵运算、矩阵变换  
**预计时间：** 约 120 分钟  
**使用语言：** Python / NumPy / Julia  

---

## 1. 提出问题：为什么 AI 算法工程师必须理解 SVD

你有一个 `1000 x 2000` 的矩阵。它可能是：

```txt
用户-电影评分矩阵
文档-词频矩阵
图像像素矩阵
embedding 矩阵
实验观测矩阵
特征矩阵
```

你希望对它做几件事：

```txt
压缩
去噪
发现隐藏结构
求低秩近似
求最小二乘解
做推荐系统
做文本语义分析
做 PCA 降维
```

问题是：特征分解只适用于方阵，而且还要求矩阵有足够好的特征向量结构。  
现实中的数据矩阵通常是长方形的、不满秩的、带噪声的。

SVD 的强大之处在于：

```txt
任何矩阵都可以做 SVD。
任意形状。
任意秩。
不需要额外条件。
```

它把一个矩阵拆成三个部分，让你看清这个矩阵到底如何变换空间，以及哪些方向最重要。

---

### 1.1 数学概念历史出现缘由

SVD 最初来自线性代数和数值分析中的矩阵分解问题。  
人们希望找到一种方法，可以稳定地分析任意矩阵的结构，而不局限于方阵。

| 数学概念 | 历史出现缘由 |
|---|---|
| 矩阵分解 | 为了把复杂矩阵拆成更容易理解和计算的因子 |
| SVD | 为了分解任意形状矩阵，揭示输入方向、缩放强度和输出方向 |
| 奇异值 | 为了衡量矩阵在每个主方向上的拉伸强度 |
| 左奇异向量 | 为了描述矩阵作用后的输出空间主方向 |
| 右奇异向量 | 为了描述矩阵输入空间中的主方向 |
| 低秩近似 | 为了用更少参数近似原始矩阵 |
| 截断 SVD | 为了只保留最重要的奇异值和奇异向量 |
| Eckart-Young 定理 | 为了证明截断 SVD 是最优低秩近似 |
| 伪逆 | 为了给非方阵和奇异矩阵定义广义逆 |
| 最小二乘 | 为了求解无精确解的过定方程组 |
| 条件数 | 为了度量矩阵对误差的敏感程度 |
| PCA 与 SVD | 为了用更稳定的方法计算主成分 |
| LSA | 为了从文档-词矩阵中提取潜在语义结构 |
| 推荐系统矩阵分解 | 为了从稀疏评分矩阵中发现用户和物品的潜在因子 |

可以这样理解：

```txt
现实问题：矩阵不是方阵，不能直接特征分解怎么办？
数学工具：SVD

现实问题：如何看出矩阵最重要的方向？
数学对象：奇异值和奇异向量

现实问题：如何用少量参数近似大矩阵？
数学方法：截断 SVD

现实问题：线性方程组无精确解怎么办？
数学工具：SVD 伪逆

现实问题：PCA 如何更稳定地计算？
工程实现：对中心化数据做 SVD
```

---

### 1.2 原始问题是什么

本课要解决的原始问题是：

```txt
任意矩阵如何被分解成可解释的结构？
SVD 中的 U、Sigma、V^T 分别表示什么？
为什么矩阵可以理解成“旋转 -> 缩放 -> 旋转”？
奇异值为什么可以衡量矩阵中模式的重要性？
如何用截断 SVD 做低秩近似？
为什么截断 SVD 是最优 rank-k 近似？
如何用 SVD 压缩图片？
如何用 SVD 去噪？
如何用 SVD 求伪逆和最小二乘解？
SVD 和 PCA 是什么关系？
推荐系统和 LSA 为什么可以用 SVD？
```

换成 AI 语言，就是：

```txt
为什么 PCA 底层常用 SVD 实现？
为什么低秩结构能压缩模型和数据？
为什么 LoRA 也可以理解成低秩思想？
为什么 embedding 矩阵或评分矩阵可以被分解成 latent factors？
为什么 SVD 可以用于推荐系统和语义分析？
为什么用 np.linalg.svd(A) 比 eig(A.T @ A) 更稳定？
```

---

### 1.3 AI 中的现代问题

| 数学知识点 | AI 中的现代问题 |
|---|---|
| SVD | 任意矩阵结构分析 |
| 奇异值 | 模式强度、低秩结构、条件数 |
| 左奇异向量 | 输出空间主方向 |
| 右奇异向量 | 输入空间主方向 |
| 截断 SVD | 矩阵压缩、低秩近似 |
| 低秩近似 | 模型压缩、LoRA、推荐系统 |
| 图像压缩 | 用少量奇异值保留图像主体 |
| 降噪 | 丢弃小奇异值对应的噪声方向 |
| 伪逆 | 最小二乘、线性系统求解 |
| 条件数 | 数值稳定性、病态矩阵 |
| PCA | 中心化数据的 SVD |
| 推荐系统 | user/movie latent factor |
| LSA / LSI | 文档-词矩阵的潜在语义 |
| Frobenius norm | 低秩近似误差 |
| Eckart-Young | 最优 rank-k 近似理论保证 |

---

### 1.4 本课要解决的核心问题

学完本课，你应该能够回答：

1. SVD 为什么适用于任意矩阵？
2. `A = U Sigma V^T` 中每个矩阵分别代表什么？
3. SVD 的几何意义为什么是“旋转 -> 缩放 -> 旋转”？
4. 左奇异向量、右奇异向量和奇异值分别是什么？
5. `A v_i = sigma_i u_i` 如何解释矩阵作用？
6. SVD 的外积形式为什么能解释低秩近似？
7. 截断 SVD 为什么是最优 rank-k 近似？
8. 如何用 SVD 做图像压缩？
9. 如何用 SVD 做噪声过滤？
10. 如何通过 SVD 计算 Moore-Penrose 伪逆？
11. SVD 为什么比对 `A^T A` 做特征分解更数值稳定？
12. SVD 和 PCA 的精确关系是什么？
13. 推荐系统和 LSA 如何利用 SVD 发现潜在因子？
14. 如何从零用 power iteration 实现一个简化 SVD？

---

## 2. 概念定位：这节课学什么

---

### 2.1 所属模块

```txt
所属模块：线性代数 / 矩阵分解 / 降维
前置知识：矩阵乘法、矩阵变换、特征值、特征向量、PCA
后续连接：PCA、推荐系统、低秩模型、LoRA、LSA、图像压缩、矩阵补全、数值线性代数
```

前面的课程已经解决了：

```txt
矩阵如何变换空间？
特征值和特征向量如何描述方阵中的特殊方向？
PCA 如何找到最大方差方向？
```

本课进一步解决：

```txt
如何对任意矩阵做最通用、最稳定、最可解释的分解？
```

---

### 2.2 本课学习目标

完成本课后，你应该能够：

- 解释 SVD 的几何意义
- 理解 `U`、`Sigma`、`V^T` 的维度和作用
- 解释左奇异向量、右奇异向量和奇异值
- 写出 SVD 的外积形式
- 说明 SVD 与 `A^T A`、`A A^T` 特征分解的关系
- 实现一个基于 power iteration 的简化 SVD
- 使用截断 SVD 做低秩近似
- 使用 SVD 做图像压缩并计算压缩率和重构误差
- 使用 SVD 做噪声过滤
- 使用 SVD 计算伪逆并求解最小二乘问题
- 解释 SVD 与 PCA、推荐系统、LSA 的关系

---

### 2.3 本课在整体路线中的位置

```txt
矩阵变换
    ↓
特征值与特征向量
    ↓
PCA
    ↓
SVD
    ↓
低秩近似 / 压缩 / 去噪 / 伪逆
    ↓
推荐系统 / LSA / LoRA / 矩阵补全
```

本课的核心能力是：

```txt
看到一个矩阵时，不只知道它能相乘，还能判断它有哪些主要模式、能否低秩近似、能否稳定求解，以及如何压缩和去噪。
```

---

## 3. 理解概念：直觉、公式、几何解释、AI 连接

---

### 3.1 SVD 的几何意义：旋转、缩放、再旋转

**一句话直觉：**  
任何矩阵都可以理解成三步：先旋转输入空间，再沿坐标轴缩放，最后旋转到输出空间。

SVD 公式：

```text
A = U Sigma V^T
```

对于一个 `m x n` 矩阵 `A`：

| 矩阵 | 形状 | 几何意义 |
|---|---|---|
| `V^T` | `n x n` | 在输入空间中旋转坐标 |
| `Sigma` | `m x n` | 沿各轴缩放、压缩或压扁 |
| `U` | `m x m` | 在输出空间中旋转坐标 |

整体过程：

```txt
输入向量
    ↓
V^T：旋转到右奇异向量坐标系
    ↓
Sigma：按奇异值缩放
    ↓
U：旋转到输出空间
    ↓
输出向量
```

**AI 连接：**

| 场景 | SVD 直觉 |
|---|---|
| embedding 矩阵 | 找到主要语义方向 |
| 图像矩阵 | 找到主要视觉结构 |
| 用户-物品矩阵 | 找到潜在偏好因子 |
| 线性层权重 | 分析权重矩阵有效秩 |
| LoRA | 用低秩更新近似大矩阵变化 |

---

### 3.2 完整 SVD 分解

**一句话直觉：**  
SVD 把任意矩阵拆成两个正交矩阵和一个非负对角缩放矩阵。

对于：

```text
A: m x n
```

完整分解为：

```text
A = U Sigma V^T
```

其中：

| 符号 | 形状 | 含义 |
|---|---|---|
| `U` | `m x m` | 左奇异向量组成的正交矩阵 |
| `Sigma` | `m x n` | 对角线上为奇异值的矩阵 |
| `V` | `n x n` | 右奇异向量组成的正交矩阵 |
| `V^T` | `n x n` | `V` 的转置 |
| `sigma_i` | 非负标量 | 第 i 个奇异值 |

正交性质：

```text
U^T U = I
V^T V = I
```

奇异值排序：

```text
sigma_1 >= sigma_2 >= ... >= sigma_r > 0
```

其中：

```text
r = rank(A)
```

---

### 3.3 左奇异向量、右奇异向量和奇异值

**一句话直觉：**  
右奇异向量是输入方向，奇异值是缩放倍数，左奇异向量是输出方向。

核心关系：

```text
A v_i = sigma_i u_i
```

解释：

| 符号 | 含义 |
|---|---|
| `v_i` | 第 i 个右奇异向量，输入空间方向 |
| `sigma_i` | 第 i 个奇异值，该方向被拉伸的倍数 |
| `u_i` | 第 i 个左奇异向量，输出空间方向 |

也就是说：

```txt
矩阵 A 把输入空间中的方向 v_i，
缩放 sigma_i 倍，
映射到输出空间中的方向 u_i。
```

如果：

```txt
sigma_i = 0
```

说明该方向被矩阵完全压扁。

**AI 连接：**

| 概念 | AI 解释 |
|---|---|
| 右奇异向量 | 输入特征空间中的重要方向 |
| 左奇异向量 | 输出表示空间中的重要方向 |
| 奇异值 | 该模式的重要程度 |
| 零奇异值 | 被压扁或无效的方向 |
| 奇异值衰减快 | 矩阵有明显低秩结构 |

---

### 3.4 SVD 的外积形式：矩阵是 rank-1 模式的叠加

**一句话直觉：**  
SVD 可以把一个矩阵写成多个 rank-1 矩阵的加权和，每一项都是一种模式。

外积形式：

```text
A = sigma_1 u_1 v_1^T
  + sigma_2 u_2 v_2^T
  + ...
  + sigma_r u_r v_r^T
```

每一项：

```text
sigma_i u_i v_i^T
```

都是一个 rank-1 矩阵。

含义：

```txt
第 1 项捕捉最重要模式。
第 2 项捕捉第二重要模式。
第 k 项捕捉第 k 重要模式。
```

保留前 k 项：

```text
A_k = sum_{i=1}^k sigma_i u_i v_i^T
```

就得到 rank-k 近似。

**AI 连接：**

| 场景 | 外积形式解释 |
|---|---|
| 图像压缩 | 每个 rank-1 项增加一层视觉结构 |
| 推荐系统 | 每个因子解释一种用户偏好 |
| LSA | 每个因子解释一个潜在主题 |
| LoRA | 用低秩矩阵近似参数更新 |
| 降噪 | 保留主要模式，丢弃噪声项 |

---

### 3.5 SVD 与特征分解的关系

**一句话直觉：**  
SVD 可以通过 `A^T A` 和 `A A^T` 的特征分解理解，但直接做 SVD 更稳定。

如果：

```text
A = U Sigma V^T
```

则：

```text
A^T A = V Sigma^T Sigma V^T
```

因此：

```txt
V 是 A^T A 的特征向量。
sigma_i^2 是 A^T A 的特征值。
```

同理：

```text
A A^T = U Sigma Sigma^T U^T
```

因此：

```txt
U 是 A A^T 的特征向量。
sigma_i^2 也是 A A^T 的非零特征值。
```

重要结论：

```txt
奇异值是 A^T A 特征值的平方根。
奇异值总是实数且非负。
```

**AI 连接：**

PCA 中：

```text
Cov(X) = X^T X / (n - 1)
```

所以 PCA 的主成分其实就是中心化数据矩阵 SVD 中的右奇异向量。

---

### 3.6 截断 SVD：最优低秩近似

**一句话直觉：**  
截断 SVD 保留最大的 k 个奇异值和对应向量，得到最好的 rank-k 近似。

完整 SVD：

```text
A = U Sigma V^T
```

截断为 rank-k：

```text
A_k = U_k Sigma_k V_k^T
```

其中：

| 矩阵 | 形状 | 含义 |
|---|---|---|
| `U_k` | `m x k` | 前 k 个左奇异向量 |
| `Sigma_k` | `k x k` | 前 k 个奇异值 |
| `V_k` | `n x k` | 前 k 个右奇异向量 |

Eckart-Young-Mirsky 定理说明：

```txt
在所有 rank-k 矩阵中，截断 SVD 得到的 A_k 距离原矩阵 A 最近。
```

误差：

```text
spectral norm error = sigma_{k+1}
Frobenius error = sqrt(sigma_{k+1}^2 + ... + sigma_r^2)
```

**AI 连接：**

| 场景 | 截断 SVD 作用 |
|---|---|
| 图像压缩 | 保留主要结构 |
| 去噪 | 丢掉小奇异值对应噪声 |
| 推荐系统 | 保留主要 latent factors |
| 文本分析 | 保留主要语义主题 |
| 模型压缩 | 用低秩矩阵近似权重 |

---

### 3.7 图像压缩：只保留主要奇异值

**一句话直觉：**  
自然图像的主要结构通常集中在前几个奇异值中，因此可以用截断 SVD 压缩。

一张灰度图像可以看成矩阵：

```text
image: 800 x 600
```

原始存储：

```text
800 * 600 = 480,000 个数
```

rank-k SVD 存储：

```text
U_k: 800 x k
Sigma_k: k
V_k: 600 x k
Total = k * (800 + 600 + 1)
```

例如：

| k | 存储量 | 占原始比例 |
|---:|---:|---:|
| 10 | 14,010 | 2.9% |
| 50 | 70,050 | 14.6% |
| 100 | 140,100 | 29.2% |

解释：

```txt
前几个奇异值捕捉大轮廓、亮度梯度和主要形状。
后面的奇异值捕捉细节和噪声。
```

---

### 3.8 SVD 降噪：保留信号，丢弃噪声

**一句话直觉：**  
信号通常集中在大的奇异值里，噪声更分散在小奇异值中。

如果奇异值有明显断崖：

```txt
大奇异值：信号
小奇异值：噪声
```

则可以保留前 k 个奇异值重构：

```text
A_denoised = U_k Sigma_k V_k^T
```

**AI 连接：**

| 场景 | SVD 降噪作用 |
|---|---|
| 图像去噪 | 去除高频随机扰动 |
| 传感器数据 | 提取主要信号 |
| 科学测量 | 去掉测量噪声 |
| embedding 清理 | 去掉低能量方向 |
| anomaly detection | 异常样本重构误差较高 |

---

### 3.9 Moore-Penrose 伪逆：非方阵也能“求逆”

**一句话直觉：**  
伪逆是矩阵逆的推广，适用于非方阵和奇异矩阵。

如果：

```text
A = U Sigma V^T
```

那么伪逆：

```text
A^+ = V Sigma^+ U^T
```

其中 `Sigma^+` 的构造方法：

```txt
1. 转置 Sigma
2. 把非零奇异值 sigma_i 替换为 1/sigma_i
3. 零奇异值保持为 0
```

伪逆可以求解最小二乘问题：

```text
x_ls = A^+ b
```

如果方程：

```text
A x = b
```

没有精确解，`x_ls` 是让：

```text
||A x - b||
```

最小的解。

**AI 连接：**

| 场景 | 伪逆作用 |
|---|---|
| 线性回归 | 最小二乘解 |
| 过定系统 | 方程多于未知数 |
| 欠定系统 | 未知数多于方程 |
| 数值稳定求解 | 避免直接求逆 |
| closed-form baseline | 训练前建立线性模型基线 |

---

### 3.10 条件数与数值稳定性

**一句话直觉：**  
条件数衡量矩阵对输入误差有多敏感。

SVD 给出条件数：

```text
condition number = sigma_max / sigma_min
```

如果条件数很大：

```txt
矩阵是病态的。
输入中的微小误差可能在输出中被极大放大。
```

为什么不要用 `A^T A` 计算 SVD？

```txt
A^T A 的特征值是 A 奇异值的平方。
这会把条件数平方，放大数值误差。
```

例子：

```txt
A 的奇异值: [1000, 1, 0.001]
cond(A) = 10^6

A^T A 的特征值: [10^6, 1, 10^-6]
cond(A^T A) = 10^12
```

所以实践中应优先使用：

```python
np.linalg.svd(A)
```

而不是：

```python
np.linalg.eig(A.T @ A)
```

---

### 3.11 SVD 与 PCA 的关系

**一句话直觉：**  
PCA 就是对中心化数据矩阵做 SVD。

给定中心化数据矩阵：

```text
X: n_samples x n_features
```

协方差矩阵：

```text
C = (1 / (n - 1)) X^T X
```

如果：

```text
X = U Sigma V^T
```

则：

```text
X^T X = V Sigma^2 V^T
```

所以：

```txt
PCA 的主成分 = V 中的右奇异向量
PCA 的解释方差 = sigma_i^2 / (n - 1)
```

这说明：

```txt
PCA 不是类似 SVD，而是中心化数据上的 SVD。
```

scikit-learn 的 PCA 通常也是基于 SVD 实现，因为它更快、更稳定。

---

### 3.12 推荐系统：用户和物品的潜在因子

**一句话直觉：**  
用户-物品评分矩阵通常有低秩结构，SVD 可以发现隐藏的偏好因子。

用户-电影评分矩阵：

```txt
          Movie1  Movie2  Movie3  Movie4
User1       5       ?       3       ?
User2       ?       4       ?       2
User3       3       ?       5       ?
```

SVD 会分解成：

| 因子 | 含义 |
|---|---|
| `U` | 用户在潜在偏好空间中的表示 |
| `Sigma` | 每个潜在因子的重要程度 |
| `V^T` | 电影在潜在因子空间中的表示 |

潜在因子可能是：

```txt
动作 vs 剧情
老电影 vs 新电影
轻松娱乐 vs 深度思考
商业大片 vs 艺术电影
```

用户对电影的预测评分可以看作：

```txt
用户 latent vector 和电影 latent vector 的加权相似度。
```

---

### 3.13 LSA：文档-词矩阵中的潜在语义

**一句话直觉：**  
LSA 把文档-词矩阵分解成低维语义空间，让相似词和相似文档靠近。

文档-词矩阵：

```txt
        Doc1  Doc2  Doc3
cat       3     0     1
dog       2     0     0
fish      0     4     1
ocean     0     3     0
```

SVD 后：

```txt
每个词变成潜在语义空间中的向量。
每篇文档也变成潜在语义空间中的向量。
相似主题会聚在一起。
```

例如：

```txt
cat 和 dog 可能靠近。
fish 和 ocean 可能靠近。
```

**AI 连接：**

LSA 是早期语义表示方法之一。  
现代 word embedding 和 matrix factorization 思想与它有历史联系。

---

### 3.14 本课核心概念总表

| 概念 | 一句话直觉 | AI 中的对应位置 |
|---|---|---|
| SVD | 任意矩阵分解 | 矩阵分析、压缩、降维 |
| U | 输出空间正交基 | left singular vectors |
| Sigma | 缩放强度 | singular values |
| V | 输入空间正交基 | right singular vectors |
| 奇异值 | 模式重要程度 | 低秩结构、条件数 |
| 外积形式 | rank-1 模式叠加 | low-rank approximation |
| 截断 SVD | 保留最重要 k 个模式 | 压缩、去噪 |
| Eckart-Young | 最优 rank-k 近似 | 理论保证 |
| 伪逆 | 广义逆 | least squares |
| 条件数 | 数值敏感性 | 病态矩阵 |
| PCA | 中心化数据的 SVD | 降维 |
| 推荐系统 | latent factors | user-item matrix |
| LSA | 潜在语义空间 | NLP |
| Frobenius norm | 矩阵整体误差 | reconstruction error |

---

## 4. 手写实现：从零实现 SVD、压缩、降噪和伪逆

这一节使用 NumPy 做基础矩阵运算，但不直接调用 SVD 来实现第一版。  
目标是理解 SVD 的计算逻辑和应用方式。

---

### 4.1 用 Power Iteration 找最大特征向量

```python
import numpy as np


def power_iteration(M, num_iters=100):
    n = M.shape[1]
    v = np.random.randn(n)
    v = v / np.linalg.norm(v)

    for _ in range(num_iters):
        Mv = M @ v
        v = Mv / np.linalg.norm(Mv)

    eigenvalue = v @ M @ v

    return eigenvalue, v
```

Power iteration 的直觉：

```txt
反复用矩阵乘一个向量并归一化。
最大特征值对应的方向会逐渐占主导。
```

---

### 4.2 从零实现简化版 SVD

```python
def svd_from_scratch(A, k=None):
    m, n = A.shape

    if k is None:
        k = min(m, n)

    sigmas = []
    us = []
    vs = []

    A_residual = A.copy().astype(float)

    for _ in range(k):
        AtA = A_residual.T @ A_residual

        eigenvalue, v = power_iteration(AtA, num_iters=200)

        if eigenvalue < 1e-10:
            break

        sigma = np.sqrt(eigenvalue)
        u = A_residual @ v / sigma

        sigmas.append(sigma)
        us.append(u)
        vs.append(v)

        A_residual = A_residual - sigma * np.outer(u, v)

    U = np.column_stack(us) if us else np.empty((m, 0))
    S = np.array(sigmas)
    V = np.column_stack(vs) if vs else np.empty((n, 0))

    return U, S, V
```

说明：

```txt
1. 对 A^T A 做 power iteration，找到最大右奇异向量 v
2. 奇异值 sigma = sqrt(eigenvalue)
3. 左奇异向量 u = A v / sigma
4. 用 outer product 从残差矩阵中减去已找到的 rank-1 模式
5. 重复，得到下一个奇异值和奇异向量
```

---

### 4.3 和 NumPy SVD 对照

```python
np.random.seed(42)

A = np.random.randn(5, 4)

U_ours, S_ours, V_ours = svd_from_scratch(A)
U_np, S_np, Vt_np = np.linalg.svd(A, full_matrices=False)

print("Our singular values:", np.round(S_ours, 4))
print("NumPy singular values:", np.round(S_np, 4))

A_reconstructed = U_ours @ np.diag(S_ours) @ V_ours.T

print(f"Reconstruction error: {np.linalg.norm(A - A_reconstructed):.8f}")
```

注意：

```txt
手写版本用于教学，数值稳定性不如 np.linalg.svd。
真实工程应使用 np.linalg.svd 或 scipy/sklearn。
```

---

### 4.4 用 SVD 做图像压缩

```python
def compress_image_svd(image_matrix, k):
    U, S, Vt = np.linalg.svd(image_matrix, full_matrices=False)

    compressed = U[:, :k] @ np.diag(S[:k]) @ Vt[:k, :]

    return compressed


np.random.seed(42)

rows, cols = 200, 300
image = np.random.randn(rows, cols)

for k in [1, 5, 10, 20, 50]:
    compressed = compress_image_svd(image, k)

    error = np.linalg.norm(image - compressed) / np.linalg.norm(image)

    original_size = rows * cols
    compressed_size = k * (rows + cols + 1)
    ratio = compressed_size / original_size

    print(f"k={k:>3d}  error={error:.4f}  storage={ratio:.1%}")
```

---

### 4.5 用 SVD 做噪声过滤

```python
np.random.seed(42)

clean = np.outer(
    np.sin(np.linspace(0, 4 * np.pi, 100)),
    np.cos(np.linspace(0, 2 * np.pi, 80))
)

noise = 0.3 * np.random.randn(100, 80)

noisy = clean + noise

U, S, Vt = np.linalg.svd(noisy, full_matrices=False)

denoised = U[:, :5] @ np.diag(S[:5]) @ Vt[:5, :]

print(f"Noisy error:    {np.linalg.norm(noisy - clean):.4f}")
print(f"Denoised error: {np.linalg.norm(denoised - clean):.4f}")
print(
    f"Improvement:    {(1 - np.linalg.norm(denoised - clean) / np.linalg.norm(noisy - clean)):.1%}"
)
```

---

### 4.6 用 SVD 计算伪逆

```python
A = np.array([
    [1, 1],
    [2, 1],
    [3, 1]
], dtype=float)

b = np.array([3, 5, 6], dtype=float)

U, S, Vt = np.linalg.svd(A, full_matrices=False)

S_inv = np.diag(1.0 / S)

A_pinv = Vt.T @ S_inv @ U.T

x_svd = A_pinv @ b
x_lstsq = np.linalg.lstsq(A, b, rcond=None)[0]
x_pinv = np.linalg.pinv(A) @ b

print(f"SVD pseudoinverse solution: {x_svd}")
print(f"np.linalg.lstsq solution:   {x_lstsq}")
print(f"np.linalg.pinv solution:    {x_pinv}")
```

---

### 4.7 用 SVD 连接 PCA

```python
def pca_via_svd(X, n_components):
    X_centered = X - X.mean(axis=0)

    U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)

    components = Vt[:n_components]
    X_reduced = X_centered @ components.T

    explained_variance = (S ** 2) / (X.shape[0] - 1)
    explained_variance_ratio = explained_variance / explained_variance.sum()

    return X_reduced, components, explained_variance_ratio[:n_components]
```

---

## 5. 生产使用：用 NumPy / scikit-learn 完成 SVD 应用

---

### 5.1 NumPy SVD

```python
import numpy as np

A = np.random.randn(100, 50)

U, S, Vt = np.linalg.svd(A, full_matrices=False)

A_reconstructed = U @ np.diag(S) @ Vt

print(np.linalg.norm(A - A_reconstructed))
```

---

### 5.2 scikit-learn TruncatedSVD

```python
from sklearn.decomposition import TruncatedSVD

svd = TruncatedSVD(n_components=50, random_state=42)

X_reduced = svd.fit_transform(A)

print(f"Reduced shape: {X_reduced.shape}")
print(f"Explained variance ratio sum: {svd.explained_variance_ratio_.sum():.4f}")
```

`TruncatedSVD` 常用于稀疏矩阵，例如文档-词矩阵。  
与 PCA 不同，它不会默认中心化数据，因此非常适合 sparse text matrix。

---

### 5.3 SVD 用于 LSA

```python
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import TruncatedSVD

docs = [
    "cat dog pet",
    "dog pet animal",
    "fish ocean water",
    "ocean fish sea",
]

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(docs)

lsa = TruncatedSVD(n_components=2, random_state=42)
X_lsa = lsa.fit_transform(X)

print(X_lsa)
```

---

### 5.4 手写实现 vs 生产库

| 对比项 | 手写 SVD | NumPy / sklearn |
|---|---|---|
| 目的 | 理解 SVD 原理 | 工程实践 |
| 算法 | power iteration + deflation | 稳定数值算法 |
| 精度 | 教学可用 | 高精度 |
| 速度 | 慢 | 快 |
| 稳定性 | 较弱 | 强 |
| 稀疏矩阵 | 不适合 | TruncatedSVD 适合 |
| 推荐使用 | 学习 | 真实任务 |

---

## 6. 交付沉淀：产出物、连接关系、练习、术语表

---

### 6.1 本课产出

```txt
code/svd_from_scratch.py
code/use_numpy.py
code/use_sklearn.py
outputs/skill-svd.md
outputs/exercises-solutions.md
```

| 文件 | 作用 |
|---|---|
| `code/svd_from_scratch.py` | 从零实现 power iteration、SVD、图像压缩、降噪、伪逆 |
| `code/use_numpy.py` | NumPy SVD、伪逆、PCA via SVD |
| `code/use_sklearn.py` | TruncatedSVD、LSA、稀疏矩阵降维 |
| `outputs/skill-svd.md` | 用于判断何时使用 SVD 的 skill |
| `outputs/exercises-solutions.md` | 练习题与参考答案 |

---

### 6.2 知识连接关系

| 本课知识点 | 后续连接 |
|---|---|
| SVD | PCA、低秩近似、数值线性代数 |
| 奇异值 | 条件数、有效秩、压缩 |
| 截断 SVD | 图像压缩、推荐系统、LSA |
| 外积形式 | rank-1 模式分解 |
| 伪逆 | least squares、线性回归 |
| Frobenius norm | 低秩重构误差 |
| 条件数 | 数值稳定性 |
| PCA via SVD | 降维 |
| latent factor | 推荐系统、语义分析 |
| low-rank | LoRA、模型压缩 |

---

### 6.3 AI 应用连接

| 数学 / 工程知识点 | AI 应用 |
|---|---|
| SVD | 矩阵结构分析 |
| 截断 SVD | 压缩、去噪、低秩近似 |
| 奇异值衰减 | 判断是否存在低秩结构 |
| 图像压缩 | 保留主要视觉模式 |
| 伪逆 | 最小二乘、线性模型 |
| 推荐系统 | 用户-物品 latent factor |
| LSA | 文档语义空间 |
| PCA | SVD 在中心化数据上的应用 |
| 条件数 | 数值稳定性诊断 |
| Low-rank approximation | LoRA、模型压缩、矩阵补全 |

---

### 6.4 练习

1. **用特征分解实现 SVD。**  
   不使用 power iteration，而是对 `A^T A` 做特征分解，得到 `V` 和奇异值，再计算：

   ```text
   U = A V Sigma^{-1}
   ```

   比较它和 power iteration 版本、NumPy SVD 的数值误差。

2. **真实图像压缩。**  
   加载一张真实灰度图像，分别用 rank：

   ```txt
   1, 5, 10, 25, 50, 100
   ```

   压缩。计算压缩率和相对误差，找出视觉上可接受的最小 rank。

3. **小型推荐系统。**  
   构造一个 `10 x 8` 用户-电影评分矩阵，部分评分缺失。  
   用用户均值填补缺失值，做 rank-3 SVD 重构，预测缺失评分。

4. **合成 LSA。**  
   创建一个 `100 x 50` 文档-词矩阵，包含 3 个主题，每个主题有 5 个关键词。加入噪声后做 SVD，验证前三个奇异值明显更大。

5. **噪声水平与最优 rank。**  
   生成一个 rank-3 的干净矩阵，加入不同强度 Gaussian noise：

   ```txt
   sigma = 0.1, 0.5, 1.0, 2.0
   ```

   对每个噪声水平，扫描 `k=1...40`，找到相对干净矩阵重构误差最低的 k。

---

### 6.5 关键术语

| 术语 | 常见说法 | 真正含义 |
|---|---|---|
| SVD | 分解任意矩阵 | 把 A 分解成 `U Sigma V^T`，适用于任意形状矩阵 |
| 奇异值 | 重要程度 | 矩阵沿某个主方向的拉伸倍数，非负并降序排列 |
| 左奇异向量 | 输出方向 | `U` 的列向量，表示输出空间主方向 |
| 右奇异向量 | 输入方向 | `V` 的列向量，表示输入空间主方向 |
| 截断 SVD | 低秩近似 | 保留前 k 个奇异值和向量 |
| Rank | 有效维度 | 非零奇异值的个数 |
| 伪逆 | 广义逆 | `V Sigma+ U^T`，可解非方阵或奇异矩阵的最小二乘问题 |
| 条件数 | 数值敏感性 | 最大奇异值除以最小非零奇异值 |
| Latent factor | 隐藏因子 | 低秩空间中发现的潜在维度 |
| Frobenius norm | 矩阵整体大小 | 所有元素平方和再开方，也等于奇异值平方和开方 |
| Eckart-Young | 最优压缩定理 | 截断 SVD 给出最佳 rank-k 近似 |
| Power iteration | 最大特征向量迭代 | 反复矩阵乘法和归一化，找到最大特征方向 |
| LSA | 潜在语义分析 | 对文档-词矩阵做 SVD，得到语义空间 |

---

### 6.6 自检问题

学完本课后，你应该能回答：

- SVD 最初解决什么问题？
- 为什么 SVD 可以用于任意矩阵？
- `U`、`Sigma`、`V^T` 分别代表什么？
- SVD 的几何意义为什么是旋转、缩放、旋转？
- `A v_i = sigma_i u_i` 如何解释？
- 为什么奇异值越大，对应模式越重要？
- SVD 的外积形式如何解释低秩近似？
- 为什么截断 SVD 是最优 rank-k 近似？
- 如何用 SVD 做图像压缩？
- 如何用 SVD 做降噪？
- 如何用 SVD 计算伪逆？
- 为什么直接用 `np.linalg.svd(A)` 比 `eig(A.T @ A)` 更稳定？
- PCA 和 SVD 的关系是什么？
- 推荐系统如何用 SVD 发现 latent factors？
- LSA 如何用 SVD 找到潜在语义？

---

## 附录：本课最小代码文件建议

如果要拆成代码文件，可以这样组织：

```txt
code/
├── svd_from_scratch.py
├── use_numpy.py
└── use_sklearn.py
```

### `svd_from_scratch.py`

包含：

```txt
power_iteration
svd_from_scratch
compress_image_svd
svd_denoising_demo
pseudoinverse_demo
pca_via_svd
```

### `use_numpy.py`

包含：

```txt
np.linalg.svd
np.linalg.pinv
np.linalg.lstsq
condition_number_from_singular_values
```

### `use_sklearn.py`

包含：

```txt
TruncatedSVD
CountVectorizer
LSA demo
sparse matrix dimensionality reduction
```
