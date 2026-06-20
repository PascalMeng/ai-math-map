---
title: 凸优化
description: 面向 AI 算法工程师的凸优化课程：从凸集、凸函数、Hessian、Newton 方法、Lagrange multiplier、KKT 条件，到正则化、对偶性和深度学习非凸优化。
---

# 凸优化

> 凸问题只有一个山谷。神经网络有成千上万个山谷。理解这个差别非常重要。

**课程类型：** 实现 / 应用  
**所属模块：** 优化方法 / 数值优化 / 机器学习理论  
**前置知识：** Phase 1 Lesson 04：机器学习中的微积分；Lesson 08：优化方法  
**预计时间：** 约 90 分钟  
**使用语言：** Python / NumPy / SciPy / scikit-learn  

---

## 1. 提出问题：为什么 AI 算法工程师必须理解凸优化

前面的优化课程已经讲过：

```txt
gradient descent
momentum
Adam
learning rate schedule
non-convex loss landscape
```

这些优化器可以在任意 loss surface 上往下走。  
但问题是：

```txt
在非凸地形上，往下走不等于一定能到全局最低点。
```

非凸问题可能有：

```txt
多个局部最小值
鞍点
平坦区域
尖锐谷底
震荡路径
初始化敏感性
```

神经网络就是典型非凸问题。  
你用 SGD / Adam 训练它，并不是因为它有全局最优保证，而是因为实践中它通常能找到足够好的解。

但机器学习里有很多问题是凸的：

```txt
线性回归
逻辑回归
SVM
LASSO
Ridge regression
凸约束下的资源分配
某些最大熵模型
某些概率模型的参数估计
```

凸优化的强大之处在于：

```txt
如果问题是凸的，任何局部最小值都是全局最小值。
```

这意味着：

```txt
不用反复随机初始化。
不用担心坏的 local minimum。
可以证明收敛。
可以使用 Newton、L-BFGS 等更强工具。
可以理解 SVM duality、regularization constraint 和 KKT 条件。
```

本课的目标不是让你把神经网络变成凸问题，而是让你知道：

```txt
什么时候问题是“好解的”。
什么时候问题没有全局保证。
什么时候可以用凸优化工具。
什么时候深度学习虽然非凸，但仍然能工作。
```

---

### 1.1 数学概念历史出现缘由

凸优化来自数学优化、几何、经济学、工程控制和数值分析。  
它研究的是一类具有良好结构的优化问题：目标函数和可行域都是凸的。

| 数学 / 工程概念 | 出现缘由 |
|---|---|
| 凸集 | 为了描述没有洞、没有凹陷的可行域 |
| 凸函数 | 为了描述只有一个整体山谷的目标函数 |
| 凸性定义 | 为了用线段不低于函数图像来判断凸性 |
| 二阶导数判定 | 为了在一维中通过曲率判断凸性 |
| Hessian 判定 | 为了在多维中通过曲率矩阵判断凸性 |
| 正半定矩阵 | 为了描述所有方向曲率非负 |
| 全局最优保证 | 因为凸函数的局部最优就是全局最优 |
| Newton 方法 | 为了利用二阶曲率信息快速收敛 |
| 约束优化 | 为了解决现实中带预算、范数、边界、等式/不等式约束的问题 |
| Lagrange multiplier | 为了把等式约束优化转化为无约束问题 |
| KKT 条件 | 为了处理不等式约束下的最优性条件 |
| 对偶性 | 为了把原始优化问题转换成有时更容易求解的 dual problem |
| 正则化约束 | 为了把模型复杂度约束理解成优化可行域 |
| SVM dual | 为了通过对偶形式引入 kernel trick |
| 非凸深度学习 | 为了解释神经网络为何没有凸保证但仍能优化成功 |

可以这样理解：

```txt
现实问题：优化问题是否只有一个真正的最低点？
数学概念：凸函数

现实问题：参数必须满足约束，怎么办？
数学工具：约束优化

现实问题：等式约束如何处理？
数学工具：Lagrange multiplier

现实问题：不等式约束如何处理？
数学工具：KKT conditions

现实问题：正则化到底是在做什么？
几何解释：限制参数落在 L1/L2 范数球内

现实问题：SVM 为什么能使用 kernel trick？
优化解释：对偶问题只依赖样本点之间的内积

现实问题：神经网络不是凸的，为什么还能训练？
现代解释：高维、过参数化、SGD 噪声和鞍点结构
```

---

### 1.2 原始问题是什么

本课要解决的原始问题是：

```txt
什么是凸集？
什么是凸函数？
如何判断一个函数是否凸？
为什么凸函数的局部最小值就是全局最小值？
Hessian 矩阵如何描述多维曲率？
Newton 方法为什么比 gradient descent 快？
为什么 Newton 方法不适合大规模神经网络？
约束优化如何求解？
Lagrange multiplier 的几何意义是什么？
KKT 条件中的 complementary slackness 是什么意思？
正则化为什么可以理解成约束优化？
L1 和 L2 约束为什么产生不同模型行为？
什么是 primal 和 dual？
为什么 SVM 的 dual formulation 能支持 kernel trick？
神经网络非凸，为什么实践中依然能优化成功？
```

换成 AI 语言，就是：

```txt
为什么 logistic regression 是凸优化，但 neural network 不是？
为什么 linear regression 有全局闭式解？
为什么 LASSO / Ridge 有凸保证？
为什么 Adam 训练神经网络没有全局最优保证？
为什么 L-BFGS 适合传统 ML，但不适合大模型？
为什么 mini-batch SGD 的噪声有时是好事？
为什么 flat minima 通常泛化更好？
为什么 L1 regularization 会产生稀疏性？
```

---

### 1.3 AI 中的现代问题

| 凸优化知识点 | AI 中的现代问题 |
|---|---|
| 凸集 | 参数约束、可行域 |
| 凸函数 | 全局最优保证 |
| Hessian | 曲率、二阶优化 |
| 正半定 | 多维凸性判定 |
| Newton 方法 | 二阶优化、快速收敛 |
| L-BFGS | 传统 ML 中的高效优化 |
| Lagrange multiplier | 等式约束优化 |
| KKT 条件 | SVM、约束学习、互补松弛 |
| 对偶性 | SVM dual、kernel trick |
| L1/L2 constraint | LASSO、Ridge、正则化 |
| Strong duality | 凸问题 primal/dual 一致 |
| Condition number | 梯度下降速度、长窄山谷 |
| Saddle point | 深度学习非凸障碍 |
| Overparameterization | 深度网络优化成功原因之一 |
| SGD noise | 逃离鞍点、偏向 flat minima |
| Natural gradient / K-FAC | 近似二阶方法 |
| Hessian-free | 大规模二阶优化近似 |

---

### 1.4 本课要解决的核心问题

学完本课，你应该能够回答：

1. 凸集的定义是什么？
2. 凸函数的定义是什么？
3. 二阶导数如何判断一维凸性？
4. Hessian 如何判断多维凸性？
5. 为什么正半定 Hessian 意味着凸？
6. 为什么凸函数所有 local minimum 都是 global minimum？
7. 机器学习中哪些问题是凸的，哪些不是？
8. Newton 方法为什么可以快速收敛？
9. Newton 方法为什么不适合百万参数神经网络？
10. Lagrange multiplier 如何求解等式约束优化？
11. KKT 条件如何处理不等式约束？
12. complementary slackness 表示什么？
13. 正则化如何等价于约束优化？
14. L1 为什么产生稀疏，L2 为什么只收缩权重？
15. duality 在 SVM 中有什么作用？
16. 神经网络非凸，为什么 SGD 仍然能找到好解？
17. 二阶方法在现代 ML 中如何近似使用？

---

## 2. 概念定位：这节课学什么

---

### 2.1 所属模块

```txt
所属模块：优化方法 / 凸优化 / 机器学习理论
前置知识：导数、梯度、Hessian、梯度下降、Newton 方法直觉
后续连接：逻辑回归、SVM、LASSO、Ridge、dual optimization、KKT、深度学习非凸优化
```

前面的优化课程已经解决了：

```txt
如何沿着梯度下降？
如何用 Momentum 和 Adam 加速训练？
```

本课进一步解决：

```txt
什么时候优化有全局保证？
什么时候可以用二阶方法和约束优化理论？
为什么非凸神经网络没有这些保证但仍然有效？
```

---

### 2.2 本课学习目标

完成本课后，你应该能够：

- 判断集合是否凸
- 判断函数是否凸
- 使用定义、二阶导数和 Hessian 判定凸性
- 解释正半定 Hessian 的几何意义
- 实现一个经验凸性检查器
- 从零实现 2D Newton 方法
- 比较 Newton 方法和 gradient descent 的收敛速度
- 使用 Lagrange multiplier 求解等式约束优化
- 解释 KKT 条件和互补松弛
- 将 L1/L2 regularization 解释成约束优化
- 解释 duality 与 SVM kernel trick 的关系
- 说明为什么神经网络 loss 是非凸的
- 解释 SGD 为什么仍然能在非凸深度网络中找到好解

---

### 2.3 本课在整体路线中的位置

```txt
微积分与梯度
    ↓
梯度下降 / Adam
    ↓
凸优化
    ↓
Newton / 约束优化 / KKT / Duality
    ↓
SVM / Regularization / 二阶优化
    ↓
深度学习非凸优化
```

本课的核心能力是：

```txt
面对一个优化问题时，先判断它是否凸，再选择合适的优化方法和理论工具。
```

---

## 3. 理解概念：直觉、公式、几何解释、AI 连接

---

### 3.1 凸集：两点之间的线段不离开集合

**一句话直觉：**  
凸集没有洞、没有凹陷；集合中任意两点连线仍然完全在集合内。

定义：

```text
对任意 x, y in S 和任意 t in [0, 1]，
如果 tx + (1-t)y 仍在 S 中，
则 S 是凸集。
```

凸集例子：

- 直线
- 平面
- 整个 `R^n`
- 圆 / 球 / 超球
- 矩形
- 三角形
- 半空间 `{x : a^T x <= b}`
- 任意多个凸集的交集

非凸集例子：

- 甜甜圈 / annulus
- 两个不相交圆的并集
- 有洞的集合
- 有凹陷的星形或月牙形

**AI 连接：**

| 概念 | 机器学习含义 |
|---|---|
| 可行域 | 参数允许出现的区域 |
| L2 ball | Ridge 约束 |
| L1 ball | LASSO 约束 |
| halfspace | SVM margin constraint |
| convex intersection | 多个约束同时满足 |

---

### 3.2 凸函数：函数图像上任意两点连线在函数上方

**一句话直觉：**  
凸函数像一个碗，任意两点之间的线段不会低于函数曲线。

定义：

```text
f(tx + (1-t)y) <= t f(x) + (1-t) f(y)
```

其中：

```text
t in [0,1]
```

几何解释：

```txt
函数图像上任意两点连成一条线段。
这条线段应该在函数图像上方或重合。
```

常见凸函数：

| 函数 | 是否凸 |
|---|---|
| `f(x)=x^2` | 凸 |
| `f(x)=|x|` | 凸 |
| `f(x)=e^x` | 凸 |
| `f(x)=max(0,x)` | 凸 |
| `f(x)=-log(x), x>0` | 凸 |
| 线性函数 `a^T x + b` | 既凸又凹 |

非凸函数例子：

```txt
sin(x)
x^3 在全实数域上
神经网络 loss
k-means objective
matrix factorization objective
```

---

### 3.3 凸性为什么重要

**一句话直觉：**  
凸函数没有坏的局部最小值，任何局部最小值都是全局最小值。

凸优化核心定理：

```txt
对于凸函数，每个 local minimum 都是 global minimum。
```

这带来巨大好处：

```txt
不怕陷入坏 local minimum。
不需要随机重启。
可以证明收敛。
可以判断解的最优性。
可以使用更强的优化理论。
```

非凸优化没有这些保证：

```txt
可能有多个 local minima。
可能卡在 saddle point。
结果依赖初始化。
学习率和优化路径很重要。
```

**AI 连接：**

| 问题 | 凸性 |
|---|---|
| Linear regression | 凸 |
| Logistic regression | 凸 |
| SVM | 凸 |
| LASSO | 凸 |
| Ridge | 凸 |
| Neural network | 非凸 |
| k-means | 非凸 |
| Matrix factorization | 非凸 |

---

### 3.4 如何判断凸性：定义、二阶导数、Hessian

**一句话直觉：**  
一维看二阶导数，多维看 Hessian 是否正半定。

#### 定义判定

直接检查：

```text
f(tx + (1-t)y) <= t f(x) + (1-t) f(y)
```

适合：

```txt
不可导函数
分段函数
理论证明
```

#### 一维二阶导数判定

如果：

```text
f''(x) >= 0
```

对所有 `x` 成立，则 `f` 是凸函数。

例子：

```text
f(x)=x^2
f''(x)=2 >= 0
```

所以凸。

```text
f(x)=x^3
f''(x)=6x
```

在 `x<0` 时为负，所以不是全域凸。

#### 多维 Hessian 判定

如果 Hessian：

```text
H(x) = ∇²f(x)
```

在所有点都 positive semidefinite，则 `f` 是凸函数。

---

### 3.5 Hessian 矩阵：多维曲率

**一句话直觉：**  
Hessian 是所有二阶偏导数组成的矩阵，用来描述函数在各个方向上的曲率。

定义：

```text
H[i][j] = ∂²f / ∂x_i ∂x_j
```

例子：

```text
f(x,y) = x^2 + 3xy + y^2
```

梯度：

```text
df/dx = 2x + 3y
df/dy = 3x + 2y
```

Hessian：

```text
H = [[2, 3],
     [3, 2]]
```

Hessian 的特征值解释：

| 特征值情况 | 含义 |
|---|---|
| 全部 > 0 | 所有方向向上弯，局部最小 |
| 全部 < 0 | 所有方向向下弯，局部最大 |
| 有正有负 | 鞍点 |
| 有 0 | 某些方向平坦 |
| 全部 >= 0 | 正半定，凸性条件 |

**AI 连接：**

Hessian 用于：

```txt
Newton 方法
条件数分析
loss landscape 曲率
sharp / flat minima 判断
二阶优化近似
```

---

### 3.6 Condition number 与长窄山谷

**一句话直觉：**  
Hessian 最大和最小特征值差距越大，loss valley 越狭长，gradient descent 越慢。

条件数：

```text
kappa = lambda_max / lambda_min
```

如果 condition number 很大：

```txt
某些方向曲率很大。
某些方向曲率很小。
梯度下降会 zigzag 或非常慢。
```

**AI 连接：**

| 问题 | 影响 |
|---|---|
| ill-conditioned quadratic | GD 收敛慢 |
| feature scaling 差 | 优化难 |
| batch normalization / layer normalization | 改善尺度 |
| second-order methods | 自动适配曲率 |
| Adam | 对角近似曲率信息 |

---

### 3.7 Newton 方法：使用二阶信息一步跳向二次近似最小点

**一句话直觉：**  
Gradient descent 只看坡度；Newton 方法同时看坡度和曲率。

Gradient descent：

```text
x_new = x - lr * gradient
```

Newton 方法：

```text
x_new = x - H^{-1} gradient
```

其中：

| 符号 | 含义 |
|---|---|
| `gradient` | 一阶导数，下降方向 |
| `H` | Hessian，局部曲率 |
| `H^{-1}` | 用曲率自适应缩放方向 |

优势：

```txt
接近最优点时二次收敛。
二次函数上一步到位。
不需要手动学习率。
对尺度更不敏感。
```

劣势：

```txt
Hessian 需要 O(n^2) 存储。
求逆需要 O(n^3)。
百万参数模型完全不可行。
```

**AI 连接：**

Newton 方法适合小中型凸问题。  
深度学习中通常用近似二阶方法或一阶方法。

---

### 3.8 约束优化：最低点必须在可行域内

**一句话直觉：**  
约束优化不是在整个空间找最低点，而是在满足约束的可行区域里找最低点。

无约束：

```text
minimize f(x)
```

有约束：

```text
minimize f(x)
subject to g(x) = 0
subject to h(x) <= 0
```

几何直觉：

```txt
无约束最低点可能不满足约束。
约束最优点通常在可行域内部或边界上。
```

**AI 连接：**

| 约束 | AI 例子 |
|---|---|
| 范数约束 | regularization |
| margin constraint | SVM |
| probability simplex | 分类概率、topic models |
| budget constraint | 模型大小、推理成本 |
| fairness constraint | 公平性优化 |
| safety constraint | 控制和 RL |

---

### 3.9 Lagrange multiplier：等式约束优化

**一句话直觉：**  
在约束最优点，目标函数梯度和约束梯度平行，否则还能沿约束方向继续下降。

问题：

```text
minimize f(x)
subject to g(x) = 0
```

构造 Lagrangian：

```text
L(x, lambda) = f(x) + lambda * g(x)
```

最优条件：

```text
∇_x L = ∇f(x) + lambda ∇g(x) = 0
∂L/∂lambda = g(x) = 0
```

例子：

```text
minimize f(x,y)=x^2+y^2
subject to x+y=1
```

Lagrangian：

```text
L = x^2 + y^2 + lambda(x+y-1)
```

求导：

```text
2x + lambda = 0
2y + lambda = 0
x + y - 1 = 0
```

得到：

```text
x = y = 0.5
```

**AI 连接：**

Lagrange multiplier 是理解：

```txt
约束优化
regularization
SVM dual
KKT
probability simplex optimization
```

的基础。

---

### 3.10 KKT 条件：处理不等式约束

**一句话直觉：**  
KKT 条件告诉我们，不等式约束要么正在起作用，要么对应 multiplier 为 0。

问题：

```text
minimize f(x)
subject to g_i(x) <= 0
```

KKT 条件：

```text
1. Stationarity:
   ∇f(x) + sum_i lambda_i ∇g_i(x) = 0

2. Primal feasibility:
   g_i(x) <= 0

3. Dual feasibility:
   lambda_i >= 0

4. Complementary slackness:
   lambda_i * g_i(x) = 0
```

Complementary slackness 是核心：

```txt
如果约束没有卡住最优点，则 g_i(x)<0，lambda_i=0。
如果约束正在起作用，则 g_i(x)=0，lambda_i 可以 >0。
```

**AI 连接：**

在 SVM 中：

```txt
support vectors 对应 active constraints。
它们的 lambda > 0。
非 support vectors 的 lambda = 0。
```

---

### 3.11 正则化就是约束优化

**一句话直觉：**  
L1 和 L2 正则化可以看成把参数限制在某个范数球内。

L2 constrained form：

```text
minimize Loss(w)
subject to ||w||_2^2 <= t
```

等价 unconstrained form：

```text
minimize Loss(w) + lambda ||w||_2^2
```

L1 constrained form：

```text
minimize Loss(w)
subject to ||w||_1 <= t
```

等价：

```text
minimize Loss(w) + lambda ||w||_1
```

几何差异：

| 正则化 | 约束形状 | 结果 |
|---|---|---|
| L2 | 圆 / 球 | 权重变小但通常不为 0 |
| L1 | 菱形 / 高维多面体 | 更容易在轴上取点，产生 0 权重 |

为什么 L1 稀疏？

```txt
L1 ball 有尖角，尖角落在坐标轴上。
loss contour 更容易先碰到尖角。
因此某些权重会正好为 0。
```

---

### 3.12 Duality：原问题和对偶问题

**一句话直觉：**  
一个约束优化问题通常有一个 companion dual problem；在凸问题中，dual 有时更容易解，而且可以和 primal 有相同最优值。

Primal：

```text
minimize f(x)
subject to g(x) <= 0
```

Lagrangian：

```text
L(x, lambda) = f(x) + lambda g(x)
```

Dual function：

```text
d(lambda) = min_x L(x, lambda)
```

Dual problem：

```text
maximize d(lambda)
subject to lambda >= 0
```

为什么重要？

```txt
dual problem 有时更容易求解。
dual 给出 primal 的 lower bound。
凸问题满足条件时有 strong duality。
SVM dual 只依赖样本点之间的 inner product。
```

**AI 连接：SVM kernel trick**

SVM dual 中只出现：

```text
x_i^T x_j
```

因此可以替换为 kernel：

```text
K(x_i, x_j)
```

这就是 kernel trick 的来源。

---

### 3.13 凸 vs 非凸：机器学习问题分类

| 问题 | 是否凸 | 原因 |
|---|---|---|
| Linear regression | 是 | loss 对权重是二次函数 |
| Logistic regression | 是 | log-loss 对权重凸 |
| SVM | 是 | hinge loss + convex constraints |
| Ridge regression | 是 | 二次 loss + 二次正则 |
| LASSO | 是 | 二次 loss + L1 凸正则 |
| Neural network | 否 | 多层非线性组合破坏凸性 |
| k-means | 否 | 离散 cluster assignment |
| Matrix factorization | 否 | 两组未知矩阵相乘 |
| Deep RL | 否 | 策略、环境、估计噪声共同作用 |

**经验判断：**

```txt
线性模型 + 凸 loss 通常是凸的。
多层网络 + nonlinear activation 通常是非凸的。
```

---

### 3.14 为什么神经网络非凸但仍然有效

**一句话直觉：**  
深度网络虽然非凸，但高维过参数化、SGD 噪声和 loss landscape 结构让它通常能找到足够好的解。

几个原因：

#### 大多数 critical points 是 saddle points

高维中，一个点要成为 local minimum，需要所有方向曲率都非负。  
如果维度很高，这种情况相对少见。  
更多 critical points 是 saddle points。

#### 坏 local minima 不常见

在大规模过参数化网络中，很多 local minima 的 loss 接近 global minimum。

#### 过参数化让 landscape 更平滑

参数多于样本时，loss valley 可能变得更连通，坏谷底减少。

#### SGD noise 帮助逃离鞍点

mini-batch 噪声能让参数从 saddle point 或 sharp minima 中逃出来。

#### flat minima 泛化更好

SGD 噪声倾向于让模型停在更平坦的区域，而不是尖锐谷底。

**AI 连接：**

这解释了为什么 deep learning 违反传统凸优化直觉，但仍然可训练。

---

### 3.15 二阶方法在现代 ML 中如何使用

**一句话直觉：**  
完整 Newton 方法太贵，但二阶信息可以通过近似方法部分利用。

| 方法 | 思想 | 适用场景 |
|---|---|---|
| Newton | 显式 Hessian inverse | 小型凸问题 |
| L-BFGS | 用最近 m 次梯度差近似 inverse Hessian | 中型传统 ML |
| Natural Gradient | 用 Fisher information 描述概率分布几何 | 概率模型、RL |
| K-FAC | Kronecker 分解近似 Fisher | 大 batch 神经网络研究 |
| Hessian-free | 用 Hessian-vector product + CG | 大规模二阶优化 |
| Adam | 用二阶矩做对角尺度调整 | 深度学习默认 |
| AdaHessian | 估计 Hessian diagonal | 研究用途 |

**AI 连接：**

深度学习主流仍是一阶方法：

```txt
SGD
Adam
AdamW
```

但二阶思想会出现在：

```txt
优化器设计
曲率分析
loss landscape 研究
natural gradient
RL 和概率模型
```

---

### 3.16 本课核心概念总表

| 概念 | 一句话直觉 | AI 中的对应位置 |
|---|---|---|
| Convex set | 两点连线不离开集合 | 可行域 |
| Convex function | 单一山谷 | 全局最优保证 |
| Hessian | 多维曲率矩阵 | 二阶优化 |
| PSD | 所有方向曲率非负 | 凸性判定 |
| Condition number | 山谷狭长程度 | GD 收敛速度 |
| Newton method | 使用曲率一步校正 | 小型凸问题 |
| Constrained optimization | 带约束的最小化 | 正则化、SVM |
| Lagrange multiplier | 等式约束求解工具 | 约束优化 |
| KKT | 不等式约束最优条件 | SVM、active constraint |
| Complementary slackness | 约束要么 active，要么无影响 | support vector |
| Regularization constraint | 范数球限制参数 | L1/L2 |
| Duality | primal 的 companion problem | SVM kernel trick |
| Non-convexity | 多山谷地形 | 神经网络 |
| Saddle point | 非最小的梯度零点 | 深度学习障碍 |
| Overparameterization | 参数很多 | smoother landscape |
| L-BFGS / K-FAC | 近似二阶 | 高效优化研究 |

---

## 4. 手写实现：凸性检查、Newton 方法和 Lagrange 优化

这一节使用基础 Python 实现核心思想。

---

### 4.1 Empirical convexity checker

```python
import random
import math


def check_convexity(f, dim, bounds=(-5, 5), samples=1000):
    violations = 0

    for _ in range(samples):
        x = [
            random.uniform(*bounds)
            for _ in range(dim)
        ]

        y = [
            random.uniform(*bounds)
            for _ in range(dim)
        ]

        t = random.uniform(0, 1)

        mid = [
            t * xi + (1 - t) * yi
            for xi, yi in zip(x, y)
        ]

        lhs = f(mid)
        rhs = t * f(x) + (1 - t) * f(y)

        if lhs > rhs + 1e-10:
            violations += 1

    return violations == 0, violations
```

示例：

```python
def square(x):
    return x[0] ** 2

def sine(x):
    return math.sin(x[0])

print(check_convexity(square, dim=1))
print(check_convexity(sine, dim=1))
```

注意：

```txt
这是经验检查，不是严格数学证明。
```

---

### 4.2 Gradient descent 对照

```python
def gradient_descent(f, grad_f, x0, lr=0.01, steps=1000, tol=1e-12):
    x = list(x0)
    history = [x[:]]

    for _ in range(steps):
        g = grad_f(x)

        x = [
            xi - lr * gi
            for xi, gi in zip(x, g)
        ]

        history.append(x[:])

        if sum(gi ** 2 for gi in g) < tol:
            break

    return history
```

---

### 4.3 Newton 方法 2D 实现

```python
def newtons_method(f, grad_f, hessian_f, x0, steps=50, tol=1e-12):
    x = list(x0)
    history = [x[:]]

    for _ in range(steps):
        g = grad_f(x)
        H = hessian_f(x)

        det = H[0][0] * H[1][1] - H[0][1] * H[1][0]

        if abs(det) < 1e-15:
            break

        H_inv = [
            [H[1][1] / det, -H[0][1] / det],
            [-H[1][0] / det, H[0][0] / det],
        ]

        dx = [
            H_inv[0][0] * g[0] + H_inv[0][1] * g[1],
            H_inv[1][0] * g[0] + H_inv[1][1] * g[1],
        ]

        x = [
            x[0] - dx[0],
            x[1] - dx[1],
        ]

        history.append(x[:])

        if sum(gi ** 2 for gi in g) < tol:
            break

    return history
```

---

### 4.4 Newton vs Gradient Descent

```python
def quadratic(x):
    return 5 * x[0] ** 2 + x[1] ** 2


def quadratic_grad(x):
    return [10 * x[0], 2 * x[1]]


def quadratic_hessian(x):
    return [[10, 0], [0, 2]]


x0 = [10.0, 10.0]

gd_hist = gradient_descent(
    quadratic,
    quadratic_grad,
    x0,
    lr=0.05,
    steps=1000
)

newton_hist = newtons_method(
    quadratic,
    quadratic_grad,
    quadratic_hessian,
    x0,
    steps=10
)

print("GD steps:", len(gd_hist))
print("Newton steps:", len(newton_hist))
print("GD final:", gd_hist[-1])
print("Newton final:", newton_hist[-1])
```

解释：

```txt
对二次函数，Newton 方法可以一步跳到最优点。
Gradient descent 会因为不同方向曲率不同而慢很多。
```

---

### 4.5 Lagrange multiplier 迭代求解

```python
def lagrange_solve(
    f_grad,
    g_val,
    g_grad,
    x0,
    lr=0.01,
    lr_lambda=0.01,
    steps=5000,
):
    x = list(x0)
    lam = 0.0
    history = []

    for _ in range(steps):
        fg = f_grad(x)
        gv = g_val(x)
        gg = g_grad(x)

        x = [
            xi - lr * (fgi + lam * ggi)
            for xi, fgi, ggi in zip(x, fg, gg)
        ]

        lam = lam + lr_lambda * gv

        history.append((x[:], lam, gv))

    return history
```

例子：

```python
# minimize x^2 + y^2 subject to x + y = 1

def f_grad(x):
    return [2 * x[0], 2 * x[1]]

def g_val(x):
    return x[0] + x[1] - 1

def g_grad(x):
    return [1, 1]

history = lagrange_solve(
    f_grad,
    g_val,
    g_grad,
    x0=[2.0, -1.0],
)

print(history[-1])
```

预期结果接近：

```text
x = [0.5, 0.5]
```

---

### 4.6 Hessian eigenvalue 分析

```python
import numpy as np


def hessian_eigenvalues(H):
    return np.linalg.eigvalsh(np.array(H, dtype=float))


H_convex = [[2, 0], [0, 2]]
H_saddle = [[2, 0], [0, -2]]

print(hessian_eigenvalues(H_convex))
print(hessian_eigenvalues(H_saddle))
```

判断：

```txt
全部 >= 0：PSD，局部凸。
有正有负：saddle。
```

---

## 5. 生产使用：SciPy / scikit-learn 中的凸优化工具

---

### 5.1 SciPy L-BFGS-B

```python
import numpy as np
from scipy.optimize import minimize

X = np.random.randn(100, 5)
y = np.random.randn(100)
d = X.shape[1]

def objective(w):
    residual = y - X @ w
    return np.sum(residual ** 2) + 0.1 * np.sum(w ** 2)

def gradient(w):
    return -2 * X.T @ (y - X @ w) + 0.2 * w

result = minimize(
    fun=objective,
    x0=np.zeros(d),
    method="L-BFGS-B",
    jac=gradient,
)

print(result.x)
print(result.fun)
```

---

### 5.2 scikit-learn SVM

```python
from sklearn.svm import SVC

svm = SVC(kernel="rbf", C=1.0)

svm.fit(X_train, y_train)

print(f"Support vectors: {svm.n_support_}")
```

SVM 的 dual formulation 使 kernel trick 成为可能。

---

### 5.3 scikit-learn Logistic Regression

```python
from sklearn.linear_model import LogisticRegression

clf = LogisticRegression(
    penalty="l2",
    solver="lbfgs",
    max_iter=1000,
)

clf.fit(X_train, y_train)

print(clf.coef_)
```

Logistic regression 是凸优化问题，因此专用 solver 可以找到全局最优解。

---

### 5.4 CVXPY 约束优化

```python
import cvxpy as cp
import numpy as np

w = cp.Variable(d)

objective = cp.Minimize(
    cp.sum_squares(X @ w - y) + 0.1 * cp.norm(w, 1)
)

constraints = [
    cp.norm(w, 2) <= 10
]

problem = cp.Problem(objective, constraints)

problem.solve()

print(w.value)
```

适合明确写出凸目标和约束的问题。

---

### 5.5 手写实现 vs 生产库

| 对比项 | 手写实现 | 生产库 |
|---|---|---|
| 凸性检查 | 经验采样 | 数学证明 / DCP rules |
| Newton | 手写 Hessian inverse | scipy.optimize |
| L-BFGS | 不建议手写 | SciPy / sklearn |
| Lagrange | 教学实现 | CVXPY / scipy.optimize |
| KKT | 理论分析 | 优化器内部处理 |
| SVM dual | 理解原理 | sklearn.svm |
| 正则化 | 手写 penalty | sklearn / PyTorch |
| 深度学习非凸 | 概念分析 | PyTorch AdamW / SGD |

---

## 6. 交付沉淀：产出物、连接关系、练习、术语表

---

### 6.1 本课产出

```txt
code/convex_optimization.py
code/use_scipy.py
code/use_sklearn.py
outputs/prompt-convex-optimization-guide.md
outputs/exercises-solutions.md
```

| 文件 | 作用 |
|---|---|
| `code/convex_optimization.py` | 从零实现 convexity checker、gradient descent、Newton、Lagrange solver |
| `code/use_scipy.py` | SciPy L-BFGS-B 示例 |
| `code/use_sklearn.py` | LogisticRegression、SVM、Ridge 等凸模型示例 |
| `outputs/prompt-convex-optimization-guide.md` | 判断优化问题是否凸及选择 solver 的 prompt |
| `outputs/exercises-solutions.md` | 练习题与参考答案 |

---

### 6.2 知识连接关系

| 本课知识点 | 后续连接 |
|---|---|
| 凸集 | feasible region |
| 凸函数 | 全局最优保证 |
| Hessian | 二阶优化、曲率分析 |
| Newton | 二阶方法 |
| Lagrange multiplier | 约束优化 |
| KKT | SVM、active constraints |
| Duality | SVM kernel trick |
| L1/L2 constraint | LASSO / Ridge |
| 非凸优化 | 深度学习 |
| Saddle point | loss landscape |
| L-BFGS / K-FAC | 近似二阶优化 |
| Overparameterization | 深度网络优化理论 |

---

### 6.3 AI 应用连接

| 凸优化知识点 | AI 应用 |
|---|---|
| Convexity | 判断问题难度 |
| Logistic regression | 凸分类模型 |
| SVM | 凸 margin optimization |
| LASSO | 稀疏特征选择 |
| Ridge | L2 正则化 |
| KKT | Support vectors |
| Duality | Kernel trick |
| Newton / L-BFGS | 传统 ML solver |
| Hessian | loss curvature |
| Non-convexity | deep learning |
| SGD noise | 逃离 saddle，偏向 flat minima |
| K-FAC / Natural gradient | 二阶近似 |

---

### 6.4 练习

1. **Convexity gallery。**  
   用 convexity checker 测试：

   ```text
   f(x)=x^4
   f(x)=sin(x)
   f(x,y)=x^2+y^2
   f(x,y)=x*y
   f(x)=max(x,0)
   ```

   判断结果是否符合直觉，并解释原因。

2. **Newton vs Gradient Descent。**  
   在函数：

   ```text
   f(x,y)=50x^2 + y^2
   ```

   上从 `(10,10)` 出发，比较 gradient descent 和 Newton 方法达到 `loss < 1e-10` 所需步数。  
   增大 Hessian condition number 后，观察 GD 为什么更慢。

3. **Lagrange multiplier 几何。**  
   求解：

   ```text
   minimize (x-3)^2 + (y-3)^2
   subject to x + 2y = 4
   ```

   验证在最优点处 `∇f` 与 `∇g` 平行。

4. **L1 constrained optimization。**  
   求解：

   ```text
   minimize (x-3)^2 + (y-2)^2
   subject to |x| + |y| <= 1
   ```

   观察解是否有一个坐标为 0，解释 L1 diamond constraint 如何产生稀疏性。

5. **Rosenbrock Hessian。**  
   计算 Rosenbrock 函数在 `(1,1)` 和 `(-1,1)` 的 Hessian eigenvalues。  
   解释它们分别代表 minimum 附近和远离 minimum 时的曲率。

---

### 6.5 关键术语

| 术语 | 常见说法 | 真正含义 |
|---|---|---|
| Convex set | 凸集 | 任意两点连线仍在集合内 |
| Convex function | 凸函数 | 两点连线在函数图像上方或重合 |
| Local minimum | 局部最小值 | 附近没有更低点 |
| Global minimum | 全局最小值 | 整个定义域上的最低点 |
| Hessian | Hessian 矩阵 | 二阶偏导数组成的曲率矩阵 |
| Positive semidefinite | 正半定 | 所有特征值非负 |
| Condition number | 条件数 | 最大和最小曲率比，决定 GD 难度 |
| Newton method | 牛顿法 | 用 Hessian inverse 调整更新方向和步长 |
| Lagrange multiplier | 拉格朗日乘子 | 把等式约束加入目标函数的变量 |
| KKT conditions | KKT 条件 | 不等式约束下的最优性条件 |
| Complementary slackness | 互补松弛 | 约束 active 或 multiplier 为 0 |
| Duality | 对偶性 | 原问题对应的 dual problem |
| Strong duality | 强对偶 | primal 和 dual 最优值相等 |
| L-BFGS | 近似二阶方法 | 用有限历史梯度近似 Hessian inverse |
| Saddle point | 鞍点 | 梯度为 0，但有正负混合曲率 |
| Overparameterization | 过参数化 | 参数多于样本，常改善深度网络优化地形 |

---

### 6.6 自检问题

学完本课后，你应该能回答：

- 凸优化为什么重要？
- 凸集和凸函数的定义分别是什么？
- 如何用定义判断函数凸性？
- 一维二阶导数和多维 Hessian 如何判断凸性？
- 正半定矩阵和凸函数有什么关系？
- 为什么凸函数没有坏 local minimum？
- 哪些机器学习问题是凸的？
- 为什么神经网络通常非凸？
- Newton 方法和 gradient descent 有什么区别？
- Newton 方法为什么在二次函数上可以一步到位？
- Lagrange multiplier 的几何意义是什么？
- KKT 条件四部分分别是什么意思？
- complementary slackness 如何解释 support vectors？
- L1/L2 regularization 为什么可以看成约束优化？
- L1 为什么稀疏，L2 为什么收缩？
- duality 为什么对 SVM 和 kernel trick 重要？
- 神经网络非凸但为什么仍然训练成功？
- 二阶方法在现代 ML 中有哪些近似形式？

---

## 附录：本课最小代码文件建议

如果要拆成代码文件，可以这样组织：

```txt
code/
├── convex_optimization.py
├── use_scipy.py
└── use_sklearn.py
```

### `convex_optimization.py`

包含：

```txt
check_convexity
gradient_descent
newtons_method
quadratic
quadratic_grad
quadratic_hessian
lagrange_solve
hessian_eigenvalues
```

### `use_scipy.py`

包含：

```txt
scipy.optimize.minimize
L-BFGS-B example
constrained optimization example
```

### `use_sklearn.py`

包含：

```txt
LogisticRegression
SVC
Ridge
Lasso
regularization comparison
support vector inspection
```
