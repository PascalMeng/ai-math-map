---
title: 概率与分布
description: 面向 AI 算法工程师的概率课程：从样本空间、条件概率、PMF/PDF、常见分布、期望方差，到 softmax、log-softmax、交叉熵和采样。
---

# 概率与分布

> 概率是 AI 表达不确定性的语言。

**课程类型：** 学习 / 实现  
**所属模块：** 概率统计  
**前置知识：** Phase 1 Lesson 01-04：线性代数、矩阵运算、矩阵变换、微积分基础  
**预计时间：** 约 75 分钟  
**使用语言：** Python / NumPy / SciPy / PyTorch  

---

## 1. 提出问题：为什么 AI 算法工程师必须理解概率与分布

一个分类器输出：

```txt
[0.03, 0.91, 0.06]
```

一个语言模型要从 50,000 个候选 token 中选择下一个词。  
一个 diffusion model 通过不断采样噪声并去噪来生成图片。

这些都是概率在 AI 中的真实用法。

在 AI 中：

```txt
模型预测的是概率分布。
损失函数衡量预测分布和真实分布的差距。
训练过程调整参数，让一个分布更接近另一个分布。
采样过程根据分布生成新的输出。
```

如果不理解概率，你就很难理解分类、语言模型、生成模型、交叉熵、KL 散度、采样和训练不稳定问题。

---

### 1.1 数学概念历史出现缘由

概率论最早不是为 AI 出现的，而是为了解决不确定性问题。

| 数学概念 | 历史出现缘由 |
|---|---|
| 样本空间 | 为了明确一个随机实验所有可能发生的结果 |
| 事件 | 为了描述样本空间中我们关心的一部分结果 |
| 概率 | 为了给不确定事件赋予可计算的数值 |
| 条件概率 | 为了描述“已知某事发生后，另一件事发生的可能性” |
| 独立性 | 为了判断两个事件是否互不影响 |
| PMF | 为了描述离散随机变量每个结果的概率 |
| PDF | 为了描述连续随机变量在不同位置的概率密度 |
| 期望 | 为了描述随机变量的长期平均结果 |
| 方差 | 为了描述随机变量围绕均值的波动程度 |
| 联合分布 | 为了描述多个随机变量同时出现的概率结构 |
| 边缘分布 | 为了从联合分布中只看某一个变量 |
| 中心极限定理 | 为了解释大量独立随机因素叠加后为什么接近正态分布 |
| log probability | 为了避免许多小概率相乘导致数值下溢 |
| softmax | 为了把模型原始分数转换成概率分布 |
| 交叉熵 | 为了衡量预测分布和真实分布之间的差距 |
| 采样 | 为了根据概率分布生成随机结果 |

可以这样理解：

```txt
现实问题：所有可能结果是什么？
数学抽象：样本空间

现实问题：某个结果集合发生的可能性是多少？
数学抽象：事件和概率

现实问题：已知一个事件发生后，另一个事件概率如何变化？
数学抽象：条件概率

现实问题：离散类别如何分配概率？
数学抽象：PMF

现实问题：连续变量如何描述概率？
数学抽象：PDF

现实问题：一个随机变量长期平均会是多少？
数学抽象：期望

现实问题：预测分布和真实标签差多少？
数学抽象：交叉熵

现实问题：如何从分布中生成样本？
数学抽象：采样
```

---

### 1.2 原始问题是什么

本课要解决的原始问题是：

```txt
如何用数学方式描述不确定性？
如何区分离散概率和连续概率？
如何计算一个随机变量的平均结果和波动程度？
如何描述多个随机变量之间的关系？
为什么正态分布会频繁出现？
为什么概率计算要使用 log？
神经网络如何把 logits 转换成概率？
交叉熵为什么是分类和语言模型的核心损失函数？
模型如何根据概率分布进行采样？
```

换成 AI 语言，就是：

```txt
分类模型输出的概率是什么意思？
softmax 为什么能把 logits 变成概率？
语言模型为什么可以从 token 分布中采样？
cross-entropy loss 到底在优化什么？
为什么 PyTorch 的 CrossEntropyLoss 接收 logits 而不是 softmax 后的概率？
为什么 log-softmax 更稳定？
为什么训练 loss 有时会 NaN？
diffusion、dropout、data augmentation 为什么都和采样有关？
```

---

### 1.3 AI 中的现代问题

| 数学知识点 | AI 中的现代问题 |
|---|---|
| 样本空间 | 分类任务中所有可能类别 |
| 事件 | 某个类别或某组类别发生 |
| 概率分布 | 模型对不同输出的置信度 |
| 条件概率 | `P(label | input)`，分类模型预测目标 |
| 独立性 | 朴素贝叶斯、建模假设 |
| PMF | 离散分类输出、token 分布 |
| PDF | 连续变量、VAE latent space、diffusion 噪声 |
| Bernoulli | 二分类 |
| Categorical | 多分类和语言模型 token 选择 |
| Normal | 初始化、噪声建模、latent variable |
| Poisson | 稀有事件计数建模 |
| 期望 | expected loss、risk minimization |
| 方差 | 训练波动、梯度噪声 |
| 中心极限定理 | 为什么高斯分布到处出现 |
| log probability | 防止概率连乘下溢 |
| softmax | logits 到概率分布 |
| cross-entropy | 分类和语言模型 loss |
| sampling | dropout、LLM decoding、diffusion generation |

---

### 1.4 本课要解决的核心问题

学完本课，你应该能够回答：

1. 样本空间、事件和概率分别是什么？
2. 条件概率和独立性有什么关系？
3. PMF 和 PDF 的区别是什么？
4. Bernoulli、Categorical、Uniform、Normal、Poisson 分布分别适合什么场景？
5. 期望和方差如何计算？
6. 联合分布和边缘分布如何互相转换？
7. 中心极限定理为什么解释了正态分布的普遍性？
8. 为什么机器学习中经常使用 log probability？
9. softmax 如何把 logits 转换成概率？
10. 为什么要使用 subtract max trick？
11. cross-entropy 和 negative log-likelihood 有什么关系？
12. 采样在 LLM、diffusion、dropout 和数据增强中如何出现？
13. 如何从零实现常见分布、softmax、log-softmax 和交叉熵？

---

## 2. 概念定位：这节课学什么

---

### 2.1 所属模块

```txt
所属模块：概率统计
前置知识：函数、指数、对数、基础微积分、Python
后续连接：贝叶斯、最大似然估计、信息论、交叉熵、KL 散度、语言模型 loss、采样方法、diffusion、VAE
```

本课是从“确定性计算”进入“不确定性建模”的关键一课。  
线性代数告诉我们模型如何表示对象，微积分告诉我们模型如何学习，概率告诉我们模型如何表达不确定性。

---

### 2.2 本课学习目标

完成本课后，你应该能够：

- 解释样本空间、事件、概率、公理化概率的基本含义
- 计算条件概率，并判断两个事件是否独立
- 区分 PMF 和 PDF
- 从零实现 Bernoulli、Categorical、Poisson、Uniform、Normal 分布
- 计算期望和方差
- 理解联合分布和边缘分布
- 用中心极限定理解释为什么正态分布频繁出现
- 理解 log probability 为什么能提升数值稳定性
- 从零实现 softmax、log-softmax 和 cross-entropy loss
- 说明概率和分布如何连接到分类模型、语言模型和生成模型

---

### 2.3 本课在整体路线中的位置

```txt
微积分：模型如何通过梯度学习
    ↓
概率与分布：模型如何表达不确定性
    ↓
贝叶斯与最大似然：模型如何根据数据更新信念
    ↓
信息论：分布之间的差距如何度量
    ↓
交叉熵 / KL 散度 / 语言模型 loss
    ↓
采样方法 / diffusion / VAE / LLM decoding
```

本课的核心能力是：

```txt
看到模型输出概率、softmax、cross entropy、sampling 时，知道它背后对应的概率分布和数值计算逻辑。
```

---

## 3. 理解概念：直觉、公式、概率解释、AI 连接

---

### 3.1 样本空间、事件与概率

**一句话直觉：**  
样本空间是所有可能结果，事件是其中一部分结果，概率是事件发生可能性的数值。

掷硬币：

```txt
S = {H, T}
P(H) = 0.5
P(T) = 0.5
```

掷骰子：

```txt
S = {1, 2, 3, 4, 5, 6}
even = {2, 4, 6}
P(even) = 3/6 = 0.5
```

概率的三个基本公理：

```txt
1. P(A) >= 0
2. P(S) = 1
3. 如果 A 和 B 不可能同时发生，则 P(A or B) = P(A) + P(B)
```

**AI 连接：**

| 概念 | AI 场景 |
|---|---|
| 样本空间 | 所有类别、所有 token、所有动作 |
| 事件 | 某个类别被选中、某个 token 出现 |
| 概率 | 模型对某个输出的置信度 |
| 总概率为 1 | softmax 输出必须归一化 |

---

### 3.2 条件概率与独立性

**一句话直觉：**  
条件概率描述在已知某个事件发生后，另一个事件发生的概率。

公式：

```text
P(A | B) = P(A and B) / P(B)
```

例子：一副牌中，已知抽到的是人头牌，抽到 King 的概率是：

```text
P(King | Face card) = P(King and Face card) / P(Face card)
                    = (4/52) / (12/52)
                    = 1/3
```

两个事件独立，表示知道一个事件发生，不会改变另一个事件的概率：

```text
P(A | B) = P(A)
```

等价于：

```text
P(A and B) = P(A) * P(B)
```

**AI 连接：**

| 概念 | AI 场景 |
|---|---|
| 条件概率 | `P(y | x)` 分类模型 |
| 独立性 | 朴素贝叶斯假设 |
| 非独立性 | 语言模型 token 依赖上下文 |
| 条件生成 | `P(output | prompt)` |

---

### 3.3 PMF 与 PDF

**一句话直觉：**  
PMF 描述离散变量每个结果的概率；PDF 描述连续变量的概率密度，区间下面积才是概率。

离散随机变量使用 PMF：

```text
PMF: P(X = k)
```

例子：公平骰子

```txt
P(X = 1) = 1/6
P(X = 2) = 1/6
...
P(X = 6) = 1/6
```

所有概率相加：

```text
sum_k P(X = k) = 1
```

连续随机变量使用 PDF：

```text
PDF: f(x)
```

概率来自积分：

```text
P(a <= X <= b) = integral_a^b f(x) dx
```

注意：

```txt
PDF 在某个点的值不是概率。
PDF 可以大于 1。
总面积必须等于 1。
```

**AI 连接：**

| 类型 | AI 场景 |
|---|---|
| PMF | 分类输出、token 分布、动作分布 |
| PDF | 连续 latent space、扩散噪声、连续特征 |
| 离散分布 | softmax |
| 连续分布 | Gaussian latent variable |

---

### 3.4 常见分布

**一句话直觉：**  
分布是对随机变量可能取值及其概率规律的描述。

#### Bernoulli 分布

用于一次二元试验：

```text
P(X = 1) = p
P(X = 0) = 1 - p
```

均值和方差：

```text
E[X] = p
Var(X) = p(1 - p)
```

**AI 场景：**

- 二分类
- dropout mask
- yes/no 事件

#### Categorical 分布

用于一次多类别选择：

```text
P(X = i) = p_i
sum_i p_i = 1
```

例子：

```txt
P(cat) = 0.7
P(dog) = 0.2
P(bird) = 0.1
```

**AI 场景：**

- 多分类
- softmax 输出
- language model next-token distribution

#### Uniform 分布

所有结果等可能。

离散均匀：

```text
P(X = k) = 1/n
```

连续均匀：

```text
f(x) = 1 / (b - a), x in [a, b]
```

**AI 场景：**

- 随机初始化
- 随机数据增强参数
- 随机采样 baseline

#### Normal / Gaussian 分布

钟形曲线，由均值和标准差控制：

```text
f(x) = (1 / sqrt(2*pi*sigma^2)) * exp(-(x - mu)^2 / (2*sigma^2))
```

标准正态分布：

```txt
mu = 0
sigma = 1
```

经验规律：

```txt
约 68% 数据在 1 个 sigma 内
约 95% 数据在 2 个 sigma 内
约 99.7% 数据在 3 个 sigma 内
```

**AI 场景：**

- 权重初始化
- 噪声建模
- VAE latent space
- diffusion model noise
- SGD 梯度噪声近似

#### Poisson 分布

用于固定时间或空间中稀有事件计数：

```text
P(X = k) = (lambda^k * e^(-lambda)) / k!
```

均值和方差：

```text
E[X] = lambda
Var(X) = lambda
```

**AI 场景：**

- 点击次数
- 到达事件
- 稀有事件计数
- 流量和请求建模

---

### 3.5 期望与方差

**一句话直觉：**  
期望是概率加权平均，方差描述结果围绕期望的波动程度。

离散变量期望：

```text
E[X] = sum_i x_i * P(X = x_i)
```

连续变量期望：

```text
E[X] = integral x * f(x) dx
```

方差：

```text
Var(X) = E[(X - E[X])^2]
```

也可以写成：

```text
Var(X) = E[X^2] - (E[X])^2
```

标准差：

```text
SD(X) = sqrt(Var(X))
```

**AI 连接：**

| 概念 | AI 场景 |
|---|---|
| 期望 | expected loss、risk minimization |
| 训练 loss | 对真实 expected loss 的样本近似 |
| 方差 | 模型波动、估计不稳定 |
| 梯度方差 | SGD 训练噪声 |
| 标准差 | 初始化尺度、归一化 |

---

### 3.6 联合分布与边缘分布

**一句话直觉：**  
联合分布描述多个随机变量一起出现的概率；边缘分布是把其他变量“加掉”后，只看一个变量。

联合分布：

```text
P(X, Y)
```

边缘化：

```text
P(X = x) = sum_y P(X = x, Y = y)
```

例子：

|  | Y=0 不带伞 | Y=1 带伞 | P(X) |
|---|---|---|---|
| X=0 晴天 | 0.40 | 0.10 | 0.50 |
| X=1 下雨 | 0.05 | 0.45 | 0.50 |
| P(Y) | 0.45 | 0.55 | 1.00 |

**AI 连接：**

| 概念 | AI 场景 |
|---|---|
| 联合分布 | 多变量建模、生成模型 |
| 边缘分布 | 隐变量积分、贝叶斯推断 |
| 条件分布 | `P(y | x)` |
| 边缘化 | latent variable model、VAE |

---

### 3.7 中心极限定理：为什么高斯到处出现

**一句话直觉：**  
许多独立随机变量的和或平均值，会趋近于正态分布，不管原始分布是什么。

直觉例子：

```txt
掷 1 个骰子：近似均匀
掷 2 个骰子的平均值：更集中
掷 30 个骰子的平均值：接近钟形曲线
```

这解释了为什么正态分布到处出现：

- 测量误差通常来自许多小扰动
- 神经网络权重常用正态初始化
- SGD 梯度噪声可以近似正态
- 给定均值和方差时，正态分布是最大熵分布

**AI 连接：**

| 场景 | CLT 直觉 |
|---|---|
| 初始化 | 大量随机因素叠加 |
| 梯度噪声 | mini-batch 梯度是样本梯度平均 |
| 噪声建模 | 高斯是常见默认选择 |
| diffusion | 加噪过程常与高斯噪声相关 |

---

### 3.8 Log Probability：避免小概率连乘下溢

**一句话直觉：**  
log probability 把概率相乘变成 log 概率相加，从而避免数值下溢。

语言模型中，一个句子的概率可能写成：

```text
P(sentence) = P(word1) * P(word2) * ... * P(word_n)
```

如果每个词概率都很小：

```txt
0.01 * 0.003 * 0.02 * ...
```

很快会下溢成 0。

使用 log 后：

```text
log P(sentence) = log P(word1) + log P(word2) + ... + log P(word_n)
```

性质：

```txt
log(a * b) = log(a) + log(b)
log probability 通常 <= 0
越负表示越不可能
```

**AI 连接：**

| 场景 | log probability 作用 |
|---|---|
| 语言模型 | 计算长序列概率 |
| cross-entropy | 正确类别的 negative log probability |
| beam search | 累加 token log probability |
| VAE / diffusion | 对数似然目标 |
| 数值稳定 | 防止 underflow |

---

### 3.9 Softmax：把 logits 变成概率分布

**一句话直觉：**  
softmax 把任意实数分数转换成和为 1 的概率分布。

模型输出的原始分数叫 logits：

```txt
z = [2.0, 1.0, 0.1]
```

softmax：

```text
softmax(z_i) = exp(z_i) / sum_j exp(z_j)
```

性质：

```txt
所有输出都在 (0, 1)
所有输出相加为 1
保持输入大小顺序
exp 会放大 logit 之间的差异
```

数值稳定技巧：

```txt
先减去最大 logit，再做 exp
```

例子：

```txt
z = [100, 101, 102]
exp(102) 会溢出

z_shifted = z - max(z) = [-2, -1, 0]
exp(0) = 1，安全
```

**AI 连接：**

| 场景 | softmax 作用 |
|---|---|
| 分类模型 | 输出类别概率 |
| 语言模型 | 输出下一个 token 概率 |
| attention | 把注意力分数归一化 |
| policy network | 输出动作概率 |

---

### 3.10 Log-softmax 与 Cross-entropy

**一句话直觉：**  
log-softmax 是 softmax 后取 log 的稳定版本；cross-entropy 是正确类别的负 log probability。

log-softmax：

```text
log_softmax(z_i) = z_i - log(sum_j exp(z_j))
```

为了稳定，使用 log-sum-exp trick：

```text
log_sum_exp(z) = max(z) + log(sum_j exp(z_j - max(z)))
```

交叉熵损失：

```text
CrossEntropy = -log P(correct class)
```

对于 one-hot 标签，它等价于：

```text
cross_entropy = -log_softmax(logits)[target_index]
```

**AI 连接：**

| 场景 | 作用 |
|---|---|
| 多分类 | 分类 loss |
| 语言模型 | next-token prediction loss |
| PyTorch CrossEntropyLoss | 内部结合 log-softmax 和 NLLLoss |
| 数值稳定 | 避免先 softmax 再 log 导致溢出或下溢 |

---

### 3.11 采样：根据分布生成结果

**一句话直觉：**  
采样就是按照概率分布随机抽取结果。

在 ML 中，采样无处不在：

| 场景 | 采样作用 |
|---|---|
| dropout | 随机丢弃神经元 |
| 数据增强 | 随机旋转、裁剪、缩放 |
| 语言模型 | 根据 token 概率采样下一个词 |
| diffusion | 采样噪声并逐步去噪 |
| VAE | 从 latent distribution 中采样 |
| 强化学习 | 根据 policy 采样动作 |

常见采样方法：

- inverse transform sampling
- rejection sampling
- categorical sampling
- Box-Muller normal sampling
- reparameterization trick

---

### 3.12 本课核心概念总表

| 概念 | 一句话直觉 | AI 中的对应位置 |
|---|---|---|
| 样本空间 | 所有可能结果 | 类别集合、token vocabulary |
| 事件 | 样本空间子集 | 某类输出发生 |
| 条件概率 | 已知条件下的概率 | `P(y | x)` |
| 独立性 | 一个事件不影响另一个 | 朴素贝叶斯假设 |
| PMF | 离散结果概率 | 分类、token 分布 |
| PDF | 连续概率密度 | latent space、噪声 |
| Bernoulli | 二元随机变量 | 二分类、dropout |
| Categorical | 多类别随机变量 | softmax、LLM token |
| Normal | 钟形分布 | 初始化、噪声、diffusion |
| Poisson | 稀有事件计数 | 点击、请求、事件率 |
| 期望 | 概率加权平均 | expected loss |
| 方差 | 波动程度 | gradient noise |
| 联合分布 | 多变量一起建模 | 生成模型 |
| 边缘分布 | sum out 其他变量 | latent variable models |
| CLT | 平均值趋近正态 | 梯度噪声、误差建模 |
| log probability | 防止小概率下溢 | sequence likelihood |
| softmax | logits 到概率 | 分类、attention |
| cross-entropy | 预测分布与真实分布差距 | classification / LM loss |
| sampling | 按分布抽样 | LLM、diffusion、dropout |

---

## 4. 手写实现：从零实现概率分布、softmax 和交叉熵

这一节先不用 NumPy / SciPy。  
目标是让你知道概率库和深度学习框架底层在计算什么。

---

### 4.1 概率基础函数

```python
import math
import random


def factorial(n):
    result = 1

    for i in range(2, n + 1):
        result *= i

    return result


def combinations(n, k):
    return factorial(n) // (factorial(k) * factorial(n - k))


def conditional_probability(p_a_and_b, p_b):
    return p_a_and_b / p_b


p_king_given_face = conditional_probability(4 / 52, 12 / 52)

print(f"P(King | Face card) = {p_king_given_face:.4f}")
```

---

### 4.2 从零实现 PMF 和 PDF

```python
def bernoulli_pmf(k, p):
    return p if k == 1 else (1 - p)


def categorical_pmf(k, probs):
    return probs[k]


def poisson_pmf(k, lam):
    return (lam ** k) * math.exp(-lam) / factorial(k)


def uniform_pdf(x, a, b):
    if a <= x <= b:
        return 1.0 / (b - a)

    return 0.0


def normal_pdf(x, mu, sigma):
    coeff = 1.0 / (sigma * math.sqrt(2 * math.pi))
    exponent = -0.5 * ((x - mu) / sigma) ** 2

    return coeff * math.exp(exponent)
```

对应关系：

| 函数 | 分布 |
|---|---|
| `bernoulli_pmf` | Bernoulli |
| `categorical_pmf` | Categorical |
| `poisson_pmf` | Poisson |
| `uniform_pdf` | Uniform continuous |
| `normal_pdf` | Normal / Gaussian |

---

### 4.3 从零计算期望和方差

```python
def expected_value(values, probabilities):
    return sum(v * p for v, p in zip(values, probabilities))


def variance(values, probabilities):
    mu = expected_value(values, probabilities)

    return sum(p * (v - mu) ** 2 for v, p in zip(values, probabilities))


die_values = [1, 2, 3, 4, 5, 6]
die_probs = [1 / 6] * 6

mu = expected_value(die_values, die_probs)
var = variance(die_values, die_probs)

print(f"Die: E[X] = {mu:.4f}, Var(X) = {var:.4f}, SD = {var ** 0.5:.4f}")
```

---

### 4.4 从零实现采样

```python
def sample_bernoulli(p, n=1):
    return [
        1 if random.random() < p else 0
        for _ in range(n)
    ]


def sample_categorical(probs, n=1):
    cumulative = []
    total = 0

    for p in probs:
        total += p
        cumulative.append(total)

    samples = []

    for _ in range(n):
        r = random.random()

        for i, c in enumerate(cumulative):
            if r <= c:
                samples.append(i)
                break

    return samples


def sample_normal_box_muller(mu, sigma, n=1):
    samples = []

    for _ in range(n):
        u1 = random.random()
        u2 = random.random()

        z = math.sqrt(-2 * math.log(u1)) * math.cos(2 * math.pi * u2)
        samples.append(mu + sigma * z)

    return samples
```

---

### 4.5 从零实现 softmax、log-softmax 和交叉熵

```python
def softmax(logits):
    max_logit = max(logits)
    shifted = [z - max_logit for z in logits]
    exps = [math.exp(z) for z in shifted]
    total = sum(exps)

    return [e / total for e in exps]


def log_softmax(logits):
    max_logit = max(logits)
    shifted = [z - max_logit for z in logits]
    log_sum_exp = max_logit + math.log(sum(math.exp(z) for z in shifted))

    return [z - log_sum_exp for z in logits]


def cross_entropy_loss(logits, target_index):
    log_probs = log_softmax(logits)

    return -log_probs[target_index]
```

测试：

```python
logits = [2.0, 0.5, -1.0, 3.0, 0.1]
target_index = 3

probs = softmax(logits)
loss = cross_entropy_loss(logits, target_index)

print(f"probs = {probs}")
print(f"cross entropy loss = {loss:.4f}")
```

---

### 4.6 中心极限定理演示

```python
def demonstrate_clt(dist_fn, n_samples, n_averages):
    averages = []

    for _ in range(n_averages):
        samples = [dist_fn() for _ in range(n_samples)]
        averages.append(sum(samples) / len(samples))

    return averages
```

示例：用骰子平均值观察分布形状变化：

```python
def roll_die():
    return random.randint(1, 6)


averages = demonstrate_clt(roll_die, n_samples=30, n_averages=10000)

print(f"First 10 averages: {averages[:10]}")
```

当 `n_samples` 变大时，平均值分布会越来越接近正态分布。

---

## 5. 生产使用：用 NumPy / SciPy / PyTorch 完成同样任务

---

### 5.1 NumPy / SciPy 版本：常见分布

```python
import numpy as np
from scipy import stats

normal = stats.norm(loc=0, scale=1)

samples = normal.rvs(size=10000)

print(f"Mean: {np.mean(samples):.4f}")
print(f"Std: {np.std(samples):.4f}")
print(f"P(X < 1.96) = {normal.cdf(1.96):.4f}")
```

---

### 5.2 SciPy 版本：softmax 和 log-softmax

```python
import numpy as np
from scipy.special import softmax, log_softmax

logits = np.array([2.0, 1.0, 0.1])

probs = softmax(logits)
log_probs = log_softmax(logits)

print(f"Softmax: {probs}")
print(f"Log-softmax: {log_probs}")
```

---

### 5.3 PyTorch 版本：CrossEntropyLoss

```python
import torch
import torch.nn as nn

logits = torch.tensor([[2.0, 0.5, -1.0, 3.0, 0.1]])
target = torch.tensor([3])

criterion = nn.CrossEntropyLoss()
loss = criterion(logits, target)

print(f"Cross entropy loss: {loss.item():.4f}")
```

注意：

```txt
PyTorch 的 CrossEntropyLoss 接收的是 logits，而不是 softmax 后的概率。
它内部会做 log-softmax + negative log-likelihood。
```

---

### 5.4 NumPy 版本：log probability 累加

```python
import numpy as np

word_probs = [0.01] * 50

raw_prob = np.prod(word_probs)
log_prob = np.sum(np.log(word_probs))

print(f"Raw probability: {raw_prob}")
print(f"Log probability: {log_prob}")
print(f"Recovered probability: {np.exp(log_prob)}")
```

长序列中直接乘概率可能下溢，而 log probability 仍然可计算。

---

### 5.5 手写实现 vs 生产库

| 对比项 | 手写实现 | NumPy / SciPy / PyTorch |
|---|---|---|
| 目的 | 理解概率和分布底层逻辑 | 工程实践 |
| 分布函数 | 自己写 PMF / PDF | `scipy.stats` |
| 采样 | 自己写随机采样 | `rvs` / `torch.multinomial` |
| softmax | 手动实现稳定技巧 | `scipy.special.softmax` |
| cross entropy | 手动 log-softmax | `nn.CrossEntropyLoss` |
| 数值稳定 | 需要自己处理 | 框架通常已处理 |
| 使用场景 | 学习、验证原理 | 真实训练和评估 |

---

## 6. 交付沉淀：产出物、连接关系、练习、术语表

---

### 6.1 本课产出

```txt
code/probability.py
code/use_scipy.py
code/use_pytorch.py
outputs/prompt-probability-distributions.md
outputs/exercises-solutions.md
```

| 文件 | 作用 |
|---|---|
| `code/probability.py` | 从零实现 PMF、PDF、采样、softmax、log-softmax、cross-entropy |
| `code/use_scipy.py` | SciPy 版本概率分布和 softmax |
| `code/use_pytorch.py` | PyTorch CrossEntropyLoss 验证 |
| `outputs/prompt-probability-distributions.md` | 用于教学概率与分布直觉的 prompt |
| `outputs/exercises-solutions.md` | 练习题与参考答案 |

---

### 6.2 知识连接关系

| 本课知识点 | 后续连接 |
|---|---|
| 条件概率 | 贝叶斯公式、Naive Bayes |
| PMF / PDF | 生成模型、概率建模 |
| Bernoulli | 二分类、dropout |
| Categorical | 多分类、语言模型 |
| Normal | VAE、diffusion、初始化 |
| 期望 | risk minimization、loss |
| 方差 | 估计稳定性、梯度噪声 |
| 联合 / 边缘分布 | 隐变量模型、贝叶斯推断 |
| log probability | sequence likelihood、beam search |
| softmax | 分类、attention、policy network |
| cross-entropy | 分类 loss、语言模型 loss |
| 采样 | LLM decoding、diffusion、RL |

---

### 6.3 AI 应用连接

| 数学知识点 | AI 应用 |
|---|---|
| 样本空间 | 分类类别、token vocabulary |
| 条件概率 | `P(y | x)`、`P(token | context)` |
| Categorical 分布 | softmax 输出和 token 采样 |
| Normal 分布 | 初始化、噪声、latent variable |
| 期望 | expected loss |
| 方差 | 训练波动和不确定性 |
| log probability | 长序列概率和语言模型打分 |
| softmax | 分类概率和 attention weight |
| log-softmax | 稳定计算概率对数 |
| cross-entropy | 分类和语言模型训练 |
| sampling | 生成式 AI 输出 |

---

### 6.4 练习

1. **实现指数分布的 inverse transform sampling。**  
   采样 10,000 个值，并把直方图和真实 PDF 比较。

2. **构造两个 loaded dice 的联合分布。**  
   计算边缘分布，并判断两个骰子是否独立。

3. **手算 5 类分类器的交叉熵。**  
   给定 logits：

   ```txt
   [2.0, 0.5, -1.0, 3.0, 0.1]
   ```

   正确类别是 index `3`。  
   先手写计算 cross-entropy，再用 PyTorch 的 `nn.CrossEntropyLoss` 验证。

4. **计算长句子的 log probability。**  
   写一个函数，输入一组 log probabilities，输出最可能序列、总 log probability 和等价 raw probability。  
   用一个 50 个词的句子测试，每个词概率为 `0.01`。

5. **实现 temperature sampling。**  
   给定 logits，加入 temperature 参数：

   ```text
   softmax(logits / temperature)
   ```

   观察 temperature 变大或变小时，分布如何变化。

---

### 6.5 关键术语

| 术语 | 常见说法 | 真正含义 |
|---|---|---|
| 样本空间 | 所有可能性 | 一个随机实验所有可能结果的集合 |
| 事件 | 某件事发生 | 样本空间的一个子集 |
| 概率 | 可能性大小 | 把事件映射到 0 到 1 之间的数 |
| 条件概率 | 已知某事后的概率 | `P(A|B) = P(A and B) / P(B)` |
| 独立性 | 互不影响 | `P(A and B) = P(A)P(B)` |
| PMF | 离散概率函数 | 给出每个离散结果的概率，总和为 1 |
| PDF | 概率曲线 | 连续变量的密度，区间积分才是概率 |
| Bernoulli | 二选一 | 单次二元随机试验 |
| Categorical | 多选一 | 单次多类别随机试验 |
| Normal | 钟形曲线 | 由均值和方差控制，因 CLT 普遍出现 |
| Poisson | 事件计数 | 固定区间内稀有事件发生次数 |
| 期望 | 平均值 | 概率加权平均结果 |
| 方差 | 波动程度 | 偏离均值的平方期望 |
| 联合分布 | 多变量一起看 | `P(X, Y)` 描述变量组合概率 |
| 边缘分布 | 只看一个变量 | 从联合分布中 sum out 其他变量 |
| 中心极限定理 | 平均值趋向正态 | 大量独立随机变量的平均趋近高斯 |
| log probability | 概率取 log | 把概率乘法变成加法，避免下溢 |
| logits | 原始分数 | softmax 之前的未归一化模型输出 |
| softmax | 分数转概率 | 把 logits 映射为合法概率分布 |
| log-softmax | 稳定 log 概率 | softmax + log 的数值稳定版本 |
| cross-entropy | 分类损失 | 真实分布和预测分布差距，one-hot 时等于正确类负 log probability |
| sampling | 按概率抽样 | 根据分布随机生成结果 |

---

### 6.6 自检问题

学完本课后，你应该能回答：

- 概率论最初为什么会出现？
- 样本空间、事件和概率有什么区别？
- 条件概率和独立性如何判断？
- PMF 和 PDF 有什么本质区别？
- Bernoulli、Categorical、Normal、Poisson 分别适合什么场景？
- 期望和方差如何计算？
- 联合分布如何边缘化？
- 中心极限定理为什么解释了正态分布的普遍性？
- 为什么长序列概率要用 log probability？
- softmax 为什么能把 logits 转换成概率？
- subtract max trick 为什么能避免 overflow？
- cross-entropy 和 negative log-likelihood 有什么关系？
- PyTorch 的 `CrossEntropyLoss` 为什么接收 logits？
- 采样为什么是生成式 AI 的核心机制之一？

---

## 附录：本课最小代码文件建议

如果要拆成代码文件，可以这样组织：

```txt
code/
├── probability.py
├── use_scipy.py
└── use_pytorch.py
```

### `probability.py`

包含：

```txt
factorial
combinations
conditional_probability
bernoulli_pmf
categorical_pmf
poisson_pmf
uniform_pdf
normal_pdf
expected_value
variance
sample_bernoulli
sample_categorical
sample_normal_box_muller
softmax
log_softmax
cross_entropy_loss
demonstrate_clt
```

### `use_scipy.py`

包含：

```txt
scipy.stats.norm
normal.rvs
normal.cdf
scipy.special.softmax
scipy.special.log_softmax
```

### `use_pytorch.py`

包含：

```txt
torch.tensor
nn.CrossEntropyLoss
logits
target
loss.item
```
