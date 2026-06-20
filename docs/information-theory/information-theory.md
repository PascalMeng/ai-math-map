---
title: 信息论
description: 面向 AI 算法工程师的信息论课程：从信息量、熵、交叉熵、KL 散度、互信息，到 perplexity、label smoothing 和语言模型损失。
---

# 信息论

> 信息论度量“惊讶程度”。很多机器学习损失函数，都是建立在这个思想之上。

**课程类型：** 学习 / 实现  
**所属模块：** 信息论与损失函数  
**前置知识：** Phase 1 Lesson 06：概率与分布  
**预计时间：** 约 60 分钟  
**使用语言：** Python / NumPy / PyTorch  

---

## 1. 提出问题：为什么 AI 算法工程师必须理解信息论

你训练分类模型时，经常会写：

```python
loss = CrossEntropyLoss(logits, labels)
```

你读语言模型论文时，会看到：

```txt
perplexity
```

你读 VAE、知识蒸馏、RLHF、DPO 或 PPO 时，又会看到：

```txt
KL divergence
```

这些不是互不相关的术语，而是同一个核心思想的不同形式：

```txt
信息论用来度量不确定性、惊讶程度、分布差异和预测代价。
```

在 AI 中，模型训练本质上是在做一件事：

```txt
让模型预测分布越来越接近真实分布。
```

交叉熵、KL 散度、负对数似然、perplexity，都是在不同角度描述这个过程。

---

### 1.1 数学概念历史出现缘由

信息论最初不是为神经网络发明的，而是为通信问题而生。  
Claude Shannon 在 1948 年提出信息论，用来回答：

```txt
如何度量一条消息包含多少信息？
如何用最短编码传输消息？
有噪声的通信信道最多能传多少信息？
```

后来人们发现，机器学习也可以被看成一种“信息传输”问题：

```txt
模型试图把输入中的信息传递成正确标签。
损失函数衡量模型预测分布与真实分布之间的信息差距。
训练过程就是减少这种信息浪费。
```

| 信息论概念 | 历史出现缘由 |
|---|---|
| 信息量 | 为了度量一个事件发生后带来的“惊讶程度” |
| 熵 | 为了度量一个分布平均有多不确定 |
| 交叉熵 | 为了度量用错误分布编码真实事件时的平均代价 |
| KL 散度 | 为了度量两个分布之间的信息浪费 |
| 互信息 | 为了度量一个变量能告诉我们另一个变量多少信息 |
| 条件熵 | 为了度量已知一个变量后，另一个变量还剩多少不确定性 |
| 联合熵 | 为了度量两个变量一起的不确定性 |
| bits / nats | 为了区分信息量使用的对数单位 |
| perplexity | 为了把语言模型损失解释成“有效候选词数量” |
| label smoothing | 为了避免模型对 one-hot 标签过度自信 |

可以这样理解：

```txt
现实问题：一个事件发生后带来多少信息？
数学抽象：信息量

现实问题：一个随机系统平均有多不确定？
数学抽象：熵

现实问题：用模型分布解释真实标签会浪费多少信息？
数学抽象：交叉熵

现实问题：两个分布差多少？
数学抽象：KL 散度

现实问题：一个特征对标签有多少帮助？
数学抽象：互信息

现实问题：语言模型平均还在多少个词之间犹豫？
评估指标：perplexity
```

---

### 1.2 原始问题是什么

本课要解决的原始问题是：

```txt
一个事件发生到底带来多少信息？
为什么低概率事件更“有信息”？
如何度量一个概率分布的不确定性？
交叉熵为什么能作为分类模型损失函数？
KL 散度为什么可以度量两个分布之间的差异？
互信息为什么可以用来做特征选择？
perplexity 为什么能衡量语言模型困惑程度？
label smoothing 为什么能改善模型校准？
```

换成 AI 语言，就是：

```txt
CrossEntropyLoss 到底在优化什么？
为什么分类标签是 one-hot 时，交叉熵等于正确类的负 log 概率？
为什么最小化交叉熵等价于最大化训练数据似然？
为什么 KL divergence 会出现在 VAE、RLHF 和蒸馏里？
为什么语言模型论文常用 perplexity？
为什么 label smoothing 能防止模型过度自信？
```

---

### 1.3 AI 中的现代问题

| 信息论知识点 | AI 中的现代问题 |
|---|---|
| 信息量 | 一个事件出现后有多“惊讶” |
| 熵 | 预测分布有多不确定 |
| 交叉熵 | 分类和语言模型的核心 loss |
| KL 散度 | VAE、蒸馏、RLHF、DPO、PPO |
| 互信息 | 特征选择、表示学习 |
| 条件熵 | 决策树、信息增益 |
| 联合熵 | 多变量联合建模 |
| label smoothing | 防止过度自信，提高校准 |
| 负对数似然 | 最大似然训练目标 |
| bits / nats | 信息单位，PyTorch 默认用 nats |
| perplexity | 语言模型评价指标 |
| softmax | logits 转概率分布 |
| log probability | 稳定计算序列概率 |

---

### 1.4 本课要解决的核心问题

学完本课，你应该能够回答：

1. 信息量为什么定义为 `-log(p)`？
2. 熵为什么是平均惊讶程度？
3. 为什么均匀分布熵更大，确定性分布熵更小？
4. 交叉熵为什么是分类模型最常用的 loss？
5. KL 散度为什么等于 cross-entropy 减 entropy？
6. 为什么最小化 cross-entropy 等价于最大化 log-likelihood？
7. KL 散度为什么不是对称距离？
8. 互信息为什么能衡量特征和标签之间的关系？
9. 条件熵和联合熵之间有什么关系？
10. label smoothing 为什么能防止模型过度自信？
11. bits 和 nats 有什么区别？
12. perplexity 如何解释语言模型的困惑程度？
13. 如何从零实现 entropy、cross-entropy、KL divergence、mutual information 和 perplexity？

---

## 2. 概念定位：这节课学什么

---

### 2.1 所属模块

```txt
所属模块：信息论与损失函数
前置知识：概率分布、log probability、softmax、分类问题
后续连接：交叉熵损失、KL 散度、VAE、蒸馏、RLHF、语言模型 loss、决策树、特征选择
```

前面的概率课程解决了：

```txt
模型如何表达不确定性？
```

本课进一步解决：

```txt
如何度量这种不确定性，以及如何用它定义机器学习损失函数？
```

---

### 2.2 本课学习目标

完成本课后，你应该能够：

- 从零计算信息量、熵、交叉熵和 KL 散度
- 解释熵、交叉熵和 KL 散度之间的关系
- 说明为什么交叉熵损失等价于负对数似然
- 计算互信息，并用它评估特征重要性
- 理解条件熵、联合熵和互信息之间的关系
- 解释 label smoothing 为什么能作为正则化
- 区分 bits 和 nats
- 解释 perplexity 为什么是语言模型常见指标
- 用 NumPy 实现信息论核心公式
- 理解 `torch.nn.CrossEntropyLoss` 背后的数学含义

---

### 2.3 本课在整体路线中的位置

```txt
概率与分布
    ↓
log probability / softmax
    ↓
信息量与熵
    ↓
交叉熵与 KL 散度
    ↓
分类 loss / 语言模型 loss
    ↓
perplexity / label smoothing
    ↓
VAE / 蒸馏 / RLHF / DPO / 信息增益
```

本课的核心能力是：

```txt
看到 CrossEntropyLoss、KL divergence、perplexity 时，能知道它们都在度量分布之间的信息差异。
```

---

## 3. 理解概念：直觉、公式、信息解释、AI 连接

---

### 3.1 信息量：事件的惊讶程度

**一句话直觉：**  
越不可能发生的事件，一旦发生，带来的信息越多。

信息量公式：

```text
I(x) = -log(p(x))
```

如果使用 `log2`，单位是 bits。  
如果使用自然对数 `ln`，单位是 nats。

| 事件 | 概率 | 信息量 bits |
|---|---:|---:|
| 公平硬币正面 | 0.5 | 1.0 |
| 掷骰子得到 6 | 1/6 | 2.58 |
| 1/1000 的事件 | 0.001 | 9.97 |
| 必然事件 | 1.0 | 0.0 |

确定会发生的事件信息量为 0，因为你已经知道它会发生。

**AI 连接：**

| 场景 | 信息量含义 |
|---|---|
| 正确类别概率很低 | loss 高，模型很“惊讶” |
| 正确类别概率很高 | loss 低，模型不惊讶 |
| 语言模型预测罕见 token | 信息量高 |
| anomaly detection | 低概率事件更异常 |

---

### 3.2 熵：平均惊讶程度

**一句话直觉：**  
熵是一个分布中所有事件信息量的平均值。

公式：

```text
H(P) = -sum_x p(x) log(p(x))
```

公平硬币：

```text
H = -(0.5 log2 0.5 + 0.5 log2 0.5) = 1 bit
```

高度偏置硬币：

```text
P(head)=0.99, P(tail)=0.01
H ≈ 0.08 bits
```

解释：

```txt
公平硬币最不确定，所以熵高。
偏置硬币几乎总是正面，所以熵低。
```

**AI 连接：**

| 场景 | 熵含义 |
|---|---|
| 分类输出很均匀 | 模型不确定，熵高 |
| 分类输出集中在一个类别 | 模型很确定，熵低 |
| active learning | 优先标注高熵样本 |
| 决策树 | 用熵衡量节点不纯度 |
| 语言模型 | 预测分布熵反映困惑程度 |

---

### 3.3 交叉熵：用模型分布编码真实分布的代价

**一句话直觉：**  
交叉熵衡量：如果真实数据来自分布 P，但你用模型分布 Q 去解释它，平均会有多惊讶。

公式：

```text
H(P, Q) = -sum_x p(x) log(q(x))
```

其中：

| 符号 | 含义 |
|---|---|
| `P` | 真实分布 |
| `Q` | 模型预测分布 |
| `p(x)` | 真实分布下事件 x 的概率 |
| `q(x)` | 模型给事件 x 的概率 |

如果模型预测分布 `Q` 和真实分布 `P` 完全一致，那么：

```text
H(P, Q) = H(P)
```

如果 `Q` 和 `P` 不一致，交叉熵会更大。

**分类中的简化：**

分类标签通常是 one-hot：

```txt
真实类别概率为 1
其他类别概率为 0
```

此时交叉熵变成：

```text
H(P, Q) = -log(q(true_class))
```

这就是分类模型的交叉熵损失。

**AI 连接：**

| 场景 | 交叉熵作用 |
|---|---|
| 多分类 | 分类 loss |
| 语言模型 | next-token prediction loss |
| 图像分类 | softmax cross-entropy |
| 语音识别 | token-level CE |
| 序列模型 | 每个位置的 CE 求和或平均 |

---

### 3.4 KL 散度：两个分布之间的信息浪费

**一句话直觉：**  
KL 散度衡量用 Q 代替 P 会多浪费多少信息。

公式：

```text
D_KL(P || Q) = sum_x p(x) log(p(x) / q(x))
```

也可以写成：

```text
D_KL(P || Q) = H(P, Q) - H(P)
```

因此：

```text
Cross-entropy = Entropy + KL divergence
```

训练时，真实分布 `P` 是固定的，所以 `H(P)` 是常数。  
最小化交叉熵等价于最小化 KL 散度：

```txt
让模型分布 Q 尽量接近真实分布 P。
```

注意：

```text
D_KL(P || Q) != D_KL(Q || P)
```

所以 KL 散度不是严格意义上的距离。

**AI 连接：**

| 场景 | KL 散度作用 |
|---|---|
| VAE | 让 latent 分布接近先验 |
| 知识蒸馏 | 学生模型拟合教师模型分布 |
| RLHF / PPO | 限制新策略偏离参考策略 |
| DPO | 控制偏好优化中的分布偏移 |
| domain adaptation | 衡量源域和目标域分布差异 |

---

### 3.5 互信息：一个变量告诉我们另一个变量多少信息

**一句话直觉：**  
互信息衡量知道一个变量后，另一个变量的不确定性减少了多少。

公式：

```text
I(X; Y) = H(X) - H(X|Y)
```

也可以写成：

```text
I(X; Y) = H(X) + H(Y) - H(X, Y)
```

或者：

```text
I(X; Y) = sum_x sum_y p(x,y) log(p(x,y) / (p(x)p(y)))
```

性质：

```txt
I(X;Y) >= 0
I(X;Y) = 0 当且仅当 X 和 Y 独立
I(X;Y) = I(Y;X)
I(X;X) = H(X)
```

**AI 连接：**

| 场景 | 互信息作用 |
|---|---|
| 特征选择 | 衡量 feature 和 target 依赖程度 |
| 表示学习 | 保留与任务相关的信息 |
| 对比学习 | 增强相关表示之间的信息 |
| 决策树 | 信息增益 |
| 模态对齐 | 图像和文本之间共享信息 |

互信息比相关系数更通用：

| 方法 | 能检测什么关系 |
|---|---|
| Pearson correlation | 线性关系 |
| Spearman correlation | 单调关系 |
| Mutual information | 任意统计依赖 |

---

### 3.6 条件熵：已知一个变量后还剩多少不确定性

**一句话直觉：**  
条件熵衡量在知道 X 之后，对 Y 还剩多少不确定性。

公式：

```text
H(Y|X) = H(X,Y) - H(X)
```

两个极端：

```txt
如果 X 完全决定 Y，则 H(Y|X) = 0。
如果 X 对 Y 没有任何帮助，则 H(Y|X) = H(Y)。
```

性质：

```text
0 <= H(Y|X) <= H(Y)
```

**AI 连接：**

| 场景 | 条件熵作用 |
|---|---|
| 决策树 | 选择让 label 条件熵最小的特征 |
| 信息增益 | `H(Y) - H(Y|X)` |
| 特征选择 | 判断 feature 是否减少 label 不确定性 |
| 表征学习 | 好表示应该降低目标不确定性 |

---

### 3.7 联合熵：多个变量一起的不确定性

**一句话直觉：**  
联合熵度量多个变量作为整体的不确定性。

公式：

```text
H(X,Y) = -sum_x sum_y p(x,y) log(p(x,y))
```

关键关系：

```text
H(X,Y) <= H(X) + H(Y)
```

当 `X` 和 `Y` 独立时：

```text
H(X,Y) = H(X) + H(Y)
```

如果 `X` 和 `Y` 共享信息，联合熵会小于各自熵之和。  
少掉的那部分就是互信息：

```text
I(X;Y) = H(X) + H(Y) - H(X,Y)
```

**AI 连接：**

| 场景 | 联合熵作用 |
|---|---|
| 多变量建模 | 衡量多个变量整体不确定性 |
| 图模型 | 联合分布 |
| 多模态学习 | 图像和文本联合信息 |
| 表示学习 | 判断表示是否共享信息 |

---

### 3.8 Label Smoothing：让标签不那么绝对

**一句话直觉：**  
Label smoothing 把 one-hot 硬标签变成带一点不确定性的软标签，防止模型过度自信。

原始 hard target：

```txt
[0, 0, 1, 0]
```

label smoothing：

```text
soft_target = (1 - epsilon) * hard_target + epsilon / num_classes
```

如果 `epsilon = 0.1`，有 4 类：

```txt
hard target: [0, 0, 1, 0]
soft target: [0.025, 0.025, 0.925, 0.025]
```

从信息论角度看：

```txt
one-hot 标签熵为 0。
label smoothing 提高了标签分布的熵。
```

好处：

- 防止模型 logits 走向极端
- 降低过度自信
- 改善概率校准
- 作为正则化
- 减少训练和推理之间的置信度落差

---

### 3.9 为什么交叉熵是分类任务核心 loss

**一句话直觉：**  
交叉熵同时可以从信息论、最大似然和梯度三个角度解释。

#### 信息论视角

交叉熵衡量：

```txt
用模型分布编码真实事件时浪费了多少信息。
```

最小化交叉熵就是让模型成为更高效的真实标签编码器。

#### 最大似然视角

对于训练样本：

```text
Likelihood = product_i q(y_i)
```

log-likelihood：

```text
Log-likelihood = sum_i log(q(y_i))
```

negative log-likelihood：

```text
NLL = -sum_i log(q(y_i))
```

这正是 one-hot 分类下的交叉熵。

所以：

```txt
最小化 cross-entropy = 最大化训练数据 likelihood。
```

#### 梯度视角

softmax + cross-entropy 对 logits 的梯度非常简洁：

```text
gradient = predicted_probability - true_distribution
```

这也是它数值稳定、训练高效的重要原因。

---

### 3.10 Bits vs Nats：信息单位

**一句话直觉：**  
bits 和 nats 的区别只是 log 的底数不同。

| 单位 | 对数底数 | 常见场景 |
|---|---|---|
| bits | `log2` | 信息论传统 |
| nats | `ln` | 机器学习框架默认 |
| hartleys | `log10` | 较少使用 |

换算：

```text
1 nat = 1 / ln(2) bits ≈ 1.4427 bits
```

PyTorch 和 TensorFlow 默认使用自然对数，因此交叉熵通常以 nats 为单位。

---

### 3.11 Perplexity：语言模型的困惑程度

**一句话直觉：**  
Perplexity 是交叉熵的指数，表示模型平均相当于在多少个等可能选项中选择。

如果使用 bits：

```text
Perplexity = 2^H(P,Q)
```

如果使用 nats：

```text
Perplexity = e^H(P,Q)
```

例子：

```txt
perplexity = 50
```

表示语言模型平均像是在 50 个等可能 token 中做选择。

越低越好：

```txt
perplexity 越低，模型越不困惑。
```

**AI 连接：**

| 场景 | perplexity 含义 |
|---|---|
| 语言模型评估 | 衡量 next-token prediction 难度 |
| 模型比较 | 同数据集上 perplexity 越低越好 |
| 领域适配 | 特定领域 perplexity 下降表示适配更好 |
| 数据质量 | 高 perplexity 可能说明数据与训练分布不匹配 |

---

### 3.12 本课核心概念总表

| 概念 | 一句话直觉 | AI 中的对应位置 |
|---|---|---|
| 信息量 | 单个事件的惊讶程度 | token loss、异常检测 |
| 熵 | 平均惊讶程度 | 预测不确定性 |
| 交叉熵 | 用模型分布解释真实分布的代价 | 分类 loss、LM loss |
| KL 散度 | 用 Q 替代 P 的额外信息浪费 | VAE、RLHF、蒸馏 |
| 互信息 | 一个变量告诉另一个变量多少 | 特征选择、表示学习 |
| 条件熵 | 已知 X 后 Y 还剩多少不确定性 | 决策树、信息增益 |
| 联合熵 | 多变量整体不确定性 | 联合分布、多模态 |
| label smoothing | 软化标签分布 | 正则化、校准 |
| NLL | 正确类别负 log 概率 | 最大似然训练 |
| bits / nats | 信息单位 | PyTorch 默认 nats |
| perplexity | 有效候选数量 | 语言模型评估 |

---

## 4. 手写实现：从零实现信息论核心公式

这一节先不用 PyTorch。  
目标是亲手实现 entropy、cross-entropy、KL divergence、mutual information 和 perplexity。

---

### 4.1 实现信息量和熵

```python
import math


def information_content(p, base=2):
    if p <= 0 or p > 1:
        return float("inf") if p <= 0 else 0.0

    return -math.log(p) / math.log(base)


def entropy(probs, base=2):
    return sum(
        p * information_content(p, base)
        for p in probs
        if p > 0
    )


fair_coin = [0.5, 0.5]
biased_coin = [0.99, 0.01]
fair_die = [1 / 6] * 6

print(f"Fair coin entropy:   {entropy(fair_coin):.4f} bits")
print(f"Biased coin entropy: {entropy(biased_coin):.4f} bits")
print(f"Fair die entropy:    {entropy(fair_die):.4f} bits")
```

---

### 4.2 实现交叉熵和 KL 散度

```python
def cross_entropy(p, q, base=2):
    total = 0.0

    for pi, qi in zip(p, q):
        if pi > 0:
            if qi <= 0:
                return float("inf")

            total += pi * (-math.log(qi) / math.log(base))

    return total


def kl_divergence(p, q, base=2):
    return cross_entropy(p, q, base) - entropy(p, base)


true_dist = [0.7, 0.2, 0.1]
good_model = [0.6, 0.25, 0.15]
bad_model = [0.1, 0.1, 0.8]

print(f"Entropy of true dist: {entropy(true_dist):.4f} bits")
print(f"CE good model:        {cross_entropy(true_dist, good_model):.4f} bits")
print(f"CE bad model:         {cross_entropy(true_dist, bad_model):.4f} bits")
print(f"KL good model:        {kl_divergence(true_dist, good_model):.4f} bits")
print(f"KL bad model:         {kl_divergence(true_dist, bad_model):.4f} bits")
```

预期现象：

```txt
good_model 的 cross-entropy 和 KL 都比 bad_model 更低。
```

---

### 4.3 实现分类交叉熵损失

```python
def softmax(logits):
    max_logit = max(logits)
    exps = [math.exp(z - max_logit) for z in logits]
    total = sum(exps)

    return [e / total for e in exps]


def cross_entropy_loss(true_class, logits):
    probs = softmax(logits)

    return -math.log(probs[true_class])


logits = [2.0, 1.0, 0.1]
true_class = 0

probs = softmax(logits)
loss = cross_entropy_loss(true_class, logits)

print(f"Logits:     {logits}")
print(f"Softmax:    {[f'{p:.4f}' for p in probs]}")
print(f"True class: {true_class}")
print(f"Loss:       {loss:.4f} nats")
print(f"Perplexity: {math.exp(loss):.2f}")
```

---

### 4.4 验证交叉熵等于负对数似然

```python
import random

random.seed(42)

n_samples = 1000
n_classes = 3

true_labels = [
    random.randint(0, n_classes - 1)
    for _ in range(n_samples)
]

model_logits = [
    [random.gauss(0, 1) for _ in range(n_classes)]
    for _ in range(n_samples)
]

ce_loss = sum(
    cross_entropy_loss(label, logits)
    for label, logits in zip(true_labels, model_logits)
) / n_samples

nll = -sum(
    math.log(softmax(logits)[label])
    for label, logits in zip(true_labels, model_logits)
) / n_samples

print(f"Cross-entropy loss:      {ce_loss:.6f}")
print(f"Negative log-likelihood: {nll:.6f}")
print(f"Difference:              {abs(ce_loss - nll):.2e}")
```

如果差异接近 0，说明：

```txt
分类交叉熵就是负对数似然。
```

---

### 4.5 实现互信息

```python
def mutual_information(joint_probs, base=2):
    rows = len(joint_probs)
    cols = len(joint_probs[0])

    margin_x = [
        sum(joint_probs[i][j] for j in range(cols))
        for i in range(rows)
    ]

    margin_y = [
        sum(joint_probs[i][j] for i in range(rows))
        for j in range(cols)
    ]

    mi = 0.0

    for i in range(rows):
        for j in range(cols):
            pxy = joint_probs[i][j]

            if pxy > 0:
                mi += pxy * math.log(
                    pxy / (margin_x[i] * margin_y[j])
                ) / math.log(base)

    return mi


independent = [
    [0.25, 0.25],
    [0.25, 0.25],
]

dependent = [
    [0.45, 0.05],
    [0.05, 0.45],
]

print(f"MI independent: {mutual_information(independent):.4f} bits")
print(f"MI dependent:   {mutual_information(dependent):.4f} bits")
```

预期结果：

```txt
独立变量互信息接近 0。
相关变量互信息大于 0。
```

---

### 4.6 实现 perplexity

```python
def perplexity_from_losses(losses, base="e"):
    avg_loss = sum(losses) / len(losses)

    if base == "e":
        return math.exp(avg_loss)

    if base == 2:
        return 2 ** avg_loss

    raise ValueError("base must be 'e' or 2")


losses = [1.2, 1.6, 1.4, 1.5]

print(f"Perplexity: {perplexity_from_losses(losses):.2f}")
```

---

## 5. 生产使用：用 NumPy / PyTorch 完成同样任务

---

### 5.1 NumPy 实现 entropy、cross-entropy、KL

```python
import numpy as np


def np_entropy(p):
    p = np.asarray(p, dtype=float)
    mask = p > 0

    result = np.zeros_like(p)
    result[mask] = p[mask] * np.log(p[mask])

    return -result.sum()


def np_cross_entropy(p, q):
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)

    mask = p > 0

    return -(p[mask] * np.log(q[mask])).sum()


def np_kl_divergence(p, q):
    return np_cross_entropy(p, q) - np_entropy(p)


true = np.array([0.7, 0.2, 0.1])
pred = np.array([0.6, 0.25, 0.15])

print(f"Entropy:    {np_entropy(true):.4f} nats")
print(f"Cross-ent:  {np_cross_entropy(true, pred):.4f} nats")
print(f"KL div:     {np_kl_divergence(true, pred):.4f} nats")
```

---

### 5.2 PyTorch CrossEntropyLoss

```python
import torch
import torch.nn as nn

logits = torch.tensor([[2.0, 1.0, 0.1]])
target = torch.tensor([0])

criterion = nn.CrossEntropyLoss()
loss = criterion(logits, target)

print(f"CrossEntropyLoss: {loss.item():.4f} nats")
print(f"Perplexity: {torch.exp(loss).item():.2f}")
```

注意：

```txt
PyTorch 的 CrossEntropyLoss 接收 logits。
它内部会做 log-softmax + negative log-likelihood。
```

---

### 5.3 PyTorch KLDivLoss

```python
import torch
import torch.nn.functional as F

p = torch.tensor([0.7, 0.2, 0.1])
q_logits = torch.tensor([1.8, 0.9, 0.2])

log_q = F.log_softmax(q_logits, dim=0)

kl = F.kl_div(log_q, p, reduction="sum")

print(f"KL divergence: {kl.item():.4f} nats")
```

---

### 5.4 手写实现 vs 生产库

| 对比项 | 手写实现 | NumPy / PyTorch |
|---|---|---|
| 目的 | 理解信息论公式 | 工程实践 |
| 熵 | 手动求和 | NumPy 向量化 |
| 交叉熵 | 手动 softmax + log | `CrossEntropyLoss` |
| KL | `CE - H` | `KLDivLoss` |
| 单位 | 可选 bits / nats | 默认 nats |
| 数值稳定 | 需要自己处理 | 框架内置稳定实现 |
| 使用场景 | 教学、验证 | 真实训练 |

---

## 6. 交付沉淀：产出物、连接关系、练习、术语表

---

### 6.1 本课产出

```txt
code/information_theory.py
code/use_numpy.py
code/use_pytorch.py
outputs/prompt-information-theory.md
outputs/exercises-solutions.md
```

| 文件 | 作用 |
|---|---|
| `code/information_theory.py` | 从零实现信息量、熵、交叉熵、KL、互信息、perplexity |
| `code/use_numpy.py` | NumPy 版本信息论计算 |
| `code/use_pytorch.py` | PyTorch CrossEntropyLoss / KLDivLoss 示例 |
| `outputs/prompt-information-theory.md` | 用于教学信息论和损失函数直觉的 prompt |
| `outputs/exercises-solutions.md` | 练习题与参考答案 |

---

### 6.2 知识连接关系

| 本课知识点 | 后续连接 |
|---|---|
| 信息量 | token-level loss、异常检测 |
| 熵 | 不确定性、决策树 |
| 交叉熵 | 分类 loss、语言模型 loss |
| KL 散度 | VAE、蒸馏、RLHF |
| 互信息 | 特征选择、表示学习 |
| 条件熵 | 信息增益、决策树 |
| 联合熵 | 多变量建模 |
| label smoothing | 校准、正则化 |
| NLL | 最大似然估计 |
| perplexity | 语言模型评估 |

---

### 6.3 AI 应用连接

| 数学知识点 | AI 应用 |
|---|---|
| 信息量 | 正确类别概率越低，loss 越高 |
| 熵 | 模型不确定性 |
| 交叉熵 | 分类、语言模型训练 |
| KL 散度 | VAE、RLHF、DPO、distillation |
| 互信息 | 特征选择、多模态对齐 |
| 条件熵 | 决策树 split |
| label smoothing | 防止过度自信 |
| NLL | 最大似然训练 |
| perplexity | 语言模型评估 |
| bits / nats | loss 单位解释 |

---

### 6.4 练习

1. **英文字符熵。**  
   假设英文字母 26 个字符均匀分布，计算其熵。再使用真实英文字母频率估计熵。哪个更高？为什么？

2. **手算交叉熵。**  
   模型输出 logits：

   ```txt
   [5.0, 2.0, 0.5]
   ```

   真实类别是 index `1`。  
   手算交叉熵，再用 `cross_entropy_loss` 验证。什么样的 logits 会让 loss 接近 0？

3. **证明 KL 不对称。**  
   选择两个分布 `P` 和 `Q`，分别计算：

   ```text
   D_KL(P || Q)
   D_KL(Q || P)
   ```

   解释为什么二者不同。

4. **实现序列 perplexity。**  
   写一个函数，输入一组 `(true_token_index, predicted_logits)`，输出整段序列 perplexity。

5. **label smoothing 实验。**  
   实现 label smoothing 后的 cross-entropy，比较 hard target 和 soft target 下 loss 的变化。

---

### 6.5 关键术语

| 术语 | 常见说法 | 真正含义 |
|---|---|---|
| 信息量 | 惊讶程度 | `-log(p)`，事件越小概率，信息量越大 |
| 熵 | 随机性 | 一个分布的平均信息量或平均惊讶程度 |
| 交叉熵 | loss function | 用模型分布编码真实事件的平均惊讶程度 |
| KL 散度 | 分布距离 | 用 Q 代替 P 额外浪费的信息，不对称 |
| 互信息 | 相关程度 | 一个变量能减少另一个变量多少不确定性 |
| 条件熵 | 剩余不确定性 | 已知 X 后 Y 还剩多少不确定性 |
| 联合熵 | 联合不确定性 | 多个变量一起的不确定性 |
| Softmax | 分数转概率 | 把 logits 映射成合法概率分布 |
| NLL | 负对数似然 | one-hot 分类下等价于交叉熵 |
| Perplexity | 困惑度 | 交叉熵的指数，表示有效候选数量 |
| Bits | 比特 | 以 2 为底的 log 信息单位 |
| Nats | 自然单位 | 以 e 为底的 log 信息单位，ML 框架默认 |
| Label smoothing | 标签软化 | 把 one-hot 标签变成带少量不确定性的软分布 |

---

### 6.6 自检问题

学完本课后，你应该能回答：

- 信息论最初解决什么问题？
- 为什么信息量定义为 `-log(p)`？
- 熵为什么可以理解为平均惊讶程度？
- 为什么均匀分布的熵更高？
- 交叉熵为什么是分类模型的常用 loss？
- KL 散度为什么等于 cross-entropy 减 entropy？
- 为什么 KL 散度不是对称距离？
- 最小化交叉熵为什么等价于最大化 log-likelihood？
- 互信息为什么可以用于特征选择？
- 条件熵和信息增益有什么关系？
- label smoothing 为什么能改善校准？
- bits 和 nats 有什么区别？
- perplexity 如何解释语言模型困惑程度？
- PyTorch 的 CrossEntropyLoss 内部做了什么？

---

## 附录：本课最小代码文件建议

如果要拆成代码文件，可以这样组织：

```txt
code/
├── information_theory.py
├── use_numpy.py
└── use_pytorch.py
```

### `information_theory.py`

包含：

```txt
information_content
entropy
cross_entropy
kl_divergence
softmax
cross_entropy_loss
mutual_information
perplexity_from_losses
label_smoothing_cross_entropy
```

### `use_numpy.py`

包含：

```txt
np_entropy
np_cross_entropy
np_kl_divergence
```

### `use_pytorch.py`

包含：

```txt
torch.nn.CrossEntropyLoss
torch.nn.functional.kl_div
torch.exp(loss) for perplexity
```
