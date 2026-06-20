---
title: 贝叶斯公式
description: 面向 AI 算法工程师的贝叶斯课程：从先验、似然、后验、证据，到朴素贝叶斯、MLE、MAP、共轭先验、序贯更新和 Bayesian A/B Testing。
---

# 贝叶斯公式

> 概率关心你原本相信什么；贝叶斯公式关心你看到证据后应该如何更新相信。

**课程类型：** 实现 / 应用  
**所属模块：** 概率统计  
**前置知识：** Phase 1 Lesson 06：概率与分布  
**预计时间：** 约 75 分钟  
**使用语言：** Python / scikit-learn  

---

## 1. 提出问题：为什么 AI 算法工程师必须理解贝叶斯公式

一个医疗测试准确率是 99%。你测试阳性。你真的有 99% 的概率患病吗？

大多数人的直觉会说：是的，差不多 99%。

但如果这种病非常罕见，比如 10,000 人里只有 1 人患病，那么测试阳性之后，你真正患病的概率可能不到 1%。  
原因是：大多数阳性结果其实来自健康人群中的假阳性。

这不是文字游戏，而是贝叶斯公式。

贝叶斯公式回答的是：

```txt
在看到新证据之后，应该如何更新原来的信念？
```

在 AI 中，贝叶斯思想无处不在：

```txt
垃圾邮件过滤
医疗诊断
异常检测
A/B 测试
少样本学习
模型不确定性
贝叶斯优化
正则化
在线学习
```

如果不理解贝叶斯，你可能会误解模型概率、设置错误阈值、过度相信模型输出，或者在小样本场景下做出过度自信的决策。

---

### 1.1 数学概念历史出现缘由

贝叶斯公式最初不是为机器学习出现的，而是为了解决“如何根据证据更新判断”的问题。

| 数学概念 | 历史出现缘由 |
|---|---|
| 条件概率 | 为了描述已知某个事件发生后，另一个事件发生的概率 |
| 贝叶斯公式 | 为了把 `P(证据|假设)` 反过来变成 `P(假设|证据)` |
| 先验 | 为了表达在看到证据之前的原始信念 |
| 似然 | 为了表达某个假设为真时，观察到当前证据的可能性 |
| 后验 | 为了表达看到证据之后更新后的信念 |
| 证据 | 为了对所有可能假设下观察到证据的概率进行归一化 |
| 全概率公式 | 为了把证据在所有可能情况下的概率加总 |
| 朴素贝叶斯 | 为了在多特征场景下简化贝叶斯分类 |
| MLE | 为了从数据中估计最可能产生观测结果的参数 |
| MAP | 为了把先验加入参数估计，相当于带正则化的估计 |
| Laplace smoothing | 为了避免未见过的特征导致概率为 0 |
| 共轭先验 | 为了让贝叶斯更新有闭式解，避免复杂积分 |
| 序贯贝叶斯更新 | 为了让新数据不断更新旧信念 |
| Bayesian A/B Testing | 为了直接估计一个版本优于另一个版本的概率 |

可以这样理解：

```txt
现实问题：看到证据后，原来的判断应该如何更新？
数学抽象：贝叶斯公式

现实问题：某个结果在某个假设下出现的概率是多少？
数学抽象：似然

现实问题：在看到数据之前，我原本相信什么？
数学抽象：先验

现实问题：看到数据之后，我应该相信什么？
数学抽象：后验

现实问题：训练集中某个词从未出现，概率是否应该为 0？
工程方法：Laplace smoothing

现实问题：小样本下如何避免过拟合？
数学思想：MAP 和先验正则化
```

---

### 1.2 原始问题是什么

本课要解决的原始问题是：

```txt
如何根据新证据更新原来的概率判断？
如何区分 P(A|B) 和 P(B|A)？
为什么高准确率测试不一定意味着高患病概率？
如何把文本中的词转化为垃圾邮件概率？
如何从数据中估计 P(feature|class)？
为什么概率为 0 会毁掉整个分类器？
如何把先验知识引入参数估计？
如何让今天的后验成为明天的先验？
如何用贝叶斯方法做 A/B 测试？
```

换成 AI 语言，就是：

```txt
模型预测概率是否真的可信？
为什么小样本下需要正则化？
Naive Bayes 如何做文本分类？
MLE 和 MAP 有什么区别？
为什么 L2 正则化可以看作高斯先验？
为什么线上系统可以不断用新数据更新信念？
Bayesian A/B Testing 为什么能直接输出 P(B > A)？
```

---

### 1.3 AI 中的现代问题

| 数学知识点 | AI 中的现代问题 |
|---|---|
| 先验 | 小样本学习、正则化、领域知识 |
| 似然 | 模型对观测数据的解释能力 |
| 后验 | 看到数据后的更新信念 |
| 证据 | 归一化常数、模型比较 |
| 全概率公式 | 计算总证据概率 |
| 贝叶斯公式 | 概率更新、诊断、分类 |
| 基础率 | 避免忽视类别先验导致误判 |
| 朴素贝叶斯 | 文本分类、垃圾邮件检测 |
| 条件独立假设 | 简化高维特征建模 |
| Laplace smoothing | 解决未见词概率为 0 |
| log probability | 避免概率连乘下溢 |
| MLE | 不带先验的参数估计 |
| MAP | 带先验的参数估计，连接正则化 |
| 共轭先验 | 高效闭式贝叶斯更新 |
| Beta-Binomial | 点击率、转化率、A/B 测试 |
| 序贯更新 | 在线学习、流式系统 |
| Bayesian A/B | 直接估计版本优劣概率 |

---

### 1.4 本课要解决的核心问题

学完本课，你应该能够回答：

1. 贝叶斯公式如何从条件概率推导出来？
2. 先验、似然、后验、证据分别是什么意思？
3. 为什么基础率会极大影响后验概率？
4. 医疗测试阳性为什么不等于高概率患病？
5. 垃圾邮件过滤如何使用贝叶斯公式？
6. 朴素贝叶斯为什么要假设特征条件独立？
7. 为什么朴素贝叶斯虽然假设很粗糙，但实际分类效果常常不错？
8. MLE 如何从频率估计概率？
9. 为什么 Laplace smoothing 能避免零概率问题？
10. MAP 和正则化有什么关系？
11. 为什么 L2 正则化可以看作高斯先验？
12. 什么是共轭先验？
13. Beta-Binomial 如何用于序贯更新和 A/B 测试？
14. 如何从零实现 Naive Bayes 文本分类器？
15. scikit-learn 的 MultinomialNB 和手写版本有什么对应关系？

---

## 2. 概念定位：这节课学什么

---

### 2.1 所属模块

```txt
所属模块：概率统计
前置知识：条件概率、PMF、log probability、分类问题、Python 字典
后续连接：最大似然估计、MAP、正则化、朴素贝叶斯、贝叶斯优化、A/B 测试、在线学习、模型不确定性
```

上一课你已经知道如何描述概率分布。  
本课进一步学习：**看到证据后，如何更新概率分布和信念。**

---

### 2.2 本课学习目标

完成本课后，你应该能够：

- 用条件概率推导贝叶斯公式
- 解释先验、似然、证据和后验
- 计算医疗测试、垃圾邮件等实际问题的后验概率
- 从零实现一个 Naive Bayes 文本分类器
- 使用 Laplace smoothing 防止零概率
- 使用 log-space 避免概率连乘下溢
- 区分 MLE 和 MAP
- 解释 MAP 与 L1 / L2 正则化的关系
- 理解 Bayesian vs Frequentist 的实践差异
- 使用 Beta-Binomial 共轭先验做序贯贝叶斯更新
- 理解 Bayesian A/B Testing 的基本流程
- 使用 scikit-learn 的 MultinomialNB 做生产级对照

---

### 2.3 本课在整体路线中的位置

```txt
概率与分布
    ↓
条件概率
    ↓
贝叶斯公式
    ↓
Naive Bayes / MLE / MAP
    ↓
共轭先验与序贯更新
    ↓
Bayesian A/B Testing / 在线学习
    ↓
不确定性估计 / 贝叶斯优化 / 概率模型
```

本课的核心能力是：

```txt
看到一个模型输出概率时，不只问“概率是多少”，还要问“这个概率是在什么先验、证据和假设下得到的”。
```

---

## 3. 理解概念：直觉、公式、概率解释、AI 连接

---

### 3.1 从条件概率到贝叶斯公式

**一句话直觉：**  
贝叶斯公式把“假设产生证据的概率”反过来，变成“看到证据后假设成立的概率”。

条件概率定义：

```text
P(A|B) = P(A and B) / P(B)
```

同理：

```text
P(B|A) = P(A and B) / P(A)
```

因此：

```text
P(A and B) = P(A|B)P(B) = P(B|A)P(A)
```

整理得到贝叶斯公式：

```text
P(A|B) = P(B|A)P(A) / P(B)
```

这就是从 joint probability 到 Bayes 的推导。

**AI 连接：**

| 表达式 | AI 中的解释 |
|---|---|
| `P(A)` | 类别先验、参数先验 |
| `P(B|A)` | 给定类别时观测到特征的概率 |
| `P(A|B)` | 看到输入后属于某个类别的概率 |
| `P(B)` | 输入证据的总概率 |

---

### 3.2 贝叶斯公式的四个部分

**一句话直觉：**  
贝叶斯公式就是“先验 × 似然，再除以证据进行归一化”。

公式：

```text
Posterior = Likelihood * Prior / Evidence
```

也就是：

```text
P(A|B) = P(B|A) * P(A) / P(B)
```

| 部分 | 数学符号 | 含义 |
|---|---|---|
| Prior | `P(A)` | 看到证据前，对 A 的原始信念 |
| Likelihood | `P(B|A)` | 如果 A 为真，看到证据 B 的概率 |
| Evidence | `P(B)` | 在所有可能情况下看到证据 B 的总概率 |
| Posterior | `P(A|B)` | 看到证据 B 后，对 A 的更新信念 |

证据项可以用全概率公式展开：

```text
P(B) = P(B|A)P(A) + P(B|not A)P(not A)
```

**AI 连接：**

| 部分 | AI 例子 |
|---|---|
| Prior | 垃圾邮件比例、疾病患病率、类别分布 |
| Likelihood | 某词在垃圾邮件中出现的概率 |
| Evidence | 某词在所有邮件中出现的概率 |
| Posterior | 邮件是垃圾邮件的概率 |

---

### 3.3 医疗测试：基础率为什么重要

**一句话直觉：**  
即使测试很准确，如果疾病非常罕见，阳性结果也可能大多是假阳性。

假设：

```text
P(sick) = 0.0001
P(positive | sick) = 0.99
P(positive | healthy) = 0.01
```

证据概率：

```text
P(positive)
= P(positive|sick)P(sick) + P(positive|healthy)P(healthy)
= 0.99 * 0.0001 + 0.01 * 0.9999
= 0.010098
```

后验概率：

```text
P(sick | positive)
= P(positive|sick)P(sick) / P(positive)
= 0.99 * 0.0001 / 0.010098
≈ 0.0098
= 0.98%
```

结论：

```txt
测试阳性后，真正患病概率不到 1%。
```

原因是：

```txt
疾病基础率太低，健康人群中的假阳性数量远远多于真正患病者中的真阳性。
```

**AI 连接：**

| 场景 | 类比 |
|---|---|
| 罕见病检测 | 异常检测 |
| 欺诈识别 | 正样本极少，误报很多 |
| 安全告警 | 高准确率也可能高误报 |
| 分类阈值 | 必须考虑 base rate |
| 医疗 AI | 不能只看模型 accuracy |

---

### 3.4 垃圾邮件过滤：一个词如何更新概率

**一句话直觉：**  
一个关键词可以显著改变邮件属于垃圾邮件的后验概率。

假设：

```text
P(spam) = 0.3
P("lottery" | spam) = 0.05
P("lottery" | not spam) = 0.001
```

证据概率：

```text
P("lottery") = 0.05 * 0.3 + 0.001 * 0.7
             = 0.0157
```

后验概率：

```text
P(spam | "lottery") = 0.05 * 0.3 / 0.0157
                    ≈ 0.955
                    = 95.5%
```

一个词就能把垃圾邮件概率从 30% 提高到 95.5%。

**AI 连接：**

这就是文本分类中的基本思想：

```txt
不同词对不同类别有不同的证据强度。
分类器根据这些证据更新类别概率。
```

---

### 3.5 Naive Bayes：条件独立假设

**一句话直觉：**  
Naive Bayes 假设在给定类别后，各个特征之间相互独立。

对于多个特征：

```text
P(class | feature_1, feature_2, ..., feature_n)
```

根据贝叶斯公式：

```text
P(class | features)
∝ P(class) * P(feature_1|class) * P(feature_2|class) * ... * P(feature_n|class)
```

因为分母对所有类别都相同，分类时只需要比较分子：

```text
score(class) = P(class) * product_i P(feature_i | class)
```

“Naive” 的地方在于：

```txt
词之间并不真的独立。
例如 "New" 和 "York" 明显相关。
```

但实际分类中，Naive Bayes 往往效果不错，因为分类器只需要比较类别分数，不一定需要输出完美校准的概率。

**AI 连接：**

| 场景 | Naive Bayes 作用 |
|---|---|
| 垃圾邮件分类 | 根据词出现概率判断 spam / ham |
| 文本分类 baseline | 快速、可解释、轻量 |
| 小数据分类 | 简单模型不容易过拟合 |
| 特征重要性 | 可以检查哪些词最支持某一类 |

---

### 3.6 MLE：从数据频率估计概率

**一句话直觉：**  
MLE 选择最能让观测数据出现的参数；在离散计数中，它通常就是相对频率。

例如：

```text
P("free" | spam)
= spam 邮件中包含 "free" 的次数 / spam 邮件总词数
```

这就是 maximum likelihood estimation：

```text
选择最大化 P(data | parameters) 的参数。
```

问题：

```txt
如果某个词在训练集中从未出现在 spam 类中，那么 MLE 会给它概率 0。
```

在 Naive Bayes 中，多个词概率要相乘。  
只要其中一个词概率为 0，整个类别分数就变成 0。

---

### 3.7 Laplace Smoothing：避免零概率

**一句话直觉：**  
Laplace smoothing 给每个词都加一个小计数，避免未见过的词概率为 0。

公式：

```text
P(word | class)
= (count(word, class) + 1) / (total_words_in_class + vocabulary_size)
```

更一般地，可以加 `alpha`：

```text
P(word | class)
= (count(word, class) + alpha) / (total_words_in_class + alpha * vocabulary_size)
```

**AI 连接：**

| 问题 | smoothing 解决方式 |
|---|---|
| 未见词概率为 0 | 加一个伪计数 |
| 小样本不稳定 | 把概率拉回更平滑的分布 |
| 过度自信 | 防止某些特征一票否决 |
| 文本分类 | 提升 Naive Bayes 稳定性 |

---

### 3.8 Log-space：避免概率连乘下溢

**一句话直觉：**  
很多小概率相乘会下溢成 0，取 log 后乘法变加法，更稳定。

原始 Naive Bayes 分数：

```text
score(class) = P(class) * product_i P(word_i | class)
```

log-space：

```text
log_score(class) = log P(class) + sum_i log P(word_i | class)
```

数学等价：

```text
log(a*b) = log(a) + log(b)
```

**AI 连接：**

log-space 广泛用于：

- Naive Bayes
- 语言模型 log probability
- beam search
- cross-entropy
- Viterbi / HMM
- 序列建模

---

### 3.9 MAP：MLE 加上先验

**一句话直觉：**  
MLE 只看数据，MAP 同时看数据和先验。

MLE 优化：

```text
maximize P(data | parameters)
```

MAP 优化：

```text
maximize P(parameters | data)
```

根据贝叶斯公式：

```text
P(parameters | data) ∝ P(data | parameters) * P(parameters)
```

也就是：

```txt
MAP = likelihood + prior
```

如果你认为参数应该比较小，就可以给参数一个偏向小值的先验。  
这和机器学习中的正则化是同一件事。

| 估计方式 | 优化目标 | ML 等价物 |
|---|---|---|
| MLE | `P(data | params)` | 无正则化训练 |
| MAP | `P(data | params) * P(params)` | 带正则化训练 |

**AI 连接：**

| 先验 | 正则化 |
|---|---|
| Gaussian prior | L2 regularization |
| Laplace prior | L1 regularization |

所以：

```txt
每次你添加 regularization，本质上就是在表达一种参数先验。
```

---

### 3.10 Bayesian vs Frequentist

**一句话直觉：**  
频率派把参数看成固定未知数；贝叶斯派把参数看成可以用分布描述的不确定量。

| 维度 | Frequentist | Bayesian |
|---|---|---|
| 参数 | 固定但未知 | 随机变量，有分布 |
| 输出 | 点估计 | 分布 |
| 不确定性 | confidence interval | credible interval |
| 小数据 | 容易过拟合 | 先验可以约束 |
| 计算 | 通常更快 | 可能需要 MCMC / VI |
| 典型问题 | 如果重复实验会怎样 | 给定观察数据，我现在相信什么 |

**AI 连接：**

生产 ML 大多数是频率派方法：

```txt
SGD + point estimate
```

贝叶斯方法特别适合：

- 医疗
- 安全系统
- 小样本学习
- 冷启动推荐
- 模型不确定性
- 风险敏感决策

---

### 3.11 共轭先验：让贝叶斯更新变成加法

**一句话直觉：**  
如果先验和后验属于同一类分布，这个先验就叫共轭先验。

共轭先验的好处：

```txt
不用复杂积分。
不用采样。
直接用公式更新参数。
```

常见共轭关系：

| Likelihood | Conjugate Prior | Posterior | 例子 |
|---|---|---|---|
| Bernoulli | Beta(a, b) | Beta(a + successes, b + failures) | 硬币偏置、点击率 |
| Normal | Normal | Normal | 传感器校准 |
| Poisson | Gamma | Gamma | 到达率、事件计数 |
| Multinomial | Dirichlet | Dirichlet | 主题模型、语言模型 |

Beta 分布常用于描述一个概率参数。

```text
Beta(a, b)
```

均值：

```text
mean = a / (a + b)
```

特殊情况：

| Beta prior | 含义 |
|---|---|
| Beta(1, 1) | 均匀分布，没有明显先验偏好 |
| Beta(10, 10) | 强烈相信概率接近 0.5 |
| Beta(1, 10) | 相信概率偏小 |

更新规则：

```text
Prior:     Beta(a, b)
Data:      s successes, f failures
Posterior: Beta(a + s, b + f)
```

---

### 3.12 序贯贝叶斯更新

**一句话直觉：**  
今天的后验可以成为明天的先验。

例子：估计硬币正面概率。

Day 1：没有数据。

```text
Prior = Beta(1, 1)
mean = 0.5
```

Day 2：观察到 7 次正面、3 次反面。

```text
Posterior = Beta(1 + 7, 1 + 3) = Beta(8, 4)
mean = 8 / 12 = 0.667
```

Day 3：又观察到 5 次正面、5 次反面。

```text
Posterior = Beta(8 + 5, 4 + 5) = Beta(13, 9)
mean = 13 / 22 = 0.591
```

这种方式的核心是：

```txt
新数据不需要从头训练。
只需要更新当前分布的参数。
```

**AI 连接：**

| 场景 | 序贯更新意义 |
|---|---|
| 在线学习 | 新数据持续到来 |
| bandit | Thompson sampling |
| 推荐系统 | 用户行为不断更新 |
| A/B testing | 每天更新版本胜率 |
| 异常检测 | 持续更新背景分布 |

---

### 3.13 Bayesian A/B Testing

**一句话直觉：**  
Bayesian A/B Testing 直接回答：“B 比 A 更好的概率是多少？”

例子：测试两个按钮颜色的点击率。

数据：

```txt
A：50 clicks / 1000 views
B：65 clicks / 1000 views
```

先验：

```text
A_prior = Beta(1, 1)
B_prior = Beta(1, 1)
```

后验：

```text
A_posterior = Beta(1 + 50, 1 + 950) = Beta(51, 951)
B_posterior = Beta(1 + 65, 1 + 935) = Beta(66, 936)
```

后验均值：

```text
A_mean = 51 / (51 + 951) ≈ 0.051
B_mean = 66 / (66 + 936) ≈ 0.066
```

决策问题：

```txt
P(B > A) 是多少？
```

可以用 Monte Carlo 估计：

```txt
1. 从 Beta(51, 951) 采样 100,000 次，得到 samples_A
2. 从 Beta(66, 936) 采样 100,000 次，得到 samples_B
3. 计算 samples_B > samples_A 的比例
```

如果：

```txt
P(B > A) > 0.95
```

就可以考虑上线 B。

| 维度 | Frequentist A/B | Bayesian A/B |
|---|---|---|
| 输出 | p-value | `P(B > A)` |
| 解释 | 如果 A=B，当前数据有多罕见 | B 真实优于 A 的概率 |
| 提前查看 | 容易引入 peeking problem | 可以持续更新 |
| 先验 | 不直接使用 | 可以编码先验 |
| 决策 | p < 0.05 | `P(B > A)` 超过阈值 |

---

### 3.14 本课核心概念总表

| 概念 | 一句话直觉 | AI 中的对应位置 |
|---|---|---|
| 条件概率 | 已知 B 后 A 的概率 | `P(y|x)` |
| 贝叶斯公式 | 用证据更新信念 | 诊断、分类、不确定性 |
| 先验 | 看数据前的信念 | regularization、小样本学习 |
| 似然 | 假设下看到数据的概率 | model likelihood |
| 后验 | 看数据后的信念 | uncertainty estimation |
| 证据 | 归一化常数 | model comparison |
| 基础率 | 类别本身稀有程度 | class imbalance |
| Naive Bayes | 条件独立分类器 | spam filtering |
| Laplace smoothing | 防止零概率 | text classification |
| MLE | 只用数据估参数 | unregularized training |
| MAP | 数据 + 先验估参数 | regularized training |
| 共轭先验 | 后验和先验同族 | closed-form Bayesian update |
| Beta-Binomial | 成功/失败建模 | CTR、A/B testing |
| 序贯更新 | 后验变先验 | online learning |
| Bayesian A/B | 直接比较版本优劣概率 | experiment decision |

---

## 4. 手写实现：从零实现贝叶斯更新和 Naive Bayes

这一节先不用 scikit-learn。  
目标是理解贝叶斯分类器和 log-space 计算到底在做什么。

---

### 4.1 实现贝叶斯公式

```python
def bayes(prior, likelihood, false_positive_rate):
    evidence = likelihood * prior + false_positive_rate * (1 - prior)
    posterior = likelihood * prior / evidence
    return posterior


result = bayes(
    prior=0.0001,
    likelihood=0.99,
    false_positive_rate=0.01
)

print(f"P(sick|positive) = {result:.4f}")
```

这里：

| 参数 | 含义 |
|---|---|
| `prior` | `P(sick)` |
| `likelihood` | `P(positive|sick)` |
| `false_positive_rate` | `P(positive|healthy)` |
| `posterior` | `P(sick|positive)` |

---

### 4.2 从零实现 Naive Bayes 分类器

```python
import math
from collections import defaultdict


class NaiveBayes:
    def __init__(self, smoothing=1.0):
        self.smoothing = smoothing
        self.class_counts = defaultdict(int)
        self.word_counts = defaultdict(lambda: defaultdict(int))
        self.class_word_totals = defaultdict(int)
        self.vocab = set()

    def train(self, documents, labels):
        for doc, label in zip(documents, labels):
            self.class_counts[label] += 1

            words = doc.lower().split()

            for word in words:
                self.word_counts[label][word] += 1
                self.class_word_totals[label] += 1
                self.vocab.add(word)

    def predict(self, document):
        words = document.lower().split()
        total_docs = sum(self.class_counts.values())
        vocab_size = len(self.vocab)

        best_class = None
        best_score = float("-inf")

        for cls in self.class_counts:
            score = math.log(self.class_counts[cls] / total_docs)

            for word in words:
                count = self.word_counts[cls].get(word, 0)
                total = self.class_word_totals[cls]

                prob = (count + self.smoothing) / (
                    total + self.smoothing * vocab_size
                )

                score += math.log(prob)

            if score > best_score:
                best_score = score
                best_class = cls

        return best_class
```

关键点：

```txt
1. 用 class_counts 估计 P(class)
2. 用 word_counts 估计 P(word|class)
3. 用 Laplace smoothing 避免概率为 0
4. 用 log probability 避免下溢
5. 比较每个类别的 log score
```

---

### 4.3 训练垃圾邮件分类器

```python
train_docs = [
    "win free money now",
    "free lottery ticket winner",
    "claim your prize today free",
    "urgent offer free cash",
    "congratulations you won free",
    "meeting tomorrow at noon",
    "project update attached",
    "can we schedule a call",
    "quarterly report review",
    "lunch on thursday sounds good",
    "team standup notes attached",
    "please review the pull request",
]

train_labels = [
    "spam", "spam", "spam", "spam", "spam",
    "ham", "ham", "ham", "ham", "ham", "ham", "ham",
]

classifier = NaiveBayes()
classifier.train(train_docs, train_labels)

test_messages = [
    "free money waiting for you",
    "meeting rescheduled to friday",
    "you won a free prize",
    "please review the attached report",
]

for msg in test_messages:
    print(f"'{msg}' -> {classifier.predict(msg)}")
```

---

### 4.4 查看模型学到了什么

```python
def show_top_words(classifier, cls, n=5):
    vocab_size = len(classifier.vocab)
    total = classifier.class_word_totals[cls]

    probs = {}

    for word in classifier.vocab:
        count = classifier.word_counts[cls].get(word, 0)

        probs[word] = (
            count + classifier.smoothing
        ) / (
            total + classifier.smoothing * vocab_size
        )

    sorted_words = sorted(probs.items(), key=lambda x: x[1], reverse=True)

    for word, prob in sorted_words[:n]:
        print(f"    {word}: {prob:.4f}")


print("Top spam words:")
show_top_words(classifier, "spam")

print("Top ham words:")
show_top_words(classifier, "ham")
```

这可以帮助你解释模型：

```txt
哪些词最支持 spam？
哪些词最支持 ham？
模型为什么会做出这个分类？
```

---

### 4.5 实现 Beta-Binomial 序贯更新

```python
def beta_update(a, b, successes, failures):
    return a + successes, b + failures


def beta_mean(a, b):
    return a / (a + b)


# Day 1: no data
a, b = 1, 1
print(f"Prior: Beta({a}, {b}), mean={beta_mean(a, b):.3f}")

# Day 2: observe 7 heads, 3 tails
a, b = beta_update(a, b, successes=7, failures=3)
print(f"After day 2: Beta({a}, {b}), mean={beta_mean(a, b):.3f}")

# Day 3: observe 5 heads, 5 tails
a, b = beta_update(a, b, successes=5, failures=5)
print(f"After day 3: Beta({a}, {b}), mean={beta_mean(a, b):.3f}")
```

---

### 4.6 用 Monte Carlo 做 Bayesian A/B Testing

```python
import random


def sample_beta_approx(a, b, n=10000):
    # 教学版：使用 gamma 采样构造 beta 采样
    samples = []

    for _ in range(n):
        x = random.gammavariate(a, 1.0)
        y = random.gammavariate(b, 1.0)
        samples.append(x / (x + y))

    return samples


# A: 50 clicks / 1000 views
# B: 65 clicks / 1000 views

a_A, b_A = 1 + 50, 1 + 950
a_B, b_B = 1 + 65, 1 + 935

samples_A = sample_beta_approx(a_A, b_A)
samples_B = sample_beta_approx(a_B, b_B)

prob_B_better = sum(
    b > a for a, b in zip(samples_A, samples_B)
) / len(samples_A)

print(f"P(B > A) = {prob_B_better:.4f}")
```

---

## 5. 生产使用：用 scikit-learn 完成同样任务

---

### 5.1 scikit-learn 版本 Naive Bayes

```python
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

vectorizer = CountVectorizer()

X_train = vectorizer.fit_transform(train_docs)

clf = MultinomialNB()
clf.fit(X_train, train_labels)

X_test = vectorizer.transform(test_messages)

predictions = clf.predict(X_test)

for msg, pred in zip(test_messages, predictions):
    print(f"'{msg}' -> {pred}")
```

对应关系：

| 手写实现 | scikit-learn |
|---|---|
| `doc.lower().split()` | `CountVectorizer` |
| `vocab` | `vectorizer.vocabulary_` |
| `word_counts` | 稀疏词频矩阵 |
| `Laplace smoothing` | `MultinomialNB(alpha=1.0)` |
| `log score` | 内部 log probability |
| `predict` | `clf.predict` |

---

### 5.2 手写实现 vs 生产库

| 对比项 | 手写 NaiveBayes | scikit-learn MultinomialNB |
|---|---|---|
| 目的 | 理解贝叶斯分类原理 | 工程实践 |
| 分词 | 简单 split | CountVectorizer |
| 特征矩阵 | 手写字典 | 稀疏矩阵 |
| smoothing | 手动实现 | `alpha` 参数 |
| log-space | 手动实现 | 内部已处理 |
| 性能 | 适合小样本教学 | 可用于真实文本分类 |
| 可解释性 | 容易查看词概率 | 可查看 `feature_log_prob_` |

---

## 6. 交付沉淀：产出物、连接关系、练习、术语表

---

### 6.1 本课产出

```txt
code/bayes.py
code/use_sklearn.py
outputs/prompt-bayes-theorem.md
outputs/exercises-solutions.md
```

| 文件 | 作用 |
|---|---|
| `code/bayes.py` | 从零实现 Bayes、NaiveBayes、Laplace smoothing、Beta update、Bayesian A/B |
| `code/use_sklearn.py` | scikit-learn MultinomialNB 生产库版本 |
| `outputs/prompt-bayes-theorem.md` | 用于教学贝叶斯直觉的 prompt |
| `outputs/exercises-solutions.md` | 练习题与参考答案 |

---

### 6.2 知识连接关系

| 本课知识点 | 后续连接 |
|---|---|
| 条件概率 | 贝叶斯公式、生成式分类 |
| 先验 | 正则化、小样本学习 |
| 似然 | 最大似然估计、生成模型 |
| 后验 | 不确定性估计 |
| 证据 | 模型比较、边际似然 |
| Naive Bayes | 文本分类、生成式学习算法 |
| Laplace smoothing | 平滑估计、语言模型 |
| MLE | 监督学习训练目标 |
| MAP | 正则化、贝叶斯学习 |
| 共轭先验 | 贝叶斯闭式更新 |
| Beta-Binomial | CTR、A/B 测试 |
| 序贯更新 | 在线学习 |
| Bayesian A/B | 实验决策 |

---

### 6.3 AI 应用连接

| 数学知识点 | AI 应用 |
|---|---|
| 贝叶斯公式 | 概率更新、诊断、分类 |
| 基础率 | 异常检测、医疗 AI、欺诈识别 |
| Naive Bayes | 垃圾邮件过滤、文本分类 |
| log probability | 稳定计算长文本概率 |
| Laplace smoothing | 处理未见词 |
| MLE | 无正则化模型训练 |
| MAP | L1 / L2 正则化 |
| Beta-Binomial | 点击率建模、A/B 测试 |
| 序贯更新 | 在线推荐、流式学习 |
| Bayesian A/B | 产品实验决策 |

---

### 6.4 练习

1. **多次测试。**  
   一个病人连续两次独立测试阳性。两次测试准确率都是 99%，疾病患病率是 1 / 10,000。  
   第一次测试后的 posterior 作为第二次测试的 prior，计算两次阳性后真正患病的概率。

2. **观察 smoothing 影响。**  
   分别使用 smoothing = `0.01`、`0.1`、`1.0`、`10.0` 训练 spam classifier。  
   观察 top words 的概率如何变化。  
   再试试 smoothing = `0`，看看遇到只在 ham 中出现的词会发生什么。

3. **加入消息长度特征。**  
   扩展 `NaiveBayes` 类，把消息长度分成 `short` 和 `long`，并估计：

   ```text
   P(short | spam)
   P(short | ham)
   ```

   再把这个特征加入预测分数。

4. **手算 MAP。**  
   观察到 10 次抛硬币中有 7 次正面。使用 `Beta(2, 2)` 先验，计算硬币正面概率的 MAP 估计，并和 MLE `7/10` 比较。

5. **Bayesian A/B Testing。**  
   A 版本：120 clicks / 2000 views。  
   B 版本：150 clicks / 2100 views。  
   用 Beta(1,1) 先验和 Monte Carlo 估计 `P(B > A)`。

---

### 6.5 关键术语

| 术语 | 常见说法 | 真正含义 |
|---|---|---|
| 先验 | 初始猜测 | 看到证据前对假设的概率信念 |
| 似然 | 数据有多匹配 | 某个假设为真时，观察到当前数据的概率 |
| 后验 | 更新后的信念 | 看到证据后对假设的概率信念 |
| 证据 | 归一化常数 | 所有假设下观察到数据的总概率 |
| 贝叶斯公式 | 用证据更新概率 | 把先验和似然结合成后验 |
| 基础率 | 类别本身比例 | 先验概率，罕见事件判断中特别重要 |
| Naive Bayes | 简单文本分类器 | 假设特征在给定类别下条件独立的贝叶斯分类器 |
| Laplace smoothing | 加一平滑 | 给每个特征加伪计数，避免零概率 |
| MLE | 只看数据 | 最大化 `P(data|params)` 的参数估计 |
| MAP | 数据加先验 | 最大化 `P(data|params)P(params)`，等价于带先验的 MLE |
| log probability | log 空间概率 | 用 log 概率相加代替原始概率相乘 |
| 共轭先验 | 更新后同族 | 先验和后验属于同一分布族，更新有闭式解 |
| Beta 分布 | 概率的分布 | 常用于描述点击率、转化率等概率参数 |
| 序贯更新 | 后验变先验 | 新数据来时持续更新信念 |
| Bayesian A/B | 概率式实验决策 | 直接估计一个版本优于另一个版本的概率 |

---

### 6.6 自检问题

学完本课后，你应该能回答：

- 贝叶斯公式最初解决什么问题？
- `P(A|B)` 和 `P(B|A)` 为什么不能混淆？
- prior、likelihood、posterior、evidence 分别是什么？
- 为什么医疗测试阳性不等于 99% 患病概率？
- 基础率为什么重要？
- Naive Bayes 为什么叫 naive？
- 为什么 Naive Bayes 要用 log probability？
- Laplace smoothing 解决了什么问题？
- MLE 和 MAP 有什么区别？
- MAP 为什么和正则化有关？
- Gaussian prior 为什么对应 L2 regularization？
- 什么是共轭先验？
- Beta-Binomial 如何做序贯更新？
- Bayesian A/B Testing 如何计算 `P(B > A)`？
- 手写 Naive Bayes 和 scikit-learn 的 MultinomialNB 有什么对应关系？

---

## 附录：本课最小代码文件建议

如果要拆成代码文件，可以这样组织：

```txt
code/
├── bayes.py
└── use_sklearn.py
```

### `bayes.py`

包含：

```txt
bayes
NaiveBayes
NaiveBayes.train
NaiveBayes.predict
show_top_words
beta_update
beta_mean
sample_beta_approx
bayesian_ab_demo
```

### `use_sklearn.py`

包含：

```txt
CountVectorizer
MultinomialNB
fit
predict
feature_log_prob_
classification_report
```
