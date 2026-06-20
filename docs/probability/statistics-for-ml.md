---
title: 机器学习中的统计学
description: 面向 AI 算法工程师的统计学课程：从描述性统计、相关性、协方差矩阵、假设检验、p-value、置信区间，到 bootstrap、A/B 测试和统计显著性。
---

# 机器学习中的统计学

> 统计学告诉你：模型是真的变好了，还是只是碰巧看起来变好了。

**课程类型：** 实现 / 应用  
**所属模块：** 概率统计 / 实验评估  
**前置知识：** Phase 1 Lesson 06-07：概率与分布、贝叶斯公式  
**预计时间：** 约 120 分钟  
**使用语言：** Python / NumPy / SciPy / scikit-learn  

---

## 1. 提出问题：为什么 AI 算法工程师必须理解统计学

你训练了两个模型：

```txt
Model A 测试集 accuracy = 0.87
Model B 测试集 accuracy = 0.89
```

看起来 Model B 更好，于是你上线了 B。  
三周后，线上指标反而变差了。

可能原因是：

```txt
Model B 并没有真正优于 Model A。
0.02 的差异只是测试集噪声。
测试集太小。
指标方差太大。
模型在某些子群体上变差了。
你只看了 point estimate，没有看置信区间。
```

这类问题在机器学习中非常常见：

```txt
Kaggle leaderboard shakeup
论文无法复现
A/B 测试提前宣布胜利
多个模型只报告最好的一个结果
小样本测试集上的偶然提升
```

统计学的作用就是：

```txt
区分 signal 和 noise。
判断差异是否可靠。
估计结果有多不确定。
避免把随机波动当作模型进步。
```

---

### 1.1 数学概念历史出现缘由

统计学最早是为了解决“如何从有限样本推断总体规律”的问题。  
在机器学习中，它变成了实验评估、模型比较、A/B 测试和不确定性判断的基础。

| 统计学概念 | 出现缘由 |
|---|---|
| 描述性统计 | 为了用少数数字概括数据分布形状 |
| 均值 / 中位数 / 众数 | 为了描述数据的中心位置 |
| 方差 / 标准差 | 为了描述数据的离散程度 |
| 分位数 / IQR | 为了稳健描述分布和尾部 |
| 样本统计 vs 总体统计 | 为了区分有限样本估计和真实总体参数 |
| Bessel correction | 为了修正样本方差低估总体方差的问题 |
| 相关性 | 为了描述两个变量如何一起变化 |
| 协方差矩阵 | 为了描述多维特征之间的线性关系 |
| 假设检验 | 为了在不确定性下做决策 |
| p-value | 为了衡量在零假设成立时，观察到当前极端结果的概率 |
| 置信区间 | 为了给参数估计提供不确定性范围 |
| t-test | 为了比较均值是否显著不同 |
| chi-squared test | 为了比较分类频数是否偏离期望 |
| effect size | 为了区分“显著”与“有意义” |
| 多重比较修正 | 为了避免大量测试中偶然显著 |
| bootstrap | 为了不用分布假设也能估计统计量不确定性 |
| 参数检验 / 非参数检验 | 为了适应不同数据分布和样本规模 |
| 中心极限定理 | 为了解释样本均值为何近似正态 |
| A/B 测试 | 为了比较两个系统或模型在线上的真实效果 |

可以这样理解：

```txt
现实问题：数据的典型水平是多少？
统计工具：均值、中位数、众数

现实问题：数据波动有多大？
统计工具：方差、标准差、IQR、分位数

现实问题：两个变量是否一起变化？
统计工具：相关性和协方差

现实问题：模型 B 比模型 A 好，是不是只是运气？
统计工具：假设检验、置信区间、bootstrap

现实问题：差异虽然显著，但是否值得上线？
统计工具：effect size 和 practical significance

现实问题：测试很多指标后总有几个显著，怎么办？
统计工具：多重比较修正
```

---

### 1.2 原始问题是什么

本课要解决的原始问题是：

```txt
如何概括一个数据集的中心、波动、尾部和异常值？
如何判断两个变量是否相关？
如何构造协方差矩阵并理解特征之间的关系？
如何判断两个模型的指标差异是否可靠？
p-value 到底是什么意思，不能怎么解释？
置信区间到底表示什么？
什么时候用 t-test？
什么时候用 chi-squared test？
如何判断统计显著性和实际价值之间的区别？
多重比较为什么会制造假阳性？
bootstrap 如何给任意指标构造置信区间？
参数检验和非参数检验如何选择？
中心极限定理在 ML 实验中有什么实际意义？
```

换成 AI 语言，就是：

```txt
0.89 accuracy 是否真的比 0.87 accuracy 好？
测试集上的 2% 提升是否可能只是噪声？
A/B 测试什么时候可以宣布 winner？
模型指标应该只报一个数，还是要报 confidence interval？
为什么 cross-validation 应该比较 paired scores？
为什么 accuracy 在类别极不平衡时会误导？
为什么论文里应该报告 effect size？
为什么 hyperparameter search 后不能只报告最好的 p-value？
```

---

### 1.3 AI 中的现代问题

| 统计学知识点 | AI 中的现代问题 |
|---|---|
| 均值 / 中位数 | 平均 loss、平均 latency、典型用户体验 |
| P95 / P99 | 推理延迟、tail latency、安全风险 |
| 方差 / 标准差 | 模型指标波动、训练稳定性 |
| IQR | 稳健异常值检测 |
| Pearson correlation | 线性特征关系 |
| Spearman correlation | 单调关系、rank 指标 |
| 协方差矩阵 | PCA、特征相关性、Mahalanobis |
| p-value | 模型差异显著性 |
| 置信区间 | 指标不确定性 |
| t-test | 比较均值、cross-validation fold scores |
| paired t-test | 同一数据切分上比较两个模型 |
| chi-squared test | 分类频数、分布偏差 |
| bootstrap | 任意 metric 的置信区间 |
| effect size | 判断差异是否值得上线 |
| 多重比较 | 多模型、多指标、多数据集评估 |
| A/B testing | 在线实验和产品决策 |
| 非参数检验 | 小样本、非正态、ordinal data |
| CLT | 平均指标、mini-batch gradient、ensemble |

---

### 1.4 本课要解决的核心问题

学完本课，你应该能够回答：

1. 均值、中位数、众数分别适合描述什么？
2. 方差、标准差、IQR、分位数分别有什么用途？
3. 样本方差为什么要除以 `n - 1`？
4. Pearson 和 Spearman 相关有什么区别？
5. 协方差矩阵如何连接 PCA？
6. 假设检验中的 H0 和 H1 是什么？
7. p-value 到底是什么意思？
8. 置信区间应该如何解释？
9. one-sample、two-sample、paired t-test 分别适合什么场景？
10. chi-squared test 适合什么问题？
11. ML 模型 A/B 测试应该如何设计？
12. 统计显著和实际显著有什么区别？
13. effect size 如何衡量差异大小？
14. 多重比较为什么会带来假阳性？
15. bootstrap 为什么适合 ML 指标置信区间？
16. 参数检验和非参数检验如何选择？
17. CLT 在 ML 中有哪些实际含义？
18. ML 论文和实验中最常见的统计错误有哪些？

---

## 2. 概念定位：这节课学什么

---

### 2.1 所属模块

```txt
所属模块：概率统计 / 实验评估
前置知识：概率、分布、期望、方差、贝叶斯思想
后续连接：模型评估、A/B testing、cross-validation、bootstrap、显著性检验、实验设计
```

前面的概率课程解决了：

```txt
如何描述不确定性？
```

本课进一步解决：

```txt
如何从有限样本中判断一个结论是否可靠？
```

---

### 2.2 本课学习目标

完成本课后，你应该能够：

- 从零实现均值、中位数、众数、方差、标准差、分位数和 IQR
- 从零实现 Pearson 和 Spearman correlation
- 从零计算协方差矩阵
- 正确解释 p-value 和 confidence interval
- 实现 one-sample t-test、two-sample t-test 和 paired t-test 的核心统计量
- 实现 chi-squared test 的统计量
- 使用 bootstrap 为任意 metric 构造置信区间
- 区分 statistical significance 和 practical significance
- 计算 Cohen's d 等 effect size
- 理解多重比较和 Bonferroni correction
- 判断什么时候应该使用非参数检验
- 识别 ML 实验中常见统计错误

---

### 2.3 本课在整体路线中的位置

```txt
概率与分布
    ↓
贝叶斯公式
    ↓
统计学与实验评估
    ↓
模型比较 / A/B testing / bootstrap
    ↓
可靠机器学习实验
    ↓
生产模型上线决策
```

本课的核心能力是：

```txt
看到两个模型指标差异时，能判断它是否可靠、是否有实际价值、是否足以支持上线。
```

---

## 3. 理解概念：直觉、公式、统计解释、AI 连接

---

### 3.1 描述性统计：先看清数据长什么样

**一句话直觉：**  
描述性统计用少数几个数字概括数据的中心、波动和尾部。

在建模前，你至少应该知道：

```txt
数据中心在哪里？
数据有多分散？
是否有 outlier？
分布是否偏斜？
尾部是否很重？
```

**AI 连接：**

| 数据 | 应看统计量 |
|---|---|
| loss distribution | mean、median、P95 |
| inference latency | P50、P95、P99 |
| prediction confidence | histogram、percentiles |
| error distribution | mean、std、tail errors |
| feature distribution | mean、std、IQR、outliers |

---

### 3.2 中心趋势：mean、median、mode

**一句话直觉：**  
mean 是平衡点，median 是中间位置，mode 是最常见值。

均值：

```text
mean = (1/n) * sum_i x_i
```

优点：

```txt
使用全部数据。
适合对称分布。
```

缺点：

```txt
对 outlier 非常敏感。
```

中位数：

```txt
排序后位于中间的值。
```

优点：

```txt
对 outlier 稳健。
```

例子：

```txt
[1, 2, 3, 4, 1000]
mean = 202
median = 3
```

众数：

```txt
出现次数最多的值。
```

适合：

```txt
categorical data
离散类别
最常见标签
```

**AI 连接：**

| 场景 | 推荐 |
|---|---|
| 平均 loss | mean |
| 延迟典型值 | median / P50 |
| 类别最多标签 | mode |
| heavy-tailed error | median 更稳健 |
| income / spend data | median 通常比 mean 更有代表性 |

---

### 3.3 离散程度：variance、standard deviation、range、IQR

**一句话直觉：**  
离散程度告诉我们数据围绕中心波动有多大。

总体方差：

```text
sigma^2 = (1/N) * sum_i (x_i - mu)^2
```

样本方差：

```text
s^2 = (1/(n-1)) * sum_i (x_i - x_bar)^2
```

标准差：

```text
s = sqrt(s^2)
```

标准差和原数据单位相同，因此更容易解释。

Range：

```text
max - min
```

但它对 outlier 很敏感。

IQR：

```text
IQR = Q3 - Q1
```

表示中间 50% 数据的范围，对 outlier 更稳健。

**AI 连接：**

| 指标 | 用途 |
|---|---|
| variance | 训练波动、模型不稳定性 |
| std | 指标不确定性、置信区间 |
| range | 快速查看极端差异 |
| IQR | 稳健异常值检测 |
| P95 / P99 | tail latency 和极端错误 |

---

### 3.4 分位数与 Tail Metrics

**一句话直觉：**  
分位数告诉你有多少比例的数据低于某个值。

常见分位数：

| 分位数 | 含义 |
|---|---|
| P50 | 中位数，50% 数据低于该值 |
| P90 | 90% 数据低于该值 |
| P95 | 95% 数据低于该值 |
| P99 | 99% 数据低于该值 |

在系统监控中：

```txt
P50 latency：典型用户体验
P95 latency：较差用户体验
P99 latency：尾部延迟，常常决定线上稳定性
```

**AI 连接：**

一个模型可能平均 latency 很低，但 P99 极差。  
在生产系统里，这可能比平均值更重要。

---

### 3.5 样本统计 vs 总体统计

**一句话直觉：**  
总体参数是真实但通常未知的；样本统计量是我们用有限数据估计出来的。

总体方差：

```text
sigma^2 = (1/N) * sum_i (x_i - mu)^2
```

样本方差：

```text
s^2 = (1/(n-1)) * sum_i (x_i - x_bar)^2
```

为什么样本方差除以 `n - 1`？

```txt
样本均值 x_bar 本身是从样本估计出来的。
如果除以 n，会系统性低估总体方差。
除以 n - 1 是 Bessel correction，可以得到无偏估计。
```

**AI 连接：**

| 场景 | 影响 |
|---|---|
| 小测试集 | 方差估计差异明显 |
| 大样本 | n 和 n-1 差异很小 |
| cross-validation | fold 数少时不确定性大 |
| bootstrap | 用样本近似总体不确定性 |

---

### 3.6 Pearson correlation：线性相关

**一句话直觉：**  
Pearson correlation 衡量两个变量之间的线性关系强度和方向。

公式：

```text
r = cov(X, Y) / (std(X) * std(Y))
```

范围：

```txt
+1：完美正线性关系
-1：完美负线性关系
 0：无线性关系
```

注意：

```txt
r = 0 不代表没有关系，只代表没有线性关系。
```

Pearson 的限制：

- 假设关系大致线性
- 对 outlier 敏感
- 不表示因果关系

**AI 连接：**

| 场景 | Pearson 用途 |
|---|---|
| 特征线性关系 | feature correlation |
| 多重共线性检查 | correlated features |
| 模型预测 vs 真实值 | 回归评估 |
| embedding dimensions | 线性依赖分析 |

---

### 3.7 Spearman correlation：秩相关

**一句话直觉：**  
Spearman correlation 先把值变成排名，再计算 Pearson，因此衡量单调关系。

步骤：

```txt
1. 把每个变量的值替换成 rank。
2. 对 rank 计算 Pearson correlation。
```

适合：

- ordinal data
- 非正态数据
- 单调但非线性关系
- 存在 outlier 的情况

例子：

```txt
y = x^3
```

这是严格单调关系。  
Pearson 可能小于 1，但 Spearman 可以接近 1。

**AI 连接：**

| 场景 | Spearman 用途 |
|---|---|
| ranking metrics | 排序一致性 |
| rating data | 用户评分 |
| non-linear monotonic relation | 单调关系检测 |
| 模型排名能力 | rank correlation |

---

### 3.8 协方差矩阵

**一句话直觉：**  
协方差矩阵记录所有特征之间如何一起变化。

两个变量协方差：

```text
Cov(X, Y) = (1/n) * sum_i (x_i - x_bar)(y_i - y_bar)
```

多维数据的协方差矩阵：

```text
C[i][j] = Cov(feature_i, feature_j)
```

结构：

```text
C =
| Var(x1)      Cov(x1,x2)  Cov(x1,x3) |
| Cov(x2,x1)  Var(x2)      Cov(x2,x3) |
| Cov(x3,x1)  Cov(x3,x2)  Var(x3)     |
```

性质：

```txt
对称矩阵。
对角线是各特征方差。
非对角线是特征之间协方差。
半正定，特征值非负。
```

**AI 连接：**

| 场景 | 协方差矩阵作用 |
|---|---|
| PCA | 对协方差矩阵做特征分解 |
| Mahalanobis distance | 用协方差校正距离 |
| 特征相关性 | 发现冗余特征 |
| Gaussian model | 描述多元正态分布形状 |

---

### 3.9 假设检验：在不确定性下做判断

**一句话直觉：**  
假设检验判断观测数据是否足以推翻“没有效果”的默认假设。

基本结构：

```txt
H0：Null hypothesis，零假设，通常表示没有差异或没有效果。
H1：Alternative hypothesis，备择假设，表示你想证明的效果。
```

例子：

```txt
H0：Model A 和 Model B 准确率相同。
H1：Model B 准确率更高。
```

流程：

```txt
1. 定义 H0 和 H1。
2. 选择统计检验方法。
3. 计算 test statistic。
4. 计算 p-value。
5. 与 alpha 比较，例如 alpha = 0.05。
6. 决定是否 reject H0。
```

---

### 3.10 p-value：最容易被误解的统计量

**一句话直觉：**  
p-value 是在零假设成立时，观察到当前这么极端或更极端数据的概率。

公式化理解：

```text
p-value = P(data this extreme | H0 is true)
```

它不是：

```txt
H0 为真的概率。
结果由随机产生的概率。
模型 B 没有效果的概率。
```

判断：

```txt
p < alpha：reject H0，结果统计显著。
p >= alpha：fail to reject H0，证据不足。
```

注意：

```txt
fail to reject H0 不等于证明 H0 为真。
```

**AI 连接：**

一个模型提升 p-value 很小，说明差异不太可能由 H0 下的随机波动解释。  
但它仍不告诉你这个提升是否值得上线。

---

### 3.11 Confidence Interval：估计的不确定范围

**一句话直觉：**  
置信区间给出参数的一个合理范围，而不是只给一个点估计。

均值的 95% 置信区间：

```text
x_bar ± 1.96 * s / sqrt(n)
```

更一般地：

```text
estimate ± critical_value * standard_error
```

正确解释：

```txt
如果重复实验很多次，构造出的 95% 置信区间中，大约 95% 会包含真实参数。
```

常见误解：

```txt
不能说“真实参数有 95% 概率在这个具体区间里”。
```

**AI 连接：**

模型指标建议报告：

```txt
accuracy = 0.89
95% CI = [0.86, 0.92]
```

而不是只报告：

```txt
accuracy = 0.89
```

---

### 3.12 t-test：比较均值

**一句话直觉：**  
t-test 用来判断一个或两个均值之间的差异是否显著。

#### One-sample t-test

用于比较样本均值和某个假设均值：

```text
t = (x_bar - mu_0) / (s / sqrt(n))
```

自由度：

```text
df = n - 1
```

#### Two-sample t-test

用于比较两个独立样本均值。  
实践中通常推荐 Welch's t-test，因为它不要求两个组方差相等：

```text
t = (mean_1 - mean_2) / sqrt(s1^2/n1 + s2^2/n2)
```

#### Paired t-test

用于成对数据，例如两个模型在同一批 cross-validation folds 上的结果：

```txt
先计算每对差异 d_i = b_i - a_i。
再对 d_i 做 one-sample t-test，检验均值是否为 0。
```

**AI 连接：**

| 场景 | 推荐检验 |
|---|---|
| 单模型指标是否超过 baseline | one-sample t-test |
| 两个独立实验组 | two-sample t-test |
| 两个模型在同样 folds 上比较 | paired t-test |
| 小样本且非正态 | Wilcoxon signed-rank |

---

### 3.13 Chi-squared test：比较分类频数

**一句话直觉：**  
chi-squared test 判断观察到的分类频数是否偏离期望频数。

统计量：

```text
chi^2 = sum((observed - expected)^2 / expected)
```

例子：

| 类别 | Observed | Expected |
|---|---:|---:|
| Positive | 120 | 100 |
| Negative | 80 | 100 |

计算：

```text
chi^2 = (120-100)^2/100 + (80-100)^2/100
      = 4 + 4
      = 8
```

**AI 连接：**

| 场景 | chi-squared 用途 |
|---|---|
| 分类输出分布是否偏移 | prediction distribution shift |
| 语言模型类别分布 | output category distribution |
| A/B 测试转化频数 | categorical count comparison |
| 数据漂移 | label distribution drift |

---

### 3.14 机器学习中的 A/B 测试

**一句话直觉：**  
模型 A/B 测试不是只看谁的指标更高，而是要判断差异是否可靠、是否值得上线。

可靠比较需要：

```txt
同一个测试集。
同样的数据切分。
多个指标。
方差估计。
置信区间。
效应大小。
防止数据泄漏。
```

模型比较流程：

```txt
1. 定义 metric 和 alpha，例如 alpha = 0.05。
2. 在同一 k-fold cross-validation splits 上跑两个模型。
3. 得到 paired scores：[(a1,b1), ..., (ak,bk)]。
4. 计算每折差异 d_i = b_i - a_i。
5. 对 d_i 做 paired t-test 或 bootstrap。
6. 给出 mean difference 和 confidence interval。
7. 计算 effect size。
8. 决定是否上线。
```

**AI 连接：**

如果两个模型不是在同一测试集上评估，比较就不可靠。  
如果 test set 曾经参与模型选择，也会产生偏差。

---

### 3.15 Statistical significance vs practical significance

**一句话直觉：**  
统计显著说明差异不像随机噪声；实际显著说明差异大到值得关心。

例子：

```txt
Model A accuracy = 0.9234
Model B accuracy = 0.9237
n = 1,000,000
p-value = 0.001
```

统计显著：

```txt
是。
```

实际显著：

```txt
未必。0.03% 的提升可能不值得上线成本和风险。
```

Effect size 衡量差异大小：

```text
Cohen's d = (mean_1 - mean_2) / pooled_std
```

经验解释：

| Cohen's d | 效应大小 |
|---:|---|
| 0.2 | small |
| 0.5 | medium |
| 0.8 | large |

**AI 连接：**

上线模型不应只看 p-value。  
还要看：

```txt
业务收益
延迟成本
计算成本
稳定性
公平性影响
维护成本
```

---

### 3.16 Multiple Comparison Problem

**一句话直觉：**  
测试越多，偶然显著的概率越高。

如果你做 `m` 次独立检验，每次 alpha = 0.05，那么至少一次假阳性的概率：

```text
P(at least one false positive) = 1 - (1 - alpha)^m
```

当：

```txt
m = 20
alpha = 0.05
```

有：

```text
1 - 0.95^20 ≈ 0.64
```

也就是说，即使所有效果都是假的，也有 64% 概率至少出现一个“显著结果”。

Bonferroni correction：

```text
adjusted alpha = alpha / m
```

例如：

```text
0.05 / 20 = 0.0025
```

**AI 连接：**

多重比较常见于：

- 同时比较很多模型
- 测试很多超参数
- 同时报告很多指标
- 多个数据集上寻找胜利点
- 反复看 A/B test 中间结果

---

### 3.17 Bootstrap：不依赖分布假设的置信区间

**一句话直觉：**  
Bootstrap 通过有放回重采样，估计任意统计量的不确定性。

算法：

```txt
1. 原始数据有 n 个样本。
2. 有放回采样 n 个样本，构成一个 bootstrap sample。
3. 在这个样本上计算统计量。
4. 重复 B 次，例如 1000 到 10000 次。
5. 得到统计量的经验分布。
```

Percentile bootstrap CI：

```txt
95% CI = bootstrap statistics 的 2.5% 和 97.5% 分位数
```

为什么适合 ML？

```txt
很多 ML 指标没有简单闭式置信区间。
例如 AUC、F1、precision@k、median latency。
Bootstrap 可以直接估计。
```

模型比较 bootstrap：

```txt
1. 同一测试集上有 Model A 和 B 的预测。
2. 每次 bootstrap 重采样测试样本 index。
3. 分别计算 metric_A 和 metric_B。
4. 记录 diff = metric_B - metric_A。
5. 看 diff 的 95% CI 是否包含 0。
```

---

### 3.18 参数检验 vs 非参数检验

**一句话直觉：**  
参数检验假设数据来自某类分布；非参数检验不强依赖分布假设。

参数检验：

| 方法 | 假设 |
|---|---|
| t-test | 数据或均值近似正态 |
| ANOVA | 正态性和方差齐性 |
| Pearson correlation | 线性关系、近似正态 |

非参数检验：

| 方法 | 替代 |
|---|---|
| Mann-Whitney U | independent t-test |
| Wilcoxon signed-rank | paired t-test |
| Spearman rho | Pearson |
| Kruskal-Wallis | ANOVA |

什么时候用非参数？

```txt
样本很小，例如 n < 30。
数据明显非正态。
有 heavy outliers。
数据是 ordinal。
分布极度偏斜。
```

**AI 连接：**

很多 ML cross-validation 只有 5 或 10 个 folds。  
此时 paired t-test 的正态性假设可能较弱，Wilcoxon signed-rank 更稳健。

---

### 3.19 Central Limit Theorem：为什么平均值常常近似正态

**一句话直觉：**  
只要样本独立同分布且方差有限，样本均值会随着样本量增加而趋近正态分布。

形式：

```text
X_bar ~ Normal(mu, sigma^2 / n)
```

这解释了：

```txt
为什么均值的置信区间可以用正态近似。
为什么 t-test 在样本量较大时可用。
为什么 mini-batch gradient 是 true gradient 的 noisy estimate。
为什么 ensemble 平均通常更稳定。
```

CLT 不意味着：

```txt
原始数据变成正态。
相关数据也自动满足。
重尾且方差无穷的数据也满足。
```

**AI 连接：**

| 场景 | CLT 作用 |
|---|---|
| 平均 accuracy | 近似正态推断 |
| cross-validation mean | 稳定估计 |
| mini-batch gradient | 近似 true gradient |
| ensemble | 平均降低方差 |
| bootstrap | 估计采样分布 |

---

### 3.20 ML 实验中的常见统计错误

| 错误 | 后果 |
|---|---|
| 在训练集上测试 | 高估性能 |
| 不报告置信区间 | 结果不可判断可靠性 |
| 忽略多重比较 | 假阳性膨胀 |
| 只看统计显著 | 可能上线无实际价值改动 |
| 类别不平衡只看 accuracy | 指标严重误导 |
| cherry-picking metrics | 只报告赢的指标 |
| 数据泄漏 | 测试结果虚高 |
| 小测试集无方差估计 | 把噪声当提升 |
| 样本不独立却当独立 | 低估不确定性 |
| p-hacking | 结果不可复现 |

---

### 3.21 本课核心概念总表

| 概念 | 一句话直觉 | AI 中的对应位置 |
|---|---|---|
| Mean | 平均水平 | 平均 loss / accuracy |
| Median | 中间值 | latency P50、稳健中心 |
| Variance / Std | 波动大小 | 训练稳定性、指标波动 |
| Percentile | 尾部表现 | P95 / P99 latency |
| IQR | 中间 50% 范围 | 稳健异常检测 |
| Pearson | 线性相关 | feature correlation |
| Spearman | 单调相关 | rank correlation |
| Covariance matrix | 特征共同变化 | PCA、Mahalanobis |
| Hypothesis test | 证据是否足够 | 模型比较 |
| p-value | H0 下当前结果极端程度 | 显著性判断 |
| Confidence interval | 参数范围估计 | 指标不确定性 |
| t-test | 均值比较 | cross-validation scores |
| Chi-squared | 频数比较 | 分类分布漂移 |
| Effect size | 差异大小 | practical significance |
| Bootstrap | 重采样估计不确定性 | 任意 metric CI |
| Bonferroni | 多重比较修正 | 多模型多指标 |
| CLT | 均值趋近正态 | CI、mini-batch gradient |

---

## 4. 手写实现：从零实现统计量、相关性、检验和 bootstrap

这一节先不用 NumPy / SciPy。  
目标是理解统计方法的核心计算逻辑。

---

### 4.1 描述性统计

```python
import math
from collections import Counter


def mean(xs):
    return sum(xs) / len(xs)


def median(xs):
    xs = sorted(xs)
    n = len(xs)
    mid = n // 2

    if n % 2 == 1:
        return xs[mid]

    return (xs[mid - 1] + xs[mid]) / 2


def mode(xs):
    counts = Counter(xs)
    max_count = max(counts.values())

    return [x for x, c in counts.items() if c == max_count]


def variance(xs, sample=True):
    mu = mean(xs)
    denom = len(xs) - 1 if sample else len(xs)

    return sum((x - mu) ** 2 for x in xs) / denom


def std(xs, sample=True):
    return math.sqrt(variance(xs, sample=sample))
```

---

### 4.2 Percentile 和 IQR

```python
def percentile(xs, q):
    xs = sorted(xs)

    if not xs:
        raise ValueError("empty data")

    pos = (len(xs) - 1) * q / 100
    lower = math.floor(pos)
    upper = math.ceil(pos)

    if lower == upper:
        return xs[int(pos)]

    weight = pos - lower

    return xs[lower] * (1 - weight) + xs[upper] * weight


def iqr(xs):
    return percentile(xs, 75) - percentile(xs, 25)
```

---

### 4.3 Pearson 与 Spearman correlation

```python
def covariance(x, y, sample=True):
    mx = mean(x)
    my = mean(y)
    denom = len(x) - 1 if sample else len(x)

    return sum((a - mx) * (b - my) for a, b in zip(x, y)) / denom


def pearson_corr(x, y):
    return covariance(x, y) / (std(x) * std(y))


def ranks(xs):
    sorted_pairs = sorted((value, i) for i, value in enumerate(xs))
    result = [0] * len(xs)

    for rank, (_, i) in enumerate(sorted_pairs, start=1):
        result[i] = rank

    return result


def spearman_corr(x, y):
    return pearson_corr(ranks(x), ranks(y))
```

---

### 4.4 协方差矩阵

```python
def covariance_matrix(X):
    # X: list of rows, shape = n_samples x n_features
    n_features = len(X[0])
    columns = [
        [row[j] for row in X]
        for j in range(n_features)
    ]

    return [
        [
            covariance(columns[i], columns[j])
            for j in range(n_features)
        ]
        for i in range(n_features)
    ]
```

---

### 4.5 t-test 的核心统计量

```python
def one_sample_t_stat(xs, mu0):
    n = len(xs)
    xbar = mean(xs)
    s = std(xs, sample=True)

    return (xbar - mu0) / (s / math.sqrt(n))


def welch_t_stat(x, y):
    nx = len(x)
    ny = len(y)

    mx = mean(x)
    my = mean(y)

    vx = variance(x, sample=True)
    vy = variance(y, sample=True)

    return (mx - my) / math.sqrt(vx / nx + vy / ny)


def paired_t_stat(x, y):
    diffs = [a - b for a, b in zip(x, y)]

    return one_sample_t_stat(diffs, mu0=0.0)
```

说明：

```txt
这里先实现 t statistic。
实际 p-value 通常使用 scipy.stats 计算 t-distribution CDF。
```

---

### 4.6 Chi-squared statistic

```python
def chi_squared_stat(observed, expected):
    return sum(
        (o - e) ** 2 / e
        for o, e in zip(observed, expected)
    )


observed = [120, 80]
expected = [100, 100]

print(chi_squared_stat(observed, expected))
```

---

### 4.7 Cohen's d

```python
def cohens_d(x, y):
    nx = len(x)
    ny = len(y)

    sx2 = variance(x, sample=True)
    sy2 = variance(y, sample=True)

    pooled_var = ((nx - 1) * sx2 + (ny - 1) * sy2) / (nx + ny - 2)
    pooled_std = math.sqrt(pooled_var)

    return (mean(x) - mean(y)) / pooled_std
```

---

### 4.8 Bootstrap confidence interval

```python
import random


def bootstrap_ci(data, statistic_fn, n_bootstrap=1000, confidence=0.95):
    stats = []
    n = len(data)

    for _ in range(n_bootstrap):
        sample = [
            data[random.randrange(n)]
            for _ in range(n)
        ]

        stats.append(statistic_fn(sample))

    stats.sort()

    lower_q = (1 - confidence) / 2
    upper_q = 1 - lower_q

    lower = stats[int(lower_q * n_bootstrap)]
    upper = stats[int(upper_q * n_bootstrap)]

    return lower, upper
```

用于模型差异：

```python
def bootstrap_metric_diff(y_true, pred_a, pred_b, metric_fn, n_bootstrap=1000):
    n = len(y_true)
    diffs = []

    for _ in range(n_bootstrap):
        idxs = [random.randrange(n) for _ in range(n)]

        yt = [y_true[i] for i in idxs]
        pa = [pred_a[i] for i in idxs]
        pb = [pred_b[i] for i in idxs]

        diff = metric_fn(yt, pb) - metric_fn(yt, pa)
        diffs.append(diff)

    diffs.sort()

    return (
        diffs[int(0.025 * n_bootstrap)],
        diffs[int(0.975 * n_bootstrap)],
    )
```

---

## 5. 生产使用：NumPy / SciPy / scikit-learn 中的统计评估

---

### 5.1 NumPy 描述性统计

```python
import numpy as np

xs = np.array([1, 2, 3, 4, 1000])

print(np.mean(xs))
print(np.median(xs))
print(np.std(xs, ddof=1))
print(np.percentile(xs, [25, 50, 75]))
```

---

### 5.2 SciPy 统计检验

```python
from scipy import stats

model_a_scores = [0.85, 0.86, 0.87, 0.86, 0.88]
model_b_scores = [0.86, 0.87, 0.89, 0.88, 0.90]

# paired t-test
result = stats.ttest_rel(model_b_scores, model_a_scores)

print(result.statistic)
print(result.pvalue)
```

Welch's t-test：

```python
stats.ttest_ind(model_b_scores, model_a_scores, equal_var=False)
```

Chi-squared test：

```python
observed = [120, 80]
expected = [100, 100]

stats.chisquare(f_obs=observed, f_exp=expected)
```

---

### 5.3 Bootstrap accuracy confidence interval

```python
import numpy as np
from sklearn.metrics import accuracy_score

def bootstrap_accuracy_ci(y_true, y_pred, n_bootstrap=2000):
    n = len(y_true)
    scores = []

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    for _ in range(n_bootstrap):
        idx = np.random.randint(0, n, size=n)
        scores.append(accuracy_score(y_true[idx], y_pred[idx]))

    return np.percentile(scores, [2.5, 97.5])
```

---

### 5.4 模型比较建议流程

```txt
1. 固定最终 test set，不参与模型选择。
2. 在同一数据切分上评估模型 A 和 B。
3. 选择主要 metric，例如 AUC、F1、accuracy、latency。
4. 计算 point estimate。
5. 用 bootstrap 计算 confidence interval。
6. 对 metric difference 计算 confidence interval。
7. 报告 effect size。
8. 同时检查 practical significance。
9. 多指标或多模型比较时做多重比较修正。
10. 再决定是否上线。
```

---

### 5.5 手写实现 vs 生产库

| 对比项 | 手写实现 | NumPy / SciPy / sklearn |
|---|---|---|
| 目的 | 理解统计量和检验原理 | 工程实践 |
| 描述性统计 | 手写函数 | NumPy / pandas |
| 相关性 | Pearson / Spearman 手写 | scipy.stats |
| 协方差矩阵 | 手写矩阵 | np.cov |
| t-test | 手写 t statistic | scipy.stats.ttest_* |
| chi-squared | 手写 statistic | scipy.stats.chisquare |
| bootstrap | 手写 resampling | 自定义或 sklearn utilities |
| ML metric | 手写 accuracy 等 | sklearn.metrics |
| 生产报告 | 需手动整理 | notebook / report / dashboard |

---

## 6. 交付沉淀：产出物、连接关系、练习、术语表

---

### 6.1 本课产出

```txt
code/statistics.py
code/use_scipy.py
code/model_comparison.py
outputs/prompt-statistical-evaluation.md
outputs/exercises-solutions.md
```

| 文件 | 作用 |
|---|---|
| `code/statistics.py` | 从零实现描述性统计、相关性、协方差、t statistic、chi-square、bootstrap |
| `code/use_scipy.py` | SciPy 统计检验示例 |
| `code/model_comparison.py` | 模型比较、bootstrap CI、effect size |
| `outputs/prompt-statistical-evaluation.md` | 用于模型评估统计审查的 prompt |
| `outputs/exercises-solutions.md` | 练习题与参考答案 |

---

### 6.2 知识连接关系

| 本课知识点 | 后续连接 |
|---|---|
| 描述性统计 | 数据理解、EDA |
| 分位数 | latency、tail risk |
| 相关性 | feature analysis |
| 协方差矩阵 | PCA、Mahalanobis |
| 假设检验 | 模型比较 |
| p-value | 显著性判断 |
| 置信区间 | 指标不确定性 |
| t-test | cross-validation comparison |
| chi-squared | 分类分布漂移 |
| effect size | 上线决策 |
| bootstrap | 任意 metric CI |
| 多重比较 | 超参搜索、论文实验 |
| CLT | mini-batch、ensemble、CI |

---

### 6.3 AI 应用连接

| 统计知识点 | AI 应用 |
|---|---|
| Mean / Median | loss、latency、指标汇总 |
| P95 / P99 | 推理尾部延迟 |
| Variance / Std | 指标波动和训练稳定性 |
| Correlation | 特征分析 |
| Covariance | PCA、异常检测 |
| t-test | 模型均值比较 |
| Chi-squared | label distribution shift |
| Bootstrap | accuracy / F1 / AUC CI |
| Effect size | 判断是否值得上线 |
| Multiple comparison | 多模型多指标评估 |
| Non-parametric tests | 小样本 fold comparison |
| CLT | mini-batch gradient、ensemble |

---

### 6.4 练习

1. **描述性统计对 outlier 的敏感性。**  
   对数据：

   ```txt
   [1, 2, 3, 4, 1000]
   ```

   计算 mean、median、std、IQR。解释为什么 mean 和 std 被 outlier 严重影响。

2. **Pearson vs Spearman。**  
   构造：

   ```text
   y = x^3
   ```

   的数据，分别计算 Pearson 和 Spearman。解释为什么 Spearman 更接近 1。

3. **模型 paired t-test。**  
   给定两个模型在 10 个 fold 上的 accuracy，计算每个 fold 的差异，并做 paired t-test。解释 p-value。

4. **Bootstrap CI。**  
   给定一个测试集预测结果，使用 bootstrap 构造 accuracy 的 95% CI。

5. **统计显著 vs 实际显著。**  
   构造一个大样本实验，使得两个模型 accuracy 差异只有 0.03%，但 p-value < 0.05。讨论是否值得上线。

6. **多重比较修正。**  
   假设你比较了 30 个模型，每个模型都做显著性检验。计算 Bonferroni adjusted alpha。

7. **类别不平衡。**  
   构造一个 99% negative 的二分类数据集。一个总是预测 negative 的模型 accuracy 是多少？为什么这个指标误导？

---

### 6.5 关键术语

| 术语 | 常见说法 | 真正含义 |
|---|---|---|
| Mean | 平均数 | 总和除以数量，对 outlier 敏感 |
| Median | 中位数 | 排序后的中间值，对 outlier 稳健 |
| Mode | 众数 | 最常见值，适合分类数据 |
| Variance | 方差 | 偏离均值的平方平均 |
| Standard deviation | 标准差 | 方差开方，与原数据同单位 |
| Percentile | 分位数 | 有多少比例数据低于该值 |
| IQR | 四分位距 | Q3 - Q1，中间 50% 的范围 |
| Pearson correlation | 线性相关 | 衡量线性关系，范围 [-1,1] |
| Spearman correlation | 秩相关 | 衡量单调关系，基于排名 |
| Covariance matrix | 协方差矩阵 | 所有特征两两协方差组成的矩阵 |
| Null hypothesis | 零假设 | 默认没有效果或没有差异 |
| Alternative hypothesis | 备择假设 | 想证明存在效果或差异 |
| p-value | p 值 | H0 成立时观察到当前极端数据的概率 |
| Confidence interval | 置信区间 | 参数估计的不确定范围 |
| t-test | t 检验 | 判断均值差异是否显著 |
| Chi-squared test | 卡方检验 | 判断观察频数是否偏离期望频数 |
| Effect size | 效应大小 | 差异大小，独立于样本量 |
| Bootstrap | 自助法 | 有放回重采样估计统计量分布 |
| Type I error | 一类错误 | 假阳性，H0 真却拒绝 H0 |
| Type II error | 二类错误 | 假阴性，H0 假却未拒绝 H0 |
| Statistical power | 检验功效 | 正确拒绝错误 H0 的概率 |
| CLT | 中心极限定理 | 样本均值随样本量增加趋近正态 |
| Parametric test | 参数检验 | 假设数据服从某类分布 |
| Non-parametric test | 非参数检验 | 不强依赖分布假设 |

---

### 6.6 自检问题

学完本课后，你应该能回答：

- 统计学为什么是模型评估的基础？
- mean 和 median 在有 outlier 时有什么区别？
- 为什么样本方差除以 `n - 1`？
- Pearson 和 Spearman 有什么区别？
- 协方差矩阵为什么和 PCA 有关？
- H0 和 H1 分别是什么？
- p-value 能解释为什么，不能解释什么？
- 置信区间的正确解释是什么？
- one-sample、two-sample、paired t-test 如何选择？
- chi-squared test 适合什么数据？
- ML 模型比较为什么应该使用 paired scores？
- statistical significance 和 practical significance 有什么区别？
- effect size 为什么重要？
- 多重比较为什么会增加假阳性？
- bootstrap 为什么适合 ML 指标？
- 参数检验和非参数检验怎么选？
- CLT 在 mini-batch gradient 和 ensemble 中有什么意义？
- ML 论文和实验中最常见的统计错误有哪些？

---

## 附录：本课最小代码文件建议

如果要拆成代码文件，可以这样组织：

```txt
code/
├── statistics.py
├── use_scipy.py
└── model_comparison.py
```

### `statistics.py`

包含：

```txt
mean
median
mode
variance
std
percentile
iqr
covariance
pearson_corr
spearman_corr
covariance_matrix
one_sample_t_stat
welch_t_stat
paired_t_stat
chi_squared_stat
cohens_d
bootstrap_ci
```

### `use_scipy.py`

包含：

```txt
scipy.stats.ttest_1samp
scipy.stats.ttest_ind
scipy.stats.ttest_rel
scipy.stats.chisquare
scipy.stats.spearmanr
scipy.stats.pearsonr
```

### `model_comparison.py`

包含：

```txt
bootstrap_accuracy_ci
bootstrap_metric_diff
paired_model_comparison
effect_size_report
multiple_comparison_correction
```
