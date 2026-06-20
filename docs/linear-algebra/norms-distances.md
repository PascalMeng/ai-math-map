---
title: 范数与距离
description: 面向 AI 算法工程师的距离度量课程：从 L1/L2/Lp/L∞、余弦相似度、Mahalanobis、Jaccard、编辑距离，到 KL、Wasserstein、最近邻搜索和正则化。
---

# 范数与距离

> 你的距离函数定义了什么叫“相似”。选错距离，后面的模型、检索和聚类都会跟着错。

**课程类型：** 实现 / 应用  
**所属模块：** 表征、距离与检索  
**前置知识：** Phase 1 Lesson 01-02：线性代数直觉、向量与矩阵基本运算  
**预计时间：** 约 90 分钟  
**使用语言：** Python / NumPy / scikit-learn  

---

## 1. 提出问题：为什么 AI 算法工程师必须理解范数与距离

你有两个向量。它们可能是：

```txt
word embeddings
sentence embeddings
user profiles
item embeddings
image pixel arrays
feature vectors
probability distributions
sets of tags
strings
```

你想知道：

```txt
它们有多相似？
它们有多远？
哪个样本是最近邻？
哪个文档最接近 query？
哪个用户最像当前用户？
```

但问题是：

```txt
“近”不是一个天然概念。
“相似”也不是一个唯一答案。
```

同一组数据，在不同距离度量下，最近邻可能完全不同。

```txt
L2 认为最近的点，cosine 可能认为很远。
Jaccard 只关心集合重叠，不关心频率。
Edit distance 适合字符串，但不适合 embedding。
Mahalanobis 会考虑特征相关性，L2 不会。
```

你的 KNN、推荐系统、向量数据库、聚类算法、异常检测、loss function 和正则化，全部都依赖距离或范数。  
选错距离，模型就在优化错误的“相似性”。

---

### 1.1 数学概念历史出现缘由

范数和距离最早来自几何和分析，用来度量对象的大小和对象之间的差异。  
在机器学习中，它们变成了“相似性”的基础定义。

| 数学 / 工程概念 | 出现缘由 |
|---|---|
| 范数 | 为了度量向量的大小 |
| 距离 | 为了度量两个对象之间的差异 |
| L1 norm | 为了度量沿坐标轴移动的总距离 |
| L2 norm | 为了度量欧氏空间中的直线距离 |
| Lp norm | 为了统一表示一族距离度量 |
| L∞ norm | 为了度量最坏维度上的最大偏差 |
| Cosine similarity | 为了只比较方向，不比较长度 |
| Dot product similarity | 为了同时考虑方向和向量长度 |
| Mahalanobis distance | 为了考虑特征尺度和相关性 |
| Jaccard similarity | 为了度量集合之间的重叠比例 |
| Edit distance | 为了度量字符串之间的最小编辑成本 |
| KL divergence | 为了度量概率分布之间的信息差异 |
| Wasserstein distance | 为了度量把一个分布搬成另一个分布的最小代价 |
| Nearest neighbor search | 为了在数据集中寻找最相似对象 |
| ANN | 为了在大规模向量库中快速近似检索 |
| HNSW | 为了用图结构高效做向量近邻搜索 |
| L1/L2 regularization | 为了用范数约束模型参数 |

可以这样理解：

```txt
现实问题：一个向量有多大？
数学抽象：范数

现实问题：两个向量有多远？
数学抽象：距离

现实问题：文本长度不同但语义方向相似，如何比较？
数学工具：cosine similarity

现实问题：特征有不同尺度和相关性，如何比较？
数学工具：Mahalanobis distance

现实问题：两个集合重叠多少？
数学工具：Jaccard similarity

现实问题：两个字符串要改几步才能一样？
算法工具：Edit distance

现实问题：两个概率分布差多少？
信息论工具：KL divergence / Wasserstein distance

现实问题：海量 embedding 如何快速找最近邻？
工程系统：ANN / HNSW / FAISS / vector database
```

---

### 1.2 原始问题是什么

本课要解决的原始问题是：

```txt
什么是范数？
范数和距离是什么关系？
L1、L2、Lp、L∞ 分别度量什么？
为什么同一数据在不同距离下会得到不同最近邻？
为什么 NLP 和 embedding 检索常用 cosine？
为什么 dot product 和 cosine 有时一样、有时不一样？
为什么 Mahalanobis 能识别相关特征下的异常点？
为什么集合相似度要用 Jaccard？
为什么字符串相似度要用 edit distance？
KL divergence 为什么不是严格距离？
Wasserstein distance 为什么适合分布不重叠的场景？
距离函数和 loss function、regularization 有什么关系？
```

换成 AI 语言，就是：

```txt
向量数据库应该选 cosine 还是 dot product？
为什么 embedding 要做 L2 normalization？
为什么 KNN 在高维下很难？
为什么 L1 regularization 会产生稀疏权重？
为什么 L2 regularization 只会缩小权重，不会让权重精确为 0？
为什么 GAN 里会用 Wasserstein distance？
为什么 outlier detection 不能只用 Euclidean distance？
为什么图像、文本、集合、字符串要用不同距离？
```

---

### 1.3 AI 中的现代问题

| 数学 / 工程知识点 | AI 中的现代问题 |
|---|---|
| L1 distance | 稀疏高维特征、MAE、LASSO |
| L2 distance | 图像、连续特征、MSE、Ridge |
| Lp norm | 统一距离族 |
| L∞ distance | 最坏维度偏差、鲁棒性约束 |
| Cosine similarity | NLP embedding、semantic search |
| Dot product | 推荐系统、向量检索排序 |
| Mahalanobis | 异常检测、相关特征距离 |
| Jaccard | 标签集合、文档去重、IoU |
| Edit distance | 拼写纠错、字符串匹配、DNA 序列 |
| KL divergence | VAE、蒸馏、RLHF、概率分布差异 |
| Wasserstein | WGAN、optimal transport、分布搬运 |
| Nearest neighbor | KNN、vector DB、retrieval |
| ANN | 大规模语义检索 |
| HNSW | 主流向量数据库索引 |
| L1 regularization | 特征选择、稀疏模型 |
| L2 regularization | weight decay、Ridge |
| Triplet / contrastive loss | 表征学习、metric learning |

---

### 1.4 本课要解决的核心问题

学完本课，你应该能够回答：

1. 范数和距离有什么关系？
2. L1、L2、Lp、L∞ 的公式和几何直觉是什么？
3. L1 为什么更鲁棒、更容易产生稀疏？
4. L2 为什么对大误差更敏感？
5. cosine similarity 为什么忽略向量长度？
6. dot product 和 cosine similarity 什么时候等价？
7. Mahalanobis distance 为什么能考虑特征相关性？
8. Jaccard similarity 为什么适合集合？
9. edit distance 如何用动态规划计算？
10. KL divergence 为什么不是距离？
11. Wasserstein distance 为什么对不重叠分布更有意义？
12. 不同任务应该如何选择距离函数？
13. 距离函数如何对应 loss function？
14. L1/L2 regularization 为什么会有不同几何效果？
15. ANN 和 HNSW 为什么是向量数据库的核心？

---

## 2. 概念定位：这节课学什么

---

### 2.1 所属模块

```txt
所属模块：表征、距离与检索
前置知识：向量、点积、矩阵、概率分布、信息论
后续连接：KNN、聚类、向量数据库、推荐系统、metric learning、正则化、检索增强生成 RAG
```

前面的课程已经解决了：

```txt
如何表示向量？
如何计算点积？
如何理解 embedding？
```

本课进一步解决：

```txt
如何定义两个对象是否相似，以及如何为任务选择合适距离？
```

---

### 2.2 本课学习目标

完成本课后，你应该能够：

- 从零实现 L1、L2、Lp、L∞ 距离
- 从零实现 cosine similarity 和 cosine distance
- 区分 dot product similarity 和 cosine similarity
- 从零实现 Mahalanobis distance
- 从零实现 Jaccard similarity
- 从零实现 Levenshtein edit distance
- 解释 KL divergence 和 Wasserstein distance 的适用场景
- 比较不同距离下的 nearest neighbor 结果
- 说明 L1/L2 与 LASSO/Ridge regularization 的关系
- 根据任务选择合适的距离度量
- 理解 ANN、HNSW 和 vector database 的基本思想

---

### 2.3 本课在整体路线中的位置

```txt
向量与矩阵
    ↓
范数与距离
    ↓
相似度搜索 / 最近邻
    ↓
聚类 / 推荐系统 / RAG
    ↓
Metric learning / Contrastive learning
    ↓
向量数据库 / ANN / HNSW
```

本课的核心能力是：

```txt
面对不同数据类型时，能判断“相似”应该如何定义，而不是默认全部用 L2。
```

---

## 3. 理解概念：直觉、公式、几何解释、AI 连接

---

### 3.1 范数：向量有多大

**一句话直觉：**  
范数是把一个向量映射成非负数，用来表示它的大小。

一般来说，距离可以写成：

```text
d(a, b) = ||a - b||
```

也就是说：

```txt
两个向量之间的距离，就是它们差向量的大小。
```

一个合格范数通常满足：

```txt
非负性：||x|| >= 0
零向量唯一性：||x|| = 0 当且仅当 x = 0
齐次性：||c x|| = |c| ||x||
三角不等式：||x + y|| <= ||x|| + ||y||
```

**AI 连接：**

| 场景 | 范数作用 |
|---|---|
| 参数大小 | regularization |
| 梯度大小 | gradient clipping |
| embedding normalization | cosine / dot equivalence |
| 向量距离 | KNN、检索、聚类 |
| loss function | prediction vs target error |

---

### 3.2 L1 Norm：Manhattan distance

**一句话直觉：**  
L1 距离是每个维度绝对差值的总和，像在城市街区里沿横竖道路走。

公式：

```text
||x||_1 = |x_1| + |x_2| + ... + |x_n|
```

两个点：

```txt
A = (1, 1)
B = (4, 5)
```

L1 距离：

```text
|4 - 1| + |5 - 1| = 3 + 4 = 7
```

**适用场景：**

- 高维稀疏特征
- 文本 bag-of-words / one-hot 特征
- 对 outlier 更鲁棒的误差度量
- 特征选择
- L1 regularization / LASSO

**AI 连接：**

MAE 是 L1 loss：

```text
MAE = |y - y_hat|
```

它对异常值比 MSE 更鲁棒，因为大误差只被线性惩罚。

---

### 3.3 L2 Norm：Euclidean distance

**一句话直觉：**  
L2 距离是直线距离，也就是几何中最熟悉的欧氏距离。

公式：

```text
||x||_2 = sqrt(x_1^2 + x_2^2 + ... + x_n^2)
```

两个点：

```txt
A = (1, 1)
B = (4, 5)
```

L2 距离：

```text
sqrt((4 - 1)^2 + (5 - 1)^2)
= sqrt(9 + 16)
= 5
```

**适用场景：**

- 低到中维连续特征
- 空间坐标
- 传感器数据
- 图像像素级比较
- feature scale 已经可比的场景

**AI 连接：**

MSE 是 L2 squared loss：

```text
MSE = (y - y_hat)^2
```

它对大误差惩罚更重，因此对 outlier 更敏感。

---

### 3.4 Lp Norm：统一的范数家族

**一句话直觉：**  
Lp norm 用一个参数 p 统一表示 L1、L2、L3 直到 L∞。

公式：

```text
||x||_p = (|x_1|^p + |x_2|^p + ... + |x_n|^p)^(1/p)
```

特殊情况：

| p | 名称 | 几何形状 |
|---:|---|---|
| 1 | L1 | 菱形 |
| 2 | L2 | 圆 / 球 |
| 3 | L3 | 更接近方形的圆 |
| ∞ | L∞ | 正方形 / 超立方体 |

**AI 连接：**

Lp norm 常用于：

- adversarial robustness 约束
- regularization
- 距离度量选择
- 优化几何分析

---

### 3.5 L∞ Norm：Chebyshev distance

**一句话直觉：**  
L∞ 距离只看所有维度中最大的绝对差异。

公式：

```text
||x||_inf = max_i |x_i|
```

两个点：

```txt
A = (1, 1)
B = (4, 5)
```

L∞ 距离：

```text
max(|4-1|, |5-1|) = max(3, 4) = 4
```

**适用场景：**

- 最坏维度偏差最重要
- 棋盘移动，例如国王一步可向任意方向走
- 制造公差，每个维度都必须在规格内
- adversarial perturbation 的 L∞ 约束

---

### 3.6 Cosine Similarity：只看方向，不看长度

**一句话直觉：**  
cosine similarity 衡量两个向量夹角是否接近，忽略向量长度。

公式：

```text
cos_sim(a, b) = (a · b) / (||a||_2 ||b||_2)
```

范围：

```txt
1：方向完全相同
0：互相垂直
-1：方向完全相反
```

cosine distance：

```text
cosine_distance = 1 - cosine_similarity
```

例子：

```txt
a = (1, 0)
b = (1, 1)
```

```text
cos_sim = 1 / sqrt(2) ≈ 0.707
cos_dist = 0.293
```

**为什么 NLP 常用 cosine？**

文本长度不应该决定语义相似度。  
一篇关于猫的短文和一篇关于猫的长文可能长度不同，但语义方向相近。

**AI 连接：**

| 场景 | cosine 作用 |
|---|---|
| word embedding | 语义方向 |
| sentence embedding | 语义检索 |
| RAG | query 和 document 相似度 |
| vector database | 常见检索 metric |
| 推荐系统 | 用户偏好方向 |

---

### 3.7 Dot Product vs Cosine Similarity

**一句话直觉：**  
dot product 同时包含方向和长度；cosine 是归一化后的 dot product，只看方向。

公式：

```text
a · b = ||a|| ||b|| cos(theta)
```

如果两个向量都已经 L2-normalized：

```text
||a|| = 1
||b|| = 1
```

则：

```text
a · b = cos(theta)
```

因此：

```txt
单位向量上，dot product 和 cosine similarity 等价。
```

区别：

```txt
dot product 会奖励大范数向量。
cosine similarity 不关心范数。
```

**AI 连接：**

| 选择 | 适合场景 |
|---|---|
| Cosine | 只关心语义方向 |
| Dot product | 向量长度代表置信度、热度、质量或强度 |
| L2-normalized dot | 等价于 cosine |
| Vector DB | 通常可选择 cosine / dot / L2 |

---

### 3.8 Mahalanobis Distance：考虑尺度和相关性的距离

**一句话直觉：**  
Mahalanobis distance 是在“白化后的空间”里计算 L2 距离，会考虑特征方差和相关性。

公式：

```text
d_M(x, y) = sqrt((x - y)^T S^(-1) (x - y))
```

其中：

```txt
S 是数据协方差矩阵。
```

如果：

```text
S = I
```

那么 Mahalanobis distance 退化为 Euclidean distance。

直觉：

```txt
先用协方差矩阵把数据 decorrelate 和 normalize，
再计算 L2 距离。
```

**例子：**

身高和体重是相关的。  
一个 `6'2"` 且 180 lbs 的人可能正常；  
一个 `5'0"` 且 180 lbs 的人可能异常。

L2 可能只看数值距离；Mahalanobis 会考虑“这种组合是否符合数据分布”。

**AI 连接：**

| 场景 | Mahalanobis 作用 |
|---|---|
| outlier detection | 多变量异常检测 |
| quality control | 多指标过程监控 |
| correlated features | 考虑协方差 |
| Gaussian model | 距离均值的统计异常程度 |
| classification | 特征尺度不同且相关 |

---

### 3.9 Jaccard Similarity：集合重叠

**一句话直觉：**  
Jaccard similarity 衡量两个集合交集占并集的比例。

公式：

```text
J(A, B) = |A ∩ B| / |A ∪ B|
```

Jaccard distance：

```text
d_J(A, B) = 1 - J(A, B)
```

例子：

```txt
A = {cat, dog, fish}
B = {cat, bird, fish, snake}
```

交集：

```txt
{cat, fish}，大小为 2
```

并集：

```txt
{cat, dog, fish, bird, snake}，大小为 5
```

所以：

```text
J(A, B) = 2 / 5 = 0.4
```

**AI 连接：**

| 场景 | Jaccard 作用 |
|---|---|
| 标签集合 | tag overlap |
| 文档去重 | word set overlap |
| near-duplicate detection | MinHash |
| segmentation | IoU = Jaccard |
| binary features | presence / absence |

---

### 3.10 Edit Distance：字符串最小编辑成本

**一句话直觉：**  
edit distance 衡量把一个字符串变成另一个字符串所需的最少编辑次数。

常见 Levenshtein distance 操作：

```txt
insert
delete
substitute
```

例子：

```txt
kitten -> sitting
```

步骤：

```txt
kitten -> sitten  替换 k 为 s
sitten -> sittin  替换 e 为 i
sittin -> sitting 插入 g
```

编辑距离：

```text
3
```

通常用动态规划计算：

```txt
dp[i][j] = 字符串 A 前 i 个字符与字符串 B 前 j 个字符的编辑距离
```

**AI 连接：**

| 场景 | edit distance 作用 |
|---|---|
| 拼写纠错 | 找最接近词 |
| fuzzy matching | 脏数据去重 |
| DNA sequence | 序列比对 |
| OCR 后处理 | 文本修正 |
| ASR 评估 | word error rate 类似思想 |

---

### 3.11 KL Divergence：不是距离但常被当作距离用

**一句话直觉：**  
KL divergence 衡量用分布 Q 代替分布 P 会多浪费多少信息，但它不是对称距离。

公式：

```text
D_KL(P || Q) = sum_x p(x) log(p(x) / q(x))
```

关键性质：

```text
D_KL(P || Q) != D_KL(Q || P)
```

所以它不是严格意义上的 distance metric。

Forward KL：

```text
D_KL(P || Q)
```

倾向于让 Q 覆盖 P 的所有模式，常被称为 mean-seeking。

Reverse KL：

```text
D_KL(Q || P)
```

倾向于让 Q 集中在 P 的某个模式上，常被称为 mode-seeking。

**AI 连接：**

| 场景 | KL 作用 |
|---|---|
| VAE | latent distribution 接近 prior |
| distillation | student 拟合 teacher |
| RLHF / PPO | 限制策略偏离 base model |
| probability matching | 分布差异 |
| DPO / preference learning | 分布约束 |

---

### 3.12 Wasserstein Distance：搬土距离

**一句话直觉：**  
Wasserstein distance 衡量把一个概率分布的质量搬成另一个分布所需的最小工作量。

直觉：

```txt
一个分布是一堆土。
另一个分布是目标坑。
距离 = 搬多少土 × 搬多远 的最小总成本。
```

一维情况下：

```text
W_1(P, Q) = integral |CDF_P(x) - CDF_Q(x)| dx
```

为什么重要？

```txt
当两个分布没有重叠时，KL 可能是无穷大，
但 Wasserstein 仍然给出有意义的距离和梯度。
```

例子：

```txt
P = [1, 0, 0, 0, 0]
Q = [0, 0, 0, 0, 1]

KL 可能无穷大。
Wasserstein = 把质量从 bin 1 搬到 bin 5 的代价。
```

**AI 连接：**

| 场景 | Wasserstein 作用 |
|---|---|
| WGAN | 提供更稳定的分布距离 |
| optimal transport | 最优传输 |
| image histogram | 比较颜色分布 |
| distribution shift | 分布差异 |
| generative modeling | 生成分布与真实分布比较 |

---

### 3.13 不同任务应该选什么距离

| 任务 | 推荐距离 | 原因 |
|---|---|---|
| 文本相似度 | Cosine | 长度是噪声，方向是语义 |
| embedding search | Cosine / Dot | 取决于向量范数是否有意义 |
| 图像像素比较 | L2 | 像素值连续且尺度可比 |
| 稀疏高维特征 | L1 | 鲁棒，不放大少数大差异 |
| 标签集合 | Jaccard | 数据天然是集合 |
| 字符串匹配 | Edit distance | 编辑操作符合任务直觉 |
| 异常检测 | Mahalanobis | 考虑特征相关性和尺度 |
| 概率分布比较 | KL | 信息损失视角 |
| GAN 训练 | Wasserstein | 分布不重叠时仍有梯度 |
| 制造质控 | L∞ | 最坏维度偏差重要 |
| 推荐系统 | Dot product | 范数可能表示热度或置信度 |

---

### 3.14 距离函数与 Loss Function 的关系

**一句话直觉：**  
很多 loss function 本质上就是预测值和目标值之间的距离或 divergence。

| Loss function | 对应距离 / divergence | 行为 |
|---|---|---|
| MAE | L1 | 对 outlier 更鲁棒 |
| MSE | L2 squared | 大误差惩罚更重 |
| Huber loss | 小误差 L2，大误差 L1 | 兼顾平滑和鲁棒 |
| Cross-entropy | KL / NLL | 分布匹配 |
| Hinge loss | margin-based distance | 分类间隔 |
| Triplet loss | 常用 L2 / cosine | 拉近正样本，推远负样本 |
| Contrastive loss | 常用 L2 / cosine | 表征学习 |

**AI 连接：**

当你选 loss 时，本质上也在选：

```txt
什么错误更严重？
什么相似性应该被优化？
模型应该对 outlier 敏感还是鲁棒？
```

---

### 3.15 距离函数与 Regularization 的关系

**一句话直觉：**  
正则化是在 loss 上额外加一个参数范数惩罚，限制模型复杂度。

L1 regularization / LASSO：

```text
loss + lambda * ||w||_1
```

效果：

```txt
推动一些权重精确变成 0。
产生稀疏模型。
自动做特征选择。
```

L2 regularization / Ridge / Weight Decay：

```text
loss + lambda * ||w||_2^2
```

效果：

```txt
把所有权重往 0 缩小。
通常不会让权重精确为 0。
提升稳定性和泛化。
```

Elastic Net：

```text
loss + lambda_1 ||w||_1 + lambda_2 ||w||_2^2
```

效果：

```txt
结合 L1 的稀疏性和 L2 的稳定性。
```

几何直觉：

```txt
L1 constraint region 是菱形，角在坐标轴上，更容易得到某些权重为 0。
L2 constraint region 是圆，没有轴上的尖角，权重通常只变小而不为 0。
```

---

### 3.16 最近邻搜索与向量数据库

**一句话直觉：**  
最近邻搜索就是给定 query，按某种距离函数找到最相似的数据点。

精确搜索复杂度：

```text
O(n * d)
```

其中：

| 符号 | 含义 |
|---|---|
| `n` | 数据点数量 |
| `d` | 向量维度 |

大规模向量数据库中，精确搜索太慢。  
所以使用 Approximate Nearest Neighbor，也就是 ANN。

常见 ANN 方法：

| 算法 | 思想 | 使用场景 |
|---|---|---|
| KD-tree | 轴对齐切分空间 | 低维 |
| Ball tree | 超球划分空间 | 中维 |
| LSH | 随机哈希投影 | 近似去重 |
| HNSW | 多层小世界图 | 主流 vector DB |
| IVF | 聚类倒排索引 | 大规模检索 |
| Product Quantization | 压缩向量再搜索 | 内存受限场景 |

HNSW 的直觉：

```txt
构建多层图。
上层边少，适合长跳。
下层边密，适合精细搜索。
查询时从上层快速靠近目标，再逐层下降。
```

**AI 连接：**

| 系统 | 距离核心 |
|---|---|
| RAG | query embedding vs document embedding |
| vector DB | cosine / dot / L2 |
| recommendation | user vector vs item vector |
| duplicate detection | Jaccard / MinHash |
| semantic search | cosine |
| image retrieval | embedding distance |

---

### 3.17 本课核心概念总表

| 概念 | 一句话直觉 | AI 中的对应位置 |
|---|---|---|
| Norm | 向量大小 | regularization、gradient norm |
| Distance | 两对象差异 | KNN、retrieval、clustering |
| L1 | 绝对差之和 | MAE、LASSO、sparse features |
| L2 | 直线距离 | MSE、Ridge、image similarity |
| Lp | 范数家族 | robustness、regularization |
| L∞ | 最大维度差异 | worst-case constraint |
| Cosine | 方向相似度 | NLP embeddings |
| Dot product | 方向 × 长度 | recommendation、retrieval |
| Mahalanobis | 协方差校正距离 | outlier detection |
| Jaccard | 集合重叠 | tags、IoU、dedup |
| Edit distance | 字符串编辑成本 | spell check、DNA |
| KL | 信息差异 | VAE、RLHF |
| Wasserstein | 搬运质量代价 | WGAN、OT |
| ANN | 近似最近邻 | vector database |
| HNSW | 图索引搜索 | FAISS、Qdrant、Weaviate |
| L1/L2 regularization | 参数范数惩罚 | LASSO、Ridge、weight decay |

---

## 4. 手写实现：从零实现主要范数与距离

这一节不用专业库，先用 Python 和基础数学实现核心距离函数。

---

### 4.1 L1、L2、Lp、L∞ 距离

```python
import math


def l1_distance(a, b):
    return sum(abs(x - y) for x, y in zip(a, b))


def l2_distance(a, b):
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def lp_distance(a, b, p):
    return sum(abs(x - y) ** p for x, y in zip(a, b)) ** (1 / p)


def linf_distance(a, b):
    return max(abs(x - y) for x, y in zip(a, b))


a = [1, 1]
b = [4, 5]

print(l1_distance(a, b))
print(l2_distance(a, b))
print(linf_distance(a, b))
```

---

### 4.2 Cosine similarity 与 dot product

```python
def dot_product(a, b):
    return sum(x * y for x, y in zip(a, b))


def norm_l2(a):
    return math.sqrt(sum(x ** 2 for x in a))


def cosine_similarity(a, b, eps=1e-12):
    denom = norm_l2(a) * norm_l2(b)

    if denom < eps:
        return 0.0

    return dot_product(a, b) / denom


def cosine_distance(a, b):
    return 1 - cosine_similarity(a, b)


a = [1, 0]
b = [1, 1]

print(cosine_similarity(a, b))
print(cosine_distance(a, b))
```

---

### 4.3 Mahalanobis distance

```python
import numpy as np


def mahalanobis_distance(x, y, covariance, eps=1e-8):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    cov = np.asarray(covariance, dtype=float)

    cov_reg = cov + eps * np.eye(cov.shape[0])
    inv_cov = np.linalg.inv(cov_reg)

    diff = x - y

    return float(np.sqrt(diff.T @ inv_cov @ diff))
```

说明：

```txt
eps 用于避免 covariance matrix 奇异或数值不稳定。
```

---

### 4.4 Jaccard similarity

```python
def jaccard_similarity(a, b):
    set_a = set(a)
    set_b = set(b)

    intersection = len(set_a & set_b)
    union = len(set_a | set_b)

    if union == 0:
        return 1.0

    return intersection / union


def jaccard_distance(a, b):
    return 1 - jaccard_similarity(a, b)


A = {"cat", "dog", "fish"}
B = {"cat", "bird", "fish", "snake"}

print(jaccard_similarity(A, B))
print(jaccard_distance(A, B))
```

---

### 4.5 Edit distance

```python
def edit_distance(a, b):
    m, n = len(a), len(b)

    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(m + 1):
        dp[i][0] = i

    for j in range(n + 1):
        dp[0][j] = j

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1

            dp[i][j] = min(
                dp[i - 1][j] + 1,
                dp[i][j - 1] + 1,
                dp[i - 1][j - 1] + cost,
            )

    return dp[m][n]


print(edit_distance("kitten", "sitting"))
```

---

### 4.6 最近邻搜索对比

```python
def nearest_neighbor(dataset, query, distance_fn):
    best_idx = None
    best_dist = float("inf")

    for i, point in enumerate(dataset):
        dist = distance_fn(query, point)

        if dist < best_dist:
            best_dist = dist
            best_idx = i

    return best_idx, best_dist


dataset = [
    [1, 1],
    [4, 5],
    [10, 10],
    [2, 8],
]

query = [3, 4]

for name, fn in [
    ("L1", l1_distance),
    ("L2", l2_distance),
    ("Linf", linf_distance),
    ("Cosine", cosine_distance),
]:
    idx, dist = nearest_neighbor(dataset, query, fn)

    print(f"{name}: nearest={dataset[idx]}, distance={dist:.4f}")
```

---

### 4.7 Embedding similarity search

```python
import numpy as np


def cosine_similarity_matrix(X):
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1, norms)

    X_normalized = X / norms

    return X_normalized @ X_normalized.T


np.random.seed(42)

embeddings = np.random.randn(1000, 768)

sim_matrix = cosine_similarity_matrix(embeddings)

query_idx = 0
similarities = sim_matrix[query_idx]

top_k = np.argsort(similarities)[::-1][1:6]

print(f"Top 5 most similar to item 0: {top_k}")
print(f"Similarities: {similarities[top_k]}")
```

---

## 5. 生产使用：NumPy / scikit-learn 中的距离计算

---

### 5.1 NumPy 向量化距离

```python
import numpy as np

X = np.random.randn(1000, 128)
query = np.random.randn(128)

l2 = np.linalg.norm(X - query, axis=1)

top_k = np.argsort(l2)[:5]

print(top_k)
```

cosine：

```python
X_norm = X / np.linalg.norm(X, axis=1, keepdims=True)
q_norm = query / np.linalg.norm(query)

cos_sim = X_norm @ q_norm

top_k = np.argsort(cos_sim)[::-1][:5]

print(top_k)
```

---

### 5.2 scikit-learn pairwise distances

```python
from sklearn.metrics import pairwise_distances

X = np.random.randn(100, 10)

D_l2 = pairwise_distances(X, metric="euclidean")
D_l1 = pairwise_distances(X, metric="manhattan")
D_cosine = pairwise_distances(X, metric="cosine")

print(D_l2.shape)
print(D_l1.shape)
print(D_cosine.shape)
```

---

### 5.3 KNN 中选择距离

```python
from sklearn.neighbors import KNeighborsClassifier

knn_l2 = KNeighborsClassifier(n_neighbors=5, metric="euclidean")
knn_l1 = KNeighborsClassifier(n_neighbors=5, metric="manhattan")
knn_cosine = KNeighborsClassifier(n_neighbors=5, metric="cosine")
```

---

### 5.4 手写实现 vs 生产库

| 对比项 | 手写实现 | NumPy / scikit-learn / vector DB |
|---|---|---|
| 目的 | 理解距离定义 | 工程实践 |
| L1/L2 | Python 循环 | NumPy 向量化 |
| Cosine | 手写 dot + norm | sklearn / vector DB |
| Mahalanobis | 手写 inv covariance | scipy / sklearn |
| Jaccard | set 操作 | sklearn / MinHash |
| Edit distance | 动态规划 | rapidfuzz / python-Levenshtein |
| ANN | 不适合手写 | FAISS / HNSW / Qdrant |
| 大规模检索 | 慢 | vector database |

---

## 6. 交付沉淀：产出物、连接关系、练习、术语表

---

### 6.1 本课产出

```txt
code/distances.py
code/use_numpy.py
code/use_sklearn.py
outputs/prompt-distance-metric-guide.md
outputs/exercises-solutions.md
```

| 文件 | 作用 |
|---|---|
| `code/distances.py` | 从零实现 L1、L2、Lp、L∞、cosine、Mahalanobis、Jaccard、edit distance |
| `code/use_numpy.py` | NumPy 向量化距离和 cosine search |
| `code/use_sklearn.py` | pairwise distances、KNN metric 示例 |
| `outputs/prompt-distance-metric-guide.md` | 选择距离度量的 prompt |
| `outputs/exercises-solutions.md` | 练习题与参考答案 |

---

### 6.2 知识连接关系

| 本课知识点 | 后续连接 |
|---|---|
| L1/L2 | loss function、regularization |
| Cosine | embedding search、RAG |
| Dot product | recommendation、retrieval |
| Mahalanobis | anomaly detection |
| Jaccard | set similarity、IoU、MinHash |
| Edit distance | fuzzy matching、sequence alignment |
| KL divergence | information theory、VAE、RLHF |
| Wasserstein | WGAN、optimal transport |
| ANN / HNSW | vector database |
| Regularization geometry | LASSO、Ridge、Elastic Net |

---

### 6.3 AI 应用连接

| 距离 / 范数 | AI 应用 |
|---|---|
| L1 | MAE、稀疏特征、LASSO |
| L2 | MSE、图像距离、Ridge |
| L∞ | adversarial constraint、最坏偏差 |
| Cosine | 文本 embedding、语义检索 |
| Dot product | 推荐系统、向量召回 |
| Mahalanobis | outlier detection |
| Jaccard | 标签集合、分割 IoU、文档去重 |
| Edit distance | 拼写纠错、DNA、OCR |
| KL | 分布匹配、VAE、RLHF |
| Wasserstein | WGAN、分布搬运 |
| HNSW | 向量数据库搜索 |
| L1/L2 regularization | 模型泛化和特征选择 |

---

### 6.4 练习

1. **比较 L1、L2、L∞。**  
   计算 `(1, 2, 3)` 和 `(4, 0, 6)` 的 L1、L2、L∞ 距离。  
   验证：

   ```text
   L∞ <= L2 <= L1
   ```

   并解释为什么这个关系通常成立。

2. **构造 cosine 高但 L2 大的向量。**  
   创建两个向量，使得 cosine similarity > 0.9，但 L2 distance > 10。  
   再创建两个向量，使得 cosine similarity < 0.3，但 L2 distance < 0.5。  
   解释几何含义。

3. **不同距离最近邻不一致。**  
   写一个函数，输入 dataset 和 query，分别返回 L1、L2、cosine、Mahalanobis 下的最近邻。  
   构造一个数据集，让四种距离选择不同最近邻。

4. **手算 Wasserstein。**  
   用 CDF 方法计算：

   ```txt
   [0.5, 0.5, 0, 0]
   [0, 0, 0.5, 0.5]
   ```

   的 Wasserstein distance。  
   再计算：

   ```txt
   [0.25, 0.25, 0.25, 0.25]
   [0, 0, 0.5, 0.5]
   ```

   哪个更大？为什么？

5. **实现 MinHash。**  
   用 MinHash 近似 Jaccard similarity。生成 100 个随机集合，比较 exact Jaccard 和 MinHash approximation。分别使用 50、100、200 个 hash function，观察误差如何变化。

---

### 6.5 关键术语

| 术语 | 常见说法 | 真正含义 |
|---|---|---|
| Norm | 向量大小 | 满足非负性、齐次性、三角不等式等性质的向量大小函数 |
| L1 norm | 曼哈顿距离 | 绝对值之和，对 outlier 更鲁棒，促进稀疏 |
| L2 norm | 欧氏距离 | 平方和开根号，直线距离 |
| Lp norm | 广义范数 | L1、L2、L∞ 的统一形式 |
| L∞ norm | 最大范数 | 最大绝对分量，关注最坏维度 |
| Cosine similarity | 夹角相似度 | 归一化 dot product，忽略向量长度 |
| Cosine distance | 1 - cosine | 把相似度转成距离 |
| Dot product | 未归一化 cosine | 同时包含方向和向量长度 |
| Mahalanobis distance | 协方差校正距离 | 白化空间中的 L2 距离 |
| Jaccard similarity | 集合重叠 | 交集大小除以并集大小 |
| Edit distance | 编辑距离 | 字符串最少插入、删除、替换次数 |
| KL divergence | 信息差异 | 非对称，不是真正距离 |
| Wasserstein distance | 搬土距离 | 把一个分布搬成另一个分布的最小代价 |
| ANN | 近似最近邻 | 用少量精度损失换大幅搜索加速 |
| HNSW | 小世界图索引 | 主流向量数据库近邻搜索算法 |
| L1 regularization | LASSO | L1 参数惩罚，促使权重为 0 |
| L2 regularization | Ridge / weight decay | L2 参数惩罚，缩小权重但通常不置零 |
| Elastic Net | L1 + L2 | 同时具备稀疏性和稳定性 |

---

### 6.6 自检问题

学完本课后，你应该能回答：

- 范数和距离有什么关系？
- L1、L2、Lp、L∞ 的公式分别是什么？
- L1 为什么会产生稀疏性？
- L2 为什么对 outlier 更敏感？
- cosine similarity 为什么适合文本 embedding？
- dot product 和 cosine 什么时候等价？
- Mahalanobis distance 为什么适合异常检测？
- Jaccard similarity 为什么适合集合数据？
- edit distance 如何用动态规划实现？
- KL divergence 为什么不是 distance metric？
- Wasserstein distance 为什么适合比较不重叠分布？
- 不同任务如何选择距离？
- loss function 和 distance 有什么关系？
- L1/L2 regularization 的几何直觉是什么？
- ANN 和 HNSW 如何支持大规模向量检索？

---

## 附录：本课最小代码文件建议

如果要拆成代码文件，可以这样组织：

```txt
code/
├── distances.py
├── use_numpy.py
└── use_sklearn.py
```

### `distances.py`

包含：

```txt
l1_distance
l2_distance
lp_distance
linf_distance
dot_product
cosine_similarity
cosine_distance
mahalanobis_distance
jaccard_similarity
jaccard_distance
edit_distance
nearest_neighbor
```

### `use_numpy.py`

包含：

```txt
np.linalg.norm
cosine_similarity_matrix
vectorized l2 search
embedding top-k search
```

### `use_sklearn.py`

包含：

```txt
pairwise_distances
KNeighborsClassifier with euclidean / manhattan / cosine
NearestNeighbors
```
