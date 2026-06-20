---
title: 降维方法
description: 面向 AI 算法工程师的降维课程：从维度灾难、PCA、解释方差、重构误差，到 t-SNE、UMAP、Kernel PCA 和高维数据可视化。
---

# 降维方法

> 高维数据并不是没有结构。关键是从正确的角度看它。

**课程类型：** 实现 / 应用  
**所属模块：** 表征、距离与降维  
**前置知识：** Phase 1 Lesson 01-03：线性代数直觉、矩阵运算、特征值与特征向量；Lesson 06：概率与分布  
**预计时间：** 约 90 分钟  
**使用语言：** Python / NumPy / scikit-learn / UMAP  

---

## 1. 提出问题：为什么 AI 算法工程师必须理解降维

你有一个数据集，每个样本有 784 个特征。  
它可能是一张 `28 x 28` 的手写数字图片，也可能是基因表达数据、用户行为数据、embedding 向量或传感器特征。

问题是：

```txt
784 维无法直接画出来。
784 维很难直观理解。
784 维可能包含大量冗余、噪声和相关特征。
```

但真实信息往往并不需要这么多独立维度。  
一张手写数字 “7” 可能主要由几个因素决定：

```txt
笔画角度
横杠长度
倾斜程度
书写粗细
位置偏移
```

降维要解决的核心问题就是：

```txt
在尽量保留重要结构的前提下，把高维数据压缩到更低维空间。
```

---

### 1.1 数学概念历史出现缘由

降维最早不是为深度学习出现的，而是为了处理高维数据难以理解、计算和可视化的问题。

| 数学 / 工程概念 | 历史出现缘由 |
|---|---|
| 维度灾难 | 为了解释维度升高后距离、体积和样本密度变得反直觉 |
| PCA | 为了找到数据中方差最大的方向，并用少数方向表示主要信息 |
| 协方差矩阵 | 为了描述不同特征如何一起变化 |
| 特征值 / 特征向量 | 为了找到矩阵变换中最重要的方向和强度 |
| 解释方差比 | 为了衡量每个主成分保留了多少信息 |
| 肘部法 | 为了选择保留多少主成分 |
| 重构误差 | 为了衡量压缩后丢失了多少信息 |
| t-SNE | 为了把高维数据映射到 2D/3D，同时保留局部邻域关系 |
| UMAP | 为了更快地做高维可视化，并更好保留部分全局结构 |
| Kernel PCA | 为了处理标准 PCA 无法发现的非线性结构 |
| 流形 | 为了描述高维空间中实际低维的数据表面 |

可以这样理解：

```txt
现实问题：高维数据无法可视化，且很多特征冗余。
数学抽象：降维

现实问题：哪些方向保留了最多数据变化？
数学方法：PCA

现实问题：如何判断一个方向有多重要？
数学指标：解释方差比

现实问题：低维表示丢了多少信息？
评估指标：重构误差

现实问题：数据结构是非线性的，PCA 不够用怎么办？
方法扩展：t-SNE、UMAP、Kernel PCA
```

---

### 1.2 原始问题是什么

本课要解决的原始问题是：

```txt
为什么高维空间会让距离和数据密度变得反直觉？
如何找到高维数据中最重要的方向？
PCA 为什么要先中心化数据？
为什么 PCA 要计算协方差矩阵？
为什么协方差矩阵的特征向量就是主成分方向？
如何判断保留多少个主成分？
降维后如何衡量信息损失？
t-SNE 和 UMAP 为什么更适合可视化？
Kernel PCA 如何处理非线性结构？
```

换成 AI 语言，就是：

```txt
为什么 784 维 MNIST 可以压缩到几十维？
embedding 为什么可以降到 2D 画图？
PCA、t-SNE、UMAP 的用途有什么区别？
为什么 t-SNE 的 cluster 间距离不能随便解释？
为什么 UMAP 在大数据上更常用？
如何用 PCA 作为模型训练前的预处理？
如何用重构误差做异常检测？
```

---

### 1.3 AI 中的现代问题

| 数学 / 工程知识点 | AI 中的现代问题 |
|---|---|
| 维度灾难 | 高维 embedding 检索、KNN 失效、数据稀疏 |
| PCA | 特征压缩、去噪、可视化、预处理 |
| 协方差矩阵 | 特征相关性、主方向分析 |
| 特征值 | 每个主方向的方差大小 |
| 特征向量 | 主成分方向 |
| 解释方差比 | 选择保留多少维度 |
| 重构误差 | 降维信息损失、异常检测 |
| t-SNE | 高维数据 2D 可视化 |
| UMAP | 大规模 embedding 可视化 |
| Kernel PCA | 非线性结构降维 |
| 流形 | 高维数据可能位于低维表面 |
| 肘部法 | 选择主成分数量 |
| 下游性能 | 用任务指标选择降维维度 |

---

### 1.4 本课要解决的核心问题

学完本课，你应该能够回答：

1. 什么是维度灾难？
2. 为什么高维空间中距离会变得不可靠？
3. PCA 的核心目标是什么？
4. PCA 为什么要中心化数据？
5. 协方差矩阵在 PCA 中起什么作用？
6. 为什么 PCA 要做特征分解？
7. explained variance ratio 如何选择主成分数量？
8. 重构误差如何衡量信息损失？
9. PCA、t-SNE、UMAP 分别适合什么场景？
10. 为什么 t-SNE 更适合可视化，而不适合预处理训练？
11. UMAP 相比 t-SNE 有什么优势？
12. Kernel PCA 如何处理非线性数据结构？
13. 如何从零实现 PCA？
14. 如何用 scikit-learn 完成 PCA、t-SNE、UMAP 和分类预处理？

---

## 2. 概念定位：这节课学什么

---

### 2.1 所属模块

```txt
所属模块：表征、距离与降维
前置知识：向量、矩阵、协方差、特征值、特征向量、概率分布
后续连接：PCA、SVD、embedding 可视化、异常检测、特征压缩、t-SNE、UMAP、Kernel PCA
```

前面的线性代数课程已经解决了：

```txt
矩阵如何变换空间？
特征值和特征向量如何表示主方向？
```

本课进一步解决：

```txt
如何利用这些数学工具，把高维数据压缩到低维空间，并尽量保留重要结构？
```

---

### 2.2 本课学习目标

完成本课后，你应该能够：

- 解释维度灾难为什么影响距离、体积和样本密度
- 从零实现 PCA：中心化、协方差矩阵、特征分解、投影
- 计算 explained variance ratio 和 cumulative explained variance
- 使用肘部法和阈值法选择主成分数量
- 计算 PCA 重构误差
- 比较 PCA、t-SNE、UMAP 的用途、优势和限制
- 使用 PCA 对 MNIST 或 embedding 做降维可视化
- 使用 PCA 作为分类器预处理
- 理解 Kernel PCA 如何处理非线性数据结构
- 说明降维和异常检测之间的关系

---

### 2.3 本课在整体路线中的位置

```txt
线性代数
    ↓
特征值与特征向量
    ↓
PCA
    ↓
降维、去噪、可视化
    ↓
t-SNE / UMAP / Kernel PCA
    ↓
embedding 分析、异常检测、特征压缩
```

本课的核心能力是：

```txt
面对高维数据时，能判断应该用 PCA 压缩、用 t-SNE/UMAP 可视化，还是用 Kernel PCA 处理非线性结构。
```

---

## 3. 理解概念：直觉、公式、几何解释、AI 连接

---

### 3.1 维度灾难：高维空间的反直觉问题

**一句话直觉：**  
维度越高，距离、体积和样本密度越不符合低维直觉。

高维空间会出现三个问题：

#### 距离变得不可靠

在高维空间中，随机点之间的距离会越来越接近。  
如果所有点之间距离都差不多，那么最近邻搜索就会变得困难。

```txt
维度越高，最大距离和最小距离的比例越接近 1。
```

#### 体积集中在角落

一个 d 维超立方体有：

```text
2^d
```

个角。  
当维度很高时，大部分体积都远离中心，集中在边缘和角落附近。

#### 需要指数级更多数据

如果你想在高维空间维持同样的数据密度，维度每增加一点，需要的数据量都会快速膨胀。

**AI 连接：**

| 问题 | AI 影响 |
|---|---|
| 距离失效 | KNN、向量检索变难 |
| 数据稀疏 | 模型更容易过拟合 |
| 样本不足 | 高维特征需要更多数据 |
| 冗余特征 | 训练更慢，泛化更差 |
| 噪声维度 | 干扰模型学习 |

---

### 3.2 PCA：找到最重要的方向

**一句话直觉：**  
PCA 找到数据中方差最大的方向，然后把数据投影到这些方向上。

PCA 的基本步骤：

```txt
1. 中心化数据：每个特征减去均值
2. 计算协方差矩阵：看特征如何一起变化
3. 特征分解：找主方向和对应方差
4. 按特征值排序：方差越大越重要
5. 选前 k 个主成分：保留主要结构
6. 投影：把原数据映射到低维空间
```

PCA 的核心思想：

```txt
方差大的方向通常包含更多信息。
方差小的方向通常包含噪声或冗余。
```

**AI 连接：**

| 场景 | PCA 作用 |
|---|---|
| 图像压缩 | 用少量主成分保留主要形状 |
| embedding 可视化 | 把高维向量投影到 2D/3D |
| 特征预处理 | 降低模型输入维度 |
| 去噪 | 丢掉低方差方向 |
| 异常检测 | 用重构误差发现异常样本 |

---

### 3.3 中心化：让数据围绕原点分析

**一句话直觉：**  
中心化就是把每个特征减去均值，让 PCA 关注“变化方向”，而不是数据整体位置。

公式：

```text
X_centered = X - mean(X)
```

为什么需要中心化？

```txt
PCA 关心方差方向。
如果不中心化，均值位置会影响主方向判断。
```

**AI 连接：**

| 操作 | 作用 |
|---|---|
| 中心化 | 去掉整体偏移 |
| 标准化 | 让不同特征尺度可比 |
| normalization | 改善训练稳定性 |
| preprocessing | 提升模型输入质量 |

---

### 3.4 协方差矩阵：特征如何一起变化

**一句话直觉：**  
协方差矩阵记录每个特征和其他特征如何一起变化。

协方差矩阵：

```text
Cov(X) = (1 / (n - 1)) * X_centered^T @ X_centered
```

其中：

| 符号 | 含义 |
|---|---|
| `X_centered` | 中心化后的数据 |
| `X_centered^T` | 转置 |
| `Cov(X)` | 特征之间的协方差矩阵 |

协方差矩阵的含义：

| 位置 | 含义 |
|---|---|
| 对角线 | 每个特征自己的方差 |
| 非对角线 | 两个特征一起变化的程度 |
| 正值 | 两个特征同向变化 |
| 负值 | 两个特征反向变化 |
| 接近 0 | 相关性弱 |

**AI 连接：**

| 场景 | 协方差作用 |
|---|---|
| PCA | 从协方差矩阵找主方向 |
| 特征相关性 | 发现冗余特征 |
| 多变量建模 | 理解变量之间依赖 |
| Whitening | 去相关和标准化 |
| Gaussian model | 协方差描述分布形状 |

---

### 3.5 特征值和特征向量：PCA 的主方向

**一句话直觉：**  
协方差矩阵的特征向量是数据的主方向，特征值表示该方向上的方差大小。

PCA 中：

```txt
特征向量 = principal component direction
特征值 = 该方向解释的方差
```

特征值越大：

```txt
该方向上的数据变化越大，越重要。
```

所以 PCA 会按特征值从大到小排序，保留前 k 个特征向量。

**AI 连接：**

| 概念 | AI 用途 |
|---|---|
| 最大特征值方向 | 数据主要变化方向 |
| 小特征值方向 | 噪声或低信息方向 |
| top-k eigenvectors | 降维后的坐标轴 |
| eigenvalue ranking | 选择主成分顺序 |

---

### 3.6 Explained Variance Ratio：每个主成分保留多少信息

**一句话直觉：**  
解释方差比表示某个主成分保留了总方差的多少比例。

公式：

```text
explained_ratio_k = eigenvalue_k / sum(all eigenvalues)
```

累积解释方差：

```text
cumulative_ratio_k = sum_{i=1}^k explained_ratio_i
```

示例：

| 主成分 | 特征值 | 解释方差比 | 累积解释方差 |
|---|---:|---:|---:|
| PC1 | 4.73 | 0.473 | 0.473 |
| PC2 | 2.51 | 0.251 | 0.724 |
| PC3 | 1.12 | 0.112 | 0.836 |
| PC4 | 0.89 | 0.089 | 0.925 |

如果前 4 个主成分已经解释 92.5% 的方差，后面的维度可能主要是细节或噪声。

---

### 3.7 如何选择主成分数量

**一句话直觉：**  
选择主成分数量，就是在信息保留和压缩程度之间取平衡。

常见策略：

| 方法 | 做法 | 适用场景 |
|---|---|---|
| 阈值法 | 保留 90%-95% 累积解释方差 | 通用预处理 |
| 肘部法 | 找解释方差曲线突然变平的位置 | 可视化选择 |
| 下游性能 | sweep k，观察模型准确率何时不再提升 | 任务驱动 |
| 固定维度 | 直接压到 2D/3D | 可视化 |

**AI 连接：**

| 选择 k 的方式 | 适合任务 |
|---|---|
| `k=2` | 可视化 |
| `k=50` | MNIST 压缩 |
| 保留 95% 方差 | 信息保留 |
| 准确率 plateau | 模型训练预处理 |

---

### 3.8 重构误差：降维到底丢了多少信息

**一句话直觉：**  
重构误差衡量把低维数据还原回高维时，和原始数据差多少。

步骤：

```txt
1. 低维投影：X_reduced = X @ W_k
2. 重构：X_hat = X_reduced @ W_k^T
3. 误差：MSE = mean((X - X_hat)^2)
```

PCA 中，重构误差和丢弃的特征值有关：

```text
reconstruction error = sum(dropped eigenvalues)
```

丢失比例：

```text
fraction_lost = sum(dropped eigenvalues) / sum(all eigenvalues)
```

**AI 连接：**

| 场景 | 重构误差作用 |
|---|---|
| 选择 k | 看压缩损失 |
| 图像压缩 | 衡量画质下降 |
| 异常检测 | 高重构误差样本可能异常 |
| Autoencoder | 重构误差也是核心 loss |

---

### 3.9 t-SNE：保留局部邻域结构

**一句话直觉：**  
t-SNE 把高维数据映射到 2D/3D，并尽量保留“谁和谁相邻”。

核心思想：

```txt
在高维空间中，近的点应该在低维图上仍然近。
```

t-SNE 会在高维空间中根据距离构造点对概率，然后寻找一个低维布局，使低维点对概率尽量接近高维点对概率。

特点：

| 特点 | 说明 |
|---|---|
| 非线性 | 能展开 PCA 无法处理的复杂结构 |
| 随机性 | 不同 random seed 可能有不同图形 |
| 适合可视化 | 常用于 2D cluster plot |
| 不适合预处理 | 输出维度不适合直接给下游模型 |
| cluster 间距离不可靠 | 只能重点看局部邻域和簇结构 |
| 较慢 | 默认复杂度较高，大数据慢 |

关键参数：

```txt
perplexity
```

它控制每个点大约考虑多少邻居，常见范围：

```txt
5 - 50
```

---

### 3.10 UMAP：更快并保留部分全局结构

**一句话直觉：**  
UMAP 也用于高维到低维可视化，但通常比 t-SNE 更快，并且更好保留一些全局结构。

UMAP 的基本思想：

```txt
先在高维空间构建近邻图，
再在低维空间中寻找一个尽量保留该图结构的布局。
```

关键参数：

| 参数 | 含义 |
|---|---|
| `n_neighbors` | 定义局部结构时考虑多少邻居 |
| `min_dist` | 低维空间中点可以靠得多近 |
| `n_components` | 输出维度，通常为 2 或 3 |

对比 t-SNE：

| 方法 | 优势 |
|---|---|
| t-SNE | 局部 cluster 可视化清晰 |
| UMAP | 更快，较好保留全局结构，适合更大数据 |

---

### 3.11 Kernel PCA：处理非线性结构

**一句话直觉：**  
标准 PCA 只能找线性子空间；Kernel PCA 可以通过 kernel trick 在高维特征空间中做 PCA，从而处理非线性结构。

标准 PCA 的限制：

```txt
只能通过旋转坐标轴和丢弃方向来降维。
如果数据结构是非线性流形，线性 PCA 可能无效。
```

Kernel PCA 的步骤：

```txt
1. 计算 kernel matrix K，其中 K_ij = k(x_i, x_j)
2. 在特征空间中中心化 kernel matrix
3. 对中心化后的 K 做特征分解
4. 使用 top eigenvectors 得到低维表示
```

常见 kernel：

| Kernel | 公式 | 适合场景 |
|---|---|---|
| RBF / Gaussian | `exp(-gamma * ||x - y||^2)` | 平滑非线性流形 |
| Polynomial | `(x · y + c)^d` | 多项式关系 |
| Sigmoid | `tanh(alpha * x · y + c)` | 类神经网络映射 |

经典例子：

```txt
同心圆数据无法被线性 PCA 分开。
RBF Kernel PCA 可以把内圈和外圈映射到更容易分离的表示。
```

---

### 3.12 PCA、t-SNE、UMAP、Kernel PCA 如何选择

| 方法 | 主要用途 | 保留什么 | 速度 | 是否适合预处理 |
|---|---|---|---|---|
| PCA | 压缩、去噪、预处理 | 全局方差 | 快 | 适合 |
| PCA 2D | 快速探索可视化 | 线性结构 | 快 | 可视化 |
| t-SNE | 高质量 2D 可视化 | 局部邻域 | 慢 | 不适合 |
| UMAP | 大规模可视化 | 局部 + 部分全局 | 中等到快 | 视情况 |
| Kernel PCA | 非线性降维 | kernel 定义的结构 | 慢 | 小中型数据可用 |

经验法则：

```txt
做模型预处理：优先 PCA。
做 embedding 可视化：t-SNE 或 UMAP。
数据很大：优先 UMAP。
结构明显非线性：尝试 Kernel PCA。
```

---

### 3.13 本课核心概念总表

| 概念 | 一句话直觉 | AI 中的对应位置 |
|---|---|---|
| 维度灾难 | 高维空间中距离和数据密度失效 | embedding、KNN、稀疏数据 |
| PCA | 找最大方差方向 | 压缩、去噪、预处理 |
| 协方差矩阵 | 特征如何一起变化 | 特征相关性 |
| 特征向量 | 主成分方向 | PCA axes |
| 特征值 | 主方向方差大小 | explained variance |
| explained variance ratio | 每个主成分保留多少信息 | 选择 k |
| 肘部法 | 找曲线变平的位置 | 选择主成分数 |
| 重构误差 | 降维损失多少信息 | anomaly detection |
| t-SNE | 保留局部邻域的可视化 | cluster visualization |
| UMAP | 更快的高维可视化 | embedding map |
| Kernel PCA | 非线性 PCA | nonlinear manifold |
| manifold | 高维空间中的低维表面 | 表征学习 |

---

## 4. 手写实现：从零实现 PCA 和降维评估

这一节主要使用 NumPy，但不直接调用 sklearn PCA。  
目标是亲手实现 PCA 的每个数学步骤。

---

### 4.1 从零实现 PCA 类

```python
import numpy as np


class PCA:
    def __init__(self, n_components):
        self.n_components = n_components
        self.components = None
        self.mean = None
        self.eigenvalues = None
        self.explained_variance_ratio_ = None

    def fit(self, X):
        self.mean = np.mean(X, axis=0)
        X_centered = X - self.mean

        cov_matrix = np.cov(X_centered, rowvar=False)

        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

        sorted_idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[sorted_idx]
        eigenvectors = eigenvectors[:, sorted_idx]

        self.components = eigenvectors[:, :self.n_components].T
        self.eigenvalues = eigenvalues[:self.n_components]

        total_var = np.sum(eigenvalues)
        self.explained_variance_ratio_ = self.eigenvalues / total_var

        return self

    def transform(self, X):
        X_centered = X - self.mean
        return X_centered @ self.components.T

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)
```

关键步骤对应：

| 代码 | 数学含义 |
|---|---|
| `X - mean` | 中心化 |
| `np.cov` | 协方差矩阵 |
| `np.linalg.eigh` | 对称矩阵特征分解 |
| `argsort` | 按特征值从大到小排序 |
| `components` | top-k 主成分 |
| `X_centered @ components.T` | 投影到低维空间 |

---

### 4.2 用合成数据测试 PCA

```python
np.random.seed(42)

n_samples = 500

t = np.random.uniform(0, 2 * np.pi, n_samples)

x1 = 3 * np.cos(t) + np.random.normal(0, 0.2, n_samples)
x2 = 3 * np.sin(t) + np.random.normal(0, 0.2, n_samples)
x3 = 0.5 * x1 + 0.3 * x2 + np.random.normal(0, 0.1, n_samples)

X_synthetic = np.column_stack([x1, x2, x3])

pca = PCA(n_components=2)
X_reduced = pca.fit_transform(X_synthetic)

print(f"Original shape: {X_synthetic.shape}")
print(f"Reduced shape:  {X_reduced.shape}")
print(f"Explained variance ratios: {pca.explained_variance_ratio_}")
print(f"Total variance captured: {sum(pca.explained_variance_ratio_):.4f}")
```

如果前两个主成分已经捕获大部分方差，说明第三个特征主要由前两个特征组合而来。

---

### 4.3 在 MNIST 上做 PCA

```python
from sklearn.datasets import fetch_openml

mnist = fetch_openml("mnist_784", version=1, as_frame=False, parser="auto")

X_mnist = mnist.data[:5000].astype(float)
y_mnist = mnist.target[:5000].astype(int)

pca_mnist = PCA(n_components=50)
X_pca50 = pca_mnist.fit_transform(X_mnist)

print(f"50 components capture {sum(pca_mnist.explained_variance_ratio_):.2%} of variance")

pca_2d = PCA(n_components=2)
X_pca2d = pca_2d.fit_transform(X_mnist)

print(f"2 components capture {sum(pca_2d.explained_variance_ratio_):.2%} of variance")
```

解释：

```txt
50 个主成分可能保留很多结构。
2 个主成分适合可视化，但通常会丢失大量信息。
```

---

### 4.4 和 scikit-learn PCA 对照

```python
from sklearn.decomposition import PCA as SklearnPCA
from sklearn.manifold import TSNE

sklearn_pca = SklearnPCA(n_components=2)
X_sklearn_pca = sklearn_pca.fit_transform(X_mnist)

print(f"Our PCA explained variance:     {pca_2d.explained_variance_ratio_}")
print(f"Sklearn PCA explained variance: {sklearn_pca.explained_variance_ratio_}")

diff = np.abs(np.abs(X_pca2d) - np.abs(X_sklearn_pca))
print(f"Max absolute difference: {diff.max():.10f}")
```

注意：

```txt
PCA 的方向可能整体取反，所以比较时常用绝对值。
```

---

### 4.5 t-SNE 可视化

```python
tsne = TSNE(
    n_components=2,
    perplexity=30,
    random_state=42
)

X_tsne = tsne.fit_transform(X_mnist)

print(f"t-SNE output shape: {X_tsne.shape}")
```

说明：

```txt
t-SNE 输出适合画 2D cluster。
不要过度解释 cluster 之间的距离和方向。
```

---

### 4.6 UMAP 可视化

```python
try:
    from umap import UMAP

    reducer = UMAP(
        n_components=2,
        n_neighbors=15,
        min_dist=0.1,
        random_state=42
    )

    X_umap = reducer.fit_transform(X_mnist)

    print(f"UMAP output shape: {X_umap.shape}")

except ImportError:
    print("Install umap-learn: pip install umap-learn")
```

---

### 4.7 实现 inverse_transform 和重构误差

```python
class PCAWithInverse(PCA):
    def inverse_transform(self, X_reduced):
        return X_reduced @ self.components + self.mean


def reconstruction_error(X, X_hat):
    return np.mean((X - X_hat) ** 2)
```

使用：

```python
pca = PCAWithInverse(n_components=50)

X_reduced = pca.fit_transform(X_mnist)
X_hat = pca.inverse_transform(X_reduced)

error = reconstruction_error(X_mnist, X_hat)

print(f"Reconstruction error: {error:.4f}")
```

---

## 5. 生产使用：用 scikit-learn 完成降维和模型预处理

---

### 5.1 PCA 作为分类器预处理

```python
from sklearn.decomposition import PCA as SklearnPCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

X_train, X_test, y_train, y_test = train_test_split(
    X_mnist,
    y_mnist,
    test_size=0.2,
    random_state=42
)

results = {}

for k in [10, 30, 50, 100, 200]:
    pca_k = SklearnPCA(n_components=k)

    X_tr = pca_k.fit_transform(X_train)
    X_te = pca_k.transform(X_test)

    clf = LogisticRegression(max_iter=1000, random_state=42)
    clf.fit(X_tr, y_train)

    acc = accuracy_score(y_test, clf.predict(X_te))
    var_captured = sum(pca_k.explained_variance_ratio_)

    results[k] = (acc, var_captured)

    print(f"k={k:>3d}  accuracy={acc:.4f}  variance={var_captured:.4f}")
```

观察点：

```txt
准确率通常会在某个 k 后开始 plateau。
这个 k 就是更实用的降维维度选择点。
```

---

### 5.2 Kernel PCA

```python
from sklearn.decomposition import KernelPCA

kpca = KernelPCA(
    n_components=2,
    kernel="rbf",
    gamma=10,
    fit_inverse_transform=False
)

X_kpca = kpca.fit_transform(X_synthetic)

print(f"Kernel PCA output shape: {X_kpca.shape}")
```

适合场景：

```txt
标准 PCA 无法处理的非线性结构。
例如同心圆、弯曲流形、非线性可分数据。
```

---

### 5.3 手写实现 vs 生产库

| 对比项 | 手写 PCA | scikit-learn |
|---|---|---|
| 目的 | 理解 PCA 数学流程 | 工程实践 |
| 中心化 | 手动实现 | 自动处理 |
| 协方差矩阵 | 手动计算 | 内部优化 |
| 特征分解 | `np.linalg.eigh` | 多种 solver |
| explained variance | 手动计算 | 内置属性 |
| t-SNE | 不建议手写 | `sklearn.manifold.TSNE` |
| UMAP | 不建议手写 | `umap-learn` |
| Kernel PCA | 复杂 | `KernelPCA` |
| 大数据 | 手写版不适合 | sklearn 更稳定 |

---

## 6. 交付沉淀：产出物、连接关系、练习、术语表

---

### 6.1 本课产出

```txt
code/pca_from_scratch.py
code/use_sklearn.py
outputs/skill-dimensionality-reduction.md
outputs/exercises-solutions.md
```

| 文件 | 作用 |
|---|---|
| `code/pca_from_scratch.py` | 从零实现 PCA、inverse_transform、重构误差 |
| `code/use_sklearn.py` | sklearn PCA、t-SNE、UMAP、Kernel PCA 示例 |
| `outputs/skill-dimensionality-reduction.md` | 用于选择降维方法的 skill |
| `outputs/exercises-solutions.md` | 练习题与参考答案 |

---

### 6.2 知识连接关系

| 本课知识点 | 后续连接 |
|---|---|
| 维度灾难 | 高维检索、embedding、KNN |
| PCA | SVD、压缩、去噪 |
| 协方差矩阵 | 特征相关性、Gaussian |
| 特征值 / 特征向量 | PCA、谱方法 |
| explained variance | 主成分选择 |
| reconstruction error | 异常检测、autoencoder |
| t-SNE | embedding 可视化 |
| UMAP | 大规模可视化 |
| Kernel PCA | 核方法、非线性降维 |
| manifold | 表征学习、生成模型 |

---

### 6.3 AI 应用连接

| 数学 / 工程知识点 | AI 应用 |
|---|---|
| PCA | 特征压缩、去噪、预处理 |
| t-SNE | 高维 embedding 可视化 |
| UMAP | 大规模 embedding map |
| explained variance | 选择压缩维度 |
| reconstruction error | 异常检测 |
| Kernel PCA | 非线性结构降维 |
| manifold | 表征空间分析 |
| PCA + classifier | 训练加速和降噪 |

---

### 6.4 练习

1. **实现 inverse_transform。**  
   给 PCA 类加入 `inverse_transform`。用 10、50、200 个主成分重构 MNIST 图片，并计算重构误差。

2. **比较 t-SNE perplexity。**  
   在同一 MNIST 子集上分别设置 perplexity 为 `5`、`30`、`100`。观察输出图变化，并解释 perplexity 如何影响 cluster 紧密程度。

3. **生成有效维度为 5 的数据。**  
   使用 `sklearn.datasets.make_classification` 生成 50 个特征、只有 5 个 informative features 的数据。应用 PCA 后，观察 explained variance curve 是否能发现数据近似 5 维。

4. **PCA 作为预处理。**  
   对同一个分类任务，分别使用原始特征和 PCA 压缩后的特征训练 Logistic Regression，对比训练速度和测试准确率。

5. **重构误差异常检测。**  
   用正常样本训练 PCA，再构造一些异常样本，比较它们的 reconstruction error 是否更高。

---

### 6.5 关键术语

| 术语 | 常见说法 | 真正含义 |
|---|---|---|
| 维度灾难 | 特征太多 | 高维空间中距离、体积和样本密度都变得反直觉 |
| PCA | 降维 | 旋转坐标系，让坐标轴对齐最大方差方向，再丢掉低方差方向 |
| 主成分 | 重要方向 | 协方差矩阵的特征向量 |
| 解释方差比 | 这个维度有多少信息 | 某个主成分解释的方差占总方差比例 |
| 协方差矩阵 | 特征相关性矩阵 | 描述不同特征如何一起变化 |
| t-SNE | 聚类可视化图 | 保留局部邻域关系的非线性可视化方法 |
| UMAP | 更快的 t-SNE | 基于近邻图和拓扑思想的高维可视化方法 |
| perplexity | t-SNE 参数 | 控制每个点考虑多少邻居 |
| manifold | 数据所在表面 | 嵌入在高维空间中的低维结构 |
| Kernel PCA | 非线性 PCA | 通过 kernel trick 在高维特征空间中做 PCA |
| 重构误差 | 还原误差 | 压缩再还原后与原始数据的差异 |

---

### 6.6 自检问题

学完本课后，你应该能回答：

- 降维最初解决什么问题？
- 维度灾难为什么会让距离和数据密度变得反直觉？
- PCA 为什么要找最大方差方向？
- PCA 为什么要先中心化数据？
- 协方差矩阵在 PCA 中起什么作用？
- 为什么协方差矩阵的特征向量是主成分方向？
- explained variance ratio 如何计算？
- 如何选择主成分数量？
- 重构误差如何衡量信息损失？
- PCA、t-SNE、UMAP 各自适合什么任务？
- 为什么 t-SNE 的 cluster 间距离不能随便解释？
- UMAP 的 `n_neighbors` 和 `min_dist` 分别控制什么？
- Kernel PCA 解决了标准 PCA 的什么问题？
- 如何用 PCA 作为分类器预处理？
- 如何用重构误差做异常检测？

---

## 附录：本课最小代码文件建议

如果要拆成代码文件，可以这样组织：

```txt
code/
├── pca_from_scratch.py
└── use_sklearn.py
```

### `pca_from_scratch.py`

包含：

```txt
PCA
PCA.fit
PCA.transform
PCA.fit_transform
PCAWithInverse
reconstruction_error
explained_variance_plot
```

### `use_sklearn.py`

包含：

```txt
SklearnPCA
TSNE
UMAP
KernelPCA
train_test_split
LogisticRegression
accuracy_score
PCA preprocessing sweep
```
