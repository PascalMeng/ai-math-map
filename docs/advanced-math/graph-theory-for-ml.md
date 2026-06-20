---
title: 机器学习中的图论
description: 面向 AI 算法工程师的图论课程：从图、邻接矩阵、BFS/DFS、图拉普拉斯、谱聚类，到 GNN message passing 和 GCN。
---

# 机器学习中的图论

> 图是表达关系的数据结构。只要你的数据有连接关系，你就需要图论。

**课程类型：** 实现 / 应用  
**所属模块：** 进阶数学 / 图论 / 图神经网络  
**前置知识：** Phase 1 Lesson 01-03：线性代数、矩阵、矩阵变换  
**预计时间：** 约 90 分钟  
**使用语言：** Python / NumPy / NetworkX  

---

## 1. 提出问题：为什么 AI 算法工程师必须理解图论

传统机器学习经常把数据看成表格：

```txt
每一行是一个样本。
每一列是一个特征。
样本之间默认相互独立。
```

但现实世界里，很多数据不是孤立样本，而是带有连接关系的结构：

```txt
社交网络：用户和用户之间有好友关系。
知识图谱：实体和实体之间有关系。
分子结构：原子和原子之间有化学键。
引用网络：论文引用论文。
道路网络：地点之间有道路。
欺诈网络：账号、设备、交易之间有关联。
推荐系统：用户和物品之间有交互。
```

如果你把这些数据强行压成表格，就会丢掉最重要的信息：

```txt
关系。
```

例如，在社交推荐中，一个用户自己的行为重要，但朋友的行为也很重要。  
在药物发现中，原子本身重要，但原子如何连接才是分子性质的关键。  
在欺诈检测中，单笔交易看起来正常，但交易网络结构可能暴露异常团伙。

图论解决的是：

```txt
如何表示、计算和学习关系结构。
```

Graph Neural Networks（GNN）正是建立在图论基础上。  
它们被用于：

```txt
药物发现
社交推荐
金融欺诈检测
知识图谱推理
交通预测
分子性质预测
论文引用网络分类
供应链风险分析
```

理解 GNN 前，你必须先理解四件事：

```txt
1. 如何把图表示成矩阵。
2. 如何遍历和探索图结构。
3. 图拉普拉斯为什么能揭示连通性和社区结构。
4. message passing 为什么是 GNN 的核心操作。
```

---

### 1.1 数学概念历史出现缘由

图论最早来自离散数学和网络问题，用来描述点与点之间的关系。  
在现代机器学习中，图成为表达复杂关系数据的基本结构。

| 图论概念 | 出现缘由 |
|---|---|
| 图 Graph | 为了表示对象及其关系 |
| 节点 Vertex / Node | 为了表示对象、实体或样本 |
| 边 Edge | 为了表示节点之间的连接 |
| 有向图 | 为了表达非对称关系，例如关注、引用、网页链接 |
| 无向图 | 为了表达对称关系，例如朋友、化学键 |
| 加权图 | 为了表达连接强度、距离、成本或相似度 |
| 邻接表 | 为了高效存储稀疏连接 |
| 邻接矩阵 | 为了把图变成矩阵，从而进行线性代数计算 |
| 度 Degree | 为了衡量节点连接数量 |
| 度矩阵 | 为了把节点度数写成矩阵形式 |
| BFS | 为了按层探索图并求无权最短路径 |
| DFS | 为了深入探索图、找连通分量和检测环 |
| 连通分量 | 为了识别图中彼此不连通的部分 |
| 图拉普拉斯 | 为了用矩阵谱分析图结构 |
| Fiedler value | 为了衡量图的代数连通性 |
| Fiedler vector | 为了找到图的自然二分结构 |
| 谱聚类 | 为了用拉普拉斯特征向量做图聚类 |
| 随机游走 | 为了描述节点之间的转移和扩散 |
| PageRank | 为了度量有向图中的节点重要性 |
| Message passing | 为了让节点聚合邻居信息 |
| GCN | 为了把图结构和神经网络结合起来 |

可以这样理解：

```txt
现实问题：样本之间有关联，如何表示？
数学结构：图

现实问题：图如何进入神经网络？
矩阵表示：邻接矩阵

现实问题：节点如何获得邻居信息？
GNN 操作：message passing

现实问题：如何找到图中的社区？
谱方法：图拉普拉斯和 Fiedler vector

现实问题：网页或节点谁更重要？
图算法：PageRank
```

---

### 1.2 原始问题是什么

本课要解决的原始问题是：

```txt
什么是图？
节点和边分别表示什么？
有向图、无向图、加权图有什么区别？
邻接矩阵和邻接表分别适合什么？
度矩阵如何表示节点连接数？
BFS 和 DFS 分别如何遍历图？
如何找连通分量？
图拉普拉斯 L = D - A 为什么重要？
拉普拉斯的特征值如何表示图的连通性？
Fiedler value 和 Fiedler vector 分别是什么？
谱聚类为什么可以用特征向量划分节点？
GNN message passing 的数学形式是什么？
为什么 GCN 是归一化邻接矩阵乘以特征矩阵？
PageRank 如何衡量节点重要性？
```

换成 AI 语言，就是：

```txt
为什么 GNN 输入通常是 adjacency matrix？
为什么图神经网络要聚合邻居？
为什么一层 GNN 看到 1-hop neighborhood，两层看到 2-hop？
为什么 GCN 要加 self-loop？
为什么要做 degree normalization？
为什么图拉普拉斯能做 community detection？
为什么知识图谱、分子和社交网络不能只用普通表格模型？
```

---

### 1.3 AI 中的现代问题

| 图论知识点 | AI 中的现代问题 |
|---|---|
| Graph | 关系型数据建模 |
| Node | 用户、原子、实体、论文、网页 |
| Edge | 好友、化学键、交易、引用、关系 |
| Adjacency matrix | GNN 输入结构 |
| Adjacency list | 稀疏图高效存储 |
| Degree | hub node、节点重要性 |
| BFS | 无权最短路径、知识图谱 hop search |
| DFS | 连通分量、环检测、拓扑排序 |
| Connected components | 图预处理、分块训练 |
| Laplacian | 谱聚类、图信号处理 |
| Eigenvalues of L | 连通性、社区结构 |
| Fiedler vector | 图二分、社区划分 |
| Spectral clustering | 无监督节点聚类 |
| Random walk | PageRank、node2vec、图扩散 |
| Message passing | GCN、GAT、GraphSAGE |
| Normalized adjacency | 稳定邻居聚合 |
| Self-loops | 保留节点自身特征 |
| PageRank | 节点重要性排序 |
| Degree distribution | power-law network、特征工程 |

---

### 1.4 本课要解决的核心问题

学完本课，你应该能够回答：

1. 图 `G=(V,E)` 中 `V` 和 `E` 分别是什么？
2. 有向图、无向图、加权图分别适合什么场景？
3. 邻接矩阵如何表示图？
4. 邻接表和邻接矩阵各有什么优缺点？
5. 节点 degree、in-degree、out-degree 分别是什么？
6. BFS 为什么能找到无权图最短路径？
7. DFS 为什么适合找连通分量和环？
8. 图拉普拉斯 `L=D-A` 为什么重要？
9. 拉普拉斯零特征值个数为什么等于连通分量个数？
10. Fiedler value 如何度量图连通性？
11. Fiedler vector 如何用于谱聚类？
12. Message passing 如何用矩阵乘法表达？
13. GCN 公式中的归一化邻接矩阵是什么意思？
14. PageRank 的基本迭代思想是什么？
15. 图论如何连接 GNN、知识图谱、分子建模和推荐系统？

---

## 2. 概念定位：这节课学什么

---

### 2.1 所属模块

```txt
所属模块：进阶数学 / 图论 / 图神经网络
前置知识：矩阵、特征值、特征向量、线性变换、矩阵乘法
后续连接：GCN、GAT、GraphSAGE、知识图谱、谱聚类、PageRank、图表示学习
```

前面的线性代数课程已经解决了：

```txt
如何用矩阵表示线性关系？
如何用特征值和特征向量分析矩阵？
```

本课进一步解决：

```txt
如何用矩阵表示节点连接关系，并通过矩阵谱和邻居聚合学习图结构？
```

---

### 2.2 本课学习目标

完成本课后，你应该能够：

- 从零实现一个 Graph 类
- 使用邻接表和邻接矩阵表示图
- 计算 degree matrix 和 graph Laplacian
- 实现 BFS 和 DFS
- 使用 BFS 找无权最短路径
- 使用 BFS / DFS 找 connected components
- 计算 Laplacian eigenvalues
- 用零特征值判断连通分量数量
- 理解 Fiedler value 和 Fiedler vector
- 实现简单 spectral clustering
- 实现一轮 GNN-style message passing
- 理解 GCN 的 normalized adjacency matrix
- 使用 NetworkX 做真实图分析

---

### 2.3 本课在整体路线中的位置

```txt
矩阵与特征值
    ↓
图的矩阵表示
    ↓
图遍历 BFS / DFS
    ↓
图拉普拉斯与谱分析
    ↓
谱聚类 / PageRank
    ↓
GNN message passing
    ↓
GCN / GAT / GraphSAGE / 知识图谱推理
```

本课的核心能力是：

```txt
面对关系型数据时，能把它转化成图，并用矩阵、遍历算法和 message passing 进行分析和建模。
```

---

## 3. 理解概念：直觉、公式、矩阵解释、AI 连接

---

### 3.1 什么是图

**一句话直觉：**  
图由节点和边组成，用来表示对象之间的关系。

形式：

```text
G = (V, E)
```

其中：

| 符号 | 含义 |
|---|---|
| `V` | vertices / nodes，节点集合 |
| `E` | edges，边集合 |

例子：

| 图场景 | 节点 | 边 |
|---|---|---|
| 社交网络 | 用户 | 好友关系 |
| 分子图 | 原子 | 化学键 |
| 知识图谱 | 实体 | 关系 |
| 引用网络 | 论文 | 引用 |
| 道路网络 | 地点 | 道路 |
| 推荐系统 | 用户和物品 | 交互 |

**AI 连接：**

图适合建模：

```txt
关系本身携带信息的数据。
```

---

### 3.2 有向图、无向图、加权图

**一句话直觉：**  
边可以有方向，也可以有权重；这取决于关系是否对称、强度是否重要。

| 图类型 | 含义 | 例子 |
|---|---|---|
| 无向无权图 | 连接对称，边只有存在/不存在 | Facebook 好友 |
| 有向无权图 | 连接有方向 | Twitter 关注 |
| 无向加权图 | 对称连接有强度 | 道路距离 |
| 有向加权图 | 有方向且有权重 | 网页链接权重、转账网络 |

有向图中：

```txt
(u, v) 表示 u 指向 v。
不一定有 (v, u)。
```

无向图中：

```txt
(u, v) 同时表示 u 连接 v 和 v 连接 u。
```

---

### 3.3 邻接矩阵

**一句话直觉：**  
邻接矩阵把节点之间是否相连写成一个 `n x n` 矩阵。

对于 n 个节点：

```text
A[i][j] = 1，如果 i 到 j 有边
A[i][j] = 0，否则
```

加权图中：

```text
A[i][j] = edge weight
```

无向图中：

```text
A[i][j] = A[j][i]
```

所以邻接矩阵是对称的。

例子：三角形图

```txt
Nodes: 0, 1, 2
Edges: (0,1), (1,2), (0,2)
```

邻接矩阵：

```text
A = [[0, 1, 1],
     [1, 0, 1],
     [1, 1, 0]]
```

**AI 连接：**

邻接矩阵是 GNN 的核心输入之一。  
矩阵乘法 `A @ H` 就表示邻居特征聚合。

---

### 3.4 邻接表

**一句话直觉：**  
邻接表为每个节点存它的邻居，适合稀疏图。

例子：

```text
0: [1, 2]
1: [0, 2]
2: [0, 1]
```

优缺点：

| 表示 | 优点 | 缺点 |
|---|---|---|
| 邻接矩阵 | 矩阵运算方便，判断边是否存在快 | 稀疏图浪费内存 |
| 邻接表 | 存储稀疏图高效，遍历邻居快 | 不适合直接谱分解 |

**AI 连接：**

真实图通常很稀疏：

```txt
节点很多，但每个节点只连接少数邻居。
```

大规模 GNN 通常使用稀疏矩阵或边列表，而不是 dense adjacency matrix。

---

### 3.5 Degree 与 Degree Matrix

**一句话直觉：**  
degree 表示一个节点连接了多少条边。

无向图：

```text
degree(i) = 节点 i 的邻居数量
```

有向图：

| 概念 | 含义 |
|---|---|
| in-degree | 指向该节点的边数量 |
| out-degree | 从该节点指出去的边数量 |

度矩阵 `D` 是对角矩阵：

```text
D[i][i] = degree(i)
D[i][j] = 0, i != j
```

三角形中每个节点 degree = 2：

```text
D = [[2, 0, 0],
     [0, 2, 0],
     [0, 0, 2]]
```

**AI 连接：**

degree 可以作为节点特征：

```txt
高 degree 节点可能是 hub。
社交网络 degree distribution 常呈 power law。
异常 degree 节点可能代表异常账号或核心节点。
```

---

### 3.6 BFS：按层探索

**一句话直觉：**  
BFS 从起点开始，先访问所有一跳邻居，再访问两跳邻居。

BFS 使用：

```txt
queue
FIFO
```

流程：

```txt
1. 起点入队。
2. 每次从队首取出节点。
3. 把未访问邻居加入队尾。
4. 重复直到队列为空。
```

BFS 的重要性质：

```txt
在无权图中，BFS 第一次访问某节点时的层数就是最短路径长度。
```

**AI 连接：**

| 场景 | BFS 用途 |
|---|---|
| 社交网络 | k-hop friend search |
| 知识图谱 | 多跳关系查询 |
| 无权最短路径 | hop distance |
| 推荐系统 | 邻域扩展 |
| GNN sampling | 按 hop 采样邻居 |

---

### 3.7 DFS：一路深入再回溯

**一句话直觉：**  
DFS 沿着一条路径尽可能走到底，然后再回退探索其他路径。

DFS 使用：

```txt
stack
LIFO
```

或者递归实现。

适合：

- 找 connected components
- 检测 cycle
- topological sorting
- 深度遍历树结构
- 搜索所有路径

**AI 连接：**

| 场景 | DFS 用途 |
|---|---|
| 图预处理 | 连通分量 |
| 有向无环图 | 拓扑排序 |
| 计算图 | 依赖遍历 |
| 知识图谱 | 路径搜索 |
| 数据清洗 | 删除孤立组件 |

---

### 3.8 Connected Components

**一句话直觉：**  
连通分量是图中彼此可达的一整块区域。

如果两个节点之间存在路径，它们属于同一个 connected component。  
如果图被分成几块互不连接的部分，每一块就是一个 component。

算法：

```txt
从未访问节点开始 BFS/DFS。
访问到的所有节点构成一个 component。
重复直到所有节点都访问过。
```

**AI 连接：**

| 场景 | 作用 |
|---|---|
| 图预处理 | 分离不连通子图 |
| 社区分析 | 找独立群体 |
| 训练 GNN | batch subgraph |
| 异常检测 | 小孤立组件可能异常 |
| 谱分析 | zero eigenvalue 数量 |

---

### 3.9 图拉普拉斯 `L = D - A`

**一句话直觉：**  
图拉普拉斯是图结构中最重要的矩阵，它把 degree 和 adjacency 结合起来，用于分析连通性和聚类结构。

定义：

```text
L = D - A
```

其中：

| 矩阵 | 含义 |
|---|---|
| `A` | adjacency matrix |
| `D` | degree matrix |
| `L` | graph Laplacian |

三角形例子：

```text
D = [[2,0,0],
     [0,2,0],
     [0,0,2]]

A = [[0,1,1],
     [1,0,1],
     [1,1,0]]

L = [[ 2,-1,-1],
     [-1, 2,-1],
     [-1,-1, 2]]
```

重要性质：

```txt
L 是 positive semidefinite。
L 的零特征值数量等于 connected components 数量。
最小非零特征值衡量图连通性。
对应特征向量可用于谱聚类。
```

---

### 3.10 拉普拉斯特征值与连通性

**一句话直觉：**  
图拉普拉斯的谱能告诉你图有几块、连接有多强、如何自然分割。

性质：

| 拉普拉斯谱性质 | 图结构含义 |
|---|---|
| 最小特征值为 0 | 总是成立 |
| 零特征值个数 | connected components 数量 |
| 一个 connected graph | 恰好一个零特征值 |
| Fiedler value | 最小非零特征值，衡量代数连通性 |
| Fiedler vector | 可用于图二分 |

如果 Fiedler value 很小：

```txt
图存在瓶颈，容易被切开。
```

如果 Fiedler value 很大：

```txt
图整体连接紧密。
```

---

### 3.11 Fiedler vector 与谱聚类

**一句话直觉：**  
Fiedler vector 给每个节点一个一维坐标，符号或数值大小可以用于划分社区。

谱聚类基本流程：

```txt
1. 计算 Laplacian L。
2. 求 L 的特征值和特征向量。
3. 取最小非零特征值对应的特征向量，也就是 Fiedler vector。
4. 根据特征向量的正负号或阈值划分节点。
```

对于 `k` 个 cluster：

```txt
取前 k 个非平凡特征向量。
把每个节点表示为 k 维坐标。
再做 k-means。
```

为什么有效？

```txt
拉普拉斯特征向量是图上最平滑的函数。
连接紧密的节点会有相似的特征向量值。
瓶颈两侧节点值会分开。
```

---

### 3.12 Random Walk 与 PageRank

**一句话直觉：**  
随机游走把图看成一个转移系统，PageRank 用随机游走的长期访问概率衡量节点重要性。

在无向图中，随机游走从当前节点随机选择一个邻居移动。  
转移概率通常与 degree 有关。

PageRank 思想：

```txt
一个网页重要，如果很多重要网页指向它。
```

迭代公式：

```text
score(v) = (1-d)/n + d * sum_{u -> v} score(u)/out_degree(u)
```

其中：

| 符号 | 含义 |
|---|---|
| `d` | damping factor，常用 0.85 |
| `n` | 节点数 |
| `u -> v` | 指向 v 的节点 |
| `out_degree(u)` | u 指出去的边数 |

**AI 连接：**

PageRank 类似思想用于：

```txt
网页排序
知识图谱实体重要性
图中心性特征
推荐系统节点权重
attention initialization
```

---

### 3.13 Message Passing：GNN 的核心

**一句话直觉：**  
GNN 中每个节点从邻居收集信息，聚合后更新自己的表示。

一般形式：

```text
h_v^(k+1) =
UPDATE(
  h_v^(k),
  AGGREGATE({h_u^(k) : u in N(v)})
)
```

其中：

| 符号 | 含义 |
|---|---|
| `h_v` | 节点 v 的表示 |
| `N(v)` | 节点 v 的邻居集合 |
| `AGGREGATE` | 聚合邻居消息 |
| `UPDATE` | 更新节点状态 |

最简单 mean aggregation：

```text
h_v^(k+1) = sigma(W * mean({h_u^(k) : u in N(v)}))
```

矩阵形式：

```text
H^(k+1) = sigma(A_norm H^(k) W)
```

其中：

| 矩阵 | 含义 |
|---|---|
| `H` | 所有节点特征矩阵 |
| `A_norm` | 归一化邻接矩阵 |
| `W` | 可学习权重矩阵 |

一层 message passing：

```txt
节点看到 1-hop 邻居。
```

两层：

```txt
节点看到 2-hop 邻居。
```

K 层：

```txt
节点看到 K-hop neighborhood。
```

---

### 3.14 GCN 的归一化邻接矩阵

**一句话直觉：**  
GCN 用加 self-loop 的归一化邻接矩阵做 message passing，避免高度节点特征被过度放大。

GCN 公式：

```text
H^(l+1) =
sigma(
  D_hat^(-1/2) A_hat D_hat^(-1/2) H^(l) W^(l)
)
```

其中：

```text
A_hat = A + I
```

表示加 self-loops。

```text
D_hat
```

是 `A_hat` 的 degree matrix。

为什么加 self-loop？

```txt
节点聚合邻居时，也保留自己的特征。
```

为什么 degree normalization？

```txt
避免高 degree 节点贡献过大。
稳定不同节点的特征尺度。
```

**AI 连接：**

GCN、GraphSAGE、GAT 都是 message passing 框架的不同变体。

---

### 3.15 Adjacency Matrix Multiplication 的含义

**一句话直觉：**  
`A @ H` 的第 i 行，就是节点 i 的邻居特征求和。

设：

```text
H: n_nodes x feature_dim
A: n_nodes x n_nodes
```

则：

```text
(A @ H)[i] = sum_j A[i,j] H[j]
```

如果 `A[i,j]=1`，节点 j 是节点 i 的邻居。  
所以 `A @ H` 就是邻居特征聚合。

如果使用 row-normalized adjacency：

```text
A_norm[i,j] = A[i,j] / degree(i)
```

则：

```text
A_norm @ H
```

表示邻居特征平均。

---

### 3.16 本课核心概念总表

| 概念 | 一句话直觉 | AI 中的对应位置 |
|---|---|---|
| Graph | 节点和边 | 关系数据 |
| Node | 对象 | 用户、原子、实体 |
| Edge | 关系 | 好友、化学键、引用 |
| Directed graph | 有方向关系 | 关注、引用 |
| Weighted graph | 边有强度 | 距离、交易额 |
| Adjacency matrix | 连接矩阵 | GNN 输入 |
| Adjacency list | 邻居表 | 稀疏图存储 |
| Degree | 节点连接数 | hub 特征 |
| BFS | 按层搜索 | 无权最短路径 |
| DFS | 深度搜索 | 连通分量、环 |
| Connected component | 连通块 | 图预处理 |
| Laplacian | `D-A` | 谱图理论 |
| Fiedler value | 代数连通性 | 瓶颈检测 |
| Fiedler vector | 谱划分坐标 | 谱聚类 |
| PageRank | 随机游走重要性 | 节点排序 |
| Message passing | 邻居聚合 | GNN |
| GCN | 归一化邻接聚合 | 图神经网络 |

---

## 4. 手写实现：Graph、BFS/DFS、Laplacian、谱聚类和 Message Passing

这一节用 Python 和 NumPy 从零实现核心图算法。

---

### 4.1 Graph 类

```python
class Graph:
    def __init__(self, n_nodes, directed=False):
        self.n = n_nodes
        self.directed = directed
        self.adj = {
            i: {}
            for i in range(n_nodes)
        }

    def add_edge(self, u, v, weight=1.0):
        self.adj[u][v] = weight

        if not self.directed:
            self.adj[v][u] = weight

    def neighbors(self, node):
        return list(self.adj[node].keys())

    def degree(self, node):
        return len(self.adj[node])

    def adjacency_matrix(self):
        import numpy as np

        A = np.zeros((self.n, self.n))

        for u in range(self.n):
            for v, w in self.adj[u].items():
                A[u][v] = w

        return A

    def degree_matrix(self):
        import numpy as np

        D = np.zeros((self.n, self.n))

        for i in range(self.n):
            D[i][i] = self.degree(i)

        return D

    def laplacian(self):
        return self.degree_matrix() - self.adjacency_matrix()
```

---

### 4.2 BFS

```python
from collections import deque


def bfs(graph, start):
    visited = set()
    order = []
    distances = {}

    queue = deque([(start, 0)])
    visited.add(start)

    while queue:
        node, dist = queue.popleft()

        order.append(node)
        distances[node] = dist

        for neighbor in graph.neighbors(node):
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append((neighbor, dist + 1))

    return order, distances
```

---

### 4.3 DFS

```python
def dfs(graph, start):
    visited = set()
    order = []
    stack = [start]

    while stack:
        node = stack.pop()

        if node in visited:
            continue

        visited.add(node)
        order.append(node)

        for neighbor in reversed(graph.neighbors(node)):
            if neighbor not in visited:
                stack.append(neighbor)

    return order
```

---

### 4.4 Connected components

```python
def connected_components(graph):
    visited = set()
    components = []

    for node in range(graph.n):
        if node not in visited:
            order, _ = bfs(graph, node)

            visited.update(order)
            components.append(order)

    return components
```

---

### 4.5 Laplacian eigenvalues

```python
def laplacian_eigenvalues(graph):
    import numpy as np

    L = graph.laplacian()
    eigenvalues = np.linalg.eigvalsh(L)

    return eigenvalues
```

说明：

```txt
对无向图，L 是对称矩阵，所以可以用 eigvalsh。
```

---

### 4.6 Spectral clustering

```python
def spectral_clustering(graph, k=2):
    import numpy as np

    L = graph.laplacian()

    eigenvalues, eigenvectors = np.linalg.eigh(L)

    features = eigenvectors[:, 1:k + 1]

    labels = np.zeros(graph.n, dtype=int)

    for i in range(graph.n):
        if features[i, 0] >= 0:
            labels[i] = 0
        else:
            labels[i] = 1

    return labels
```

说明：

```txt
k=2 时，用 Fiedler vector 的符号划分两类。
k>2 时，通常对前 k 个非平凡特征向量做 k-means。
```

---

### 4.7 Message passing

```python
def message_passing(graph, features, weight_matrix):
    import numpy as np

    A = graph.adjacency_matrix()

    row_sums = A.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1

    A_norm = A / row_sums

    aggregated = A_norm @ features

    output = aggregated @ weight_matrix

    return output
```

这表示：

```txt
每个节点先取邻居特征平均，再乘一个可学习权重矩阵。
```

---

### 4.8 加 self-loop 的 normalized adjacency

```python
def normalized_adjacency_with_self_loops(graph):
    import numpy as np

    A = graph.adjacency_matrix()
    A_hat = A + np.eye(graph.n)

    D_hat = np.diag(A_hat.sum(axis=1))

    D_inv_sqrt = np.diag(
        1.0 / np.sqrt(np.diag(D_hat))
    )

    return D_inv_sqrt @ A_hat @ D_inv_sqrt
```

GCN 一层：

```python
def gcn_layer(graph, H, W, activation=None):
    A_norm = normalized_adjacency_with_self_loops(graph)

    out = A_norm @ H @ W

    if activation is not None:
        out = activation(out)

    return out
```

---

### 4.9 PageRank from scratch

```python
def pagerank(graph, damping=0.85, tol=1e-6, max_iter=100):
    n = graph.n

    scores = {
        i: 1.0 / n
        for i in range(n)
    }

    for _ in range(max_iter):
        new_scores = {
            i: (1 - damping) / n
            for i in range(n)
        }

        for u in range(n):
            out_neighbors = graph.neighbors(u)

            if not out_neighbors:
                continue

            share = scores[u] / len(out_neighbors)

            for v in out_neighbors:
                new_scores[v] += damping * share

        diff = sum(
            abs(new_scores[i] - scores[i])
            for i in range(n)
        )

        scores = new_scores

        if diff < tol:
            break

    return scores
```

---

## 5. 生产使用：NetworkX / NumPy 图分析

---

### 5.1 NetworkX 基础图分析

```python
import networkx as nx
import numpy as np

G = nx.karate_club_graph()

A = nx.adjacency_matrix(G).toarray()
L = nx.laplacian_matrix(G).toarray()

eigenvalues = np.linalg.eigvalsh(L.astype(float))

print(f"Smallest eigenvalues: {eigenvalues[:5]}")
print(f"Connected components: {nx.number_connected_components(G)}")
```

---

### 5.2 NetworkX 社区发现和 PageRank

```python
communities = nx.community.greedy_modularity_communities(G)

print(f"Communities found: {len(communities)}")

pr = nx.pagerank(G)

top_nodes = sorted(
    pr.items(),
    key=lambda x: x[1],
    reverse=True
)[:5]

print(f"Top 5 PageRank nodes: {top_nodes}")
```

---

### 5.3 NumPy 谱聚类示例

```python
import numpy as np

A = np.array([
    [0, 1, 1, 0, 0],
    [1, 0, 1, 0, 0],
    [1, 1, 0, 1, 0],
    [0, 0, 1, 0, 1],
    [0, 0, 0, 1, 0],
])

D = np.diag(A.sum(axis=1))
L = D - A

eigenvalues, eigenvectors = np.linalg.eigh(L)

print(f"Eigenvalues: {np.round(eigenvalues, 4)}")
print(f"Fiedler value: {eigenvalues[1]:.4f}")
print(f"Fiedler vector: {np.round(eigenvectors[:, 1], 4)}")

fiedler = eigenvectors[:, 1]

group_a = np.where(fiedler >= 0)[0]
group_b = np.where(fiedler < 0)[0]

print(f"Cluster A: {group_a}")
print(f"Cluster B: {group_b}")
```

---

### 5.4 PyTorch Geometric / DGL 的位置

真实 GNN 工程通常使用：

```txt
PyTorch Geometric
DGL
Deep Graph Library
NetworkX + NumPy
GraphBolt
```

典型 GNN 框架会处理：

```txt
稀疏边索引
mini-batch subgraph sampling
message passing abstraction
neighbor sampling
GPU sparse operations
```

本课的手写实现用于理解原理，真实大图训练不要使用 dense adjacency matrix。

---

### 5.5 手写实现 vs 生产库

| 对比项 | 手写实现 | 生产库 |
|---|---|---|
| 图存储 | dict adjacency list | NetworkX / PyG edge_index |
| 邻接矩阵 | NumPy dense matrix | sparse matrix |
| BFS / DFS | 手写 queue / stack | NetworkX |
| 连通分量 | 手写 BFS | nx.connected_components |
| Laplacian | `D-A` | nx.laplacian_matrix |
| 谱聚类 | NumPy eigen | sklearn / scipy sparse eigensolver |
| PageRank | 手写迭代 | nx.pagerank |
| GNN | 手写 `A@H@W` | PyG / DGL |

---

## 6. 交付沉淀：产出物、连接关系、练习、术语表

---

### 6.1 本课产出

```txt
code/graph_theory.py
code/use_networkx.py
code/gcn_message_passing.py
outputs/skill-graph-analysis.md
outputs/exercises-solutions.md
```

| 文件 | 作用 |
|---|---|
| `code/graph_theory.py` | 从零实现 Graph、BFS、DFS、Laplacian、spectral clustering、PageRank |
| `code/use_networkx.py` | NetworkX 图分析、PageRank、社区发现 |
| `code/gcn_message_passing.py` | normalized adjacency、GCN layer、message passing demo |
| `outputs/skill-graph-analysis.md` | 图结构分析和建模策略 skill |
| `outputs/exercises-solutions.md` | 练习题与参考答案 |

---

### 6.2 知识连接关系

| 本课知识点 | 后续连接 |
|---|---|
| Graph | 关系建模 |
| Adjacency matrix | GNN 输入 |
| Degree matrix | Laplacian、normalization |
| BFS | shortest path、knowledge graph traversal |
| DFS | connected components、cycle detection |
| Laplacian | spectral clustering |
| Fiedler vector | graph partition |
| PageRank | node importance |
| Message passing | GCN、GAT、GraphSAGE |
| Normalized adjacency | GCN |
| Spectral gap | graph connectivity、random walk mixing |
| Degree distribution | power-law network analysis |

---

### 6.3 AI 应用连接

| 图论知识点 | AI 应用 |
|---|---|
| Adjacency matrix | GCN、GAT、GraphSAGE |
| Laplacian | Spectral clustering、ChebNet |
| BFS | 知识图谱多跳查询 |
| DFS | 图连通性和拓扑分析 |
| Degree | 节点重要性特征 |
| PageRank | ranking、node centrality |
| Message passing | 所有主流 GNN 层 |
| Spectral clustering | 无监督社区发现 |
| Connected components | 图预处理 |
| Normalized adjacency | 稳定 GNN 聚合 |
| Self-loop | 保留节点自身特征 |
| Fiedler vector | 图二分 |

---

### 6.4 练习

1. **从零实现 PageRank。**  
   从均匀分数开始，使用：

   ```text
   score(v) = (1-d)/n + d * sum(score(u)/out_degree(u))
   ```

   其中 `d=0.85`。  
   迭代到变化量 `<1e-6`，在一个小网页图上测试。

2. **用谱聚类找社区。**  
   构造两个 clique，只用一条边连接它们。  
   计算 Fiedler vector，验证谱聚类能把两个 clique 分开。  
   逐渐增加跨社区边，观察结果如何变化。

3. **实现 Dijkstra。**  
   对加权图实现 Dijkstra shortest path。  
   在所有边权为 1 的图上，比较 Dijkstra 和 BFS 的结果是否一致。

4. **两层 message passing。**  
   用两个不同权重矩阵连续做两轮 message passing。  
   解释为什么第二轮后节点包含 2-hop neighborhood 信息。

5. **分析 Karate Club 图。**  
   使用 NetworkX 的 Karate Club graph。  
   计算 degree distribution、Laplacian eigenvalues 和 spectral clustering。  
   与已知俱乐部分裂结果比较。

6. **GCN self-loop 对比。**  
   比较不加 self-loop 和加 self-loop 的 message passing。  
   观察节点自身特征是否会在聚合中保留。

---

### 6.5 关键术语

| 术语 | 常见说法 | 真正含义 |
|---|---|---|
| Graph | 图 | 由节点和边组成的关系结构 `G=(V,E)` |
| Node / Vertex | 节点 | 图中的对象或实体 |
| Edge | 边 | 节点之间的关系 |
| Directed graph | 有向图 | 边有方向 |
| Weighted graph | 加权图 | 边有数值权重 |
| Adjacency matrix | 邻接矩阵 | `A[i][j]` 表示 i 到 j 是否有边或边权 |
| Adjacency list | 邻接表 | 每个节点存储其邻居 |
| Degree | 度 | 节点连接的边数 |
| Degree matrix | 度矩阵 | 对角线为节点 degree |
| BFS | 广度优先搜索 | 按层遍历，可求无权最短路径 |
| DFS | 深度优先搜索 | 深入遍历，可找组件和环 |
| Connected component | 连通分量 | 任意两点可达的最大子图 |
| Laplacian | 图拉普拉斯 | `L=D-A`，用于谱图分析 |
| Fiedler value | Fiedler 值 | 最小非零拉普拉斯特征值，衡量连通性 |
| Fiedler vector | Fiedler 向量 | 对应 Fiedler value 的特征向量，可用于二分 |
| Spectral clustering | 谱聚类 | 用拉普拉斯特征向量聚类节点 |
| PageRank | 节点重要性算法 | 基于随机游走的 ranking |
| Message passing | 消息传递 | 节点聚合邻居信息更新表示 |
| GCN | 图卷积网络 | 用归一化邻接矩阵做 message passing |

---

### 6.6 自检问题

学完本课后，你应该能回答：

- 为什么图是关系型数据的自然表示？
- 有向图、无向图、加权图分别适合什么场景？
- 邻接矩阵如何表示图？
- 邻接表为什么适合稀疏图？
- degree matrix 如何构造？
- BFS 为什么能求无权最短路径？
- DFS 为什么适合连通分量和环检测？
- connected components 如何用 BFS/DFS 找到？
- 图拉普拉斯为什么定义为 `D-A`？
- 拉普拉斯零特征值个数为什么等于连通分量数？
- Fiedler value 和 Fiedler vector 分别表示什么？
- 谱聚类为什么能发现社区？
- `A @ H` 为什么表示邻居特征聚合？
- 一层 GNN 和两层 GNN 分别看到几跳邻居？
- GCN 为什么要加 self-loop？
- GCN 为什么要做 degree normalization？
- PageRank 的随机游走直觉是什么？
- 图论如何支撑 GNN、知识图谱和分子建模？

---

## 附录：本课最小代码文件建议

如果要拆成代码文件，可以这样组织：

```txt
code/
├── graph_theory.py
├── use_networkx.py
└── gcn_message_passing.py
```

### `graph_theory.py`

包含：

```txt
Graph
add_edge
neighbors
degree
adjacency_matrix
degree_matrix
laplacian
bfs
dfs
connected_components
laplacian_eigenvalues
spectral_clustering
pagerank
```

### `use_networkx.py`

包含：

```txt
nx.karate_club_graph
nx.adjacency_matrix
nx.laplacian_matrix
nx.connected_components
nx.pagerank
nx.community.greedy_modularity_communities
degree distribution
```

### `gcn_message_passing.py`

包含：

```txt
message_passing
normalized_adjacency_with_self_loops
gcn_layer
two_layer_message_passing_demo
```
