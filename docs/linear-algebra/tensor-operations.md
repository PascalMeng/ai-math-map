---
title: 张量操作
description: 面向 AI 算法工程师的张量课程：从 shape、axis、stride、reshape、transpose、broadcasting、einsum，到多头注意力中的完整 shape 流程。
---

# 张量操作

> 张量是数据和深度学习之间的共同语言。每一张图片、每一句话、每一个梯度，都会通过张量流动。

**课程类型：** 实现 / 应用  
**所属模块：** 张量与维度  
**前置知识：** Phase 1 Lesson 01-02：线性代数直觉、向量与矩阵基本运算  
**预计时间：** 约 90 分钟  
**使用语言：** Python / NumPy / PyTorch  

---

## 1. 提出问题：为什么 AI 算法工程师必须理解张量操作

你写了一个 Transformer，forward pass 看起来很干净。  
一运行，报错：

```txt
RuntimeError: mat1 and mat2 shapes cannot be multiplied (32x768 and 512x768)
```

你盯着 shape 看了半天，试着 transpose 一下。  
新的报错来了：

```txt
Expected 4D input (got 3D input)
```

你又加了一个 `unsqueeze`，结果另一个地方又炸了。

深度学习里最常见的 bug 之一就是 shape bug。  
它们看似琐碎，但会快速扩散：

```txt
一个 reshape 错了，后面的 matmul 会错。
一个 transpose 轴顺序错了，attention score 会错。
一个 broadcasting 维度错了，代码可能不报错，但结果是错的。
```

矩阵只能处理二维结构。  
真实深度学习数据往往远不止二维：

```txt
一批 RGB 图片：        (batch, channel, height, width)
一批 token embedding： (batch, seq_len, hidden_dim)
多头注意力：           (batch, heads, seq_len, head_dim)
卷积权重：             (out_channels, in_channels, kernel_h, kernel_w)
```

张量就是把向量和矩阵推广到任意维度后的统一数据结构。  
掌握张量操作，shape error 就会从“玄学报错”变成“可以逐行排查的工程问题”。

---

### 1.1 数学概念历史出现缘由

张量概念最早来自数学和物理，用来描述多维对象以及坐标变换下保持一致的量。  
在深度学习里，张量更常用作“多维数组”的工程抽象。

| 数学 / 工程概念 | 出现缘由 |
|---|---|
| 标量 | 表示单个数值 |
| 向量 | 表示一维有序数字列表 |
| 矩阵 | 表示二维数字表和线性变换 |
| 张量 | 表示任意维度的数字数组 |
| rank / order | 为了描述张量有多少个轴 |
| axis | 为了定位张量中的某一个维度 |
| shape | 为了描述每个轴的大小 |
| stride | 为了描述多维索引如何映射到底层一维内存 |
| reshape | 为了改变张量视图而不改变元素顺序 |
| transpose / permute | 为了交换或重排轴顺序 |
| broadcasting | 为了让不同 shape 的张量可以自动对齐运算 |
| einsum | 为了用统一表达式描述点积、矩阵乘法、转置、外积、批量乘法和 attention |
| contiguous | 为了区分逻辑顺序和物理内存顺序是否一致 |

可以这样理解：

```txt
现实问题：图片、文本、视频、attention 都不是简单矩阵。
工程抽象：张量

现实问题：如何知道每一维代表什么？
工程工具：shape 和 axis

现实问题：如何在不复制数据的情况下换形状？
工程机制：reshape、view、stride

现实问题：如何让 bias 自动加到 batch 上？
工程机制：broadcasting

现实问题：如何统一写 dot、matmul、batch matmul 和 attention？
数学记号：einsum
```

---

### 1.2 原始问题是什么

本课要解决的原始问题是：

```txt
什么是张量？
rank、axis、shape 分别是什么意思？
多维数组如何存储在一维内存中？
stride 如何把多维索引映射成内存位置？
reshape、squeeze、unsqueeze 分别在改变什么？
transpose 和 permute 为什么可能让张量 non-contiguous？
broadcasting 的严格规则是什么？
einsum 如何统一表达各种张量运算？
多头注意力中每一步 shape 如何变化？
```

换成 AI 语言，就是：

```txt
为什么 PyTorch 经常报 shape mismatch？
为什么 view 有时会失败，而 reshape 可以工作？
为什么 transpose 后要调用 contiguous？
为什么 bias 可以直接加到 activation 上？
为什么 attention score 的 shape 是 (B, H, T, T)？
为什么 einsum("bhtd,bhsd->bhts") 可以计算 attention scores？
为什么有些 broadcasting bug 不报错但模型结果错了？
```

---

### 1.3 AI 中的现代问题

| 张量知识点 | AI 中的现代问题 |
|---|---|
| shape | 调试模型维度错误 |
| rank / axis | 理解 batch、channel、sequence、head 等维度 |
| stride | 理解 view、transpose、contiguous |
| reshape | 拆分或合并维度 |
| squeeze / unsqueeze | 为 broadcasting 或模型接口补维 |
| transpose / permute | NCHW/NHWC、attention head 转换 |
| broadcasting | bias addition、normalization、mask |
| reduction | sum、mean、max、softmax |
| einsum | attention、batch matmul、张量 contraction |
| contiguous | PyTorch 内存布局与性能 |
| NCHW / NHWC | 视觉模型中的 layout 差异 |
| multi-head attention shapes | Transformer 工程实现核心 |
| pairwise distance broadcasting | 检索、聚类、对比学习 |
| global average pooling | CNN 输出汇聚 |
| sequence pooling | NLP 表示汇聚 |

---

### 1.4 本课要解决的核心问题

学完本课，你应该能够回答：

1. 张量和向量、矩阵的关系是什么？
2. rank、axis、shape 分别是什么意思？
3. shape `(B, C, H, W)` 和 `(B, T, D)` 在 AI 中分别表示什么？
4. stride 如何描述内存布局？
5. transpose 为什么通常不复制数据？
6. non-contiguous tensor 为什么会影响 `view`？
7. reshape、squeeze、unsqueeze 的作用分别是什么？
8. broadcasting 的规则是什么？
9. 如何用 broadcasting 做 bias addition、channel scaling 和 pairwise distance？
10. einsum 中哪些 index 会保留，哪些会被求和？
11. 如何用 einsum 表达 dot、outer、matmul、batch matmul 和 attention？
12. 多头注意力中 Q/K/V、scores、weights、output 的 shape 如何变化？
13. 如何从零实现一个简化 Tensor 类？
14. NumPy 和 PyTorch 中对应的生产级 API 是什么？

---

## 2. 概念定位：这节课学什么

---

### 2.1 所属模块

```txt
所属模块：张量与维度
前置知识：向量、矩阵、矩阵乘法、基础 Python 类
后续连接：CNN、Transformer、Attention、BatchNorm、LayerNorm、einsum、PyTorch debugging
```

前面的线性代数课程解决了：

```txt
向量和矩阵如何计算？
```

本课进一步解决：

```txt
任意维度的数据如何组织、变形、广播、收缩和传递？
```

---

### 2.2 本课学习目标

完成本课后，你应该能够：

- 解释 tensor、rank、axis、shape 和 stride
- 从零实现一个简化 Tensor 类
- 实现 shape、stride、reshape、squeeze、unsqueeze、transpose、permute
- 理解 contiguous 和 non-contiguous 的区别
- 实现 element-wise operations 和 reductions
- 掌握 broadcasting 规则
- 用 NumPy 实现常见 broadcasting 模式
- 用 einsum 表达 dot product、outer product、matrix multiplication、batch matmul 和 attention
- 追踪 multi-head attention 每一步的 shape
- 用 PyTorch 检查 shape、stride、is_contiguous 和 einsum

---

### 2.3 本课在整体路线中的位置

```txt
向量与矩阵
    ↓
张量与 shape
    ↓
reshape / transpose / broadcasting
    ↓
einsum / batch matmul
    ↓
CNN / Transformer / Attention
    ↓
复杂模型 shape debugging
```

本课的核心能力是：

```txt
看到任意模型代码时，能逐层写出每个张量的 shape，并判断每个操作是否合法。
```

---

## 3. 理解概念：直觉、公式、内存解释、AI 连接

---

### 3.1 什么是张量

**一句话直觉：**  
张量是任意维度的数字数组，是标量、向量和矩阵的统一推广。

| 对象 | rank / order | shape 示例 |
|---|---:|---|
| Scalar | 0 | `()` |
| Vector | 1 | `(3,)` |
| Matrix | 2 | `(2, 3)` |
| 3D Tensor | 3 | `(2, 2, 2)` |
| 4D Tensor | 4 | `(B, C, H, W)` |

总元素数量：

```text
numel = product(shape)
```

例如：

```txt
shape = (2, 3, 4)
numel = 2 * 3 * 4 = 24
```

**AI 连接：**

| 数据 | 常见 shape |
|---|---|
| 单个 embedding | `(D,)` |
| 一批 embedding | `(B, D)` |
| token 序列 | `(B, T, D)` |
| RGB 图片 batch | `(B, C, H, W)` |
| 多头注意力 | `(B, H, T, D)` |
| 视频 batch | `(B, T, C, H, W)` |

---

### 3.2 rank、axis 与 shape

**一句话直觉：**  
rank 是轴的数量，axis 是某一维，shape 是每个轴的大小。

以 NLP 张量为例：

```text
X.shape = (B, T, D)
```

含义：

| 轴 | 名称 | 含义 |
|---|---|---|
| axis 0 | `B` | batch size |
| axis 1 | `T` | sequence length |
| axis 2 | `D` | embedding / hidden dimension |

以图像张量为例：

```text
X.shape = (B, C, H, W)
```

含义：

| 轴 | 名称 | 含义 |
|---|---|---|
| axis 0 | `B` | batch |
| axis 1 | `C` | channel |
| axis 2 | `H` | height |
| axis 3 | `W` | width |

**AI 连接：**

如果不清楚每个 axis 代表什么，就很容易写错：

```python
x.mean(axis=1)
x.transpose(1, 2)
x.softmax(dim=-1)
```

这些操作的含义完全取决于 axis 的语义。

---

### 3.3 深度学习中的常见 shape 约定

**一句话直觉：**  
不同任务和框架对维度顺序有固定约定，搞错 layout 会导致错误或性能下降。

| 场景 | 常见 shape |
|---|---|
| PyTorch 图像 | `(B, C, H, W)` |
| TensorFlow 图像 | `(B, H, W, C)` |
| NLP hidden states | `(B, T, D)` |
| Multi-head attention | `(B, H, T, D_head)` |
| Linear weight | `(out_features, in_features)` |
| Conv2D weight | `(out_channels, in_channels, kH, kW)` |
| Embedding table | `(vocab_size, embedding_dim)` |

**AI 连接：**

| 问题 | 典型原因 |
|---|---|
| Conv2D 报 4D input 错误 | 少了 batch 或 channel 维 |
| Linear matmul mismatch | 输入最后一维和 weight in_features 不一致 |
| Attention score shape 错 | head split 或 transpose 轴顺序错 |
| 模型很慢 | layout 不符合框架优化路径 |
| silent bug | broadcasting 到了错误轴 |

---

### 3.4 内存布局与 stride

**一句话直觉：**  
张量逻辑上是多维的，但底层内存是一维的；stride 说明每个轴移动一步要跳过多少元素。

例如矩阵 shape：

```text
(3, 4)
```

row-major / C-order 下 stride 是：

```text
(4, 1)
```

含义：

```txt
行 axis 前进一步，要跳过 4 个元素。
列 axis 前进一步，要跳过 1 个元素。
```

多维 index 到 flat index 的映射：

```text
flat_index = i0 * stride0 + i1 * stride1 + ... + ik * stridek
```

**AI 连接：**

stride 解释了很多 PyTorch 行为：

```python
x.stride()
x.transpose(0, 1)
x.is_contiguous()
x.contiguous()
```

---

### 3.5 contiguous 与 non-contiguous

**一句话直觉：**  
contiguous 表示张量的逻辑顺序和内存物理顺序一致；transpose 常常只改 stride，不搬数据，因此会变成 non-contiguous。

例如：

```python
y = x.transpose(0, 1)
```

通常不会复制数据，而是：

```txt
交换 shape 和 stride 的解释方式。
```

这很高效，但会导致：

```python
y.view(...)
```

可能失败。

因为 `view` 要求张量内存连续。  
解决方式：

```python
y = y.contiguous().view(...)
```

或者：

```python
y = y.reshape(...)
```

**AI 连接：**

Transformer 中经常出现：

```python
x.transpose(...).contiguous().view(...)
```

它不是模板代码，而是在处理张量内存布局。

---

### 3.6 reshape、squeeze、unsqueeze

**一句话直觉：**  
reshape 改变张量形状，squeeze 删除大小为 1 的轴，unsqueeze 插入大小为 1 的轴。

#### reshape

```python
x.reshape(3, 4)
```

要求：

```text
新 shape 的元素总数 = 原 shape 的元素总数
```

可以使用 `-1` 自动推断一维：

```python
x.reshape(-1, 3)
```

#### squeeze

删除 size 为 1 的轴：

```python
x.shape = (1, 3, 1, 2)
x.squeeze().shape = (3, 2)
```

#### unsqueeze

插入一个 size 为 1 的轴：

```python
v.shape = (D,)
v.unsqueeze(0).shape = (1, D)
v.unsqueeze(0).unsqueeze(0).shape = (1, 1, D)
```

**AI 连接：**

bias 加到 `(B, T, D)` 上时，bias 可以视为：

```text
(1, 1, D)
```

这样就能沿 batch 和 sequence 自动 broadcast。

---

### 3.7 transpose 与 permute

**一句话直觉：**  
transpose 交换两个轴，permute 重排所有轴。

矩阵转置：

```python
x.transpose(0, 1)
```

图像 layout 转换：

```python
# NCHW -> NHWC
x.permute(0, 2, 3, 1)
```

注意：

```txt
transpose / permute 通常改变的是 axis 顺序，不一定复制数据。
```

**AI 连接：**

| 场景 | 操作 |
|---|---|
| NCHW 到 NHWC | `permute(0, 2, 3, 1)` |
| attention split head 后调整 | `(B, T, H, D) -> (B, H, T, D)` |
| matrix transpose | `(M, N) -> (N, M)` |
| batch matmul 准备 | 调整最后两个维度 |

---

### 3.8 element-wise operations 与 reductions

**一句话直觉：**  
element-wise 操作逐元素计算，reduction 会沿某些轴聚合。

Element-wise：

```python
c = a + b
d = a * 2
```

如果 shape 相同，输出 shape 不变。

Reduction：

```python
x.sum(axis=0)
x.mean(axis=1)
x.max(axis=-1)
```

会减少对应 axis。

**AI 连接：**

| 操作 | shape 变化 |
|---|---|
| CNN global average pooling | `(B, C, H, W) -> (B, C)` |
| NLP sequence mean pooling | `(B, T, D) -> (B, D)` |
| classification softmax | `(B, num_classes) -> (B, num_classes)` |
| attention softmax | `(B, H, T, S) -> (B, H, T, S)` |
| loss mean | `per-sample loss -> scalar` |

---

### 3.9 Broadcasting：不同 shape 的张量如何一起运算

**一句话直觉：**  
broadcasting 会从右对齐 shape，维度相等或其中一个为 1 时可以兼容。

规则：

```txt
1. 从右往左对齐 shape。
2. 两个维度相同，兼容。
3. 其中一个维度是 1，兼容。
4. 某个张量维度更少，则左侧补 1。
5. 其他情况不兼容。
```

例子：

```txt
A:        (8, 1, 6, 1)
B:           (7, 1, 5)
B padded: (1, 7, 1, 5)
Result:   (8, 7, 6, 5)
```

**AI 连接：**

| 场景 | Broadcasting |
|---|---|
| bias addition | `(B, D) + (D,) -> (B, D)` |
| sequence bias | `(B, T, D) + (D,) -> (B, T, D)` |
| channel scaling | `(B, C, H, W) * (1, C, 1, 1)` |
| attention mask | `(B, 1, 1, T)` broadcast 到 scores |
| pairwise distance | `(M, 1, D) - (1, N, D) -> (M, N, D)` |

---

### 3.10 Broadcasting 的 silent bug

**一句话直觉：**  
broadcasting 最危险的地方是：有时 shape 错了也不会报错，只会沿错误维度复制。

例如：

```txt
你想对 channel 做 scaling。
正确 shape 应该是 (1, C, 1, 1)。
但如果写成 (C,) 或 (1, 1, 1, C)，可能在错误轴上 broadcast。
```

这类 bug 可能不会报错，但模型结果会异常。

**AI 连接：**

调试 broadcasting 时要问：

```txt
我要对哪个 axis 共享？
我要对哪个 axis 独立？
这个 size=1 的维度是否放在正确位置？
```

---

### 3.11 Einsum：通用张量运算语言

**一句话直觉：**  
einsum 用字母给每个 axis 命名，输入中出现但输出中不出现的 index 会被求和。

规则：

```txt
输入和输出都出现的 index：保留。
输入出现但输出不出现的 index：求和消掉。
多个输入共享的 index：先相乘，再沿该 index 求和。
```

常见模式：

| 运算 | einsum |
|---|---|
| dot product | `"i,i->"` |
| outer product | `"i,j->ij"` |
| trace | `"ii->"` |
| transpose | `"ij->ji"` |
| matrix multiplication | `"ik,kj->ij"` |
| batch matmul | `"bij,bjk->bik"` |
| attention scores | `"bhtd,bhsd->bhts"` |
| attention output | `"bhts,bhsd->bhtd"` |

**AI 连接：**

einsum 非常适合表达：

- attention
- batch matmul
- tensor contraction
- covariance
- bilinear layer
- pairwise interaction
- 多维矩阵乘法

---

### 3.12 Multi-head Attention 的 shape 流程

**一句话直觉：**  
多头注意力就是一连串 projection、reshape、transpose、matmul、softmax 和再 reshape。

设：

```txt
B = batch size
T = sequence length
E = embedding dimension
H = num heads
D = head dimension
E = H * D
```

输入：

```text
X: (B, T, E)
```

#### Q/K/V projection

```text
Q = X @ W_q
K = X @ W_k
V = X @ W_v
```

shape：

```text
Q/K/V: (B, T, E)
```

#### split heads

```text
(B, T, E) -> (B, T, H, D) -> (B, H, T, D)
```

#### attention scores

```text
scores = Q @ K^T / sqrt(D)
```

einsum：

```python
scores = np.einsum("bhtd,bhsd->bhts", Q, K) / sqrt(D)
```

shape：

```text
scores: (B, H, T, T)
```

其中：

```txt
第一个 T 是 query position。
第二个 T 是 key position。
```

#### softmax

```text
weights = softmax(scores, axis=-1)
```

shape 不变：

```text
weights: (B, H, T, T)
```

#### weighted sum

```python
attn_output = np.einsum("bhts,bhsd->bhtd", weights, V)
```

shape：

```text
attn_output: (B, H, T, D)
```

#### merge heads

```text
(B, H, T, D) -> (B, T, H, D) -> (B, T, E)
```

#### output projection

```text
output = concat @ W_o
```

shape：

```text
output: (B, T, E)
```

---

### 3.13 本课核心概念总表

| 概念 | 一句话直觉 | AI 中的对应位置 |
|---|---|---|
| Tensor | 任意维数字数组 | 所有深度学习数据结构 |
| Rank | 轴的数量 | scalar/vector/matrix/ND tensor |
| Axis | 某一维 | batch、channel、time、head |
| Shape | 每个轴大小 | shape debugging |
| Stride | 轴移动对应内存跳步 | view/transpose/contiguous |
| Contiguous | 逻辑顺序和内存顺序一致 | PyTorch view |
| Reshape | 改变形状 | flatten、split、merge heads |
| Squeeze | 删除 size=1 的轴 | 清理维度 |
| Unsqueeze | 插入 size=1 的轴 | broadcasting |
| Transpose | 交换两个轴 | matrix transpose、attention |
| Permute | 重排所有轴 | NCHW/NHWC |
| Broadcasting | 自动扩展兼容维度 | bias、mask、normalization |
| Reduction | 沿轴聚合 | sum、mean、max |
| Einsum | 通用张量 contraction | attention、batch matmul |
| NCHW/NHWC | 图像 layout | PyTorch / TensorFlow 差异 |

---

## 4. 手写实现：从零实现 Tensor 的核心机制

这一节不用 PyTorch。  
目标是理解 tensor 的 shape、stride 和基础操作如何工作。

---

### 4.1 Tensor storage 与 strides

```python
from functools import reduce
import numpy as np


class Tensor:
    def __init__(self, data, shape=None):
        if isinstance(data, (list, tuple)):
            self._data, self._shape = self._flatten_nested(data)
        elif isinstance(data, np.ndarray):
            self._data = data.flatten().tolist()
            self._shape = tuple(data.shape)
        else:
            self._data = [data]
            self._shape = ()

        if shape is not None:
            total = reduce(lambda a, b: a * b, shape, 1)

            if total != len(self._data):
                raise ValueError(
                    f"Cannot reshape {len(self._data)} elements into shape {shape}"
                )

            self._shape = tuple(shape)

        self._strides = self._compute_strides(self._shape)

    @staticmethod
    def _compute_strides(shape):
        if len(shape) == 0:
            return ()

        strides = [1] * len(shape)

        for i in range(len(shape) - 2, -1, -1):
            strides[i] = strides[i + 1] * shape[i + 1]

        return tuple(strides)
```

对于：

```text
shape = (3, 4)
```

stride 为：

```text
(4, 1)
```

---

### 4.2 reshape、squeeze、unsqueeze

```python
t = Tensor(list(range(12)), shape=(2, 6))

r1 = t.reshape((3, 4))
r2 = t.reshape((-1, 3))
```

squeeze / unsqueeze：

```python
t = Tensor(list(range(6)), shape=(1, 3, 1, 2))

s = t.squeeze()

v = Tensor([1, 2, 3])
u = v.unsqueeze(0)
```

用途：

| 操作 | 作用 |
|---|---|
| `reshape` | 改变 shape，但元素总数不变 |
| `squeeze` | 删除大小为 1 的维度 |
| `unsqueeze` | 增加大小为 1 的维度，常用于 broadcasting |

---

### 4.3 transpose 和 permute

```python
mat = Tensor(list(range(6)), shape=(2, 3))

tr = mat.transpose(0, 1)
```

4D tensor permute：

```python
t4d = Tensor(list(range(24)), shape=(1, 2, 3, 4))

perm = t4d.permute((0, 2, 3, 1))
```

这相当于：

```txt
(1, 2, 3, 4) -> (1, 3, 4, 2)
```

也就是常见的：

```txt
NCHW -> NHWC
```

---

### 4.4 Element-wise operations 与 reductions

```python
a = Tensor([[1, 2], [3, 4]])
b = Tensor([[10, 20], [30, 40]])

c = a + b
d = a * 2

s0 = a.sum(axis=0)
s1 = a.sum(axis=1)
```

Reduction 示例：

| 输入 shape | 操作 | 输出 shape |
|---|---|---|
| `(B, C, H, W)` | `mean(axis=[2, 3])` | `(B, C)` |
| `(B, T, D)` | `mean(axis=1)` | `(B, D)` |
| `(B, num_classes)` | `max(axis=-1)` | `(B,)` |

---

## 5. 生产使用：NumPy / PyTorch 张量操作

---

### 5.1 NumPy Broadcasting 示例

```python
import numpy as np

activations = np.random.randn(4, 3)
bias = np.array([0.1, 0.2, 0.3])

result = activations + bias

print(result.shape)
```

channel scaling：

```python
images = np.random.randn(2, 3, 4, 4)

scale = np.array([0.5, 1.0, 1.5]).reshape(1, 3, 1, 1)

result = images * scale

print(result.shape)
```

outer product：

```python
a = np.array([1, 2, 3]).reshape(-1, 1)
b = np.array([10, 20, 30, 40]).reshape(1, -1)

outer = a * b

print(outer.shape)
```

pairwise distance：

```python
X = np.random.randn(5, 2)
Y = np.random.randn(7, 2)

diff = X[:, None, :] - Y[None, :, :]
dist = np.sqrt((diff ** 2).sum(axis=-1))

print(dist.shape)
```

---

### 5.2 NumPy einsum 示例

```python
import numpy as np

a = np.array([1.0, 2.0, 3.0])
b = np.array([4.0, 5.0, 6.0])

dot = np.einsum("i,i->", a, b)
```

矩阵乘法：

```python
A = np.array([[1, 2], [3, 4], [5, 6]], dtype=float)
B = np.array([[7, 8, 9], [10, 11, 12]], dtype=float)

matmul = np.einsum("ik,kj->ij", A, B)
```

batch matmul：

```python
batch_A = np.random.randn(4, 3, 5)
batch_B = np.random.randn(4, 5, 2)

batch_mm = np.einsum("bij,bjk->bik", batch_A, batch_B)
```

---

### 5.3 Multi-head attention via einsum

```python
import numpy as np

B, H, T, D = 2, 4, 8, 16
E = H * D

X = np.random.randn(B, T, E)

W_q = np.random.randn(E, E) * 0.02
W_k = np.random.randn(E, E) * 0.02
W_v = np.random.randn(E, E) * 0.02
W_o = np.random.randn(E, E) * 0.02

Q = np.einsum("bte,ek->btk", X, W_q)
K = np.einsum("bte,ek->btk", X, W_k)
V = np.einsum("bte,ek->btk", X, W_v)

Q = Q.reshape(B, T, H, D).transpose(0, 2, 1, 3)
K = K.reshape(B, T, H, D).transpose(0, 2, 1, 3)
V = V.reshape(B, T, H, D).transpose(0, 2, 1, 3)

scores = np.einsum("bhtd,bhsd->bhts", Q, K) / np.sqrt(D)

weights = np.exp(scores - scores.max(axis=-1, keepdims=True))
weights = weights / weights.sum(axis=-1, keepdims=True)

attn_output = np.einsum("bhts,bhsd->bhtd", weights, V)

concat = attn_output.transpose(0, 2, 1, 3).reshape(B, T, E)

output = np.einsum("bte,ek->btk", concat, W_o)

print(output.shape)
```

---

### 5.4 PyTorch 张量操作

```python
import torch

t = torch.tensor(
    [[1, 2, 3], [4, 5, 6]],
    dtype=torch.float32
)

print(t.shape)
print(t.stride())
print(t.is_contiguous())

print(t.reshape(3, 2).shape)
print(t.unsqueeze(0).shape)
print(t.transpose(0, 1).shape)
print(t.transpose(0, 1).contiguous().is_contiguous())
```

PyTorch einsum：

```python
A = torch.randn(3, 5)
B = torch.randn(5, 2)

C = torch.einsum("ik,kj->ij", A, B)

print(C.shape)
```

---

### 5.5 手写 Tensor vs NumPy / PyTorch

| 操作 | 手写 Tensor | NumPy | PyTorch |
|---|---|---|---|
| 创建 | `Tensor([[1,2],[3,4]])` | `np.array(...)` | `torch.tensor(...)` |
| shape | `t.shape` | `a.shape` | `t.shape` |
| stride | `t.strides` | `a.strides` | `t.stride()` |
| reshape | `t.reshape((3,4))` | `a.reshape(3,4)` | `t.reshape(3,4)` |
| transpose | `t.transpose(0,1)` | `a.transpose(0,1)` | `t.transpose(0,1)` |
| squeeze | `t.squeeze(0)` | `np.squeeze(a, 0)` | `t.squeeze(0)` |
| unsqueeze | `t.unsqueeze(0)` | `np.expand_dims(a, 0)` | `t.unsqueeze(0)` |
| einsum | 手写可选 | `np.einsum` | `torch.einsum` |
| autograd | 不支持 | 不支持 | 支持 |
| GPU | 不支持 | 通常不支持 | 支持 |

---

## 6. 交付沉淀：产出物、连接关系、练习、术语表

---

### 6.1 本课产出

```txt
code/tensors.py
code/use_numpy.py
code/use_pytorch.py
outputs/prompt-tensor-shapes.md
outputs/prompt-tensor-debugger.md
outputs/exercises-solutions.md
```

| 文件 | 作用 |
|---|---|
| `code/tensors.py` | 从零实现 Tensor、shape、stride、reshape、transpose、element-wise operations |
| `code/use_numpy.py` | NumPy broadcasting、einsum、attention 示例 |
| `code/use_pytorch.py` | PyTorch shape、stride、contiguous、einsum 示例 |
| `outputs/prompt-tensor-shapes.md` | 系统化排查 tensor shape mismatch 的 prompt |
| `outputs/prompt-tensor-debugger.md` | 输入报错和 shape 后生成修复建议的 prompt |
| `outputs/exercises-solutions.md` | 练习题与参考答案 |

---

### 6.2 知识连接关系

| 本课知识点 | 后续连接 |
|---|---|
| Tensor | 深度学习所有数据结构 |
| Shape | 模型 debugging |
| Stride | contiguous、view、reshape |
| Broadcasting | bias、mask、normalization |
| Reduction | pooling、loss、softmax |
| Einsum | attention、batch matmul |
| NCHW / NHWC | CNN layout |
| Multi-head attention shapes | Transformer 实现 |
| Pairwise distance | 检索、聚类、对比学习 |

---

### 6.3 AI 应用连接

| 数学 / 工程知识点 | AI 应用 |
|---|---|
| Tensor shape | 模型输入输出约定 |
| Reshape | flatten、head split、head merge |
| Transpose / permute | attention、layout conversion |
| Broadcasting | bias、mask、scale、normalization |
| Einsum | attention scores、batch matmul |
| Reduction | pooling、loss aggregation |
| Contiguous | PyTorch view 和性能 |
| NCHW/NHWC | CV 框架差异 |
| Shape tracing | Transformer debugging |

---

### 6.4 练习

1. **Reshape round-trip。**  
   构造一个 shape 为 `(2, 3, 4)` 的张量。  
   依次 reshape 为 `(6, 4)`、`(24,)`，再 reshape 回 `(2, 3, 4)`。  
   打印 flat data，验证元素顺序没有改变。

2. **实现 broadcasting。**  
   给手写 `Tensor` 类添加 `broadcast_to(shape)` 方法。  
   然后修改 `_elementwise_op`，让不同 shape 的张量可以自动 broadcast。  
   测试：

   ```txt
   (3, 1) + (1, 4) -> (3, 4)
   ```

3. **从零实现简化 einsum。**  
   实现一个基本 `einsum(subscripts, *tensors)`，至少支持：

   ```txt
   i,i->
   ij,jk->ik
   i,j->ij
   ij->ji
   ```

   和 `np.einsum` 对比结果。

4. **Attention shape tracker。**  
   写一个函数，输入：

   ```txt
   batch_size
   seq_len
   embed_dim
   num_heads
   ```

   输出 multi-head attention 每一步 shape：input、Q/K/V projection、head split、attention scores、softmax weights、weighted sum、head merge、output projection。

5. **Broadcasting bug 诊断。**  
   给定图像 tensor `(B, C, H, W)` 和 scale vector `(C,)`，分别尝试 `(C,)`、`(1,C,1,1)`、`(1,1,1,C)`，解释哪些是正确的，哪些可能造成 silent bug。

---

### 6.5 关键术语

| 术语 | 常见说法 | 真正含义 |
|---|---|---|
| Tensor | 多维矩阵 | 具有统一数据类型、shape、stride 和操作规则的多维数组 |
| Rank / Order | 维度数 | 张量的轴数量，注意不同于矩阵 rank |
| Axis | 轴 | 张量中的某一个维度 |
| Shape | 形状 | 每个 axis 的大小组成的 tuple |
| Stride | 步长 | 沿某个 axis 前进一步，需要在底层内存跳过多少元素 |
| Broadcasting | 自动扩展 | 从右对齐 shape，维度相等或其中一个为 1 时兼容 |
| Contiguous | 内存连续 | 逻辑顺序和物理内存顺序一致 |
| View | 共享内存视图 | 不复制数据，只改变 shape/stride metadata |
| Reshape | 改形状 | 可能返回 view，也可能复制数据 |
| Transpose | 交换轴 | 常通过修改 stride 实现 |
| Permute | 重排轴 | 任意重排所有维度 |
| Reduction | 归约 | 沿某些轴求和、均值、最大值等 |
| Einsum | 爱因斯坦求和 | 用 index notation 表达张量 contraction |
| Contraction | 收缩 | 对共享 index 相乘并求和 |
| NCHW / NHWC | 图像 layout | channel-first 与 channel-last 两种格式 |

---

### 6.6 自检问题

学完本课后，你应该能回答：

- 张量为什么是深度学习的核心数据结构？
- rank、axis、shape 分别是什么意思？
- `(B, C, H, W)` 和 `(B, T, D)` 每个维度代表什么？
- stride 如何把多维索引映射到一维内存？
- transpose 为什么通常不移动数据？
- non-contiguous tensor 为什么会让 `view` 失败？
- reshape、squeeze、unsqueeze 各自解决什么问题？
- broadcasting 的规则是什么？
- 为什么 broadcasting 可能产生 silent bug？
- einsum 中哪些 index 会被求和？
- 如何用 einsum 写矩阵乘法和 batch matmul？
- multi-head attention 每一步 shape 如何变化？
- PyTorch 中 `shape`、`stride()`、`is_contiguous()` 分别怎么看？
- 如何系统排查 shape mismatch？

---

## 附录：本课最小代码文件建议

如果要拆成代码文件，可以这样组织：

```txt
code/
├── tensors.py
├── use_numpy.py
└── use_pytorch.py
```

### `tensors.py`

包含：

```txt
Tensor
_compute_strides
reshape
squeeze
unsqueeze
transpose
permute
elementwise operations
sum / mean reductions
broadcast_to
```

### `use_numpy.py`

包含：

```txt
broadcasting examples
pairwise distance
np.einsum dot / outer / matmul / batch matmul
multi-head attention via einsum
```

### `use_pytorch.py`

包含：

```txt
torch.tensor
shape
stride()
is_contiguous()
reshape
view
contiguous()
transpose
permute
torch.einsum
attention shape tracker
```
