---
title: 向量、矩阵与基本运算
description: 面向 AI 算法工程师的线性代数课程：从向量、矩阵、shape、矩阵乘法、广播机制到神经网络线性层实现。
---

# 向量、矩阵与基本运算

> 每个神经网络，本质上都是矩阵乘法加上一些非线性操作。

**课程类型：** 实现 / 应用  
**所属模块：** 线性代数  
**前置知识：** Phase 1 Lesson 01：线性代数直觉  
**预计时间：** 约 60 分钟  
**使用语言：** Python / NumPy / PyTorch / Julia  

---

## 1. 提出问题：为什么 AI 算法工程师必须熟练掌握矩阵运算

当你开始写神经网络时，很快会看到这样的代码：

```python
output = activation(weights @ input + bias)
```

这行代码看起来很短，但里面包含了一个神经网络层的核心计算：

```txt
矩阵乘法 -> 加偏置 -> 激活函数
```

如果你不理解 `@`、`weights`、`input`、`bias` 的数学含义，这行代码就是黑箱。  
如果你理解向量、矩阵、shape、矩阵乘法和广播机制，那么它就是一次清晰的空间变换。

---

### 1.1 数学概念历史出现缘由

向量、矩阵和矩阵运算并不是为深度学习发明的。它们最早来自更基础的数学、物理和工程问题。

| 数学概念 | 历史出现缘由 |
|---|---|
| 向量 | 为了描述力、速度、位移等既有大小又有方向的量 |
| 矩阵 | 为了系统组织和求解多元线性方程组 |
| 矩阵乘法 | 为了表达多个线性变换的复合 |
| 转置 | 为了交换行列视角，方便表达线性映射和内积 |
| 行列式 | 为了判断线性变换对面积或体积的缩放，以及矩阵是否可逆 |
| 逆矩阵 | 为了“撤销”一个线性变换，求解线性系统 |
| 单位矩阵 | 为了表示“不改变对象”的恒等变换 |
| 广播机制 | 来自数组计算中的工程需求，用更简洁的方式表达重复加法或重复运算 |

可以这样理解：

```txt
现实问题：如何表示方向和大小？
数学抽象：向量

现实问题：如何同时处理很多线性关系？
数学抽象：矩阵

现实问题：如何连续执行多个线性变换？
数学抽象：矩阵乘法

现实问题：如何判断一个变换能不能被撤销？
数学抽象：行列式和逆矩阵

现实问题：如何在批量数据上统一加偏置？
工程抽象：广播机制
```

---

### 1.2 原始问题是什么

本课要解决的原始问题是：

```txt
如何用向量表示数据？
如何用矩阵表示变换？
如何用矩阵乘法完成神经网络层的前向传播？
如何理解 shape 匹配规则？
如何区分逐元素乘法和矩阵乘法？
如何理解 bias 在神经网络中的广播加法？
```

换成 AI 语言，就是：

```txt
input 是什么 shape？
weights 是什么 shape？
weights @ input 为什么能得到 output？
bias 为什么可以直接加到 output 上？
为什么 PyTorch 经常报 shape mismatch？
为什么 A * B 和 A @ B 完全不是一回事？
```

---

### 1.3 AI 中的现代问题

在 AI 工程里，矩阵运算几乎无处不在。

| 数学知识点 | AI 中的现代问题 |
|---|---|
| 向量 | 如何表示一个样本、一个 token、一个 embedding |
| 矩阵 | 如何表示模型权重、图像像素、线性层参数 |
| shape | 如何判断张量能不能相乘、能不能相加 |
| 矩阵乘法 | 如何实现神经网络线性层 |
| 逐元素运算 | 如何做 mask、门控、特征缩放 |
| 转置 | 如何调整维度，参与反向传播和 attention |
| 行列式 | 如何判断矩阵是否可逆，是否压缩了维度 |
| 逆矩阵 | 如何求解线性系统 |
| 单位矩阵 | 如何理解残差连接和恒等映射 |
| 广播机制 | 如何理解 bias addition 和 batch 计算 |

---

### 1.4 本课要解决的核心问题

学完本课，你应该能够回答：

1. 向量和矩阵在 AI 中分别表示什么？
2. 为什么神经网络层可以写成 `W @ x + b`？
3. 矩阵乘法的 shape 规则是什么？
4. 为什么 `(m x n) @ (n x p) = (m x p)`？
5. 逐元素乘法和矩阵乘法有什么区别？
6. bias 为什么可以自动加到每一行或每一个样本上？
7. 行列式和逆矩阵在几何上是什么意思？
8. 如何从零实现一个最小 Matrix 类？
9. 如何只用手写 Matrix 类实现一个 dense layer？
10. NumPy 和 PyTorch 中对应的生产级 API 是什么？

---

## 2. 概念定位：这节课学什么

---

### 2.1 所属模块

```txt
所属模块：线性代数
前置知识：线性代数直觉、向量、点积、基础 Python 类
后续连接：神经网络线性层、反向传播、张量操作、Transformer attention、mini neural network framework
```

本课是“线性代数直觉”之后的实现型课程。  
上一课重点是理解向量、矩阵、点积、投影和秩的直觉；本课重点是把这些直觉变成可运行的代码。

---

### 2.2 本课学习目标

完成本课后，你应该能够：

- 从零实现 `Vector` 类和 `Matrix` 类
- 实现矩阵加法、减法、标量乘法、逐元素乘法、矩阵乘法和转置
- 实现 2x2 矩阵的行列式和逆矩阵
- 区分逐元素乘法和矩阵乘法
- 解释矩阵乘法的 shape 规则
- 解释广播机制如何支持 bias addition
- 使用手写 Matrix 类实现一个 dense neural network layer
- 使用 NumPy 完成同样的矩阵运算

---

### 2.3 本课在整体路线中的位置

```txt
线性代数直觉
    ↓
向量、矩阵与基本运算
    ↓
张量操作与 broadcasting
    ↓
神经网络线性层
    ↓
自动微分与反向传播
    ↓
完整 mini neural network framework
```

本课的核心是建立一种能力：

```txt
看到神经网络代码时，能立刻判断每个张量的 shape，以及每一步矩阵运算在做什么。
```

---

## 3. 理解概念：直觉、公式、几何解释、AI 连接

---

### 3.1 向量：有顺序的数字列表

**一句话直觉：**  
向量是有顺序的一组数字，可以表示空间中的点、方向、特征或参数。

例如：

```txt
v = [3, 4]
w = [1, 0, -2]
```

二维向量 `[3, 4]` 可以看作平面中的点 `(3, 4)`，也可以看作从原点指向 `(3, 4)` 的箭头。它的长度是：

```text
||v|| = sqrt(3^2 + 4^2) = 5
```

**AI 连接：**

| AI 对象 | 向量含义 |
|---|---|
| 一个样本 | 特征向量 |
| 一个词 | word embedding |
| 一个 token | hidden state |
| 一个用户 | 用户偏好向量 |
| 模型参数的一部分 | 参数向量 |

---

### 3.2 矩阵：二维数字网格

**一句话直觉：**  
矩阵是二维数字表，也可以理解为一个线性变换。

例如：

```txt
A = | 1  2  3 |
    | 4  5  6 |
```

这是一个 `2 x 3` 矩阵：

```txt
2 行，3 列
```

在神经网络中，如果一层有 `784` 个输入和 `128` 个输出，那么它的权重矩阵通常可以写成：

```txt
W: 128 x 784
```

它把一个 `784` 维输入向量变成一个 `128` 维输出向量。

**AI 连接：**

| AI 对象 | 矩阵含义 |
|---|---|
| 图像 | 像素矩阵 |
| 权重矩阵 | 把输入特征映射为输出特征 |
| embedding table | token id 到向量的映射表 |
| attention score matrix | token 与 token 之间的关系矩阵 |

---

### 3.3 shape：矩阵乘法的合法性规则

**一句话直觉：**  
shape 决定两个矩阵能不能相乘，以及相乘后输出是什么形状。

矩阵乘法规则：

```text
(m x n) @ (n x p) = (m x p)
```

中间两个维度必须相等。

例子：

```txt
(128 x 784) @ (784 x 1) = (128 x 1)
  weights       input       output
```

解释：

| 位置 | 含义 |
|---|---|
| `128` | 输出维度 |
| `784` | 输入维度 |
| `784 = 784` | 内部维度匹配，所以可以相乘 |
| `128 x 1` | 输出向量 shape |

**AI 连接：**

PyTorch 中常见的错误：

```txt
RuntimeError: mat1 and mat2 shapes cannot be multiplied
```

本质上就是矩阵乘法的内部维度不匹配。

---

### 3.4 逐元素运算：对应位置分别计算

**一句话直觉：**  
逐元素运算就是两个同形状数组的对应位置分别计算。

逐元素乘法：

```txt
| 1  2 |   | 5  6 |   | 5   12 |
| 3  4 | * | 7  8 | = | 21  32 |
```

要求：

```txt
两个矩阵 shape 相同，或者可以广播到相同 shape。
```

**AI 连接：**

| 场景 | 逐元素运算用途 |
|---|---|
| 激活函数 | 对每个元素单独应用 ReLU / sigmoid |
| mask | 对指定位置保留或屏蔽 |
| dropout | 对每个元素随机置零 |
| 门控机制 | 对应位置控制信息通过 |
| 梯度缩放 | 每个参数单独缩放 |

---

### 3.5 矩阵乘法：行和列的点积

**一句话直觉：**  
矩阵乘法不是对应位置相乘，而是“第一矩阵的行”和“第二矩阵的列”做点积。

例子：

```txt
| 1  2 |   | 5  6 |   | 1*5+2*7  1*6+2*8 |   | 19  22 |
| 3  4 | @ | 7  8 | = | 3*5+4*7  3*6+4*8 | = | 43  50 |
```

矩阵乘法和逐元素乘法完全不同：

| 运算 | 规则 | 结果 |
|---|---|---|
| `A * B` | 对应位置相乘 | shape 通常不变 |
| `A @ B` | 行列点积 | shape 按 `(m x n) @ (n x p)` 变化 |

**AI 连接：**

神经网络线性层：

```text
output = W @ x + b
```

本质上就是用权重矩阵 `W` 对输入向量 `x` 做一次线性变换。

---

### 3.6 转置：交换行和列

**一句话直觉：**  
转置就是把矩阵的行变成列，把列变成行。

如果：

```txt
A: m x n
```

那么：

```txt
A^T: n x m
```

**AI 连接：**

| 场景 | 转置作用 |
|---|---|
| attention | `Q @ K^T` 计算 query 和 key 的相似度 |
| 反向传播 | 梯度计算中经常需要权重矩阵转置 |
| shape 调整 | 让矩阵乘法维度匹配 |
| batch 计算 | 调整不同维度顺序 |

---

### 3.7 行列式：判断变换是否压扁空间

**一句话直觉：**  
行列式衡量矩阵对面积或体积的缩放程度；如果行列式为 0，说明某个维度被压扁了，矩阵不可逆。

2x2 矩阵：

```text
A = [[a, b],
     [c, d]]
```

行列式：

```text
det(A) = ad - bc
```

几何解释：

| 行列式 | 几何含义 |
|---|---|
| `det(A) > 0` | 保持方向，面积被缩放 |
| `det(A) < 0` | 方向翻转，面积被缩放 |
| `det(A) = 0` | 空间被压扁，面积为 0，不可逆 |

**AI 连接：**

虽然深度学习里不常直接手算行列式，但它有助于理解：

- 矩阵是否可逆
- 特征是否冗余
- 线性变换是否丢失维度
- normalizing flows 中体积变化

---

### 3.8 逆矩阵：撤销一个线性变换

**一句话直觉：**  
逆矩阵是用来撤销原矩阵变换的矩阵。

如果：

```text
A @ A^-1 = I
```

那么 `A^-1` 就是 `A` 的逆矩阵。

2x2 矩阵的逆：

```text
A^-1 = 1/det(A) * [[d, -b],
                   [-c, a]]
```

前提是：

```text
det(A) != 0
```

**AI 连接：**

| 场景 | 逆矩阵思想 |
|---|---|
| 线性系统 | 解 `Ax = b` |
| 最小二乘 | 求解线性回归 |
| covariance inverse | Mahalanobis distance |
| normalizing flow | 可逆变换 |

---

### 3.9 单位矩阵：不改变对象的变换

**一句话直觉：**  
单位矩阵就是矩阵里的“1”，任何矩阵或向量乘以它都不变。

二维单位矩阵：

```txt
I = | 1  0 |
    | 0  1 |
```

满足：

```text
A @ I = A
I @ A = A
```

**AI 连接：**

| 场景 | 单位矩阵思想 |
|---|---|
| 初始化 | 从接近恒等映射开始 |
| 残差连接 | 保留原始输入 |
| 数值稳定 | 添加 identity regularization |
| 线性系统 | 验证逆矩阵是否正确 |

---

### 3.10 广播机制：自动扩展小数组

**一句话直觉：**  
广播机制会把较小的数组“自动扩展”到兼容 shape，再进行逐元素运算。

例子：

```txt
| 1  2  3 |   +   [10, 20, 30]
| 4  5  6 |
```

广播后等价于：

```txt
| 1  2  3 |   | 10  20  30 |   | 11  22  33 |
| 4  5  6 | + | 10  20  30 | = | 14  25  36 |
```

**AI 连接：**

在神经网络中，bias 经常是一个向量，但 output 是一个 batch 矩阵。框架会自动把 bias 广播到每个样本上：

```text
output = W @ x + b
```

或者 batch 场景：

```text
output = X @ W + b
```

其中 `b` 会自动加到每一行输出上。

---

### 3.11 本课核心概念总表

| 数学概念 | 一句话直觉 | AI 中的对应位置 |
|---|---|---|
| 向量 | 有顺序的数字列表 | embedding、feature vector |
| 矩阵 | 二维数字表 / 线性变换 | weight matrix、image pixels |
| shape | 张量的尺寸规则 | debug PyTorch shape error |
| 矩阵加法 | 对应元素相加 | bias / residual addition |
| 标量乘法 | 所有元素同时缩放 | learning rate * gradient |
| 逐元素乘法 | 对应位置相乘 | mask、gate、dropout |
| 矩阵乘法 | 行和列做点积 | dense layer forward pass |
| 转置 | 交换行和列 | backprop、attention |
| 行列式 | 变换对面积/体积的缩放 | invertibility、flows |
| 逆矩阵 | 撤销线性变换 | linear systems |
| 单位矩阵 | 不改变对象 | residual、identity mapping |
| 广播 | 自动扩展 shape | bias addition |

---

## 4. 手写实现：从零实现向量、矩阵和神经网络层

这一节先不用 NumPy、PyTorch 或 JAX。  
目标是让你知道生产库底层在做什么。

---

### 4.1 实现基础对象：Vector

```python
class Vector:
    def __init__(self, data):
        self.data = list(data)
        self.size = len(self.data)

    def __repr__(self):
        return f"Vector({self.data})"

    def __add__(self, other):
        return Vector([a + b for a, b in zip(self.data, other.data)])

    def __sub__(self, other):
        return Vector([a - b for a, b in zip(self.data, other.data)])

    def __mul__(self, scalar):
        return Vector([x * scalar for x in self.data])

    def dot(self, other):
        return sum(a * b for a, b in zip(self.data, other.data))

    def magnitude(self):
        return sum(x ** 2 for x in self.data) ** 0.5
```

这个类实现了：

| 方法 | 数学意义 |
|---|---|
| `__add__` | 向量加法 |
| `__sub__` | 向量减法 |
| `__mul__` | 标量乘法 |
| `dot` | 点积 |
| `magnitude` | 向量长度 |

---

### 4.2 实现基础对象：Matrix

```python
class Matrix:
    def __init__(self, data):
        self.data = [list(row) for row in data]
        self.rows = len(self.data)
        self.cols = len(self.data[0])
        self.shape = (self.rows, self.cols)

    def __repr__(self):
        rows_str = "\n  ".join(str(row) for row in self.data)
        return f"Matrix({self.shape}):\n  {rows_str}"

    def __add__(self, other):
        return Matrix([
            [self.data[i][j] + other.data[i][j] for j in range(self.cols)]
            for i in range(self.rows)
        ])

    def __sub__(self, other):
        return Matrix([
            [self.data[i][j] - other.data[i][j] for j in range(self.cols)]
            for i in range(self.rows)
        ])

    def scalar_multiply(self, scalar):
        return Matrix([
            [self.data[i][j] * scalar for j in range(self.cols)]
            for i in range(self.rows)
        ])

    def element_wise_multiply(self, other):
        return Matrix([
            [self.data[i][j] * other.data[i][j] for j in range(self.cols)]
            for i in range(self.rows)
        ])

    def matmul(self, other):
        return Matrix([
            [
                sum(self.data[i][k] * other.data[k][j] for k in range(self.cols))
                for j in range(other.cols)
            ]
            for i in range(self.rows)
        ])

    def transpose(self):
        return Matrix([
            [self.data[j][i] for j in range(self.rows)]
            for i in range(self.cols)
        ])

    def determinant(self):
        if self.shape == (1, 1):
            return self.data[0][0]

        if self.shape == (2, 2):
            return self.data[0][0] * self.data[1][1] - self.data[0][1] * self.data[1][0]

        det = 0

        for j in range(self.cols):
            minor = Matrix([
                [self.data[i][k] for k in range(self.cols) if k != j]
                for i in range(1, self.rows)
            ])
            det += ((-1) ** j) * self.data[0][j] * minor.determinant()

        return det

    def inverse_2x2(self):
        det = self.determinant()

        if det == 0:
            raise ValueError("Matrix is singular, no inverse exists")

        return Matrix([
            [self.data[1][1] / det, -self.data[0][1] / det],
            [-self.data[1][0] / det, self.data[0][0] / det]
        ])

    @staticmethod
    def identity(n):
        return Matrix([
            [1 if i == j else 0 for j in range(n)]
            for i in range(n)
        ])
```

这个类实现了：

| 方法 | 数学意义 |
|---|---|
| `__add__` | 矩阵加法 |
| `__sub__` | 矩阵减法 |
| `scalar_multiply` | 标量乘法 |
| `element_wise_multiply` | 逐元素乘法 |
| `matmul` | 矩阵乘法 |
| `transpose` | 转置 |
| `determinant` | 行列式 |
| `inverse_2x2` | 2x2 逆矩阵 |
| `identity` | 单位矩阵 |

---

### 4.3 验证矩阵运算

```python
A = Matrix([[1, 2], [3, 4]])
B = Matrix([[5, 6], [7, 8]])

print("A + B =", (A + B).data)
print("A @ B =", A.matmul(B).data)
print("A^T =", A.transpose().data)
print("det(A) =", A.determinant())
print("A^-1 =", A.inverse_2x2().data)

I = Matrix.identity(2)
print("A @ A^-1 =", A.matmul(A.inverse_2x2()).data)
```

你应该看到：

```txt
A @ A^-1 近似等于单位矩阵 I
```

这说明 `inverse_2x2` 的实现是正确的。

---

### 4.4 用手写 Matrix 实现一个 dense layer

现在用刚刚写的 Matrix 类实现一个神经网络线性层：

```python
import random

inputs = Matrix([[0.5], [0.8], [0.2]])

weights = Matrix([
    [random.uniform(-1, 1) for _ in range(3)]
    for _ in range(2)
])

bias = Matrix([[0.1], [0.1]])


def relu_matrix(m):
    return Matrix([
        [max(0, val) for val in row]
        for row in m.data
    ])


pre_activation = weights.matmul(inputs) + bias
output = relu_matrix(pre_activation)

print(f"Input shape: {inputs.shape}")
print(f"Weight shape: {weights.shape}")
print(f"Output shape: {output.shape}")
print(f"Output: {output.data}")
```

这就是一个 dense layer：

```text
output = relu(W @ x + b)
```

对应关系：

| 代码 | 数学含义 |
|---|---|
| `weights.matmul(inputs)` | `W @ x` |
| `+ bias` | 加偏置 |
| `relu_matrix` | 非线性激活 |
| `output` | 当前层输出 |

---

## 5. 生产使用：用 NumPy 完成同样任务

---

### 5.1 NumPy 版本：基础矩阵运算

```python
import numpy as np

A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])

print("A + B =\n", A + B)
print("A * B (element-wise) =\n", A * B)
print("A @ B (matrix multiply) =\n", A @ B)
print("A^T =\n", A.T)
print("det(A) =", np.linalg.det(A))
print("A^-1 =\n", np.linalg.inv(A))
print("I =\n", np.eye(2))
```

对应关系：

| 手写实现 | NumPy |
|---|---|
| `A + B` | `A + B` |
| `element_wise_multiply` | `A * B` |
| `matmul` | `A @ B` |
| `transpose` | `A.T` |
| `determinant` | `np.linalg.det(A)` |
| `inverse_2x2` | `np.linalg.inv(A)` |
| `identity` | `np.eye(2)` |

---

### 5.2 NumPy 版本：神经网络 dense layer

```python
import numpy as np

inputs = np.random.randn(3, 1)
weights = np.random.randn(2, 3)
bias = np.array([[0.1], [0.1]])

output = np.maximum(0, weights @ inputs + bias)

print(f"Neural network layer: {weights.shape} @ {inputs.shape} = {output.shape}")
print(f"Output:\n{output}")
```

这行代码：

```python
output = np.maximum(0, weights @ inputs + bias)
```

对应：

```text
output = relu(W @ x + b)
```

---

### 5.3 NumPy 版本：广播机制

```python
import numpy as np

matrix = np.array([
    [1, 2, 3],
    [4, 5, 6]
])

bias = np.array([10, 20, 30])

print(matrix + bias)
```

输出等价于：

```txt
| 1  2  3 |   | 10  20  30 |   | 11  22  33 |
| 4  5  6 | + | 10  20  30 | = | 14  25  36 |
```

NumPy 会自动把 `bias` 广播到每一行。  
这就是神经网络框架里 bias addition 的基础。

---

### 5.4 PyTorch 版本：同样的 dense layer

```python
import torch

inputs = torch.randn(3, 1)
weights = torch.randn(2, 3)
bias = torch.tensor([[0.1], [0.1]])

output = torch.relu(weights @ inputs + bias)

print(f"Input shape: {inputs.shape}")
print(f"Weight shape: {weights.shape}")
print(f"Output shape: {output.shape}")
print(output)
```

PyTorch 中同样使用：

```python
weights @ inputs
```

表示矩阵乘法。

---

### 5.5 手写实现 vs 生产库

| 对比项 | 手写 Matrix 类 | NumPy / PyTorch |
|---|---|---|
| 目的 | 理解底层原理 | 工程实践 |
| 性能 | 慢 | 快，底层使用高性能 BLAS / GPU |
| 自动求导 | 没有 | PyTorch 支持 autograd |
| shape 检查 | 需要自己写 | 框架自动检查 |
| 广播机制 | 需要自己实现 | 框架自动广播 |
| 使用场景 | 学习、教学、验证机制 | 训练、推理、部署 |

---

## 6. 交付沉淀：产出物、连接关系、练习、术语表

---

### 6.1 本课产出

```txt
code/from_scratch.py
code/use_numpy.py
code/use_pytorch.py
outputs/prompt-matrix-operations.md
outputs/exercises-solutions.md
```

| 文件 | 作用 |
|---|---|
| `code/from_scratch.py` | 从零实现 Vector、Matrix 和 dense layer |
| `code/use_numpy.py` | NumPy 版本矩阵运算和 dense layer |
| `code/use_pytorch.py` | PyTorch 版本 dense layer |
| `outputs/prompt-matrix-operations.md` | 用于教学矩阵运算的 prompt |
| `outputs/exercises-solutions.md` | 练习题和参考答案 |

---

### 6.2 知识连接关系

| 本课知识点 | 后续连接 |
|---|---|
| 向量 | embedding、hidden state、feature vector |
| 矩阵 | neural network weights、linear layer |
| shape | tensor operations、debug shape mismatch |
| 矩阵乘法 | dense layer、attention、MLP |
| 转置 | backpropagation、attention |
| 行列式 | invertibility、linear systems |
| 逆矩阵 | solving linear equations |
| broadcasting | bias addition、batch computation |
| ReLU | activation function、nonlinear neural networks |

---

### 6.3 AI 应用连接

| 数学知识点 | AI 应用 |
|---|---|
| 向量 | 输入特征、embedding、hidden state |
| 矩阵 | 权重矩阵、图像像素、attention score |
| 矩阵乘法 | 神经网络前向传播 |
| 加偏置 | dense layer bias |
| ReLU | 非线性激活函数 |
| broadcasting | batch bias addition |
| 转置 | attention 中的 `Q @ K.T` |
| 逐元素乘法 | mask、gate、dropout |
| 单位矩阵 | residual connection、identity mapping |

---

### 6.4 练习

1. **验证逆矩阵。**  
   选取三个不同的 `2 x 2` 矩阵，计算 `A @ A.inverse_2x2()`，确认结果是否接近单位矩阵。再尝试一个行列式为 0 的矩阵，观察会发生什么。

2. **实现 3x3 逆矩阵。**  
   扩展 Matrix 类，用伴随矩阵方法实现 `3 x 3` 矩阵求逆，并用 `np.linalg.inv` 验证结果。

3. **搭建两层神经网络。**  
   只使用你写的 Matrix 类，不使用 NumPy，构造一个两层网络：

   ```txt
   input(3) -> hidden(4) -> output(2)
   ```

   随机初始化权重，完成一次 forward pass，并确认所有 shape 正确。

4. **比较逐元素乘法和矩阵乘法。**  
   用同样的两个 `2 x 2` 矩阵，分别计算 `A * B` 和 `A @ B`，解释为什么结果不同。

5. **模拟 bias broadcasting。**  
   自己实现一个函数，把 bias vector 加到矩阵的每一行上，模拟 NumPy 的广播行为。

---

### 6.5 关键术语

| 术语 | 常见说法 | 真正含义 |
|---|---|---|
| 向量 | 一个箭头 | 有顺序的数字列表，在 AI 中表示高维空间中的点 |
| 矩阵 | 一张数字表 | 线性变换，把向量从一个空间映射到另一个空间 |
| shape | 尺寸 | 张量每个维度的大小，决定运算是否合法 |
| 矩阵乘法 | 把数字乘起来 | 第一矩阵的行和第二矩阵的列做点积，顺序很重要 |
| 逐元素乘法 | 普通乘法 | 对应位置相乘，要求 shape 相同或可广播 |
| 转置 | 翻转矩阵 | 交换行和列，把 `m x n` 变成 `n x m` |
| 行列式 | 从矩阵算出来的数 | 衡量面积或体积缩放，0 表示变换压扁了空间 |
| 逆矩阵 | 撤销矩阵 | 能够反向恢复原变换的矩阵 |
| 单位矩阵 | 不做事的矩阵 | 矩阵中的 1，乘上它保持不变 |
| broadcasting | 自动补 shape | 把小数组沿缺失维度重复，以匹配大数组 |
| bias | 偏置 | 在线性变换后整体平移输出 |
| ReLU | 激活函数 | 把负数变成 0，正数保持不变 |

---

### 6.6 自检问题

学完本课后，你应该能回答：

- 向量和矩阵最早为什么会出现？
- 矩阵为什么可以看成线性变换？
- 神经网络层为什么可以写成 `relu(W @ x + b)`？
- 矩阵乘法和逐元素乘法有什么区别？
- `(m x n) @ (n x p) = (m x p)` 为什么要求中间维度匹配？
- 为什么 PyTorch 会报 shape mismatch？
- bias addition 为什么可以通过 broadcasting 实现？
- 行列式为 0 为什么表示矩阵不可逆？
- 单位矩阵为什么等价于数字里的 1？
- 如何从零实现 Matrix 类？
- NumPy / PyTorch 中对应的 API 是什么？

---

## 附录：本课最小代码文件建议

如果要拆成代码文件，可以这样组织：

```txt
code/
├── from_scratch.py
├── use_numpy.py
└── use_pytorch.py
```

### `from_scratch.py`

包含：

```txt
Vector
Matrix
Matrix.__add__
Matrix.__sub__
Matrix.scalar_multiply
Matrix.element_wise_multiply
Matrix.matmul
Matrix.transpose
Matrix.determinant
Matrix.inverse_2x2
Matrix.identity
relu_matrix
dense layer forward pass
```

### `use_numpy.py`

包含：

```txt
np.array
A + B
A * B
A @ B
A.T
np.linalg.det
np.linalg.inv
np.eye
np.maximum
broadcasting example
```

### `use_pytorch.py`

包含：

```txt
torch.randn
weights @ inputs
torch.relu
bias broadcasting
shape check
```
