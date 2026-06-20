---
title: 机器学习中的微积分
description: 面向 AI 算法工程师的微积分课程：从导数、偏导数、梯度、链式法则、Hessian、Taylor 展开到梯度下降和反向传播。
---

# 机器学习中的微积分

> 导数告诉你哪边是下坡路。神经网络只需要知道这个方向，就能开始学习。

**课程类型：** 学习 / 实现  
**所属模块：** 微积分与优化  
**前置知识：** Phase 1 Lesson 01-03：线性代数直觉、向量矩阵运算、矩阵变换  
**预计时间：** 约 60 分钟  
**使用语言：** Python / NumPy / PyTorch  

---

## 1. 提出问题：为什么 AI 算法工程师必须理解微积分

训练神经网络时，我们面对的是这样一个问题：

```txt
模型有几百万甚至几百亿个参数。
每个参数都像一个旋钮。
我们要知道每个旋钮应该往哪个方向调，才能让模型更少犯错。
```

微积分提供的核心工具就是：

```txt
导数：告诉单个变量变化时，函数如何变化。
偏导数：告诉某一个参数变化时，loss 如何变化。
梯度：把所有参数的偏导数组成一个方向。
梯度下降：沿着让 loss 下降的方向更新参数。
```

没有微积分，训练模型只能靠随机尝试。  
有了导数和梯度，模型就知道每个参数应该怎么改。

---

### 1.1 数学概念历史出现缘由

微积分最早并不是为机器学习出现的，而是为了处理“变化”和“累积”这两类问题。

| 数学概念 | 历史出现缘由 |
|---|---|
| 导数 | 为了描述瞬时速度、曲线切线和局部变化率 |
| 偏导数 | 为了描述多变量函数中某一个变量单独变化的影响 |
| 梯度 | 为了把多个方向上的变化率合成一个最陡上升方向 |
| 梯度下降 | 为了寻找函数最小值或最大值 |
| 链式法则 | 为了处理复合函数的求导问题 |
| Hessian | 为了描述函数的二阶变化，也就是曲率 |
| Taylor 展开 | 为了用局部多项式近似复杂函数 |
| 积分 | 为了描述面积、累积量、概率和期望 |
| Jacobian | 为了描述向量函数对向量输入的所有一阶变化 |
| 反向传播 | 为了系统地在计算图中应用链式法则 |

可以这样理解：

```txt
现实问题：如何描述某一瞬间的速度？
数学抽象：导数

现实问题：如何知道多变量函数对某一个变量有多敏感？
数学抽象：偏导数

现实问题：如何知道函数上升最快的方向？
数学抽象：梯度

现实问题：如何让一个函数值越来越小？
数学抽象：梯度下降

现实问题：如何给多层复合函数求导？
数学抽象：链式法则

现实问题：如何理解函数表面的弯曲程度？
数学抽象：Hessian

现实问题：如何用局部简单函数近似复杂函数？
数学抽象：Taylor 展开
```

---

### 1.2 原始问题是什么

本课要解决的原始问题是：

```txt
函数变化的方向和速度如何计算？
当函数有很多输入时，如何知道每个输入分别造成多少影响？
如何把所有方向上的变化组合成一个可用于优化的方向？
如何沿着这个方向一步步找到最小值？
复合函数如何求导？
神经网络中梯度如何从 loss 传回每一层参数？
```

换成 AI 语言，就是：

```txt
loss.backward() 到底在算什么？
optimizer.step() 为什么能让 loss 降低？
learning rate 为什么太大会发散，太小会很慢？
为什么反向传播本质上就是链式法则？
为什么 Hessian 能告诉我们局部是极小值、极大值还是鞍点？
为什么 Taylor 展开能解释梯度下降和 Newton 方法？
```

---

### 1.3 AI 中的现代问题

| 数学知识点 | AI 中的现代问题 |
|---|---|
| 导数 | 判断一个参数变化时输出或 loss 如何变化 |
| 偏导数 | 计算 loss 对某一个权重的敏感性 |
| 梯度 | 给出所有参数的更新方向 |
| 梯度下降 | 神经网络训练的基础更新规则 |
| 学习率 | 控制每次参数更新步长 |
| 数值导数 | 用有限差分验证梯度是否正确 |
| 解析导数 | 快速、精确计算简单函数导数 |
| 自动微分 | PyTorch / JAX 自动计算梯度 |
| 链式法则 | 反向传播的数学基础 |
| Hessian | 描述 loss surface 曲率，连接二阶优化 |
| Taylor 展开 | 解释一阶优化和二阶优化 |
| 积分 | 概率、期望、KL 散度、贝叶斯归一化 |
| Jacobian | 向量到向量函数的导数结构 |
| 反向传播 | 从输出到输入逐层计算梯度 |

---

### 1.4 本课要解决的核心问题

学完本课，你应该能够回答：

1. 导数为什么可以表示局部变化率？
2. 偏导数如何度量单个变量对函数的影响？
3. 梯度为什么指向最陡上升方向？
4. 为什么最小化 loss 要沿着负梯度方向走？
5. 学习率为什么决定每一步走多远？
6. 数值导数和解析导数有什么区别？
7. 链式法则为什么是反向传播的数学基础？
8. Hessian 如何描述曲率？
9. Taylor 展开为什么能解释梯度下降和 Newton 方法？
10. 积分在概率、期望和 KL 散度中如何出现？
11. Jacobian 为什么是向量函数的导数矩阵？
12. 如何从零实现梯度下降并训练一个线性回归模型？

---

## 2. 概念定位：这节课学什么

---

### 2.1 所属模块

```txt
所属模块：微积分与优化
前置知识：函数、向量、矩阵、基础 Python
后续连接：自动微分、反向传播、优化器、线性回归、神经网络训练、Adam、Newton 方法
```

本课是从“矩阵表示模型”进入“模型如何学习”的关键一课。  
线性代数告诉我们模型如何计算；微积分告诉我们模型如何根据错误调整参数。

---

### 2.2 本课学习目标

完成本课后，你应该能够：

- 计算常见机器学习函数的数值导数和解析导数
- 理解导数、偏导数、梯度之间的关系
- 从零实现一维和二维函数的梯度下降
- 推导线性回归的梯度更新
- 手写一个最小线性回归训练循环
- 解释链式法则如何支持反向传播
- 理解 Hessian、Taylor 展开和二阶优化的关系
- 说明积分、Jacobian 在机器学习中的作用

---

### 2.3 本课在整体路线中的位置

```txt
线性代数：数据和参数如何表示
    ↓
微积分：loss 如何随参数变化
    ↓
梯度下降：参数如何更新
    ↓
链式法则：多层网络如何传递梯度
    ↓
自动微分与反向传播
    ↓
优化器：SGD、Momentum、Adam
```

本课的核心能力是：

```txt
看到 loss.backward() 时，知道它背后是在沿计算图应用链式法则，计算 loss 对每个参数的梯度。
```

---

## 3. 理解概念：直觉、公式、几何解释、AI 连接

---

### 3.1 导数：局部变化率

**一句话直觉：**  
导数描述的是：当输入发生一个极小变化时，输出会发生多大变化。

对于函数：

```text
y = f(x)
```

导数：

```text
f'(x)
```

表示在点 `x` 附近，`x` 轻微变化时，`y` 的变化速度。

形式定义：

```text
f'(x) = lim_{h -> 0} (f(x + h) - f(x)) / h
```

几何上，导数就是曲线在某一点的切线斜率。

以：

```text
f(x) = x^2
```

为例：

| x | f(x) | f'(x) |
|---|---|---|
| 0 | 0 | 0 |
| 1 | 1 | 2 |
| 2 | 4 | 4 |
| 3 | 9 | 6 |

在 `x = 2` 时，斜率是 `4`。  
这表示如果 `x` 轻微增加，`f(x)` 大约会以 4 倍速度增加。

**AI 连接：**

| 场景 | 导数含义 |
|---|---|
| loss 对参数求导 | 参数变化时 loss 变化多少 |
| 激活函数求导 | 反向传播时梯度如何穿过激活函数 |
| sigmoid / ReLU 导数 | 决定梯度如何传递 |
| MSE 导数 | 线性回归训练 |
| cross-entropy 导数 | 分类模型训练 |

---

### 3.2 数值导数：用有限差分近似斜率

**一句话直觉：**  
数值导数不用公式推导，而是用函数在附近两个点的差值来近似斜率。

中心差分公式：

```text
f'(x) ≈ (f(x + h) - f(x - h)) / (2h)
```

其中 `h` 是一个很小的数，例如：

```txt
h = 1e-7
```

**优点：**

- 不需要手推公式
- 适合验证梯度
- 可以用于任意黑箱函数

**缺点：**

- 比解析导数慢
- 受数值误差影响
- 不适合大规模神经网络训练

**AI 连接：**

数值导数常用于：

- gradient checking
- 验证反向传播实现是否正确
- 调试自定义 autograd function
- 教学中理解导数定义

---

### 3.3 偏导数：一次只看一个变量

**一句话直觉：**  
偏导数是在多变量函数中，只改变一个变量，其他变量保持不变，观察函数如何变化。

例如：

```text
f(x, y) = x^2 + 3xy + y^2
```

对 `x` 求偏导：

```text
df/dx = 2x + 3y
```

对 `y` 求偏导：

```text
df/dy = 3x + 2y
```

**AI 连接：**

神经网络 loss 依赖大量参数：

```text
L(w1, w2, ..., wn)
```

每个偏导数回答：

```txt
如果只轻微改变这个参数，loss 会怎么变？
```

---

### 3.4 梯度：所有偏导数组成的方向

**一句话直觉：**  
梯度是所有偏导数组成的向量，指向函数上升最快的方向。

对于：

```text
f(x, y, z)
```

梯度是：

```text
grad f = [df/dx, df/dy, df/dz]
```

如果我们要最小化函数，就沿着负梯度方向走：

```text
descent direction = -grad f
```

以：

```text
f(x, y) = x^2 + y^2
```

为例：

| 点 | grad f | 负梯度方向 |
|---|---|---|
| `(1, 1)` | `[2, 2]` | `[-2, -2]` |
| `(0, 0)` | `[0, 0]` | `[0, 0]` |

`(0, 0)` 是碗底，梯度为 0。

**AI 连接：**

| 场景 | 梯度作用 |
|---|---|
| 神经网络训练 | 告诉每个参数如何更新 |
| 梯度裁剪 | 控制梯度过大 |
| 梯度消失 | 梯度接近 0，参数难以更新 |
| 梯度爆炸 | 梯度过大，训练不稳定 |

---

### 3.5 梯度下降：沿着负梯度走

**一句话直觉：**  
梯度下降就是反复沿着让 loss 下降最快的方向迈一小步。

更新公式：

```text
w_new = w_old - learning_rate * dL/dw
```

解释：

| 符号 | 含义 |
|---|---|
| `w_old` | 当前参数 |
| `dL/dw` | loss 对参数的导数 |
| `learning_rate` | 学习率，控制步长 |
| `w_new` | 更新后的参数 |

训练时对每个参数都做类似更新：

```txt
1. 计算 loss
2. 计算 loss 对每个参数的偏导数
3. 用负梯度方向更新参数
4. 重复很多次
```

**AI 连接：**

常见训练循环：

```python
loss.backward()
optimizer.step()
optimizer.zero_grad()
```

对应关系：

| 代码 | 数学含义 |
|---|---|
| `loss.backward()` | 计算梯度 |
| `optimizer.step()` | 根据梯度更新参数 |
| `optimizer.zero_grad()` | 清空上一轮梯度 |

---

### 3.6 学习率：每一步走多远

**一句话直觉：**  
学习率控制每次参数更新的步长。

| 学习率情况 | 结果 |
|---|---|
| 太大 | 可能越过最低点，甚至发散 |
| 太小 | 收敛很慢 |
| 合适 | 稳定下降 |
| 动态调整 | 训练后期更稳定 |

**AI 连接：**

学习率是深度学习中最重要的超参数之一。  
很多训练不稳定，本质上是学习率不合适。

---

### 3.7 解析导数 vs 数值导数

**一句话直觉：**  
解析导数来自公式推导，数值导数来自近似计算。

| 方法 | 含义 | 优点 | 缺点 |
|---|---|---|---|
| 解析导数 | 手动或符号推导公式 | 精确、快速 | 需要推导 |
| 数值导数 | 用有限差分近似 | 通用、直观 | 慢、有误差 |
| 自动微分 | 按计算图机械应用链式法则 | 精确、高效 | 依赖框架实现 |

常见函数导数：

| 函数 | 导数 | ML 中用途 |
|---|---|---|
| `x^2` | `2x` | MSE loss |
| `wx + b` 对 `w` | `x` | 线性层权重梯度 |
| `wx + b` 对 `b` | `1` | 线性层 bias 梯度 |
| `e^x` | `e^x` | softmax、attention |
| `ln(x)` | `1/x` | cross-entropy |
| `sigmoid(x)` | `sigmoid(x)(1-sigmoid(x))` | sigmoid activation |

---

### 3.8 链式法则：复合函数如何求导

**一句话直觉：**  
链式法则告诉我们，复合函数的导数等于外层导数乘以内层导数。

如果：

```text
y = f(g(x))
```

那么：

```text
dy/dx = f'(g(x)) * g'(x)
```

例子：

```text
y = (3x + 1)^2
```

拆成：

```txt
外层：f(u) = u^2，f'(u) = 2u
内层：g(x) = 3x + 1，g'(x) = 3
```

所以：

```text
dy/dx = 2(3x + 1) * 3 = 6(3x + 1)
```

**AI 连接：**

神经网络就是一串复合函数：

```txt
input -> linear -> activation -> linear -> activation -> loss
```

反向传播就是从 loss 开始，反向反复应用链式法则。

---

### 3.9 多变量链式法则与计算图

**一句话直觉：**  
在神经网络中，变量会分叉、合并，梯度沿计算图反向传播，遇到多条路径时要把贡献加起来。

一个简单前向过程：

```mermaid
graph LR
    x["x"] --> z1["z1 = w*x"]
    z1 --> z2["z2 = w*x + b"]
    z2 --> a["a = sigmoid(z2)"]
    a --> L["loss"]
```

反向传播时：

```txt
每条边都乘以局部导数。
如果多个路径汇合，就把梯度贡献相加。
```

**AI 连接：**

这就是反向传播的核心：

```txt
从 loss 出发
沿计算图反向走
每一步乘局部导数
最后得到每个参数的梯度
```

---

### 3.10 Hessian：二阶导数组成的矩阵

**一句话直觉：**  
梯度告诉你斜率，Hessian 告诉你曲率。

对于多变量函数：

```text
f(x1, x2, ..., xn)
```

Hessian 矩阵的元素是二阶偏导数：

```text
H[i][j] = d^2f / (dx_i dx_j)
```

二元函数 Hessian：

```text
H = [[d^2f/dx^2,  d^2f/dxdy],
     [d^2f/dydx,  d^2f/dy^2]]
```

在梯度为 0 的临界点：

| Hessian 性质 | 含义 | 几何形状 |
|---|---|---|
| 正定 | 局部最小值 | 碗口向上 |
| 负定 | 局部最大值 | 碗口向下 |
| 不定 | 鞍点 | 马鞍面 |

例子：

```text
f(x, y) = x^2 - y^2
```

Hessian：

```text
H = [[2, 0],
     [0, -2]]
```

一个特征值为正，一个为负，所以 `(0, 0)` 是鞍点。

**AI 连接：**

Hessian 可用于：

- 判断 loss surface 曲率
- 解释鞍点
- Newton 方法
- L-BFGS
- 二阶优化
- Adam 的二阶信息近似直觉

---

### 3.11 Taylor 展开：用局部多项式近似函数

**一句话直觉：**  
Taylor 展开用函数在某一点的值、斜率和曲率，近似附近的函数形状。

公式：

```text
f(x + h) = f(x) + f'(x)h + (1/2)f''(x)h^2 + (1/6)f'''(x)h^3 + ...
```

不同阶数的意义：

| 近似阶数 | 捕捉的信息 | 对应优化方法 |
|---|---|---|
| 0 阶 | 只知道函数值 | 随机搜索 |
| 1 阶 | 斜率 | 梯度下降 |
| 2 阶 | 曲率 | Newton 方法 |
| 更高阶 | 更复杂局部结构 | ML 中较少直接使用 |

**AI 连接：**

梯度下降可以看作基于一阶 Taylor 近似。  
Newton 方法可以看作基于二阶 Taylor 近似。

学习率不能太大，也可以从 Taylor 近似理解：

```txt
梯度下降假设当前位置附近的线性近似有效。
如果一步走太远，局部近似不再准确，就可能发散。
```

---

### 3.12 积分：累积、概率与期望

**一句话直觉：**  
导数描述变化率，积分描述累积量。

在机器学习中，你很少手算积分，但它的思想到处出现。

| 积分概念 | ML 中出现位置 |
|---|---|
| 曲线下面积 | 连续概率分布 |
| 期望 | expected loss、risk minimization |
| KL 散度 | VAE、蒸馏、策略优化 |
| 归一化常数 | 贝叶斯后验、softmax 分母 |
| 边际似然 | 模型比较、ELBO |

连续随机变量的概率：

```text
P(a < X < b) = integral_a^b p(x) dx
```

期望：

```text
E[f(X)] = integral f(x) p(x) dx
```

KL 散度：

```text
KL(p || q) = integral p(x) log(p(x) / q(x)) dx
```

**AI 连接：**

| 场景 | 积分思想 |
|---|---|
| 损失函数 | 对数据分布上的期望 loss |
| 训练集 loss | 对期望 loss 的经验近似 |
| 贝叶斯推断 | 对所有参数可能性积分 |
| VAE / ELBO | 用可优化目标近似难算积分 |
| MCMC | 用采样近似积分 |

---

### 3.13 Jacobian：向量函数的导数矩阵

**一句话直觉：**  
当函数输入是向量、输出也是向量时，导数就变成一个矩阵，这个矩阵叫 Jacobian。

对于：

```text
f: R^n -> R^m
```

Jacobian 是一个 `m x n` 矩阵：

```txt
第 i 行第 j 列 = 第 i 个输出对第 j 个输入的偏导数
```

表格形式：

|  | x1 | x2 | ... | xn |
|---|---|---|---|---|
| f1 | df1/dx1 | df1/dx2 | ... | df1/dxn |
| f2 | df2/dx1 | df2/dx2 | ... | df2/dxn |
| ... | ... | ... | ... | ... |
| fm | dfm/dx1 | dfm/dx2 | ... | dfm/dxn |

**AI 连接：**

| 场景 | Jacobian 意义 |
|---|---|
| 神经网络层 | 输出向量对输入向量的导数 |
| 反向传播 | 梯度通过 Jacobian transpose 传播 |
| sensitivity analysis | 输入扰动对输出的影响 |
| adversarial examples | 输出对输入变化的敏感性 |
| normalizing flows | 需要 Jacobian determinant |

---

### 3.14 反向传播：链式法则的系统应用

**一句话直觉：**  
反向传播就是在计算图上，从 loss 向前一层层反向应用链式法则。

神经网络训练可以拆成：

```txt
Forward pass:
input -> W1 -> relu -> W2 -> softmax -> loss

Backward pass:
loss -> dW2 -> ... -> dW1
```

每个参数更新：

```text
W = W - learning_rate * dL/dW
```

**AI 连接：**

| 过程 | 作用 |
|---|---|
| forward pass | 计算预测和 loss |
| backward pass | 计算每个参数的梯度 |
| optimizer step | 用梯度更新参数 |
| 多次迭代 | 让模型逐步降低 loss |

深度学习的核心训练循环就是：

```txt
预测
计算 loss
计算梯度
更新参数
重复
```

---

### 3.15 本课核心概念总表

| 数学概念 | 一句话直觉 | AI 中的对应位置 |
|---|---|---|
| 导数 | 局部变化率 | loss 对参数变化 |
| 偏导数 | 只看一个变量的影响 | 每个权重的梯度 |
| 梯度 | 所有偏导数组成的方向 | 参数更新方向 |
| 负梯度 | 最快下降方向 | loss minimization |
| 学习率 | 每步走多远 | optimizer 超参数 |
| 数值导数 | 用有限差分近似导数 | gradient checking |
| 解析导数 | 用公式精确求导 | 手推 loss gradient |
| 链式法则 | 复合函数求导规则 | backpropagation |
| Hessian | 二阶导数矩阵 | 曲率、二阶优化 |
| Taylor 展开 | 局部多项式近似 | GD / Newton 方法 |
| 积分 | 累积量 | 概率、期望、KL |
| Jacobian | 向量函数的导数矩阵 | backprop shape |
| 反向传播 | 计算图上应用链式法则 | neural network training |

---

## 4. 手写实现：从零实现导数、梯度下降和线性回归训练

这一节先不用 PyTorch。  
目标是亲手实现神经网络训练中最核心的数学机制。

---

### 4.1 从零实现数值导数

```python
def numerical_derivative(f, x, h=1e-7):
    return (f(x + h) - f(x - h)) / (2 * h)


def f(x):
    return x ** 2


for x in [-2, -1, 0, 1, 2]:
    numerical = numerical_derivative(f, x)
    analytical = 2 * x
    print(f"x={x:2d}  f'(x) numerical={numerical:.6f}  analytical={analytical:.1f}")
```

如果数值导数和解析导数接近，说明有限差分实现正确。

---

### 4.2 从零实现多变量函数的梯度

```python
def numerical_gradient(f, point, h=1e-7):
    gradient = []

    for i in range(len(point)):
        point_plus = list(point)
        point_minus = list(point)

        point_plus[i] += h
        point_minus[i] -= h

        partial = (f(point_plus) - f(point_minus)) / (2 * h)
        gradient.append(partial)

    return gradient


def f_multi(point):
    x, y = point
    return x ** 2 + 3 * x * y + y ** 2


grad = numerical_gradient(f_multi, [1.0, 2.0])

print(f"Numerical gradient at (1,2): {[f'{g:.4f}' for g in grad]}")
print(f"Analytical gradient at (1,2): [{2*1+3*2}, {3*1+2*2}]")
```

---

### 4.3 用梯度下降最小化一维函数

目标函数：

```text
f(x) = x^2
```

它的最小值在：

```txt
x = 0
```

代码：

```python
x = 5.0
lr = 0.1

for step in range(20):
    grad = 2 * x
    x = x - lr * grad
    print(f"step {step:2d}  x={x:8.4f}  f(x)={x**2:10.6f}")
```

你会看到 `x` 逐步靠近 0。

---

### 4.4 用梯度下降最小化二维函数

目标函数：

```text
f(x, y) = x^2 + y^2
```

最小值在：

```txt
(0, 0)
```

代码：

```python
def f_2d(point):
    x, y = point
    return x ** 2 + y ** 2


point = [4.0, 3.0]
lr = 0.1

for step in range(30):
    grad = numerical_gradient(f_2d, point)
    point = [p - lr * g for p, g in zip(point, grad)]
    loss = f_2d(point)

    if step % 5 == 0 or step == 29:
        print(f"step {step:2d}  point=({point[0]:7.4f}, {point[1]:7.4f})  f={loss:.6f}")
```

---

### 4.5 比较数值导数和解析导数

```python
import math

test_functions = [
    ("x^2",    lambda x: x ** 2,       lambda x: 2 * x),
    ("x^3",    lambda x: x ** 3,       lambda x: 3 * x ** 2),
    ("sin(x)", lambda x: math.sin(x),  lambda x: math.cos(x)),
    ("e^x",    lambda x: math.exp(x),  lambda x: math.exp(x)),
    ("1/x",    lambda x: 1 / x,        lambda x: -1 / x ** 2),
]

x = 2.0

print(f"{'Function':<12} {'Numerical':>12} {'Analytical':>12} {'Error':>12}")
print("-" * 50)

for name, f, df in test_functions:
    num = numerical_derivative(f, x)
    ana = df(x)
    err = abs(num - ana)

    print(f"{name:<12} {num:12.6f} {ana:12.6f} {err:12.2e}")
```

---

### 4.6 从零计算 Hessian

```python
def hessian_2d(f, x, y, h=1e-5):
    fxx = (f(x + h, y) - 2 * f(x, y) + f(x - h, y)) / (h ** 2)
    fyy = (f(x, y + h) - 2 * f(x, y) + f(x, y - h)) / (h ** 2)
    fxy = (
        f(x + h, y + h)
        - f(x + h, y - h)
        - f(x - h, y + h)
        + f(x - h, y - h)
    ) / (4 * h ** 2)

    return [[fxx, fxy], [fxy, fyy]]


def saddle(x, y):
    return x ** 2 - y ** 2


def bowl(x, y):
    return x ** 2 + y ** 2


H_saddle = hessian_2d(saddle, 0.0, 0.0)
H_bowl = hessian_2d(bowl, 0.0, 0.0)

print(f"Saddle Hessian: {H_saddle}")
print(f"Bowl Hessian:   {H_bowl}")
```

解释：

```txt
saddle 的 Hessian 有正有负，表示鞍点。
bowl 的 Hessian 都为正，表示局部最小值。
```

---

### 4.7 Taylor 近似实验

```python
import math


def taylor_approx(f, f_prime, f_double_prime, x0, h, order=2):
    result = f(x0)

    if order >= 1:
        result += f_prime(x0) * h

    if order >= 2:
        result += 0.5 * f_double_prime(x0) * h ** 2

    return result


x0 = 0.0

for h in [0.1, 0.5, 1.0, 2.0]:
    true_val = math.sin(h)
    t1 = taylor_approx(math.sin, math.cos, lambda x: -math.sin(x), x0, h, order=1)
    t2 = taylor_approx(math.sin, math.cos, lambda x: -math.sin(x), x0, h, order=2)

    print(f"h={h:.1f}  sin(h)={true_val:.4f}  order1={t1:.4f}  order2={t2:.4f}")
```

你会看到：

```txt
h 越小，Taylor 近似越准确。
h 越大，局部近似越容易失效。
```

这就是学习率不能过大的直觉来源。

---

### 4.8 从零训练线性回归

数据来自：

```text
y = 2x + 1
```

我们让模型从随机参数开始学习：

```python
import random

random.seed(42)

w = random.gauss(0, 1)
b = random.gauss(0, 1)
lr = 0.01

xs = [1.0, 2.0, 3.0, 4.0, 5.0]
ys = [3.0, 5.0, 7.0, 9.0, 11.0]

for epoch in range(200):
    total_loss = 0
    dw = 0
    db = 0

    for x, y in zip(xs, ys):
        pred = w * x + b
        error = pred - y

        total_loss += error ** 2
        dw += 2 * error * x
        db += 2 * error

    dw /= len(xs)
    db /= len(xs)
    total_loss /= len(xs)

    w -= lr * dw
    b -= lr * db

    if epoch % 40 == 0 or epoch == 199:
        print(f"epoch {epoch:3d}  w={w:.4f}  b={b:.4f}  loss={total_loss:.6f}")

print(f"Learned: y = {w:.2f}x + {b:.2f}")
print("Actual:  y = 2x + 1")
```

这个过程完整体现了机器学习训练循环：

```txt
预测 -> 计算 loss -> 计算梯度 -> 更新参数 -> 重复
```

---

## 5. 生产使用：用 NumPy / PyTorch 完成同样任务

---

### 5.1 NumPy 版本：线性回归梯度下降

```python
import numpy as np

x = np.array([1, 2, 3, 4, 5], dtype=float)
y = np.array([3, 5, 7, 9, 11], dtype=float)

w, b = np.random.randn(), np.random.randn()
lr = 0.01

for epoch in range(200):
    pred = w * x + b
    error = pred - y

    loss = np.mean(error ** 2)

    dw = np.mean(2 * error * x)
    db = np.mean(2 * error)

    w -= lr * dw
    b -= lr * db

print(f"Learned: y = {w:.2f}x + {b:.2f}")
```

这个版本和手写版本数学完全相同，只是用 NumPy 向量化提高了效率。

---

### 5.2 PyTorch 版本：自动计算梯度

```python
import torch

x = torch.tensor([1, 2, 3, 4, 5], dtype=torch.float32)
y = torch.tensor([3, 5, 7, 9, 11], dtype=torch.float32)

w = torch.randn((), requires_grad=True)
b = torch.randn((), requires_grad=True)

lr = 0.01

for epoch in range(200):
    pred = w * x + b
    loss = torch.mean((pred - y) ** 2)

    loss.backward()

    with torch.no_grad():
        w -= lr * w.grad
        b -= lr * b.grad

        w.grad.zero_()
        b.grad.zero_()

print(f"Learned: y = {w.item():.2f}x + {b.item():.2f}")
```

这里：

| 代码 | 含义 |
|---|---|
| `requires_grad=True` | 让 PyTorch 记录计算图 |
| `loss.backward()` | 自动计算 `dL/dw` 和 `dL/db` |
| `with torch.no_grad()` | 更新参数时不记录计算图 |
| `grad.zero_()` | 清空梯度，准备下一轮 |

---

### 5.3 手写实现 vs 生产库

| 对比项 | 手写实现 | NumPy | PyTorch |
|---|---|---|---|
| 目的 | 理解导数和梯度下降 | 向量化计算 | 自动微分和深度学习训练 |
| 梯度计算 | 手动推导或数值近似 | 手动公式，批量计算 | autograd 自动计算 |
| 性能 | 慢 | 快 | 快，可用 GPU |
| 适合场景 | 教学、验证原理 | 小型数值实验 | 神经网络训练 |
| 反向传播 | 需要自己实现 | 不内置 | 内置 |

---

## 6. 交付沉淀：产出物、连接关系、练习、术语表

---

### 6.1 本课产出

```txt
code/from_scratch.py
code/use_numpy.py
code/use_pytorch.py
outputs/prompt-calculus-for-ml.md
outputs/exercises-solutions.md
```

| 文件 | 作用 |
|---|---|
| `code/from_scratch.py` | 从零实现数值导数、梯度、Hessian、Taylor 近似和线性回归训练 |
| `code/use_numpy.py` | NumPy 版本线性回归梯度下降 |
| `code/use_pytorch.py` | PyTorch autograd 版本线性回归训练 |
| `outputs/prompt-calculus-for-ml.md` | 用于教学微积分和梯度下降直觉的 prompt |
| `outputs/exercises-solutions.md` | 练习题和参考答案 |

---

### 6.2 知识连接关系

| 本课知识点 | 后续连接 |
|---|---|
| 导数 | 激活函数导数、loss gradient |
| 偏导数 | 参数梯度 |
| 梯度 | 梯度下降、优化器 |
| 链式法则 | 反向传播、自动微分 |
| Hessian | Newton 方法、二阶优化 |
| Taylor 展开 | 梯度下降、Newton 方法 |
| 积分 | 概率、期望、KL 散度 |
| Jacobian | 向量函数求导、backprop shape |
| 线性回归梯度 | 监督学习和优化基础 |

---

### 6.3 AI 应用连接

| 数学知识点 | AI 应用 |
|---|---|
| 导数 | loss 对参数的敏感性 |
| 梯度 | 神经网络参数更新方向 |
| 负梯度 | 最小化 loss |
| 学习率 | 控制训练步长 |
| 链式法则 | backpropagation |
| Hessian | loss surface 曲率、二阶优化 |
| Taylor 展开 | GD / Newton 方法直觉 |
| Jacobian | 多输出模型的导数结构 |
| 积分 | expected loss、KL、贝叶斯推断 |
| 线性回归梯度 | 最小监督学习训练循环 |

---

### 6.4 练习

1. **实现二阶数值导数。**  
   用 `numerical_derivative` 调用两次，实现 `numerical_second_derivative(f, x)`。验证 `x^3` 在 `x = 2` 处的二阶导数为 `12`。

2. **二维梯度下降。**  
   使用梯度下降寻找：

   ```text
   f(x, y) = (x - 3)^2 + (y + 1)^2
   ```

   的最小值。从 `(0, 0)` 开始，结果应该收敛到 `(3, -1)`。

3. **加入 Momentum。**  
   在梯度下降中加入速度项 `velocity`，让它累积过去的梯度。比较普通梯度下降和 Momentum 在下面函数上的收敛速度：

   ```text
   f(x) = x^4 - 3x^2
   ```

4. **手推线性回归梯度。**  
   对单个样本：

   ```text
   pred = wx + b
   loss = (pred - y)^2
   ```

   推导 `dL/dw` 和 `dL/db`。

5. **用 PyTorch 验证手推梯度。**  
   用 `requires_grad=True` 和 `loss.backward()` 验证你手推的 `dL/dw` 和 `dL/db` 是否正确。

---

### 6.5 关键术语

| 术语 | 常见说法 | 真正含义 |
|---|---|---|
| 导数 | 斜率 | 函数在某一点的局部变化率 |
| 偏导数 | 对一个变量求导 | 固定其他变量，只看一个变量变化的影响 |
| 梯度 | 最陡方向 | 所有偏导数组成的向量，指向函数上升最快方向 |
| 负梯度 | 下坡方向 | 让函数下降最快的局部方向 |
| 梯度下降 | 往下走 | 用负梯度更新参数，降低 loss |
| 学习率 | 步长 | 控制每次沿梯度方向走多远 |
| 数值导数 | 有限差分 | 用附近两点函数值近似导数 |
| 解析导数 | 手推公式 | 通过求导规则得到精确导数 |
| 自动微分 | 框架自动求导 | 按计算图机械应用链式法则 |
| 链式法则 | 复合函数求导 | 外层导数乘以内层导数，反向传播的基础 |
| 反向传播 | reverse-mode autodiff | 从 loss 开始反向计算每个参数的梯度 |
| Hessian | 二阶导数矩阵 | 描述函数曲率 |
| Taylor 展开 | 多项式近似 | 用函数值、导数、二阶导数等近似局部函数 |
| Jacobian | 导数矩阵 | 向量函数所有输出对所有输入的偏导数组成的矩阵 |
| 积分 | 曲线下面积 | 描述累积量，在 ML 中对应概率、期望和 KL |

---

### 6.6 自检问题

学完本课后，你应该能回答：

- 导数最初为什么会出现？
- 导数为什么可以理解为局部变化率？
- 数值导数和解析导数有什么区别？
- 偏导数和梯度是什么关系？
- 为什么梯度指向最陡上升方向？
- 为什么最小化 loss 要沿负梯度方向走？
- 学习率太大或太小分别会发生什么？
- 链式法则为什么是反向传播的数学基础？
- Hessian 为什么能描述曲率？
- Taylor 展开如何解释梯度下降和 Newton 方法？
- 积分为什么出现在概率、期望和 KL 散度中？
- Jacobian 为什么是向量函数的导数矩阵？
- 如何从零实现梯度下降？
- NumPy / PyTorch 中对应的实现方式是什么？

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
numerical_derivative
numerical_gradient
gradient_descent_1d
gradient_descent_2d
compare_numerical_analytical
hessian_2d
taylor_approx
manual_linear_regression_training
```

### `use_numpy.py`

包含：

```txt
np.array
vectorized linear regression
MSE loss
manual gradient update with NumPy
```

### `use_pytorch.py`

包含：

```txt
torch.tensor
requires_grad=True
loss.backward()
manual parameter update
grad.zero_()
```
