---
title: 优化方法
description: 面向 AI 算法工程师的优化课程：从梯度下降、学习率、SGD、Momentum、Adam、学习率调度，到凸优化、非凸优化、鞍点与损失地形。
---

# 优化方法

> 训练神经网络，本质上就是在损失函数的山谷里寻找最低点。

**课程类型：** 实现 / 应用  
**所属模块：** 优化方法  
**前置知识：** Phase 1 Lesson 04-05：导数、梯度、链式法则与自动微分  
**预计时间：** 约 75 分钟  
**使用语言：** Python / PyTorch  

---

## 1. 提出问题：为什么 AI 算法工程师必须理解优化

你已经有了损失函数，也已经知道如何计算梯度。

损失函数告诉你：

```txt
模型现在错得有多严重。
```

梯度告诉你：

```txt
哪个方向会让 loss 上升最快。
```

现在真正的问题是：

```txt
如何沿着合适的方向、用合适的步长，稳定而高效地走到 loss 较低的地方？
```

最朴素的方法是：

```txt
沿着负梯度方向走一步。
重复很多次。
```

这就是梯度下降。

但现实训练并不总是这么简单：

```txt
学习率太大，会越过最低点，甚至发散。
学习率太小，会训练得非常慢。
梯度方向来回震荡，会在狭长山谷里走得很慢。
遇到鞍点，梯度可能接近 0，但并不是最小值。
不同参数需要不同的有效学习率。
训练早期和后期需要不同的步长。
```

所有优化器，本质上都是在回答同一个问题：

```txt
如何更快、更稳定、更可靠地走到 loss 的低处？
```

---

### 1.1 数学概念历史出现缘由

优化方法最早不是为神经网络出现的，而是为了解决“如何在许多可能选择中找到最好选择”的问题。

| 数学 / 工程概念 | 历史出现缘由 |
|---|---|
| 优化 | 为了在多个可能解中找到使目标函数最小或最大的解 |
| 损失函数 | 为了把“模型好坏”变成可以最小化的数值 |
| 梯度下降 | 为了利用局部斜率一步步降低函数值 |
| 学习率 | 为了控制每一步更新的大小 |
| Batch Gradient Descent | 为了用整个数据集计算精确梯度 |
| SGD | 为了用随机样本快速近似梯度，提升训练效率 |
| Mini-batch | 为了在梯度质量和计算效率之间折中 |
| Momentum | 为了减少震荡，并积累稳定方向上的速度 |
| Adam | 为了给不同参数自适应分配不同的有效学习率 |
| 学习率调度 | 为了让训练早期快走、后期细调 |
| 凸优化 | 为了研究只有一个全局最优解的可控问题 |
| 非凸优化 | 为了描述神经网络这种有多个局部结构的复杂目标 |
| 鞍点 | 为了解释梯度为 0 但并非最小值的情况 |
| 损失地形 | 为了可视化参数空间中的 loss 变化 |

可以这样理解：

```txt
现实问题：如何找到一个函数的最低点？
数学抽象：优化

现实问题：如何知道模型错得有多严重？
工程抽象：损失函数

现实问题：如何决定参数往哪里更新？
数学工具：梯度

现实问题：如何避免走得太快或太慢？
工程超参数：学习率

现实问题：如何减少来回震荡？
优化方法：Momentum

现实问题：不同参数该不该用不同步长？
优化方法：Adam

现实问题：训练早期和后期是否应该用同样学习率？
工程策略：学习率调度
```

---

### 1.2 原始问题是什么

本课要解决的原始问题是：

```txt
什么是优化？
机器学习训练为什么可以看成优化问题？
梯度下降为什么能让 loss 降低？
学习率为什么如此关键？
Batch GD、SGD、Mini-batch 有什么区别？
Momentum 为什么能加速并减少震荡？
Adam 为什么可以自适应调整每个参数的学习率？
为什么训练中要使用学习率调度？
凸函数和非凸函数有什么区别？
鞍点为什么会阻碍训练？
loss landscape 如何帮助理解模型训练？
```

换成 AI 语言，就是：

```txt
optimizer.step() 到底在做什么？
为什么 Adam 的默认 lr 常用 0.001？
为什么 SGD with momentum 有时最终泛化更好？
为什么 Transformer 常用 AdamW 和 warmup？
为什么训练 loss 会发散？
为什么 saddle point 比 local minimum 更常见？
为什么 mini-batch 噪声有时是好事？
```

---

### 1.3 AI 中的现代问题

| 数学 / 工程知识点 | AI 中的现代问题 |
|---|---|
| 优化 | 模型训练本质 |
| 损失函数 | 衡量预测错误程度 |
| 梯度下降 | 最基础参数更新规则 |
| 学习率 | 控制训练稳定性和速度 |
| Batch GD | 精确但慢 |
| SGD | 快但噪声大 |
| Mini-batch | 深度学习实际默认方式 |
| Momentum | 减少震荡，加速稳定方向 |
| Adam | 自适应学习率，深度学习常用默认优化器 |
| AdamW | Transformer 训练常用优化器 |
| 学习率调度 | warmup、cosine decay、step decay |
| 凸优化 | 理解传统 ML 可解性 |
| 非凸优化 | 理解神经网络 loss |
| 鞍点 | 高维优化中的主要障碍 |
| flat minima | 泛化更好 |
| sharp minima | 可能泛化较差 |
| loss landscape | 分析训练过程和模型泛化 |

---

### 1.4 本课要解决的核心问题

学完本课，你应该能够回答：

1. 为什么训练神经网络可以看成最小化损失函数？
2. 梯度下降的更新公式是什么？
3. 学习率为什么是最重要的超参数之一？
4. Batch GD、SGD、Mini-batch GD 有什么区别？
5. 为什么 mini-batch 的噪声不一定是坏事？
6. Momentum 为什么像“滚下山的球”？
7. Adam 如何利用一阶矩和二阶矩自适应调整步长？
8. Adam 中 bias correction 为什么必要？
9. 常见学习率调度有哪些？
10. 凸函数和非凸函数有什么本质差异？
11. 为什么神经网络优化中鞍点很重要？
12. sharp minima 和 flat minima 与泛化有什么关系？
13. 如何从零实现 Gradient Descent、Momentum 和 Adam？
14. PyTorch 中对应的 optimizer API 是什么？

---

## 2. 概念定位：这节课学什么

---

### 2.1 所属模块

```txt
所属模块：优化方法
前置知识：导数、梯度、链式法则、自动微分
后续连接：SGD、Adam、AdamW、学习率调度、神经网络训练、Transformer 训练、泛化分析
```

前面的课程已经解决了：

```txt
如何计算梯度？
```

本课进一步解决：

```txt
如何使用梯度高效、稳定地更新参数？
```

---

### 2.2 本课学习目标

完成本课后，你应该能够：

- 解释优化在机器学习训练中的含义
- 从零实现 vanilla gradient descent
- 从零实现 SGD with momentum
- 从零实现 Adam
- 解释 Adam 的一阶矩、二阶矩和 bias correction
- 比较不同优化器在 Rosenbrock 函数上的收敛表现
- 区分 batch、SGD、mini-batch 三种梯度估计方式
- 理解学习率太大或太小造成的问题
- 解释常见学习率调度策略
- 区分凸优化和非凸优化
- 解释鞍点、flat minima、sharp minima 和 loss landscape
- 使用 PyTorch 配置 SGD、Adam、AdamW 和 scheduler

---

### 2.3 本课在整体路线中的位置

```txt
导数与梯度
    ↓
链式法则与自动微分
    ↓
优化方法
    ↓
SGD / Momentum / Adam / AdamW
    ↓
学习率调度与训练稳定性
    ↓
神经网络训练实践
    ↓
大模型训练、微调和泛化分析
```

本课的核心能力是：

```txt
看到 optimizer.step() 时，知道不同优化器如何利用梯度、历史梯度和梯度平方来更新参数。
```

---

## 3. 理解概念：直觉、公式、优化解释、AI 连接

---

### 3.1 优化：寻找损失函数的最低点

**一句话直觉：**  
优化就是寻找让目标函数最小或最大的输入值。

在机器学习中：

```text
minimize L(w)
```

其中：

| 符号 | 含义 |
|---|---|
| `L` | loss function，损失函数 |
| `w` | model weights，模型权重 |
| `minimize` | 寻找让 loss 最小的参数 |

训练模型就是：

```txt
找到一组参数 w，让模型在训练数据上的 loss 尽可能低。
```

**AI 连接：**

| 优化对象 | AI 中的含义 |
|---|---|
| 参数 `w` | 神经网络权重 |
| 目标函数 `L(w)` | loss |
| 最小值 | 模型较优参数 |
| 优化算法 | optimizer |
| 迭代过程 | training loop |

---

### 3.2 梯度下降：最基础的优化器

**一句话直觉：**  
梯度下降就是沿着负梯度方向更新参数。

更新公式：

```text
w = w - lr * gradient
```

其中：

| 符号 | 含义 |
|---|---|
| `w` | 当前参数 |
| `gradient` | loss 对参数的梯度 |
| `lr` | learning rate，学习率 |
| `-gradient` | loss 下降最快的局部方向 |

训练循环：

```txt
1. 前向计算 loss
2. 反向计算 gradient
3. 参数沿负梯度方向更新
4. 重复
```

**AI 连接：**

PyTorch 中对应：

```python
loss.backward()
optimizer.step()
optimizer.zero_grad()
```

---

### 3.3 学习率：最重要的超参数

**一句话直觉：**  
学习率控制每次沿梯度方向走多远。

| 学习率情况 | 训练表现 |
|---|---|
| 太大 | 越过最低点，震荡，甚至发散 |
| 太小 | 收敛很慢，浪费计算 |
| 合适 | 稳定下降 |
| 动态变化 | 早期快走，后期细调 |

常见经验值：

| 优化器 | 常见初始学习率 |
|---|---|
| Adam | `0.001` |
| SGD + momentum | `0.01` |
| AdamW | `0.001` 或更小，取决于模型和任务 |

**AI 连接：**

很多训练失败的第一排查项就是：

```txt
学习率是否过大？
```

如果 loss NaN、loss 爆炸、训练不稳定，通常先降低 learning rate。

---

### 3.4 Batch GD、SGD 与 Mini-batch

**一句话直觉：**  
三者的区别在于每次用多少数据来估计梯度。

| 方法 | Batch size | 梯度质量 | 每步速度 | 噪声 |
|---|---:|---|---|---|
| Batch GD | 全部数据 | 精确 | 慢 | 无 |
| SGD | 1 个样本 | 很 noisy | 快 | 高 |
| Mini-batch | 32-256 等 | 较好估计 | 平衡 | 中等 |

真实深度学习中，通常使用 Mini-batch：

```txt
每次取一小批数据，估计一次梯度，更新一次参数。
```

**为什么噪声不一定是坏事？**

Mini-batch 噪声有时可以帮助模型：

- 逃离浅层局部最小值
- 摆脱鞍点
- 避免过早进入 sharp minima
- 提升泛化能力

---

### 3.5 Momentum：滚下山的球

**一句话直觉：**  
Momentum 会积累过去的梯度方向，让优化路径更平滑，减少来回震荡。

Momentum 会维护一个 velocity：

```text
v = beta * v + gradient
w = w - lr * v
```

其中：

| 符号 | 含义 |
|---|---|
| `v` | velocity，历史梯度累积 |
| `beta` | momentum 系数，通常为 `0.9` |
| `gradient` | 当前梯度 |
| `lr` | 学习率 |

直觉：

```txt
像一个球沿山谷滚下去。
在一致方向上越滚越快。
在来回震荡方向上互相抵消。
```

**AI 连接：**

Momentum 常用于：

- SGD with momentum
- 加速训练
- 减少狭长山谷中的 zigzag
- 帮助逃离 saddle point

---

### 3.6 Adam：自适应学习率

**一句话直觉：**  
Adam 为每个参数维护自己的有效学习率。

Adam 跟踪两个量：

```txt
m：一阶矩，梯度的滑动平均，类似 Momentum
v：二阶矩，梯度平方的滑动平均，表示梯度规模
```

公式：

```text
m = beta1 * m + (1 - beta1) * gradient
v = beta2 * v + (1 - beta2) * gradient^2
```

bias correction：

```text
m_hat = m / (1 - beta1^t)
v_hat = v / (1 - beta2^t)
```

参数更新：

```text
w = w - lr * m_hat / (sqrt(v_hat) + epsilon)
```

关键直觉：

```txt
梯度长期很大的参数，步子会自动变小。
梯度长期很小的参数，步子会相对变大。
```

默认超参数：

```txt
lr = 0.001
beta1 = 0.9
beta2 = 0.999
epsilon = 1e-8
```

**AI 连接：**

Adam 是深度学习中最常用的默认优化器之一。  
Transformer 和大模型训练常使用 AdamW，也就是 decoupled weight decay 版本。

---

### 3.7 学习率调度：训练过程中改变步长

**一句话直觉：**  
训练早期需要大步快速下降，训练后期需要小步精细调整。

常见 learning rate schedule：

| 调度方式 | 公式 / 思想 | 适用场景 |
|---|---|---|
| Step decay | 每 N 个 epoch 乘以一个 factor | 简单手动控制 |
| Exponential decay | `lr = lr0 * decay^t` | 平滑下降 |
| Cosine annealing | 用余弦曲线从高学习率降到低学习率 | Transformer、现代训练 |
| Warmup + decay | 先线性升高，再逐步降低 | 大模型训练，防止早期不稳定 |

**AI 连接：**

大模型训练中 warmup 很常见，因为训练初期参数和梯度都不稳定。  
直接用大学习率可能导致训练崩溃。

---

### 3.8 凸优化与非凸优化

**一句话直觉：**  
凸函数像一个碗，只有一个全局最低点；非凸函数有多个山谷、鞍点和平坦区域。

凸函数例子：

```text
f(x) = x^2
```

特点：

```txt
任何局部最小值都是全局最小值。
梯度下降最终可以找到全局最优。
```

神经网络 loss 通常是非凸的：

```txt
有许多局部最小值、鞍点、平坦区域。
```

但在高维神经网络中，真正糟糕的 local minima 往往不是最大问题。  
更常见的问题是：

```txt
saddle point 和 flat region。
```

**AI 连接：**

| 概念 | AI 中的意义 |
|---|---|
| 凸优化 | 线性回归、逻辑回归、SVM 等传统模型 |
| 非凸优化 | 深度神经网络 |
| local minimum | 局部较优解 |
| saddle point | 梯度接近 0，但不是最小值 |
| flat region | 梯度很小，训练缓慢 |

---

### 3.9 鞍点：梯度为 0 但不是最低点

**一句话直觉：**  
鞍点在某些方向像最低点，在另一些方向像最高点。

例子：

```text
f(x, y) = x^2 - y^2
```

在 `(0, 0)`：

```txt
x 方向是最小值
y 方向是最大值
整体不是最小值
```

梯度可能为 0，但这不是模型真正应该停下来的地方。

**AI 连接：**

高维神经网络中鞍点很多。  
Momentum 和 mini-batch noise 有助于模型逃离鞍点。

---

### 3.10 Loss Landscape：损失地形

**一句话直觉：**  
loss landscape 是参数空间中 loss 的地形图。

一个模型可能有数百万参数。  
真实 loss landscape 是超高维空间，我们通常通过选取两个随机方向做二维切片来可视化。

| 地形 | 含义 |
|---|---|
| 高 loss 区域 | 模型预测很差 |
| 鞍点 | 梯度小但仍可继续下降 |
| local minimum | 局部低点 |
| global minimum | 全局最低点 |
| sharp minimum | 很尖锐，对参数扰动敏感 |
| flat minimum | 很平坦，对扰动更鲁棒 |

**AI 连接：**

经验上：

```txt
flat minima 通常泛化更好。
sharp minima 可能训练集 loss 很低，但对参数扰动敏感，泛化较差。
```

这也是为什么 SGD 的噪声有时比 Adam 更有利于最终泛化。

---

### 3.11 Rosenbrock 函数：优化器测试地形

**一句话直觉：**  
Rosenbrock 函数是一个经典优化测试函数，最低点容易知道，但路径在狭长弯曲山谷里，很难走。

公式：

```text
f(x, y) = (1 - x)^2 + 100 * (y - x^2)^2
```

最小点：

```txt
(x, y) = (1, 1)
```

它适合测试优化器，因为：

```txt
山谷狭长且弯曲。
普通梯度下降会走得很慢。
Momentum 和 Adam 的优势更容易体现。
```

---

### 3.12 本课核心概念总表

| 概念 | 一句话直觉 | AI 中的对应位置 |
|---|---|---|
| 优化 | 寻找函数最低点 | 模型训练 |
| 损失函数 | 错误程度的数值化 | training objective |
| 梯度下降 | 沿负梯度更新参数 | 基础 optimizer |
| 学习率 | 每步走多远 | 最关键超参数 |
| Batch GD | 全数据梯度 | 精确但慢 |
| SGD | 单样本梯度 | 快但 noisy |
| Mini-batch | 小批量梯度 | 深度学习默认 |
| Momentum | 累积历史梯度 | 加速、减少震荡 |
| Adam | 自适应每个参数学习率 | 深度学习常用 |
| AdamW | Adam + decoupled weight decay | Transformer 训练 |
| 学习率调度 | 动态改变 lr | 稳定训练 |
| 凸优化 | 单一全局最优 | 传统 ML |
| 非凸优化 | 多山谷复杂地形 | 神经网络 |
| 鞍点 | 梯度为 0 但不是最小 | 高维训练障碍 |
| Loss landscape | 参数空间地形 | 泛化和稳定性分析 |

---

## 4. 手写实现：从零实现 GD、Momentum 和 Adam

这一节先不用 PyTorch。  
目标是亲手实现优化器，理解 `optimizer.step()` 背后的逻辑。

---

### 4.1 定义 Rosenbrock 测试函数

```python
def rosenbrock(params):
    x, y = params
    return (1 - x) ** 2 + 100 * (y - x ** 2) ** 2


def rosenbrock_gradient(params):
    x, y = params

    df_dx = -2 * (1 - x) + 200 * (y - x ** 2) * (-2 * x)
    df_dy = 200 * (y - x ** 2)

    return [df_dx, df_dy]
```

---

### 4.2 实现 Vanilla Gradient Descent

```python
class GradientDescent:
    def __init__(self, lr=0.001):
        self.lr = lr

    def step(self, params, grads):
        return [
            p - self.lr * g
            for p, g in zip(params, grads)
        ]
```

数学对应：

```text
w = w - lr * gradient
```

---

### 4.3 实现 SGD with Momentum

```python
class SGDMomentum:
    def __init__(self, lr=0.001, momentum=0.9):
        self.lr = lr
        self.momentum = momentum
        self.velocity = None

    def step(self, params, grads):
        if self.velocity is None:
            self.velocity = [0.0] * len(params)

        self.velocity = [
            self.momentum * v + g
            for v, g in zip(self.velocity, grads)
        ]

        return [
            p - self.lr * v
            for p, v in zip(params, self.velocity)
        ]
```

数学对应：

```text
v = beta * v + gradient
w = w - lr * v
```

---

### 4.4 实现 Adam

```python
class Adam:
    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = None
        self.v = None
        self.t = 0

    def step(self, params, grads):
        if self.m is None:
            self.m = [0.0] * len(params)
            self.v = [0.0] * len(params)

        self.t += 1

        self.m = [
            self.beta1 * m + (1 - self.beta1) * g
            for m, g in zip(self.m, grads)
        ]

        self.v = [
            self.beta2 * v + (1 - self.beta2) * g ** 2
            for v, g in zip(self.v, grads)
        ]

        m_hat = [
            m / (1 - self.beta1 ** self.t)
            for m in self.m
        ]

        v_hat = [
            v / (1 - self.beta2 ** self.t)
            for v in self.v
        ]

        return [
            p - self.lr * mh / (vh ** 0.5 + self.epsilon)
            for p, mh, vh in zip(params, m_hat, v_hat)
        ]
```

关键点：

| 变量 | 含义 |
|---|---|
| `m` | 梯度的一阶矩估计 |
| `v` | 梯度平方的二阶矩估计 |
| `m_hat` | 修正初始化偏差后的一阶矩 |
| `v_hat` | 修正初始化偏差后的二阶矩 |
| `epsilon` | 防止除 0 |

---

### 4.5 运行并比较优化器

```python
def optimize(optimizer, func, grad_func, start, steps=5000):
    params = list(start)
    history = [params[:]]

    for _ in range(steps):
        grads = grad_func(params)
        params = optimizer.step(params, grads)
        history.append(params[:])

    return history


start = [-1.0, 1.0]

gd_history = optimize(
    GradientDescent(lr=0.0005),
    rosenbrock,
    rosenbrock_gradient,
    start
)

sgd_history = optimize(
    SGDMomentum(lr=0.0001, momentum=0.9),
    rosenbrock,
    rosenbrock_gradient,
    start
)

adam_history = optimize(
    Adam(lr=0.01),
    rosenbrock,
    rosenbrock_gradient,
    start
)

for name, history in [
    ("GD", gd_history),
    ("SGD+M", sgd_history),
    ("Adam", adam_history),
]:
    final = history[-1]
    loss = rosenbrock(final)

    print(f"{name:6s} -> x={final[0]:.6f}, y={final[1]:.6f}, loss={loss:.8f}")
```

预期现象：

```txt
Adam 通常最快接近最优点。
SGD with momentum 走得更平滑。
Vanilla GD 在狭长山谷中前进较慢。
```

---

### 4.6 实现学习率指数衰减

```python
class GradientDescentWithDecay:
    def __init__(self, lr=0.001, decay=0.999):
        self.lr0 = lr
        self.decay = decay
        self.step_count = 0

    def current_lr(self):
        return self.lr0 * (self.decay ** self.step_count)

    def step(self, params, grads):
        lr = self.current_lr()
        self.step_count += 1

        return [
            p - lr * g
            for p, g in zip(params, grads)
        ]
```

对应：

```text
lr = lr0 * decay^step
```

---

## 5. 生产使用：用 PyTorch 配置优化器

---

### 5.1 PyTorch 常见优化器

```python
import torch

model = torch.nn.Linear(784, 10)

sgd = torch.optim.SGD(
    model.parameters(),
    lr=0.01,
    momentum=0.9
)

adam = torch.optim.Adam(
    model.parameters(),
    lr=0.001
)

adamw = torch.optim.AdamW(
    model.parameters(),
    lr=0.001,
    weight_decay=0.01
)
```

---

### 5.2 PyTorch 学习率调度

```python
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    adam,
    T_max=100
)
```

训练循环中通常是：

```python
for epoch in range(num_epochs):
    for batch in dataloader:
        optimizer.zero_grad()
        loss = model_loss(batch)
        loss.backward()
        optimizer.step()

    scheduler.step()
```

---

### 5.3 优化器选择经验

| 场景 | 推荐 |
|---|---|
| 不确定先用什么 | Adam，`lr=0.001` |
| 想要更好最终泛化，能多调参 | SGD + momentum |
| Transformer / LLM | AdamW + warmup + decay |
| 训练不稳定 | 降低 learning rate |
| 训练太慢 | 适当提高 learning rate |
| loss NaN | 先检查 lr、梯度爆炸、数值稳定性 |
| 长训练任务 | 使用学习率调度 |

---

### 5.4 手写实现 vs PyTorch

| 对比项 | 手写优化器 | PyTorch optimizer |
|---|---|---|
| 目的 | 理解更新规则 | 工程实践 |
| 参数管理 | list 手动维护 | `model.parameters()` |
| 梯度来源 | 手动提供 | autograd 自动计算 |
| Momentum | 自己维护 velocity | 内部维护 state |
| Adam m/v | 自己维护 | 内部维护 optimizer state |
| GPU | 不支持 | 支持 |
| 参数组 | 不支持 | 支持 parameter groups |
| weight decay | 需手写 | 内置 |
| gradient clipping | 需手写 | 可配合工具使用 |
| scheduler | 需手写 | 内置多种 scheduler |

---

## 6. 交付沉淀：产出物、连接关系、练习、术语表

---

### 6.1 本课产出

```txt
code/optimizers.py
code/use_pytorch.py
outputs/prompt-optimizer-guide.md
outputs/exercises-solutions.md
```

| 文件 | 作用 |
|---|---|
| `code/optimizers.py` | 从零实现 GD、SGD Momentum、Adam、学习率衰减 |
| `code/use_pytorch.py` | PyTorch optimizer 和 scheduler 示例 |
| `outputs/prompt-optimizer-guide.md` | 用于选择优化器的 prompt |
| `outputs/exercises-solutions.md` | 练习题与参考答案 |

---

### 6.2 知识连接关系

| 本课知识点 | 后续连接 |
|---|---|
| 梯度下降 | 神经网络训练基础 |
| 学习率 | 训练稳定性、调参 |
| SGD | mini-batch training |
| Momentum | SGD 改进、二阶直觉 |
| Adam | Transformer 训练、LLM fine-tuning |
| AdamW | 现代大模型训练 |
| 学习率调度 | warmup、cosine decay |
| 凸优化 | 传统 ML |
| 非凸优化 | 深度学习 |
| 鞍点 | 高维优化 |
| loss landscape | 泛化、sharp / flat minima |

---

### 6.3 AI 应用连接

| 数学 / 工程知识点 | AI 应用 |
|---|---|
| 优化 | 模型训练 |
| 损失函数 | 训练目标 |
| 梯度下降 | 基础参数更新 |
| Mini-batch | 大规模数据训练 |
| Momentum | 加速收敛、减少震荡 |
| Adam | 默认深度学习优化器 |
| AdamW | Transformer / LLM |
| 学习率调度 | 稳定长训练 |
| 鞍点 | 深度学习训练障碍 |
| flat minima | 泛化能力 |
| loss landscape | 训练过程分析 |

---

### 6.4 练习

1. **学习率扫描。**  
   在 Rosenbrock 函数上运行 vanilla gradient descent，学习率取：

   ```txt
   [0.0001, 0.0005, 0.001, 0.005, 0.01]
   ```

   每个训练 5000 步，打印最终 loss。找出最大还能收敛的学习率。

2. **Momentum 对比。**  
   在 Rosenbrock 函数上运行 SGD with momentum，momentum 取：

   ```txt
   [0.0, 0.5, 0.9, 0.99]
   ```

   观察哪个收敛最快，哪个可能 overshoot。

3. **鞍点逃逸。**  
   定义：

   ```text
   f(x, y) = x^2 - y^2
   ```

   从 `(0.01, 0.01)` 开始，比较 vanilla GD、Momentum、Adam 的行为。

4. **实现学习率衰减。**  
   给 GradientDescent 类加入指数衰减：

   ```text
   lr = lr0 * 0.999^step
   ```

   比较有无 decay 的收敛过程。

5. **PyTorch 训练循环。**  
   写一个最小 PyTorch 训练循环，分别使用 SGD、Adam、AdamW，观察 loss 曲线差异。

---

### 6.5 关键术语

| 术语 | 常见说法 | 真正含义 |
|---|---|---|
| 优化 | 找最优解 | 寻找让目标函数最小或最大的参数 |
| 损失函数 | 错误大小 | 衡量模型预测和真实目标之间差距的函数 |
| 梯度下降 | 往下走 | 用负梯度方向更新参数 |
| 学习率 | 步长 | 控制每次参数更新幅度 |
| Batch GD | 全量梯度下降 | 用完整数据集计算一次精确梯度 |
| SGD | 随机梯度下降 | 用随机样本或小批量估计梯度 |
| Mini-batch | 小批数据 | 训练中最常用的梯度估计方式 |
| Momentum | 惯性 | 累积历史梯度形成 velocity，减少震荡 |
| Adam | 自适应优化器 | 使用一阶矩和二阶矩为每个参数调整学习率 |
| Bias correction | 偏差修正 | 修正 Adam 初期 m/v 从 0 初始化带来的偏差 |
| Learning rate schedule | 学习率调度 | 训练过程中动态改变学习率 |
| Convex function | 一个碗 | 任意局部最小值都是全局最小值 |
| Non-convex | 多山谷地形 | 神经网络 loss 的典型形态 |
| Saddle point | 鞍点 | 梯度为 0，但不是最小值 |
| Loss landscape | 损失地形 | 参数空间中 loss 的高维地形 |
| Convergence | 收敛 | 继续更新已经无法显著降低 loss |

---

### 6.6 自检问题

学完本课后，你应该能回答：

- 优化方法最初解决什么问题？
- 为什么神经网络训练就是优化？
- 梯度下降的更新公式是什么？
- 学习率太大或太小会发生什么？
- Batch GD、SGD、Mini-batch GD 有什么区别？
- 为什么 mini-batch noise 有时有帮助？
- Momentum 如何减少 zigzag？
- Adam 的一阶矩和二阶矩分别是什么？
- Adam 为什么要做 bias correction？
- AdamW 为什么常用于 Transformer？
- 学习率调度有什么作用？
- 凸函数和非凸函数有什么区别？
- 为什么鞍点是高维优化的重要问题？
- flat minima 为什么通常泛化更好？
- 如何从零实现 GD、Momentum 和 Adam？
- PyTorch 中 optimizer 和 scheduler 如何配置？

---

## 附录：本课最小代码文件建议

如果要拆成代码文件，可以这样组织：

```txt
code/
├── optimizers.py
└── use_pytorch.py
```

### `optimizers.py`

包含：

```txt
rosenbrock
rosenbrock_gradient
GradientDescent
SGDMomentum
Adam
optimize
GradientDescentWithDecay
learning_rate_sweep
momentum_comparison
```

### `use_pytorch.py`

包含：

```txt
torch.optim.SGD
torch.optim.Adam
torch.optim.AdamW
torch.optim.lr_scheduler.CosineAnnealingLR
minimal training loop
scheduler.step
```
