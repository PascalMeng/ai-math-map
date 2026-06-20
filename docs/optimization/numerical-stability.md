---
title: 数值稳定性
description: 面向 AI 算法工程师的数值稳定性课程：从 IEEE 754、浮点误差、overflow/underflow、log-sum-exp、稳定 softmax，到混合精度、loss scaling 和梯度检查。
---

# 数值稳定性

> 浮点数是一层会漏水的抽象。它会在训练中咬你一口，而且常常是在你看不见的地方。

**课程类型：** 实现 / 应用  
**所属模块：** 数值计算与训练稳定性  
**前置知识：** Phase 1 Lesson 01-04：线性代数、矩阵运算、微积分、梯度下降  
**预计时间：** 约 120 分钟  
**使用语言：** Python / NumPy / PyTorch  

---

## 1. 提出问题：为什么 AI 算法工程师必须理解数值稳定性

你训练一个模型，跑了三个小时。  
突然：

```txt
loss = NaN
```

你加了打印日志，发现第 9000 步 logits 正常；第 9001 步 logits 变成 `inf`；第 9002 步所有 gradient 都变成 `nan`。

或者，你复现一篇论文，结构、超参数、数据都对，但 accuracy 比论文低 2%。  
最后发现：论文用的是 float32，而你用了 float16，且没有正确 loss scaling。  
大量小梯度在 float16 中下溢为 0，模型悄悄少学了一部分东西。

再比如，你从零实现 cross-entropy loss。  
小 logits 下没问题；当 logits 超过 100 时，loss 变成 `inf`。  
原因是：

```txt
exp(100) 在 float32 中会 overflow。
```

数值稳定性不是理论细节，而是训练能不能成功的工程底线。

---

### 1.1 数学概念历史出现缘由

数值稳定性来自一个基本事实：

```txt
计算机不能精确表示所有实数。
```

浮点数只是实数的有限近似。  
当深度学习涉及大量矩阵乘法、指数、对数、归一化、梯度累积和低精度训练时，微小误差会被放大成训练失败。

| 数学 / 工程概念 | 出现缘由 |
|---|---|
| IEEE 754 | 为了统一计算机浮点数表示、舍入规则和特殊值 |
| float32 / float16 / bfloat16 | 为了在精度、范围、速度和显存之间折中 |
| machine epsilon | 为了描述某种浮点格式能分辨的最小相对变化 |
| 舍入误差 | 因为有限 mantissa 无法精确表示所有实数 |
| 灾难性抵消 | 因为两个很接近的数相减会丢失有效数字 |
| overflow | 因为结果超过浮点格式最大可表示范围 |
| underflow | 因为结果小于浮点格式最小可表示正数 |
| NaN / Inf | 为了表示未定义结果和无穷大 |
| log-sum-exp trick | 为了稳定计算 `log(sum(exp(x)))` |
| stable softmax | 为了避免 logits 过大导致 `exp` overflow |
| stable cross-entropy | 为了避免 softmax + log 的数值问题 |
| 梯度检查 | 为了验证反向传播梯度是否正确 |
| mixed precision | 为了利用低精度加速训练，同时保留稳定性 |
| loss scaling | 为了防止 float16 梯度 underflow |
| gradient clipping | 为了防止梯度爆炸毁掉参数 |
| normalization | 为了让激活值保持在安全数值范围内 |

可以这样理解：

```txt
现实问题：计算机如何表示实数？
工程标准：IEEE 754

现实问题：为什么 0.1 + 0.2 不等于 0.3？
数学原因：二进制浮点无法精确表示十进制小数

现实问题：为什么 softmax 会溢出？
数值原因：exp(logit) 超过浮点范围

现实问题：为什么 float16 训练会丢梯度？
数值原因：小梯度 underflow 为 0

现实问题：如何让训练更稳定？
工程方法：stable softmax、log-sum-exp、loss scaling、gradient clipping、normalization
```

---

### 1.2 原始问题是什么

本课要解决的原始问题是：

```txt
计算机如何存储实数？
浮点数为什么会有舍入误差？
为什么不能用 == 比较浮点数？
灾难性抵消为什么会让相对误差暴涨？
overflow 和 underflow 分别如何产生？
为什么 exp、log、softmax、cross-entropy 是数值事故高发区？
log-sum-exp trick 为什么能稳定计算？
为什么 float16 训练需要 loss scaling？
为什么 bfloat16 通常比 float16 更适合训练？
如何检测和预防 NaN / Inf？
如何用数值梯度检查 analytical gradient 是否正确？
```

换成 AI 语言，就是：

```txt
为什么 loss 会突然 NaN？
为什么 logits 稍大 softmax 就炸了？
为什么 PyTorch 的 CrossEntropyLoss 要接收 logits 而不是概率？
为什么混合精度训练需要 GradScaler？
为什么 bfloat16 在大模型训练里很常用？
为什么 gradient clipping 是训练安全机制？
为什么 LayerNorm 里要加 epsilon？
为什么不同 GPU 上结果会有微小差异？
```

---

### 1.3 AI 中的现代问题

| 数值稳定性知识点 | AI 中的现代问题 |
|---|---|
| IEEE 754 | 理解 float 格式和特殊值 |
| float32 | 默认训练精度 |
| float16 | 加速推理和训练，但范围小 |
| bfloat16 | 大模型训练常用，范围接近 float32 |
| machine epsilon | 浮点比较和误差容忍 |
| catastrophic cancellation | 方差、有限差分、log 概率差值 |
| overflow | `exp`、softmax、激活值爆炸 |
| underflow | 小概率、小梯度变成 0 |
| log-sum-exp | 稳定 softmax / log probability |
| stable cross-entropy | 分类和语言模型核心 loss |
| NaN / Inf | 训练崩溃信号 |
| gradient checking | 自定义层、自定义 loss 验证 |
| mixed precision | GPU Tensor Core 加速 |
| loss scaling | 防止 float16 gradient underflow |
| gradient clipping | 防止 exploding gradients |
| normalization | 控制 activation / gradient 范围 |
| deterministic algorithms | 复现实验结果 |

---

### 1.4 本课要解决的核心问题

学完本课，你应该能够回答：

1. IEEE 754 浮点数由哪几部分组成？
2. float32、float16、bfloat16 的 range 和 precision 有什么区别？
3. 为什么 `0.1 + 0.2 != 0.3`？
4. 为什么浮点数不能直接用 `==` 比较？
5. 什么是灾难性抵消？
6. overflow 和 underflow 分别在什么情况下出现？
7. 为什么 naive softmax 会 overflow？
8. log-sum-exp trick 为什么有效？
9. stable cross-entropy 如何避免 `inf` 和 `nan`？
10. NaN / Inf 如何检测和预防？
11. 数值梯度检查如何验证反向传播？
12. mixed precision 为什么能加速训练？
13. float16 为什么需要 loss scaling？
14. 为什么 bfloat16 更适合训练？
15. gradient clipping 和 normalization 如何提升训练稳定性？

---

## 2. 概念定位：这节课学什么

---

### 2.1 所属模块

```txt
所属模块：数值计算与训练稳定性
前置知识：指数、对数、softmax、交叉熵、梯度下降、反向传播
后续连接：训练循环、混合精度、Transformer attention、LLM 训练、PyTorch AMP、调试 NaN / Inf
```

前面的课程已经解决了：

```txt
模型如何计算？
模型如何求导？
模型如何更新参数？
```

本课进一步解决：

```txt
这些计算在有限精度机器上如何稳定执行？
```

---

### 2.2 本课学习目标

完成本课后，你应该能够：

- 解释 float32、float16、bfloat16 的结构差异
- 识别 overflow、underflow、NaN、Inf 和灾难性抵消
- 从零实现 stable softmax 和 stable log-sum-exp
- 从零实现 stable cross-entropy
- 使用 centered finite difference 做 gradient checking
- 判断 analytical gradient 是否正确
- 理解 mixed precision training 的流程
- 解释 loss scaling 如何防止 float16 gradient underflow
- 说明为什么 bfloat16 通常更适合训练
- 实现 gradient clipping by norm
- 给出训练 NaN / Inf 的排查清单

---

### 2.3 本课在整体路线中的位置

```txt
概率与信息论
    ↓
softmax / cross-entropy
    ↓
数值稳定性
    ↓
stable loss / gradient check
    ↓
mixed precision / loss scaling
    ↓
大模型训练稳定性
```

本课的核心能力是：

```txt
看到 loss NaN、grad inf、softmax overflow、float16 underflow 时，能快速定位原因并给出稳定实现。
```

---

## 3. 理解概念：直觉、公式、数值陷阱、AI 连接

---

### 3.1 IEEE 754：计算机如何存储实数

**一句话直觉：**  
浮点数用符号位、指数和尾数近似表示实数；指数决定范围，尾数决定精度。

float32 结构：

```txt
[1 sign bit] [8 exponent bits] [23 mantissa bits]
```

形式：

```text
value = (-1)^sign * 2^(exponent - bias) * 1.mantissa
```

常见格式对比：

| 格式 | 总位数 | 指数位 | 尾数位 | 十进制有效位 | 范围近似 |
|---|---:|---:|---:|---:|---|
| float64 | 64 | 11 | 52 | 15-16 | ±1.8e308 |
| float32 | 32 | 8 | 23 | 7-8 | ±3.4e38 |
| float16 | 16 | 5 | 10 | 3-4 | ±65504 |
| bfloat16 | 16 | 8 | 7 | 2-3 | ±3.4e38 |

核心差异：

```txt
float16 精度比 bfloat16 高一点，但范围小很多。
bfloat16 精度低一点，但范围接近 float32。
```

**AI 连接：**

| 格式 | 常见用途 |
|---|---|
| float32 | 默认训练精度 |
| float16 | GPU 加速、推理、AMP |
| bfloat16 | 大模型训练、TPU、A100/H100 |
| float64 | 科学计算、数值验证、gradcheck |

---

### 3.2 为什么 `0.1 + 0.2 != 0.3`

**一句话直觉：**  
很多十进制小数无法用有限二进制浮点精确表示。

十进制 `0.1` 的二进制表示是无限循环：

```txt
0.1 in binary = 0.000110011001100110011...
```

计算机只能截断或舍入它。  
所以：

```python
0.1 + 0.2
```

结果是：

```txt
0.30000000000000004
```

而不是精确的 `0.3`。

因此：

```python
0.1 + 0.2 == 0.3
```

返回：

```txt
False
```

**正确做法：**

```python
abs(a - b) < epsilon
```

或者：

```python
math.isclose(a, b)
```

**AI 连接：**

| 场景 | 风险 |
|---|---|
| loss 阈值判断 | 直接比较可能出错 |
| reproducibility test | 不同硬件有微小差异 |
| checksum | 浮点累加顺序不同导致差异 |
| early stopping | 需要 tolerance |
| gradient check | 需要 relative error |

---

### 3.3 Machine epsilon：浮点精度极限

**一句话直觉：**  
machine epsilon 是某种浮点格式能分辨的最小相对变化。

定义：

```txt
最小的 e，使得 1.0 + e != 1.0
```

float32 中约为：

```text
1.19e-7
```

这意味着：

```txt
float32 只能可靠表示约 7 位十进制有效数字。
```

**AI 连接：**

| 场景 | 意义 |
|---|---|
| gradient checking | tolerance 不能设太小 |
| convergence 判断 | loss 变化低于 eps 可能无意义 |
| float comparison | 使用相对误差 |
| accumulation | 小更新可能被舍入掉 |

---

### 3.4 灾难性抵消：相近数相减导致有效数字丢失

**一句话直觉：**  
两个很接近的浮点数相减时，前面的有效数字互相抵消，剩下的可能主要是舍入误差。

例子：

```txt
a = 1.0000001
b = 1.0000000

true difference = 0.0000001
computed difference 可能有很大相对误差
```

常见危险公式：

```text
Var(X) = E[X^2] - E[X]^2
```

如果 `X` 的均值很大，而方差很小，两个大数相减会导致严重精度损失。

更稳定的做法：

```txt
先中心化数据。
使用 Welford online algorithm。
避免大数相减。
```

**AI 连接：**

| 场景 | 风险 |
|---|---|
| 方差计算 | `E[x^2] - E[x]^2` 不稳定 |
| finite difference | h 太小会相减抵消 |
| log probability 差值 | 接近时误差大 |
| normalization | std 接近 0 时不稳定 |

---

### 3.5 Overflow 与 Underflow

**一句话直觉：**  
overflow 是数太大变成 `inf`；underflow 是数太小变成 `0.0`。

float32 近似边界：

```txt
最大值：3.4e38
最小 normal 正数：1.17e-38
最小 denormal 正数：1.4e-45
```

`exp()` 是 ML 中最常见的 overflow 来源：

```txt
exp(88.7) ≈ float32 上限
exp(89.0) -> inf
exp(-104) -> 0.0
```

`log()` 的危险：

```txt
log(0.0) -> -inf
log(-1.0) -> nan
```

**AI 连接：**

| 操作 | 风险 |
|---|---|
| softmax | `exp(logit)` overflow |
| sigmoid | 大正数 / 大负数导致饱和 |
| cross-entropy | `log(0)` |
| KL divergence | 概率为 0 |
| sequence probability | 小概率连乘 underflow |
| float16 gradient | 小梯度 underflow 成 0 |

---

### 3.6 Log-sum-exp trick

**一句话直觉：**  
计算 `log(sum(exp(x)))` 时，先减去最大值，可以避免 `exp` overflow 和 `log(0)`。

原始形式：

```text
log(sum_i exp(x_i))
```

稳定形式：

```text
c + log(sum_i exp(x_i - c))
```

取：

```text
c = max(x)
```

得到：

```text
logsumexp(x) = max(x) + log(sum_i exp(x_i - max(x)))
```

为什么稳定？

```txt
减去 max 后，最大的指数项是 exp(0)=1，不会 overflow。
至少有一个项是 1，所以求和不为 0，不会 log(0)。
```

**AI 连接：**

log-sum-exp 用在：

- softmax
- log-softmax
- cross-entropy
- sequence log probability
- mixture models
- variational inference
- CRF / HMM / Viterbi-like algorithms

---

### 3.7 Stable softmax

**一句话直觉：**  
stable softmax 和 naive softmax 数学等价，但先减最大 logit，避免 overflow。

Naive softmax：

```text
softmax(x_i) = exp(x_i) / sum_j exp(x_j)
```

如果：

```txt
x = [100, 101, 102]
```

`exp(102)` 在 float32 中会 overflow。

稳定版本：

```txt
x_shifted = x - max(x) = [-2, -1, 0]
```

然后：

```text
softmax(x_i) = exp(x_i - max(x)) / sum_j exp(x_j - max(x))
```

结果完全相同，但不会溢出。

**AI 连接：**

不要手写：

```python
np.exp(logits) / np.exp(logits).sum()
```

应使用：

```python
logits_shifted = logits - logits.max()
```

或者直接调用框架稳定实现：

```python
torch.nn.functional.softmax
torch.nn.functional.log_softmax
```

---

### 3.8 Stable cross-entropy

**一句话直觉：**  
稳定交叉熵不要先算 softmax 再取 log，而是直接用 log-sum-exp 计算正确类负 log probability。

分类交叉熵：

```text
loss = -log softmax(logits)[true_class]
```

展开：

```text
loss = -log( exp(z_y) / sum_j exp(z_j) )
     = -z_y + log(sum_j exp(z_j))
```

稳定形式：

```text
loss = -z_y + logsumexp(z)
```

也就是：

```text
loss = logsumexp(logits) - logits[true_class]
```

**AI 连接：**

PyTorch 的：

```python
torch.nn.CrossEntropyLoss
```

内部本质上做的是：

```txt
log_softmax + negative log likelihood
```

这比手动 `softmax -> log -> NLL` 更稳定。

---

### 3.9 NaN 与 Inf：检测和预防

**一句话直觉：**  
一个 NaN 或 Inf 会像病毒一样沿计算图传播，快速毁掉训练。

Inf 常见来源：

```txt
exp(large)
division by zero
float overflow
gradient explosion
```

NaN 常见来源：

```txt
0 / 0
inf - inf
inf * 0
sqrt(negative)
log(negative)
已有 NaN 参与任何计算
```

检测：

```python
import math

math.isnan(x)
math.isinf(x)
math.isfinite(x)
```

预防：

```txt
1. 对 exp 输入做 clamp
2. 分母加 epsilon
3. log 输入加 epsilon
4. 使用 log-sum-exp 和 stable softmax
5. 使用 gradient clipping
6. 降低 learning rate
7. debug 时每步检查 tensor 是否 finite
```

---

### 3.10 数值梯度检查

**一句话直觉：**  
梯度检查用有限差分近似梯度，验证 analytical gradient 是否正确。

中心差分：

```text
df/dx ≈ (f(x + h) - f(x - h)) / (2h)
```

比 forward difference 更准确：

```text
(f(x + h) - f(x)) / h
```

选择 `h`：

```txt
太大：近似误差大。
太小：灾难性抵消严重。
通常取 1e-5 到 1e-7。
```

相对误差：

```text
relative_error =
|grad_analytical - grad_numerical|
/ max(|grad_analytical|, |grad_numerical|, 1e-8)
```

经验判断：

| relative error | 判断 |
|---:|---|
| `< 1e-7` | 非常好 |
| `< 1e-5` | 通常可接受 |
| `> 1e-3` | 可能有问题 |
| `> 1` | 基本错误 |

**AI 连接：**

当你实现自定义 layer、loss 或 backward 时，应该先做 gradient check。  
PyTorch 提供：

```python
torch.autograd.gradcheck()
```

---

### 3.11 Mixed precision training

**一句话直觉：**  
混合精度训练用低精度加速大部分计算，同时保留高精度权重和关键操作以保持稳定。

典型流程：

```txt
1. 保留 float32 master weights
2. forward 使用 float16 / bfloat16 加速
3. loss 用 float32 计算
4. backward 使用低精度加速
5. gradient 适当缩放或转换
6. 用 float32 master weights 更新参数
```

优势：

```txt
更快的矩阵乘法
更低显存占用
更高吞吐
```

风险：

```txt
float16 range 小，容易 overflow / underflow。
小梯度可能变成 0。
```

---

### 3.12 Loss scaling：防止 float16 梯度下溢

**一句话直觉：**  
loss scaling 把 loss 放大，让梯度也变大，避免 float16 中小梯度 underflow 成 0；更新前再缩回来。

流程：

```txt
1. loss_scaled = loss * scale
2. backward 计算 scaled gradients
3. gradients 变成原来的 scale 倍
4. 更新参数前 gradients /= scale
5. 最终数学更新不变，但避免中间 underflow
```

动态 loss scaling：

```txt
初始使用较大 scale。
如果发现 gradient overflow，就减小 scale。
如果很多步没有 overflow，就增大 scale。
```

**AI 连接：**

PyTorch AMP 中常见：

```python
torch.cuda.amp.GradScaler
```

它自动处理 loss scaling。

---

### 3.13 float16 vs bfloat16

**一句话直觉：**  
float16 精度稍高但范围小；bfloat16 精度低一点但范围接近 float32，因此更适合训练。

结构：

```txt
float16:  [1 sign] [5 exponent] [10 mantissa]
bfloat16: [1 sign] [8 exponent] [7 mantissa]
```

比较：

| 格式 | 优势 | 劣势 | 适合 |
|---|---|---|---|
| float16 | mantissa 更多，精度略好 | range 小，容易 overflow/underflow | 推理、AMP |
| bfloat16 | range 接近 float32 | 精度略低 | 大模型训练 |
| float32 | 稳定 | 慢、占显存 | master weights、关键计算 |

为什么训练更偏好 bfloat16？

```txt
训练中 range 通常比 precision 更重要。
激活、logits、梯度可能突然变得很大或很小。
bfloat16 不容易像 float16 那样 overflow。
```

---

### 3.14 Gradient clipping

**一句话直觉：**  
gradient clipping 限制梯度大小，防止一次异常更新毁掉模型参数。

两种方式：

#### Clip by value

```text
grad = clamp(grad, -max_val, max_val)
```

缺点：

```txt
可能改变梯度方向。
```

#### Clip by norm

```text
if ||grad|| > max_norm:
    grad = grad * max_norm / ||grad||
```

优点：

```txt
保留梯度方向，只缩放整体长度。
```

PyTorch 常用：

```python
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

典型值：

| 场景 | max_norm |
|---|---:|
| Transformer | 1.0 |
| RL | 0.5 |
| 简单网络 | 5.0 |

---

### 3.15 Normalization 也是数值稳定器

**一句话直觉：**  
BatchNorm、LayerNorm、RMSNorm 不只是优化技巧，也是在控制激活值范围，防止数值爆炸或消失。

LayerNorm：

```text
LayerNorm(x) = (x - mean(x)) / (std(x) + epsilon) * gamma + beta
```

其中：

```txt
epsilon 防止 std 接近 0 时除以 0。
```

没有 normalization 时：

```txt
层数越深，activation 可能越来越大或越来越小。
最终导致 overflow、underflow、gradient explosion 或 gradient vanishing。
```

**AI 连接：**

| 方法 | 数值稳定作用 |
|---|---|
| BatchNorm | 稳定 batch 统计 |
| LayerNorm | 稳定 Transformer hidden states |
| RMSNorm | 稳定大模型训练，计算更简单 |
| epsilon | 防止除以 0 |
| residual + norm | 缓解深层训练不稳定 |

---

### 3.16 本课核心概念总表

| 概念 | 一句话直觉 | AI 中的对应位置 |
|---|---|---|
| IEEE 754 | 浮点数标准 | 所有数值计算基础 |
| mantissa | 有效数字 | 决定精度 |
| exponent | 指数范围 | 决定可表示大小 |
| machine epsilon | 最小可分辨变化 | 浮点比较、gradcheck |
| catastrophic cancellation | 相近数相减丢精度 | 方差、有限差分 |
| overflow | 数太大变 inf | exp、softmax |
| underflow | 数太小变 0 | 小概率、小梯度 |
| log-sum-exp | 稳定 log sum exp | softmax、CE |
| stable softmax | 减 max 再 exp | 分类、attention |
| stable CE | logsumexp - true logit | CrossEntropyLoss |
| NaN / Inf | 训练崩溃信号 | debug |
| gradient checking | 数值验证反传 | 自定义层 |
| mixed precision | 低精度加速 | 大模型训练 |
| loss scaling | 防止梯度下溢 | float16 training |
| bfloat16 | 大 range 的 16-bit | 训练 |
| gradient clipping | 限制梯度范数 | 防爆炸 |
| normalization | 控制激活范围 | Transformer、CNN |

---

## 4. 手写实现：从零实现稳定 softmax、log-sum-exp 和梯度检查

这一节不用 PyTorch。  
目标是亲手实现数值稳定版本，并理解为什么它们安全。

---

### 4.1 演示浮点精度限制

```python
print("=== Floating Point Precision ===")
print(f"0.1 + 0.2 = {0.1 + 0.2}")
print(f"0.1 + 0.2 == 0.3? {0.1 + 0.2 == 0.3}")
print(f"Difference: {(0.1 + 0.2) - 0.3:.2e}")
```

---

### 4.2 Naive softmax vs stable softmax

```python
import math


def softmax_naive(logits):
    exps = [math.exp(z) for z in logits]
    total = sum(exps)

    return [e / total for e in exps]


def softmax_stable(logits):
    max_logit = max(logits)
    exps = [math.exp(z - max_logit) for z in logits]
    total = sum(exps)

    return [e / total for e in exps]


safe_logits = [2.0, 1.0, 0.1]

print(f"Naive:  {softmax_naive(safe_logits)}")
print(f"Stable: {softmax_stable(safe_logits)}")

dangerous_logits = [100.0, 101.0, 102.0]

print(f"Stable dangerous: {softmax_stable(dangerous_logits)}")
```

不要直接运行：

```python
softmax_naive([100.0, 101.0, 102.0])
```

它可能 overflow。

---

### 4.3 实现 stable log-sum-exp

```python
def logsumexp_naive(values):
    return math.log(sum(math.exp(v) for v in values))


def logsumexp_stable(values):
    c = max(values)

    return c + math.log(sum(math.exp(v - c) for v in values))


safe = [1.0, 2.0, 3.0]

print(f"Naive:  {logsumexp_naive(safe):.6f}")
print(f"Stable: {logsumexp_stable(safe):.6f}")

large = [500.0, 501.0, 502.0]

print(f"Stable large: {logsumexp_stable(large):.6f}")
```

---

### 4.4 实现 stable cross-entropy

```python
def cross_entropy_naive(true_class, logits):
    probs = softmax_naive(logits)

    return -math.log(probs[true_class])


def cross_entropy_stable(true_class, logits):
    max_logit = max(logits)
    shifted = [z - max_logit for z in logits]

    log_sum_exp = math.log(sum(math.exp(s) for s in shifted))
    log_prob = shifted[true_class] - log_sum_exp

    return -log_prob


logits = [2.0, 5.0, 1.0]
true_class = 1

print(f"Naive:  {cross_entropy_naive(true_class, logits):.6f}")
print(f"Stable: {cross_entropy_stable(true_class, logits):.6f}")
```

---

### 4.5 实现数值梯度检查

```python
def numerical_gradient(f, x, h=1e-5):
    grad = []

    for i in range(len(x)):
        x_plus = x[:]
        x_minus = x[:]

        x_plus[i] += h
        x_minus[i] -= h

        grad.append((f(x_plus) - f(x_minus)) / (2 * h))

    return grad


def check_gradient(analytical, numerical, tolerance=1e-5):
    for i, (a, n) in enumerate(zip(analytical, numerical)):
        denom = max(abs(a), abs(n), 1e-8)
        rel_error = abs(a - n) / denom

        status = "OK" if rel_error < tolerance else "FAIL"

        print(
            f"param {i}: analytical={a:.8f} "
            f"numerical={n:.8f} rel_error={rel_error:.2e} [{status}]"
        )


def f(params):
    x, y = params

    return x ** 2 + 3 * x * y + y ** 3


def f_grad(params):
    x, y = params

    return [2 * x + 3 * y, 3 * x + 3 * y ** 2]


point = [2.0, 1.0]

analytical = f_grad(point)
numerical = numerical_gradient(f, point)

check_gradient(analytical, numerical)
```

---

### 4.6 实现 gradient clipping by norm

```python
def clip_by_norm(gradients, max_norm):
    total_norm = math.sqrt(sum(g ** 2 for g in gradients))

    if total_norm > max_norm:
        scale = max_norm / total_norm
        return [g * scale for g in gradients]

    return gradients


grads = [10.0, 20.0, 30.0]
clipped = clip_by_norm(grads, max_norm=5.0)

print(f"Original norm: {math.sqrt(sum(g ** 2 for g in grads)):.2f}")
print(f"Clipped norm:  {math.sqrt(sum(g ** 2 for g in clipped)):.2f}")
```

---

### 4.7 NaN / Inf 检测

```python
def check_tensor(name, values):
    has_nan = any(math.isnan(v) for v in values)
    has_inf = any(math.isinf(v) for v in values)

    if has_nan or has_inf:
        print(f"WARNING {name}: nan={has_nan} inf={has_inf}")
        return False

    return True


check_tensor("good", [1.0, 2.0, 3.0])
check_tensor("bad", [1.0, float("nan"), 3.0])
check_tensor("ugly", [1.0, float("inf"), 3.0])
```

---

## 5. 生产使用：NumPy / PyTorch 中的稳定写法

---

### 5.1 NumPy stable softmax

```python
import numpy as np


def np_softmax_stable(logits, axis=-1):
    logits = np.asarray(logits)

    shifted = logits - np.max(logits, axis=axis, keepdims=True)
    exps = np.exp(shifted)

    return exps / np.sum(exps, axis=axis, keepdims=True)
```

---

### 5.2 PyTorch stable cross-entropy

```python
import torch
import torch.nn.functional as F

logits = torch.tensor([[100.0, 101.0, 102.0]])
target = torch.tensor([2])

loss = F.cross_entropy(logits, target)

print(loss)
```

不要写：

```python
probs = torch.softmax(logits, dim=-1)
loss = -torch.log(probs[0, target])
```

更推荐直接使用：

```python
F.cross_entropy(logits, target)
```

---

### 5.3 PyTorch AMP 混合精度

```python
import torch

model = torch.nn.Linear(768, 10).cuda()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
scaler = torch.cuda.amp.GradScaler()

for x, y in dataloader:
    x = x.cuda()
    y = y.cuda()

    optimizer.zero_grad()

    with torch.cuda.amp.autocast():
        logits = model(x)
        loss = F.cross_entropy(logits, y)

    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)

    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

    scaler.step(optimizer)
    scaler.update()
```

说明：

| 代码 | 作用 |
|---|---|
| `autocast()` | 自动选择低精度或高精度运算 |
| `GradScaler()` | 自动 loss scaling |
| `scale(loss)` | 放大 loss，避免梯度 underflow |
| `unscale_(optimizer)` | 更新前还原梯度尺度 |
| `clip_grad_norm_` | 在 unscale 后裁剪梯度 |
| `step/update` | 安全更新参数并调整 scale |

---

### 5.4 PyTorch NaN / Inf 检查

```python
def check_model_finite(model):
    for name, param in model.named_parameters():
        if not torch.isfinite(param).all():
            print(f"Parameter has NaN/Inf: {name}")
            return False

        if param.grad is not None and not torch.isfinite(param.grad).all():
            print(f"Gradient has NaN/Inf: {name}")
            return False

    return True
```

---

### 5.5 手写实现 vs 生产库

| 对比项 | 手写实现 | NumPy / PyTorch |
|---|---|---|
| 目的 | 理解数值问题 | 工程实践 |
| softmax | 手写 max-subtraction | 框架内置 |
| logsumexp | 手写稳定公式 | `torch.logsumexp` |
| cross-entropy | 手写 stable CE | `F.cross_entropy` |
| gradcheck | 手写 finite difference | `torch.autograd.gradcheck` |
| mixed precision | 手动模拟 | PyTorch AMP |
| loss scaling | 手动缩放 | `GradScaler` |
| gradient clipping | 手写 norm clipping | `clip_grad_norm_` |
| NaN 检测 | `math.isfinite` | `torch.isfinite` |

---

## 6. 交付沉淀：产出物、连接关系、练习、术语表

---

### 6.1 本课产出

```txt
code/numerical.py
code/use_numpy.py
code/use_pytorch.py
outputs/prompt-numerical-debugger.md
outputs/exercises-solutions.md
```

| 文件 | 作用 |
|---|---|
| `code/numerical.py` | 从零实现 stable softmax、logsumexp、stable CE、gradcheck、clip_by_norm |
| `code/use_numpy.py` | NumPy stable softmax 和数值实验 |
| `code/use_pytorch.py` | PyTorch AMP、CrossEntropyLoss、NaN 检查 |
| `outputs/prompt-numerical-debugger.md` | 诊断 NaN / Inf 和数值问题的 prompt |
| `outputs/exercises-solutions.md` | 练习题与参考答案 |

---

### 6.2 知识连接关系

| 本课知识点 | 后续连接 |
|---|---|
| IEEE 754 | 所有数值计算 |
| overflow / underflow | softmax、CE、mixed precision |
| log-sum-exp | log-softmax、cross-entropy、sequence likelihood |
| stable softmax | attention、classification |
| stable CE | classification、language modeling |
| gradient checking | 自定义 autograd、layer、loss |
| mixed precision | GPU 加速训练 |
| loss scaling | float16 训练稳定性 |
| bfloat16 | 大模型训练 |
| gradient clipping | Transformer、RNN、RL |
| normalization | LayerNorm、RMSNorm、BatchNorm |

---

### 6.3 AI 应用连接

| 数值稳定知识点 | AI 应用 |
|---|---|
| stable softmax | attention scores、分类概率 |
| log-sum-exp | log probability、CE loss |
| stable cross-entropy | 分类和语言模型训练 |
| NaN / Inf 检测 | 训练 debug |
| gradient checking | 自定义层和 loss |
| mixed precision | 大模型训练加速 |
| loss scaling | float16 训练 |
| bfloat16 | LLM pretraining / fine-tuning |
| gradient clipping | 防止梯度爆炸 |
| normalization | 深层网络稳定训练 |

---

### 6.4 练习

1. **灾难性抵消实验。**  
   用 float32 计算：

   ```txt
   [1000000.0, 1000001.0, 1000002.0]
   ```

   的方差。  
   先用 naive formula：

   ```text
   E[x^2] - E[x]^2
   ```

   再用 Welford online algorithm。  
   比较真实方差 `0.6667` 附近的误差。

2. **寻找 machine epsilon。**  
   找到最小正 float32 数 `x`，使得：

   ```text
   1.0 + x != 1.0
   ```

   并与：

   ```python
   numpy.finfo(numpy.float32).eps
   ```

   对比。

3. **log-sum-exp 边界测试。**  
   测试 `logsumexp_stable`：

   ```txt
   所有值相等
   一个值远大于其他值
   所有值都很负，例如 -1000
   ```

   验证 naive 版本在哪里失败。

4. **线性层梯度检查。**  
   实现一个线性层：

   ```text
   y = W x + b
   ```

   以及 analytical backward。  
   用 `numerical_gradient` 验证 `W` 和 `b` 的梯度。

5. **loss scaling 实验。**  
   随机生成范围在：

   ```txt
   [1e-9, 1e-3]
   ```

   的梯度，转换成 float16，统计变成 0 的比例。  
   再先乘以 1024，转换成 float16 后除回来，比较 zero fraction。

---

### 6.5 关键术语

| 术语 | 常见说法 | 真正含义 |
|---|---|---|
| IEEE 754 | 浮点标准 | 定义二进制浮点格式、舍入规则和 NaN/Inf 等特殊值 |
| Machine epsilon | 精度极限 | 使 `1.0 + eps != 1.0` 的最小正数 |
| Catastrophic cancellation | 灾难性抵消 | 相近浮点数相减导致有效数字丢失 |
| Overflow | 数太大 | 结果超过最大可表示值，变成 `inf` |
| Underflow | 数太小 | 结果小于最小可表示正数，变成 `0.0` |
| Log-sum-exp trick | 减最大值 | 稳定计算 `log(sum(exp(x)))` |
| Stable softmax | 稳定 softmax | softmax 前先减最大 logit |
| Stable cross-entropy | 稳定 CE | 用 `logsumexp(logits) - true_logit` 计算 |
| Gradient checking | 梯度检查 | 用有限差分验证 analytical gradient |
| Mixed precision | 混合精度 | 用低精度加速，用高精度保持稳定 |
| Loss scaling | 损失缩放 | 放大 loss 防止 float16 梯度下溢 |
| bfloat16 | Brain float | 16-bit 格式，范围接近 float32，适合训练 |
| Gradient clipping | 梯度裁剪 | 限制梯度范数，防止梯度爆炸 |
| NaN | 非数 | 由未定义运算产生，并会传播 |
| Inf | 无穷大 | 由 overflow 或除零产生 |
| Numerical gradient | 数值梯度 | 用有限差分近似导数 |

---

### 6.6 自检问题

学完本课后，你应该能回答：

- 数值稳定性为什么是深度学习训练的工程底线？
- float32、float16、bfloat16 的 range 和 precision 有什么区别？
- 为什么 `0.1 + 0.2 == 0.3` 是 False？
- machine epsilon 有什么意义？
- 灾难性抵消为什么危险？
- overflow 和 underflow 如何导致 `inf`、`0`、`nan`？
- 为什么 naive softmax 会爆？
- log-sum-exp trick 为什么能避免 overflow？
- stable cross-entropy 为什么不先显式 softmax？
- 如何检测 NaN / Inf？
- centered finite difference 为什么比 forward difference 更准？
- mixed precision training 的基本流程是什么？
- loss scaling 为什么能防止 float16 梯度下溢？
- 为什么 bfloat16 常用于大模型训练？
- gradient clipping 为什么是安全机制？
- normalization 如何帮助数值稳定？

---

## 附录：本课最小代码文件建议

如果要拆成代码文件，可以这样组织：

```txt
code/
├── numerical.py
├── use_numpy.py
└── use_pytorch.py
```

### `numerical.py`

包含：

```txt
softmax_naive
softmax_stable
logsumexp_naive
logsumexp_stable
cross_entropy_naive
cross_entropy_stable
numerical_gradient
check_gradient
clip_by_norm
check_tensor
float16 / bfloat16 simulation
```

### `use_numpy.py`

包含：

```txt
np_softmax_stable
np.logaddexp / logsumexp examples
np.finfo
catastrophic cancellation demo
Welford variance
```

### `use_pytorch.py`

包含：

```txt
torch.nn.functional.cross_entropy
torch.logsumexp
torch.cuda.amp.autocast
torch.cuda.amp.GradScaler
torch.nn.utils.clip_grad_norm_
torch.isfinite
torch.autograd.gradcheck
```
