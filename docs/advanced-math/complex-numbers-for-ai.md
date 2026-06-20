---
title: AI 中的复数
description: 面向 AI 算法工程师的复数课程：从复数运算、复平面、极坐标、Euler 公式、单位根、DFT，到 Transformer 位置编码和 RoPE。
---

# AI 中的复数

> `-1` 的平方根并不“虚幻”。它是理解旋转、频率和信号处理的关键。

**课程类型：** 学习 / 实现  
**所属模块：** 进阶数学 / 信号处理 / Transformer 位置编码  
**前置知识：** Phase 1 Lesson 01-04：线性代数、矩阵运算、微积分  
**预计时间：** 约 60 分钟  
**使用语言：** Python / NumPy  

---

## 1. 提出问题：为什么 AI 算法工程师必须理解复数

你打开一篇傅里叶变换论文，里面到处都是：

```txt
i
e^(iθ)
complex exponential
roots of unity
```

你看 Transformer 位置编码，会看到：

```txt
sin
cos
不同频率
```

它们其实就是复指数的实部和虚部。

你再看 RoPE，也就是 Rotary Position Embedding，会发现它本质上是在对 query 和 key 做旋转。  
而复数乘法正是二维旋转最自然的表达方式。

复数一开始看起来很抽象：

```txt
i^2 = -1
```

似乎是一个数学技巧。  
但更有用的理解是：

```txt
i 是一个 90 度旋转操作。
```

乘一次 `i`，从实轴转到虚轴。  
再乘一次 `i`，又转 90 度，总共转 180 度，变成负实轴。  
所以：

```txt
i^2 = -1
```

复数不是“虚构的数”，而是描述旋转和振荡的自然语言。

---

### 1.1 数学概念历史出现缘由

复数最初来自解方程，后来成为几何、物理、信号处理和现代 AI 中描述旋转与频率的核心工具。

| 数学概念 | 出现缘由 |
|---|---|
| 虚数单位 `i` | 为了解决平方等于负数的问题 |
| 复数 `a+bi` | 为了把实数轴扩展为二维平面 |
| 复平面 | 为了把复数看成二维点或向量 |
| 共轭 | 为了反射虚部，并用于除法和模长计算 |
| 模长 | 为了度量复数离原点的距离 |
| 相位 | 为了度量复数相对实轴的角度 |
| 极坐标形式 | 为了用“半径 + 角度”表示复数 |
| Euler 公式 | 为了连接指数函数、三角函数和旋转 |
| 复指数 | 为了优雅表示旋转和振荡 |
| phasor | 为了用旋转箭头表示正弦波 |
| 单位根 | 为了表示单位圆上等间隔的频率基 |
| DFT | 为了把信号分解成不同频率的复指数成分 |
| IDFT | 为了从频域重构原始信号 |
| 正弦位置编码 | 为了用不同频率的 sin/cos 编码 token 位置 |
| RoPE | 为了用复数旋转编码相对位置信息 |

可以这样理解：

```txt
现实问题：如何表示二维旋转？
数学工具：复数乘法

现实问题：如何表示振荡和波？
数学工具：复指数

现实问题：如何把信号拆成频率？
数学工具：DFT

现实问题：Transformer 如何编码位置？
工程实现：sin/cos 位置编码和 RoPE

现实问题：为什么 RoPE 是旋转位置编码？
数学本质：复数乘法 = 二维旋转
```

---

### 1.2 原始问题是什么

本课要解决的原始问题是：

```txt
什么是复数？
为什么 i^2 = -1 可以理解为旋转？
复数如何做加法、乘法、除法和共轭？
复数为什么可以看成二维平面中的点？
极坐标形式和直角坐标形式有什么区别？
Euler 公式为什么重要？
复数乘法为什么等价于二维旋转？
单位根是什么？
DFT 为什么需要复数？
Transformer 正弦位置编码和复指数有什么关系？
RoPE 为什么可以看成复数旋转？
```

换成 AI 语言，就是：

```txt
为什么 Fourier transform 中有 e^(-2πikn/N)？
为什么信号可以分解成不同频率？
为什么 sin/cos 可以编码位置？
RoPE 为什么能表达相对位置？
为什么 query/key 可以通过旋转引入位置信息？
为什么复数在信号处理、频域分析、Transformer 和量子计算中都出现？
```

---

### 1.3 AI 中的现代问题

| 复数知识点 | AI 中的现代问题 |
|---|---|
| 复数平面 | 二维旋转、信号相位 |
| 极坐标形式 | 幅度和相位表示 |
| Euler 公式 | 复指数、频率、旋转 |
| 复数乘法 | 旋转和缩放 |
| Phasor | 正弦波和余弦波表示 |
| 单位根 | DFT 频率基 |
| DFT / FFT | 信号处理、频域特征 |
| 复数共轭 | 频域对称、复数除法 |
| 复数模长 | 频谱幅度 |
| 相位 | phase shift、位置信息 |
| sinusoidal PE | Transformer 原始位置编码 |
| RoPE | 旋转位置编码、相对位置 |
| complex vector space | 量子计算、频域神经网络 |
| 复指数 | 旋转、振荡、波动建模 |

---

### 1.4 本课要解决的核心问题

学完本课，你应该能够回答：

1. 复数 `a+bi` 的实部和虚部分别是什么？
2. 为什么 `i` 可以理解为 90 度旋转？
3. 如何做复数加法、乘法、除法和共轭？
4. 什么是复平面？
5. 什么是模长和相位？
6. 直角坐标形式和极坐标形式如何转换？
7. Euler 公式 `e^(iθ)=cosθ+i sinθ` 为什么重要？
8. 复数乘法为什么等价于二维旋转？
9. 什么是 phasor？
10. 什么是 N-th roots of unity？
11. DFT 如何使用单位根分解信号？
12. IDFT 如何从频域重构信号？
13. Transformer 正弦位置编码为什么是复指数的实部/虚部？
14. RoPE 为什么本质上是复数旋转？
15. 如何从零实现复数类、Euler 公式、DFT 和 IDFT？

---

## 2. 概念定位：这节课学什么

---

### 2.1 所属模块

```txt
所属模块：进阶数学 / 信号处理 / Transformer 位置编码
前置知识：二维向量、矩阵旋转、三角函数、指数函数、基础微积分
后续连接：DFT、FFT、频域分析、RoPE、Transformer、信号处理、量子计算
```

前面的线性代数课程已经解决了：

```txt
二维向量如何旋转？
矩阵如何表示线性变换？
```

本课进一步解决：

```txt
为什么复数是表达旋转、振荡和频率的最自然工具？
```

---

### 2.2 本课学习目标

完成本课后，你应该能够：

- 解释复数的实部、虚部和复平面
- 从零实现复数加法、乘法、除法和共轭
- 计算复数模长和相位
- 在 rectangular form 和 polar form 之间转换
- 使用 Euler 公式表示旋转
- 证明复数乘法等价于二维旋转矩阵
- 理解 phasor 和旋转信号
- 计算 roots of unity
- 从零实现 DFT 和 IDFT
- 解释 sinusoidal positional encoding 和复指数的关系
- 解释 RoPE 如何通过旋转编码位置

---

### 2.3 本课在整体路线中的位置

```txt
线性代数
    ↓
二维旋转
    ↓
复数与 Euler 公式
    ↓
单位根 / DFT / 频域分析
    ↓
Transformer 位置编码 / RoPE
    ↓
信号处理与生成模型中的频率方法
```

本课的核心能力是：

```txt
看到 sin、cos、e^(iθ)、DFT、RoPE 时，能把它们统一理解为复平面上的旋转和频率编码。
```

---

## 3. 理解概念：直觉、公式、几何解释、AI 连接

---

### 3.1 什么是复数

**一句话直觉：**  
复数就是把一维数轴扩展成二维平面后的数。

复数形式：

```text
z = a + bi
```

其中：

| 符号 | 含义 |
|---|---|
| `a` | real part，实部 |
| `b` | imaginary part，虚部 |
| `i` | imaginary unit，满足 `i^2=-1` |

例子：

```txt
3 + 2i
-1 + 0i
0 + 4i
```

复数不只是一个数，也可以看成二维点：

```txt
a + bi <-> (a, b)
```

**AI 连接：**

复数会出现在：

```txt
Fourier transform
signal processing
frequency domain
RoPE
sinusoidal positional encoding
quantum computing
complex-valued neural networks
```

---

### 3.2 复平面：复数是点，也是向量

**一句话直觉：**  
复平面的横轴是实部，纵轴是虚部，每个复数都是平面上的一个点。

例子：

| 复数 | 复平面坐标 |
|---|---|
| `3 + 2i` | `(3, 2)` |
| `-1 + 0i` | `(-1, 0)` |
| `0 + 4i` | `(0, 4)` |

复数既可以看成：

```txt
点：位置
向量：从原点指向该点的箭头
```

这种双重解释让复数天然适合几何旋转。

---

### 3.3 复数加法

**一句话直觉：**  
复数加法就是二维向量加法：实部加实部，虚部加虚部。

公式：

```text
(a + bi) + (c + di) = (a + c) + (b + d)i
```

例子：

```text
(3 + 2i) + (1 + 4i)
= 4 + 6i
```

**AI 连接：**

复数加法对应：

```txt
频域成分叠加
phasor 向量叠加
信号叠加
```

---

### 3.4 复数乘法

**一句话直觉：**  
复数乘法不是普通逐维相乘，而是“旋转 + 缩放”。

公式：

```text
(a + bi)(c + di)
= (ac - bd) + (ad + bc)i
```

推导：

```text
(a + bi)(c + di)
= ac + adi + bci + bd i^2
= ac + adi + bci - bd
= (ac - bd) + (ad + bc)i
```

例子：

```text
(3 + 2i)(1 + 4i)
= -5 + 14i
```

**AI 连接：**

复数乘法是理解 RoPE 的关键：  
RoPE 本质上把向量中相邻两个维度组成二维平面，然后按位置角度旋转。

---

### 3.5 共轭与除法

**一句话直觉：**  
共轭把复数沿实轴镜像反射；除法通过乘以分母共轭把分母变成实数。

共轭：

```text
conj(a + bi) = a - bi
```

重要性质：

```text
(a + bi)(a - bi) = a^2 + b^2
```

这是实数。

复数除法：

```text
(a + bi) / (c + di)
= (a + bi)(c - di) / (c^2 + d^2)
```

**AI 连接：**

共轭常见于：

```txt
傅里叶变换的对称性
频谱分析
复内积
复数除法
信号相关性计算
```

---

### 3.6 模长与相位

**一句话直觉：**  
模长是复数离原点多远，相位是它和正实轴的夹角。

对于：

```text
z = a + bi
```

模长：

```text
|z| = sqrt(a^2 + b^2)
```

相位：

```text
theta = atan2(b, a)
```

其中：

| 量 | 含义 |
|---|---|
| `|z|` | magnitude / modulus |
| `theta` | phase / argument |

**AI 连接：**

| 概念 | AI / 信号处理含义 |
|---|---|
| magnitude | 频率成分强度 |
| phase | 频率成分相位偏移 |
| phase shift | 位置、延迟、旋转 |
| amplitude | 信号幅度 |

---

### 3.7 极坐标形式

**一句话直觉：**  
复数既可以用实部虚部表示，也可以用半径和角度表示。

直角坐标形式：

```text
z = a + bi
```

极坐标形式：

```text
z = r (cos(theta) + i sin(theta))
```

其中：

```text
r = sqrt(a^2 + b^2)
theta = atan2(b, a)
```

使用 Euler 公式后：

```text
z = r e^(i theta)
```

比较：

| 形式 | 适合操作 |
|---|---|
| `a + bi` | 加法、减法 |
| `r e^(iθ)` | 乘法、旋转、幂运算 |

---

### 3.8 Euler 公式

**一句话直觉：**  
Euler 公式把复指数和三角函数连接起来，说明复指数就是单位圆上的旋转。

公式：

```text
e^(iθ) = cos(θ) + i sin(θ)
```

当：

```text
θ = π
```

有：

```text
e^(iπ) = cos(π) + i sin(π) = -1
```

所以：

```text
e^(iπ) + 1 = 0
```

这把五个基本常数联系在一起：

```txt
e, i, π, 1, 0
```

**AI 连接：**

Euler 公式是理解：

```txt
DFT
FFT
complex exponential
phasor
frequency analysis
RoPE
sinusoidal positional encoding
```

的核心。

---

### 3.9 复数乘法 = 二维旋转

**一句话直觉：**  
乘以 `e^(iθ)` 就是在复平面中旋转 θ 角度。

设：

```text
z = x + yi
```

旋转因子：

```text
e^(iθ) = cosθ + i sinθ
```

相乘：

```text
z' = z e^(iθ)
```

展开：

```text
(x + yi)(cosθ + i sinθ)
= (x cosθ - y sinθ) + (x sinθ + y cosθ)i
```

对应二维旋转矩阵：

```text
[cosθ  -sinθ] [x]
[sinθ   cosθ] [y]
```

结果相同。

**AI 连接：RoPE**

RoPE 将 query/key 的相邻维度组成二维对，然后按位置角度旋转。  
这可以看成对复数乘以：

```text
e^(i m θ)
```

其中 `m` 是位置。

---

### 3.10 Phasor：旋转的复数就是正弦波

**一句话直觉：**  
复指数 `e^(iωt)` 是单位圆上旋转的点，它的实部和虚部分别是 cos 和 sin 波。

公式：

```text
e^(iωt) = cos(ωt) + i sin(ωt)
```

实部：

```text
cos(ωt)
```

虚部：

```text
sin(ωt)
```

这说明：

```txt
正弦波可以理解为旋转复数在某个轴上的投影。
```

**AI 连接：**

| 概念 | 应用 |
|---|---|
| amplitude | 信号强度 |
| frequency | 振荡速度 |
| phase | 时间偏移 |
| phasor | 信号处理中的旋转表示 |
| complex exponential | 频域分析基础 |

---

### 3.11 单位根 Roots of Unity

**一句话直觉：**  
N 次单位根是单位圆上均匀分布的 N 个复数点。

公式：

```text
ω_k = e^(2π i k / N), k = 0, 1, ..., N-1
```

对于 `N=4`：

```text
1, i, -1, -i
```

也就是单位圆上的四个方向。

性质：

```txt
每个单位根模长都是 1。
所有 N 个单位根求和为 0。
乘以 primitive root 会移动到下一个根。
```

**AI 连接：**

单位根是 DFT 的基础。  
DFT 用这些等间隔频率来分解离散信号。

---

### 3.12 DFT：用复指数分解信号

**一句话直觉：**  
DFT 把一个离散信号分解成不同频率的复数正弦波。

给定信号：

```text
x[0], x[1], ..., x[N-1]
```

DFT：

```text
X[k] = sum_{n=0}^{N-1} x[n] e^(-2π i k n / N)
```

其中：

| 符号 | 含义 |
|---|---|
| `n` | 时间索引 |
| `k` | 频率索引 |
| `X[k]` | 第 k 个频率成分的复数系数 |
| `|X[k]|` | 第 k 个频率的幅度 |
| `arg(X[k])` | 第 k 个频率的相位 |

直觉：

```txt
每个 X[k] 衡量原信号和第 k 个复指数频率基的相关程度。
```

---

### 3.13 IDFT：从频域重构信号

**一句话直觉：**  
IDFT 把所有频率成分加回去，重构原始信号。

公式：

```text
x[n] = (1/N) sum_{k=0}^{N-1} X[k] e^(2π i k n / N)
```

区别：

```txt
DFT 指数符号是负号。
IDFT 指数符号是正号，并除以 N。
```

这说明：

```txt
频域表示没有丢信息。
DFT 后再 IDFT 可以恢复原信号。
```

---

### 3.14 Transformer 正弦位置编码与复数

**一句话直觉：**  
Transformer 原始正弦位置编码中的 sin/cos 对，可以看成不同频率复指数的实部和虚部。

原始 Transformer 位置编码：

```text
PE(pos, 2i)   = sin(pos / 10000^(2i/d))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d))
```

每一对：

```text
sin(freq * pos), cos(freq * pos)
```

可以看作：

```text
e^(i * freq * pos)
```

的虚部和实部。

不同频率提供不同分辨率：

| 频率 | 作用 |
|---|---|
| 低频 | 变化慢，编码粗位置 |
| 高频 | 变化快，编码细位置 |
| 多频组合 | 给每个位置唯一的频率指纹 |

---

### 3.15 RoPE：Rotary Position Embedding

**一句话直觉：**  
RoPE 用旋转方式把位置信息注入 query 和 key，让相对位置体现在旋转角度差中。

普通位置编码通常是加法：

```text
x_pos = token_embedding + position_embedding
```

RoPE 是旋转：

```text
q_pos = rotate(q, position)
k_pos = rotate(k, position)
```

在复数形式中：

```text
q_m = q * e^(i m θ)
k_n = k * e^(i n θ)
```

其中：

```txt
m, n 是 token 位置。
θ 是频率。
```

当 attention 计算 `q_m` 和 `k_n` 的关系时，会自然出现：

```text
e^(i (m-n) θ)
```

这意味着：

```txt
attention 可以感知相对位置 m-n。
```

**AI 连接：**

RoPE 广泛用于现代 LLM。  
它本质上是在 hidden dimension 中按二维对子进行位置相关旋转。

---

### 3.16 为什么 `i` 不是“虚幻”的

**一句话直觉：**  
`i` 与其说是“虚数”，不如说是“90 度旋转算子”。

乘以 `i`：

```text
1 -> i
```

再乘以 `i`：

```text
i -> i^2 = -1
```

这就是两次 90 度旋转，合起来 180 度，指向负实轴。

所以：

```txt
i^2 = -1
```

不是神秘的代数技巧，而是旋转几何。

**AI 连接：**

一切涉及旋转、周期、频率、振荡、相位的场景，复数都会自然出现。

---

### 3.17 本课核心概念总表

| 概念 | 一句话直觉 | AI 中的对应位置 |
|---|---|---|
| 复数 | 二维数 | 频域、旋转 |
| 实部 / 虚部 | 复平面坐标 | sin/cos 成对表示 |
| `i` | 90 度旋转 | 复数乘法 |
| 共轭 | 实轴镜像 | 频域对称 |
| 模长 | 离原点距离 | 频谱幅度 |
| 相位 | 与实轴夹角 | phase shift |
| 极坐标形式 | 半径 + 角度 | 旋转、乘法 |
| Euler 公式 | 复指数 = cos + i sin | DFT、RoPE |
| 复数乘法 | 旋转 + 缩放 | RoPE |
| Phasor | 旋转箭头 | 正弦信号 |
| 单位根 | 单位圆均匀频率点 | DFT |
| DFT | 信号分解成频率 | 频域分析 |
| IDFT | 频域重构信号 | signal reconstruction |
| Sinusoidal PE | 多频 sin/cos 编码位置 | Transformer |
| RoPE | 旋转编码相对位置 | LLM attention |

---

## 4. 手写实现：复数、Euler 公式、DFT 和 IDFT

这一节先不用 Python 内置 complex。  
目标是亲手实现复数运算，理解复数如何表示旋转和频率。

---

### 4.1 Complex 类

```python
import math


class Complex:
    def __init__(self, real, imag=0.0):
        self.real = real
        self.imag = imag

    def __add__(self, other):
        return Complex(
            self.real + other.real,
            self.imag + other.imag,
        )

    def __sub__(self, other):
        return Complex(
            self.real - other.real,
            self.imag - other.imag,
        )

    def __mul__(self, other):
        real = self.real * other.real - self.imag * other.imag
        imag = self.real * other.imag + self.imag * other.real

        return Complex(real, imag)

    def __truediv__(self, other):
        denom = other.real ** 2 + other.imag ** 2

        real = (
            self.real * other.real + self.imag * other.imag
        ) / denom

        imag = (
            self.imag * other.real - self.real * other.imag
        ) / denom

        return Complex(real, imag)

    def magnitude(self):
        return math.sqrt(self.real ** 2 + self.imag ** 2)

    def phase(self):
        return math.atan2(self.imag, self.real)

    def conjugate(self):
        return Complex(self.real, -self.imag)

    def __repr__(self):
        sign = "+" if self.imag >= 0 else "-"
        return f"{self.real:.6f} {sign} {abs(self.imag):.6f}i"
```

---

### 4.2 极坐标转换和 Euler 公式

```python
def to_polar(z):
    return z.magnitude(), z.phase()


def from_polar(r, theta):
    return Complex(
        r * math.cos(theta),
        r * math.sin(theta),
    )


def euler(theta):
    return Complex(
        math.cos(theta),
        math.sin(theta),
    )
```

验证：

```python
print(euler(0))
print(euler(math.pi / 2))
print(euler(math.pi))
print(euler(2 * math.pi))
print(euler(math.pi).magnitude())
```

预期：

```txt
euler(0) ≈ 1 + 0i
euler(pi/2) ≈ 0 + 1i
euler(pi) ≈ -1 + 0i
euler(2pi) ≈ 1 + 0i
模长始终为 1
```

---

### 4.3 用复数乘法做旋转

```python
def rotate_complex(point, theta):
    return point * euler(theta)


point = Complex(3, 4)

rotated = rotate_complex(point, math.pi / 4)

print("original:", point)
print("rotated:", rotated)
print("original magnitude:", point.magnitude())
print("rotated magnitude:", rotated.magnitude())
```

解释：

```txt
旋转不改变模长，只改变相位。
```

---

### 4.4 与旋转矩阵对照

```python
def rotate_matrix(x, y, theta):
    cos_t = math.cos(theta)
    sin_t = math.sin(theta)

    return (
        x * cos_t - y * sin_t,
        x * sin_t + y * cos_t,
    )


z = Complex(3, 4)
theta = math.pi / 3

z_rot = rotate_complex(z, theta)
x_rot, y_rot = rotate_matrix(z.real, z.imag, theta)

print(z_rot)
print(x_rot, y_rot)
```

二者应在数值误差范围内一致。

---

### 4.5 Roots of Unity

```python
def roots_of_unity(N):
    return [
        euler(2 * math.pi * k / N)
        for k in range(N)
    ]


roots = roots_of_unity(8)

for r in roots:
    print(r, "magnitude:", r.magnitude())
```

验证求和为 0：

```python
total = Complex(0, 0)

for r in roots:
    total = total + r

print(total)
```

---

### 4.6 DFT from scratch

```python
def dft(signal):
    N = len(signal)
    result = []

    for k in range(N):
        total = Complex(0, 0)

        for n in range(N):
            angle = -2 * math.pi * k * n / N
            total = total + Complex(signal[n], 0) * euler(angle)

        result.append(total)

    return result
```

解释：

```txt
每个 X[k] 是原信号和第 k 个复指数频率基的相关性。
```

---

### 4.7 IDFT from scratch

```python
def idft(spectrum):
    N = len(spectrum)
    result = []

    for n in range(N):
        total = Complex(0, 0)

        for k in range(N):
            angle = 2 * math.pi * k * n / N
            total = total + spectrum[k] * euler(angle)

        result.append(
            Complex(total.real / N, total.imag / N)
        )

    return result
```

测试：

```python
signal = [1, 2, 3, 4]

spectrum = dft(signal)
reconstructed = idft(spectrum)

print(spectrum)
print(reconstructed)
```

---

### 4.8 已知频率信号的 DFT

```python
def make_signal(N):
    signal = []

    for n in range(N):
        t = n / N

        value = (
            math.sin(2 * math.pi * 3 * t)
            + 0.5 * math.sin(2 * math.pi * 7 * t)
        )

        signal.append(value)

    return signal


signal = make_signal(32)

spectrum = dft(signal)

magnitudes = [
    z.magnitude()
    for z in spectrum
]

for k, mag in enumerate(magnitudes):
    print(k, mag)
```

预期：

```txt
频率 3 和 7 附近出现峰值。
因为真实信号由这两个频率组成。
```

---

## 5. 生产使用：Python complex 与 NumPy FFT

---

### 5.1 Python 内置 complex

Python 用 `j` 表示虚数单位：

```python
z = 3 + 2j
w = 1 + 4j

print(z + w)
print(z * w)
print(z / w)
print(abs(z))
print(z.conjugate())
```

使用 `cmath`：

```python
import cmath

print(cmath.phase(z))
print(cmath.exp(1j * cmath.pi))
```

---

### 5.2 NumPy complex array

```python
import numpy as np

z = np.array([
    1 + 2j,
    3 + 4j,
    5 + 6j,
])

print(np.abs(z))
print(np.angle(z))
print(np.conj(z))
print(np.real(z))
print(np.imag(z))
```

---

### 5.3 NumPy FFT

```python
import numpy as np

N = 128
t = np.linspace(0, 1, N, endpoint=False)

signal = np.sin(2 * np.pi * 5 * t)

spectrum = np.fft.fft(signal)
freqs = np.fft.fftfreq(N, d=1 / N)

magnitudes = np.abs(spectrum)

peak_idx = np.argmax(magnitudes[:N // 2])

print("peak frequency:", freqs[peak_idx])
```

---

### 5.4 RoPE 的最小二维旋转实现

```python
import numpy as np


def rope_rotate_pair(x_even, x_odd, theta):
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)

    new_even = x_even * cos_t - x_odd * sin_t
    new_odd = x_even * sin_t + x_odd * cos_t

    return new_even, new_odd


x_even = np.array([1.0, 2.0, 3.0])
x_odd = np.array([0.5, 1.0, 1.5])

theta = 0.25

rot_even, rot_odd = rope_rotate_pair(x_even, x_odd, theta)

print(rot_even)
print(rot_odd)
```

解释：

```txt
RoPE 会对 hidden dimension 中的相邻 pair 做类似旋转。
```

---

### 5.5 手写实现 vs 生产库

| 对比项 | 手写实现 | Python / NumPy |
|---|---|---|
| 目的 | 理解复数几何 | 工程实践 |
| 复数类 | 自己实现 real/imag | Python complex |
| 模长 | `sqrt(a^2+b^2)` | `abs(z)` / `np.abs` |
| 相位 | `atan2(b,a)` | `cmath.phase` / `np.angle` |
| 共轭 | flip imag sign | `z.conjugate()` / `np.conj` |
| Euler | `cos + i sin` | `cmath.exp(1j*theta)` |
| DFT | O(N²) 手写 | `np.fft.fft` |
| IDFT | O(N²) 手写 | `np.fft.ifft` |
| RoPE | 手写 2D 旋转 | 框架中向量化实现 |

---

## 6. 交付沉淀：产出物、连接关系、练习、术语表

---

### 6.1 本课产出

```txt
code/complex_numbers.py
code/use_numpy_fft.py
code/rope_rotation.py
outputs/skill-complex-arithmetic.md
outputs/exercises-solutions.md
```

| 文件 | 作用 |
|---|---|
| `code/complex_numbers.py` | 从零实现复数类、Euler 公式、DFT、IDFT、单位根 |
| `code/use_numpy_fft.py` | NumPy complex 和 FFT 示例 |
| `code/rope_rotation.py` | RoPE 二维旋转最小实现 |
| `outputs/skill-complex-arithmetic.md` | 复数运算和 AI 应用 skill |
| `outputs/exercises-solutions.md` | 练习题与参考答案 |

---

### 6.2 知识连接关系

| 本课知识点 | 后续连接 |
|---|---|
| 复平面 | 二维旋转 |
| 极坐标形式 | 模长、相位、旋转 |
| Euler 公式 | 复指数、信号处理 |
| Phasor | 正弦波表示 |
| 单位根 | DFT |
| DFT / IDFT | FFT、频域分析 |
| 复数乘法 | RoPE |
| sin/cos pairs | sinusoidal positional encoding |
| phase | 相对位置、频率偏移 |
| complex vectors | quantum computing、complex NN |

---

### 6.3 AI 应用连接

| 复数知识点 | AI 应用 |
|---|---|
| `e^(iθ)` | 旋转、频率 |
| DFT | 信号处理、频域特征 |
| FFT | 高效频域计算 |
| Magnitude spectrum | 频率强度分析 |
| Phase | 位移、相位、位置关系 |
| Sinusoidal PE | Transformer 位置编码 |
| RoPE | LLM 相对位置编码 |
| Roots of unity | DFT 基函数 |
| Complex multiplication | 2D rotation |
| Phasor | 振荡信号建模 |

---

### 6.4 练习

1. **手算复数运算。**  
   计算：

   ```text
   (2 + 3i)(4 - i)
   ```

   再计算：

   ```text
   (5 + 2i) / (1 - 3i)
   ```

   用代码验证，并在复平面上解释结果。

2. **旋转序列。**  
   从点 `(1,0)` 开始，连续乘以：

   ```text
   e^(iπ/6)
   ```

   共 12 次。验证最后回到 `(1,0)`，并打印每一步坐标，观察正 12 边形。

3. **已知信号的 DFT。**  
   创建信号：

   ```text
   sin(2π*3*t) + 0.5*sin(2π*7*t)
   ```

   采样 32 点。运行手写 DFT，验证频谱在 3 和 7 处有峰值，且频率 7 的峰值约为频率 3 的一半。

4. **单位根可视化。**  
   计算 8 次单位根。验证它们求和为 0。验证每个根乘以 primitive root 会得到下一个根。

5. **复数旋转 vs 矩阵旋转。**  
   对 10 个随机角度和 10 个随机点，比较复数乘法旋转和 2x2 旋转矩阵结果，打印最大数值误差。

6. **RoPE 最小实现。**  
   给定一个偶数维向量，把相邻维度组成 pair，对每个 pair 按不同频率旋转。解释它和复数乘法的关系。

---

### 6.5 关键术语

| 术语 | 常见说法 | 真正含义 |
|---|---|---|
| Complex number | 复数 | `a+bi`，由实部和虚部组成 |
| Imaginary unit | 虚数单位 | `i^2=-1`，也可理解为 90 度旋转算子 |
| Complex plane | 复平面 | 横轴为实部、纵轴为虚部的二维平面 |
| Magnitude / Modulus | 模长 | `sqrt(a^2+b^2)`，离原点距离 |
| Phase / Argument | 相位 | `atan2(b,a)`，与正实轴夹角 |
| Conjugate | 共轭 | `a+bi` 的共轭是 `a-bi` |
| Polar form | 极坐标形式 | `r e^(iθ)` |
| Euler's formula | 欧拉公式 | `e^(iθ)=cosθ+i sinθ` |
| Phasor | 相量 | 旋转的复数，用于表示正弦信号 |
| Roots of unity | 单位根 | 单位圆上等间隔的 N 个复数 |
| DFT | 离散傅里叶变换 | 用单位根把信号分解成频率成分 |
| IDFT | 逆离散傅里叶变换 | 从频域重构原信号 |
| RoPE | 旋转位置编码 | 用复数旋转思想编码 Transformer 相对位置 |
| Sinusoidal PE | 正弦位置编码 | 用多频 sin/cos 编码位置 |

---

### 6.6 自检问题

学完本课后，你应该能回答：

- 复数为什么可以看成二维平面上的点？
- `i^2=-1` 如何用旋转解释？
- 复数乘法为什么不是逐元素乘法？
- 共轭有什么用？
- 模长和相位分别表示什么？
- 极坐标形式为什么适合乘法？
- Euler 公式为什么是复数和信号处理的核心？
- 复数乘法如何等价于二维旋转矩阵？
- phasor 如何表示正弦信号？
- 单位根为什么均匀分布在单位圆上？
- DFT 中的 `e^(-2πikn/N)` 是什么？
- DFT 的幅度和相位分别代表什么？
- IDFT 如何重构原始信号？
- Transformer 正弦位置编码和复指数有什么关系？
- RoPE 为什么可以理解为复数旋转？
- 为什么复数在 Fourier、RoPE、信号处理和量子计算中都自然出现？

---

## 附录：本课最小代码文件建议

如果要拆成代码文件，可以这样组织：

```txt
code/
├── complex_numbers.py
├── use_numpy_fft.py
└── rope_rotation.py
```

### `complex_numbers.py`

包含：

```txt
Complex
to_polar
from_polar
euler
rotate_complex
rotate_matrix
roots_of_unity
dft
idft
```

### `use_numpy_fft.py`

包含：

```txt
Python complex
cmath.phase
cmath.exp
np.abs
np.angle
np.conj
np.fft.fft
np.fft.ifft
np.fft.fftfreq
```

### `rope_rotation.py`

包含：

```txt
rope_rotate_pair
pairwise hidden dimension rotation
sin/cos frequency construction
minimal RoPE demo
```
