---
title: 傅里叶变换
description: 面向 AI 算法工程师的傅里叶变换课程：从 DFT、IDFT、FFT、频谱分析、卷积定理，到 Transformer 位置编码、CNN 卷积和音频 spectrogram。
---

# 傅里叶变换

> 每个信号都可以看成正弦波的叠加。傅里叶变换告诉你：它由哪些正弦波组成。

**课程类型：** 实现 / 应用  
**所属模块：** 进阶数学 / 信号处理 / 频域分析  
**前置知识：** Phase 1 Lesson 01-04：线性代数、矩阵运算、微积分；Lesson 19：复数  
**预计时间：** 约 90 分钟  
**使用语言：** Python / NumPy / SciPy  

---

## 1. 提出问题：为什么 AI 算法工程师必须理解傅里叶变换

一段音频录音，是一串随时间变化的气压值。  
一条股价曲线，是一串随日期变化的价格。  
一张图像，是一个二维空间上的像素强度网格。

这些数据都可以在“时间域”或“空间域”中观察：

```txt
音频：时间 -> 振幅
股价：日期 -> 价格
图像：空间位置 -> 像素强度
```

但很多模式在时间域中并不明显：

```txt
这段音频是单音还是和弦？
这个时间序列有没有周周期？
这张图像有没有重复纹理？
一个信号的高频噪声在哪里？
```

这些问题关心的是：

```txt
频率成分。
```

傅里叶变换的作用就是：

```txt
把信号从时间域 / 空间域转换到频率域。
```

它告诉你：

```txt
信号里有哪些频率？
每个频率有多强？
每个频率从什么相位开始？
```

在 AI 中，频域思维无处不在：

```txt
CNN 的卷积和频域乘法
Transformer 的 sinusoidal positional encoding
FNet 用 FFT 替代 attention 的 token mixing
语音识别中的 spectrogram / mel-spectrogram
时间序列中的周期检测
图像纹理和频率分析
反走样 anti-aliasing
```

如果你理解傅里叶变换，就能把很多看似不同的 AI 模块统一到一个问题上：

```txt
数据中有哪些频率模式？
```

---

### 1.1 数学概念历史出现缘由

傅里叶变换最初来自热传导和波动方程。  
核心思想是：复杂函数可以分解成简单正弦波的叠加。

| 数学 / 工程概念 | 出现缘由 |
|---|---|
| 傅里叶级数 | 为了把周期函数表示成正弦和余弦的叠加 |
| 复指数 | 为了用 `e^(iθ)` 统一表示 sin 和 cos |
| DFT | 为了处理离散采样信号的频率分解 |
| IDFT | 为了从频率系数恢复原始离散信号 |
| FFT | 为了把 DFT 从 `O(N^2)` 加速到 `O(N log N)` |
| 频率系数 | 为了表示每个频率的幅度和相位 |
| 功率谱 | 为了表示每个频率上的能量 |
| 相位谱 | 为了表示每个频率的起始偏移 |
| Nyquist 频率 | 为了确定采样后能表示的最高频率 |
| Aliasing | 为了解释采样率不足时高频伪装成低频 |
| Windowing | 为了减少非周期截断导致的谱泄漏 |
| STFT | 为了同时知道频率在什么时候出现 |
| Spectrogram | 为了把时间-频率能量表示成二维图 |
| 卷积定理 | 为了把时域卷积转换成频域乘法 |
| Zero-padding | 为了插值频谱显示和实现线性卷积 |
| Transformer 位置编码 | 为了用多频 sin/cos 编码位置 |
| CNN 卷积 | 为了理解空间滤波和频域乘法的等价关系 |

可以这样理解：

```txt
现实问题：复杂信号由哪些简单波组成？
数学工具：傅里叶变换

现实问题：如何分析离散采样信号？
数学工具：DFT

现实问题：DFT 太慢怎么办？
算法工具：FFT

现实问题：卷积计算太慢怎么办？
数学工具：卷积定理

现实问题：音频频率随时间变化怎么办？
工程工具：STFT / spectrogram

现实问题：Transformer 为什么用 sin/cos 编码位置？
数学基础：多频 Fourier feature
```

---

### 1.2 原始问题是什么

本课要解决的原始问题是：

```txt
什么是时间域和频率域？
DFT 如何把离散信号分解成频率成分？
每个 Fourier coefficient 的幅度和相位分别表示什么？
X[0]、正频率、负频率、Nyquist frequency 分别是什么？
为什么实信号的频谱有共轭对称性？
IDFT 如何重构原始信号？
FFT 为什么比 DFT 快？
Cooley-Tukey 如何通过偶数/奇数分治计算 DFT？
功率谱如何表示信号能量？
频率分辨率由什么决定？
卷积定理为什么重要？
Windowing 如何减少 spectral leakage？
STFT 和 spectrogram 为什么适合音频模型？
Aliasing 为什么无法事后修复？
Zero-padding 为什么不真正提高频率分辨率？
傅里叶思想如何连接 CNN、Transformer 和音频模型？
```

换成 AI 语言，就是：

```txt
为什么语音模型用 spectrogram？
为什么 CNN 卷积可以在频域理解？
为什么大 kernel convolution 可以用 FFT 加速？
为什么 Transformer 位置编码是 sin/cos 多频信号？
FNet 为什么能用 Fourier transform 做 token mixing？
为什么 downsampling 前需要低通滤波？
为什么 zero-padding 的 FFT 图更平滑但不增加真实信息？
```

---

### 1.3 AI 中的现代问题

| 傅里叶知识点 | AI 中的现代问题 |
|---|---|
| DFT | 离散信号频率分解 |
| FFT | 快速频域计算 |
| IDFT | 从频域重构信号 |
| DC component | 信号均值 / 低频偏置 |
| Frequency bin | 离散频率槽 |
| Amplitude spectrum | 频率强度 |
| Phase spectrum | 相位和时间偏移 |
| Power spectrum | 能量分布 |
| Nyquist frequency | 采样率上限 |
| Aliasing | 下采样、抗混叠池化 |
| Windowing | 减少频谱泄漏 |
| STFT | 音频时频表示 |
| Spectrogram | 语音模型输入 |
| Convolution theorem | CNN、FFT convolution |
| Parseval theorem | 时域频域能量守恒 |
| Zero-padding | 频谱插值、线性卷积 |
| Sinusoidal PE | Transformer 位置编码 |
| Fourier features | 坐标网络、NeRF、位置表示 |
| FNet | 用 FFT 替代 attention mixing |

---

### 1.4 本课要解决的核心问题

学完本课，你应该能够回答：

1. DFT 的公式是什么？
2. DFT 系数为什么是复数？
3. `X[0]` 为什么是 DC component？
4. 正频率、负频率和 Nyquist frequency 分别是什么？
5. IDFT 为什么可以完美重构原信号？
6. DFT 为什么是一个 change of basis？
7. FFT 为什么能把复杂度从 `O(N^2)` 降到 `O(N log N)`？
8. Cooley-Tukey 的 even/odd split 如何工作？
9. 幅度谱、相位谱和功率谱分别表示什么？
10. 频率分辨率由采样率和样本数如何决定？
11. 卷积定理为什么能加速 convolution？
12. Windowing 为什么能减少 spectral leakage？
13. STFT 和 spectrogram 如何用于音频模型？
14. Aliasing 为什么来自采样率不足？
15. Zero-padding 为什么不增加真实频率分辨率？
16. Transformer sinusoidal PE 和 Fourier frequency decomposition 有什么关系？
17. CNN 卷积为什么可以从频域角度理解？

---

## 2. 概念定位：这节课学什么

---

### 2.1 所属模块

```txt
所属模块：进阶数学 / 信号处理 / 频域分析
前置知识：复数、Euler 公式、单位根、矩阵乘法、卷积直觉
后续连接：FFT、CNN、spectrogram、音频模型、Transformer 位置编码、Fourier features、FNet
```

前面的复数课程已经解决了：

```txt
复指数 e^(iθ) 为什么表示旋转？
单位根为什么是单位圆上的等间隔频率点？
```

本课进一步解决：

```txt
如何用这些复指数，把一个离散信号分解成不同频率成分？
```

---

### 2.2 本课学习目标

完成本课后，你应该能够：

- 从定义实现 DFT
- 从定义实现 IDFT
- 从零实现 Cooley-Tukey FFT
- 对比 DFT 和 FFT 的复杂度
- 解释 Fourier coefficient 的幅度、相位和功率
- 计算 power spectrum
- 理解 frequency bin、frequency resolution 和 Nyquist frequency
- 使用卷积定理通过 FFT 做 convolution
- 区分 circular convolution 和 linear convolution
- 理解 windowing 和 spectral leakage
- 理解 STFT 和 spectrogram
- 解释 aliasing 与 sampling rate 的关系
- 说明 zero-padding 的真实作用
- 连接 Fourier decomposition、Transformer positional encoding 和 CNN convolution

---

### 2.3 本课在整体路线中的位置

```txt
复数与 Euler 公式
    ↓
单位根
    ↓
DFT / IDFT
    ↓
FFT
    ↓
频谱分析 / 卷积定理 / STFT
    ↓
CNN / 音频模型 / Transformer 位置编码
```

本课的核心能力是：

```txt
看到一个时间序列、图像、音频或位置编码时，能从频率角度理解它的结构。
```

---

## 3. 理解概念：直觉、公式、频域解释、AI 连接

---

### 3.1 时间域 vs 频率域

**一句话直觉：**  
时间域看信号随时间怎么变，频率域看信号由哪些周期波组成。

时间域：

```txt
x[n] 表示第 n 个采样点的数值。
```

频率域：

```txt
X[k] 表示第 k 个频率成分的复数系数。
```

同一个信号可以有两种表示：

| 表示 | 关注点 |
|---|---|
| 时间域 / 空间域 | 每个位置的数值 |
| 频率域 | 周期模式、频率强度、相位 |

**AI 连接：**

| 数据 | 时间/空间域 | 频率域问题 |
|---|---|---|
| 音频 | 波形振幅 | 音高、音色、噪声 |
| 图像 | 像素强度 | 纹理、边缘、高频噪声 |
| 时间序列 | 每日数值 | 周期、季节性 |
| Transformer PE | token position | 多频位置指纹 |

---

### 3.2 DFT 定义

**一句话直觉：**  
DFT 计算信号和每个离散复指数频率基的相关程度。

给定 N 个样本：

```text
x[0], x[1], ..., x[N-1]
```

DFT 输出 N 个频率系数：

```text
X[0], X[1], ..., X[N-1]
```

公式：

```text
X[k] = sum_{n=0}^{N-1} x[n] * e^(-2π i k n / N)
```

其中：

| 符号 | 含义 |
|---|---|
| `n` | 时间索引 |
| `k` | 频率索引 |
| `N` | 样本总数 |
| `e^(-2πikn/N)` | 第 k 个频率的旋转复指数 |
| `X[k]` | 信号在第 k 个频率上的复数系数 |

直觉：

```txt
如果信号中包含频率 k，和该复指数相乘求和后会累积成大值。
如果不包含频率 k，正负相位会相互抵消，结果接近 0。
```

---

### 3.3 每个 Fourier coefficient 表示什么

**一句话直觉：**  
`X[k]` 的模长表示频率 k 有多强，相位表示这个频率从哪里开始。

对于复数系数：

```text
X[k] = a + bi
```

幅度：

```text
amplitude = |X[k]| = sqrt(a^2 + b^2)
```

相位：

```text
phase = angle(X[k]) = atan2(b, a)
```

功率：

```text
power = |X[k]|^2 = a^2 + b^2
```

**AI 连接：**

| 指标 | 含义 |
|---|---|
| magnitude spectrum | 哪些频率强 |
| phase spectrum | 这些频率的时间/空间偏移 |
| power spectrum | 能量分布 |
| dominant frequency | 主要周期 |
| high-frequency energy | 噪声、边缘、纹理 |

---

### 3.4 DC、正频率、负频率与 Nyquist

**一句话直觉：**  
DFT 输出的不同 index 表示不同频率，其中前半部分是正频率，后半部分是负频率。

#### `X[0]`：DC component

```text
X[0] = sum_n x[n]
```

因为：

```text
e^0 = 1
```

所以 `X[0]` 表示信号的常数成分，和均值成比例。

#### 正频率

对于：

```text
1 <= k <= N/2
```

`X[k]` 表示正频率，频率为：

```text
f_k = k * fs / N
```

#### Nyquist frequency

最高可表示频率：

```text
f_max = fs / 2
```

这是 Nyquist frequency。

#### 负频率

对于：

```text
N/2 < k < N
```

表示负频率。

对实值信号：

```text
X[N-k] = conjugate(X[k])
```

所以负频率是正频率的镜像。

**AI 连接：**

实信号频谱通常只需要看前 `N/2 + 1` 个系数。

---

### 3.5 IDFT：从频率域重构信号

**一句话直觉：**  
IDFT 把所有频率成分按相位加回去，恢复原始信号。

公式：

```text
x[n] = (1/N) * sum_{k=0}^{N-1} X[k] * e^(2π i k n / N)
```

与 DFT 的区别：

| 项 | DFT | IDFT |
|---|---|---|
| 指数符号 | 负号 | 正号 |
| 归一化 | 通常无 | 除以 N |
| 方向 | 时间域 -> 频率域 | 频率域 -> 时间域 |

核心结论：

```txt
DFT 不丢信息。
DFT 和 IDFT 是同一信息的两种坐标表示。
```

---

### 3.6 FFT：让 DFT 从平方复杂度变成准线性复杂度

**一句话直觉：**  
FFT 利用单位根对称性，把 DFT 拆成偶数项和奇数项两个更小的 DFT。

直接 DFT：

```text
O(N^2)
```

因为：

```txt
N 个输出，每个输出要对 N 个输入求和。
```

Cooley-Tukey FFT：

```text
O(N log N)
```

核心分治：

```txt
1. 把信号拆成偶数索引样本和奇数索引样本。
2. 分别计算两个 N/2 点 DFT。
3. 用 twiddle factors 合并。
```

公式：

```text
X[k]       = E[k] + W_N^k O[k]
X[k+N/2]   = E[k] - W_N^k O[k]
```

其中：

```text
W_N^k = e^(-2π i k / N)
```

称为 twiddle factor。

**AI 连接：**

FFT 让以下任务可行：

```txt
大规模音频频谱分析
快速卷积
图像频域滤波
spectrogram
FNet token mixing
```

---

### 3.7 频率分辨率

**一句话直觉：**  
频率分辨率由采样率和样本数量决定，样本越长，频率分得越细。

采样率：

```text
fs
```

样本数：

```text
N
```

第 k 个频率 bin：

```text
f_k = k * fs / N
```

频率分辨率：

```text
delta_f = fs / N
```

最大可表示频率：

```text
f_max = fs / 2
```

想分辨两个很接近的频率，需要更长的观测时间：

```text
T = N / fs
```

频率差 `delta_f` 需要大约：

```text
T >= 1 / delta_f
```

---

### 3.8 Power spectrum 与 Phase spectrum

**一句话直觉：**  
功率谱告诉你能量在哪些频率上，相位谱告诉你这些频率如何平移。

功率谱：

```text
P[k] = |X[k]|^2 = real(X[k])^2 + imag(X[k])^2
```

相位谱：

```text
phi[k] = atan2(imag(X[k]), real(X[k]))
```

常见做法：

```txt
信号分析中常先看 power spectrum。
相位在重构、定位、时移分析中非常重要。
```

**AI 连接：**

| 应用 | 关注 |
|---|---|
| 音频分类 | magnitude / power spectrum |
| 语音识别 | spectrogram / mel-spectrogram |
| 图像纹理 | 高频能量 |
| 图像重构 | magnitude + phase 都重要 |
| 时间序列周期 | dominant frequency |

---

### 3.9 卷积定理

**一句话直觉：**  
时域卷积等价于频域逐点乘法。

公式：

```text
x * h = IFFT(FFT(x) · FFT(h))
```

其中：

```txt
* 是卷积
· 是逐点乘法
```

意义：

| 方法 | 复杂度 |
|---|---|
| 直接卷积 | `O(NM)` |
| FFT 卷积 | `O(N log N)` |

注意：

```txt
DFT 自然计算的是 circular convolution。
如果要 linear convolution，需要 zero-pad 到 N+M-1。
```

**AI 连接：**

| 场景 | 卷积定理意义 |
|---|---|
| CNN | 卷积层可从频域理解 |
| 大 kernel convolution | FFT 可能更快 |
| 图像滤波 | 空间卷积 = 频域乘法 |
| FNet | 用 Fourier mixing 替代 attention |
| 信号处理 | filter design |

---

### 3.10 Circular convolution vs Linear convolution

**一句话直觉：**  
DFT 默认信号是周期的，所以卷积会“绕回去”；普通卷积不绕回。

Circular convolution：

```txt
尾部会 wrap around 到开头。
```

Linear convolution：

```txt
普通滑动卷积，不发生绕回。
```

要用 FFT 实现 linear convolution：

```txt
把两个信号都 zero-pad 到 len(x)+len(h)-1。
再 FFT -> multiply -> IFFT。
```

---

### 3.11 Windowing 与 Spectral Leakage

**一句话直觉：**  
DFT 假设采样片段是周期重复的；如果首尾不连续，会产生假的高频成分，也就是 spectral leakage。

如果信号在窗口边界不连续：

```txt
DFT 会把这个突变解释成很多频率成分。
```

Windowing 通过把信号两端逐渐压低来减少边界突变。

常见窗口：

| Window | 特点 | 用途 |
|---|---|---|
| Rectangular | 不处理，主瓣窄、旁瓣高 | 信号恰好周期对齐 |
| Hann | 通用，泄漏较低 | 一般频谱分析 |
| Hamming | 旁瓣更低 | 音频、语音 |
| Blackman | 旁瓣很低，主瓣更宽 | 需要强旁瓣抑制 |

Hann window：

```text
w[n] = 0.5 * (1 - cos(2πn/(N-1)))
```

Hamming window：

```text
w[n] = 0.54 - 0.46 cos(2πn/(N-1))
```

使用：

```text
X = DFT(x * w)
```

---

### 3.12 STFT 与 Spectrogram

**一句话直觉：**  
单个 FFT 只告诉你整体有哪些频率；STFT 告诉你每个时间窗口有哪些频率。

STFT 流程：

```txt
1. 选择 window size。
2. 选择 hop size。
3. 对每个窗口：
   a. 取出局部片段。
   b. 乘以 window function。
   c. 做 FFT。
   d. 保存 magnitude 或 power spectrum。
```

结果是 spectrogram：

```txt
横轴：时间
纵轴：频率
颜色：能量强度
```

**AI 连接：**

| 模型 / 任务 | 输入表示 |
|---|---|
| Speech recognition | mel-spectrogram |
| Music generation | spectrogram / audio tokens |
| Whisper | mel-spectrogram |
| Audio classification | time-frequency features |
| Sound event detection | spectrogram |

---

### 3.13 Aliasing：采样率不足导致高频伪装成低频

**一句话直觉：**  
如果信号频率超过 Nyquist frequency，采样后会伪装成较低频率。

Nyquist frequency：

```text
fs / 2
```

例子：

```txt
真实信号：90 Hz
采样率：100 Hz
Nyquist frequency：50 Hz
```

90 Hz 超过 50 Hz，会 alias 成：

```txt
10 Hz
```

采样后无法区分：

```txt
90 Hz 信号
10 Hz 信号
```

**AI 连接：**

Aliasing 出现在：

```txt
音频采样
图像下采样
feature map pooling
stride convolution
anti-aliased pooling
```

一旦 aliasing 发生，无法靠后处理恢复原始高频。  
必须在采样或下采样前做低通滤波。

---

### 3.14 Zero-padding 不提高真实频率分辨率

**一句话直觉：**  
zero-padding 让频谱图看起来更平滑，但不创造新的频率信息。

Zero-padding 会：

```txt
插值频率 bins。
让曲线更平滑。
方便 FFT 长度变成 2 的幂。
支持 linear convolution。
```

但不会：

```txt
提高真实频率分辨率。
分辨原本无法分辨的两个频率。
```

真实频率分辨率取决于：

```text
T = N / fs
```

也就是观测时间长度。

---

### 3.15 DFT 的重要性质

| 性质 | 时间域 | 频率域 |
|---|---|---|
| 线性 | `a*x + b*y` | `a*X + b*Y` |
| 时间平移 | `x[n-k]` | `X[f] * e^(-2πifk/N)` |
| 频率平移 | `x[n] * e^(2πif0n/N)` | `X[f-f0]` |
| 卷积 | `x * h` | `X · H` |
| 逐点乘法 | `x · h` | 频域卷积 |
| Parseval | 时域能量 | 频域能量 |
| 实信号共轭对称 | `x[n] real` | `X[k]=conj(X[N-k])` |

Parseval theorem：

```text
sum_n |x[n]|^2 = (1/N) sum_k |X[k]|^2
```

含义：

```txt
信号总能量在时间域和频率域中一致。
```

---

### 3.16 Transformer 位置编码与 Fourier

**一句话直觉：**  
Transformer 的 sinusoidal positional encoding 用多组不同频率的 sin/cos 给每个位置一个频率指纹。

公式：

```text
PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
```

每一对维度对应一个频率：

```txt
低频变化慢，表示粗位置。
高频变化快，表示细位置。
多频组合使每个位置具有独特编码。
```

这和 Fourier feature 的思想一致：

```txt
用多频 sin/cos basis 表示位置。
```

关键性质：

```txt
位置 p+k 的编码可以由位置 p 的编码通过线性变换得到。
因此模型可以学习相对位置关系。
```

---

### 3.17 CNN 卷积与 Fourier

**一句话直觉：**  
CNN 的卷积核可以看成空间滤波器；在频域中，卷积就是频率响应相乘。

卷积层：

```txt
输入图像和 kernel 做滑动卷积。
```

频域视角：

```txt
kernel 对不同频率有不同响应。
低通滤波器保留平滑结构。
高通滤波器强调边缘和纹理。
```

小 kernel 例如 `3x3`：

```txt
直接卷积通常更快。
```

大 kernel 或 global convolution：

```txt
FFT convolution 可能更快。
```

**AI 连接：**

| 架构 / 方法 | Fourier 连接 |
|---|---|
| CNN | 卷积 = 频域乘法 |
| Large-kernel CNN | 可用 FFT 加速 |
| FNet | FFT 做 token mixing |
| Anti-aliased CNN | 下采样前低通 |
| Spectral CNN | 频域滤波 |

---

### 3.18 本课核心概念总表

| 概念 | 一句话直觉 | AI 中的对应位置 |
|---|---|---|
| DFT | 信号分解成频率 | 频域分析 |
| IDFT | 频域重构信号 | signal reconstruction |
| FFT | 快速 DFT | 大规模频域计算 |
| Frequency bin | 离散频率槽 | 频谱 index |
| DC component | 零频率均值 | 信号偏置 |
| Nyquist | 最高可表示频率 | 采样率约束 |
| Magnitude spectrum | 频率强度 | 音频/图像分析 |
| Phase spectrum | 相位偏移 | 重构、时移 |
| Power spectrum | 能量分布 | 频率能量 |
| Convolution theorem | 卷积变乘法 | CNN、FFT convolution |
| Circular convolution | 周期卷积 | DFT 默认 |
| Linear convolution | 普通卷积 | zero-padding |
| Windowing | 减少边界泄漏 | spectral analysis |
| Spectral leakage | 假频率扩散 | 非周期截断 |
| STFT | 局部 FFT | audio spectrogram |
| Aliasing | 高频伪装低频 | 下采样 |
| Zero-padding | 频谱插值 | 不增加真实分辨率 |
| Sinusoidal PE | 多频位置编码 | Transformer |

---

## 4. 手写实现：DFT、IDFT、FFT、频谱和 FFT 卷积

这一节使用前一课的复数知识，从零实现核心算法。

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
        return Complex(
            self.real * other.real - self.imag * other.imag,
            self.real * other.imag + self.imag * other.real,
        )

    def magnitude(self):
        return math.sqrt(self.real ** 2 + self.imag ** 2)

    def phase(self):
        return math.atan2(self.imag, self.real)

    def __repr__(self):
        sign = "+" if self.imag >= 0 else "-"
        return f"{self.real:.6f} {sign} {abs(self.imag):.6f}i"
```

---

### 4.2 DFT from scratch

```python
def dft(x):
    N = len(x)
    result = []

    for k in range(N):
        total = Complex(0, 0)

        for n in range(N):
            angle = -2 * math.pi * k * n / N
            w = Complex(math.cos(angle), math.sin(angle))

            xn = x[n] if isinstance(x[n], Complex) else Complex(x[n])

            total = total + xn * w

        result.append(total)

    return result
```

---

### 4.3 IDFT from scratch

```python
def idft(X):
    N = len(X)
    result = []

    for n in range(N):
        total = Complex(0, 0)

        for k in range(N):
            angle = 2 * math.pi * k * n / N
            w = Complex(math.cos(angle), math.sin(angle))

            total = total + X[k] * w

        result.append(
            Complex(total.real / N, total.imag / N)
        )

    return result
```

验证：

```python
signal = [1, 2, 3, 4]

spectrum = dft(signal)
reconstructed = idft(spectrum)

print(spectrum)
print(reconstructed)
```

---

### 4.4 FFT: Cooley-Tukey

```python
def fft(x):
    N = len(x)

    if N <= 1:
        return [
            x[0] if isinstance(x[0], Complex) else Complex(x[0])
        ]

    if N % 2 != 0:
        return dft(x)

    even = fft([x[i] for i in range(0, N, 2)])
    odd = fft([x[i] for i in range(1, N, 2)])

    result = [Complex(0)] * N

    for k in range(N // 2):
        angle = -2 * math.pi * k / N
        twiddle = Complex(math.cos(angle), math.sin(angle))

        t = twiddle * odd[k]

        result[k] = even[k] + t
        result[k + N // 2] = even[k] - t

    return result
```

---

### 4.5 Power spectrum

```python
def power_spectrum(X):
    return [
        xk.real ** 2 + xk.imag ** 2
        for xk in X
    ]


def magnitude_spectrum(X):
    return [
        xk.magnitude()
        for xk in X
    ]


def phase_spectrum(X):
    return [
        xk.phase()
        for xk in X
    ]
```

---

### 4.6 FFT convolution

```python
def next_power_of_two(n):
    p = 1

    while p < n:
        p *= 2

    return p


def convolve_fft(x, h):
    output_len = len(x) + len(h) - 1
    padded_N = next_power_of_two(output_len)

    x_padded = x + [0.0] * (padded_N - len(x))
    h_padded = h + [0.0] * (padded_N - len(h))

    X = fft(x_padded)
    H = fft(h_padded)

    Y = [
        xk * hk
        for xk, hk in zip(X, H)
    ]

    y = idft(Y)

    return [
        y[n].real
        for n in range(output_len)
    ]
```

---

### 4.7 Direct convolution 对照

```python
def convolve_direct(x, h):
    output_len = len(x) + len(h) - 1
    y = [0.0] * output_len

    for i, xi in enumerate(x):
        for j, hj in enumerate(h):
            y[i + j] += xi * hj

    return y
```

验证：

```python
x = [1, 2, 3, 4]
h = [1, 1, 1]

print(convolve_direct(x, h))
print(convolve_fft(x, h))
```

---

### 4.8 Hann window

```python
def hann_window(N):
    return [
        0.5 * (1 - math.cos(2 * math.pi * n / (N - 1)))
        for n in range(N)
    ]


def apply_window(signal, window):
    return [
        x * w
        for x, w in zip(signal, window)
    ]
```

---

## 5. 生产使用：NumPy / SciPy 中的 Fourier 工具

---

### 5.1 NumPy FFT

```python
import numpy as np

N = 256
fs = 256

t = np.arange(N) / fs

signal = np.sin(2 * np.pi * 5 * t)

spectrum = np.fft.fft(signal)
freqs = np.fft.fftfreq(N, d=1 / fs)

power = np.abs(spectrum) ** 2

positive_freqs = freqs[:N // 2]
positive_power = power[:N // 2]

peak_idx = np.argmax(positive_power)

print("dominant frequency:", positive_freqs[peak_idx])
```

---

### 5.2 Windowing

```python
from scipy.signal import windows

window = windows.hann(N)

windowed = signal * window

spectrum_windowed = np.fft.fft(windowed)
```

---

### 5.3 FFT convolution

```python
from scipy.signal import fftconvolve

x = np.array([1, 2, 3, 4])
h = np.array([1, 1, 1])

result = fftconvolve(x, h, mode="full")

print(result)
```

---

### 5.4 STFT 与 Spectrogram

```python
from scipy.signal import stft

sample_rate = 16000

frequencies, times, Zxx = stft(
    signal,
    fs=sample_rate,
    nperseg=256,
)

spectrogram = np.abs(Zxx) ** 2

print(spectrogram.shape)
```

输出矩阵：

```txt
(n_frequencies, n_time_frames)
```

每一列是一小段时间窗口的频谱。

---

### 5.5 Transformer sinusoidal positional encoding

```python
def sinusoidal_positional_encoding(max_pos, d_model):
    positions = np.arange(max_pos)[:, None]
    dims = np.arange(d_model)[None, :]

    angle_rates = 1 / np.power(
        10000,
        (2 * (dims // 2)) / d_model
    )

    angles = positions * angle_rates

    pe = np.zeros((max_pos, d_model))
    pe[:, 0::2] = np.sin(angles[:, 0::2])
    pe[:, 1::2] = np.cos(angles[:, 1::2])

    return pe
```

---

### 5.6 手写实现 vs 生产库

| 对比项 | 手写实现 | NumPy / SciPy |
|---|---|---|
| DFT | O(N²) 公式实现 | `np.fft.fft` |
| IDFT | O(N²) 公式实现 | `np.fft.ifft` |
| FFT | 递归 Cooley-Tukey | 高性能 FFT 库 |
| Power spectrum | 手写 `real²+imag²` | `np.abs(X)**2` |
| Windowing | 手写 Hann | `scipy.signal.windows` |
| Convolution | FFT + multiply + IFFT | `scipy.signal.fftconvolve` |
| STFT | 手写滑窗 FFT | `scipy.signal.stft` |
| Spectrogram | STFT power | `scipy.signal.spectrogram` |

---

## 6. 交付沉淀：产出物、连接关系、练习、术语表

---

### 6.1 本课产出

```txt
code/fourier.py
code/use_numpy_fft.py
code/use_scipy_signal.py
outputs/prompt-spectral-analyzer.md
outputs/exercises-solutions.md
```

| 文件 | 作用 |
|---|---|
| `code/fourier.py` | 从零实现 DFT、IDFT、FFT、power spectrum、FFT convolution |
| `code/use_numpy_fft.py` | NumPy FFT、frequency bins、频谱分析 |
| `code/use_scipy_signal.py` | windowing、STFT、spectrogram、fftconvolve |
| `outputs/prompt-spectral-analyzer.md` | 用于分析信号频率结构的 prompt |
| `outputs/exercises-solutions.md` | 练习题与参考答案 |

---

### 6.2 知识连接关系

| 本课知识点 | 后续连接 |
|---|---|
| DFT | 频域分析 |
| FFT | 高效频域计算 |
| IDFT | 信号重构 |
| Power spectrum | 频率能量 |
| Phase spectrum | 相位、时移 |
| Nyquist | 采样理论 |
| Aliasing | 下采样、抗混叠 |
| Windowing | 频谱泄漏 |
| STFT | 音频特征 |
| Spectrogram | 语音识别、音乐生成 |
| Convolution theorem | CNN、FFT convolution |
| Sinusoidal PE | Transformer |
| Fourier features | NeRF、坐标网络 |
| FNet | FFT token mixing |

---

### 6.3 AI 应用连接

| 傅里叶知识点 | AI 应用 |
|---|---|
| FFT | 快速频域计算 |
| Power spectrum | 音频/图像频率分析 |
| Spectrogram | Whisper、ASR、音频分类 |
| Convolution theorem | CNN 卷积理解 |
| FFT convolution | 大 kernel 加速 |
| Aliasing | anti-aliased pooling |
| Windowing | 音频预处理 |
| Sinusoidal PE | Transformer 位置编码 |
| Fourier features | NeRF、implicit neural representations |
| FNet | 用 FFT 替代 attention mixing |

---

### 6.4 练习

1. **Pure tone identification。**  
   创建一个未知频率的 sine wave，频率在 `1-50 Hz` 之间，以 `128 Hz` 采样 1 秒。  
   用手写 DFT 找出主频。加入标准差为 `0.5` 的 Gaussian noise 后重复，观察噪声如何影响频谱。

2. **FFT vs DFT verification。**  
   生成长度为 64 的随机信号。分别用手写 DFT 和 FFT 计算频谱，验证所有系数误差小于 `1e-10`。  
   在长度 `256, 512, 1024, 2048` 上计时，观察速度差异。

3. **卷积定理验证。**  
   给定：

   ```txt
   x = [1,2,3,4,0,0,0,0]
   h = [1,1,1,0,0,0,0,0]
   ```

   先用 nested loop 做 circular convolution，再用 FFT -> multiply -> IFFT，验证结果一致。  
   再通过 zero-padding 实现 linear convolution。

4. **Windowing effects。**  
   创建两个接近频率的 sine wave：

   ```txt
   10 Hz 和 12 Hz
   ```

   在 `128 Hz` 采样 1 秒。分别使用 no window、Hann window、Hamming window 计算 power spectrum，比较谱泄漏和峰值可分辨性。

5. **Positional encoding analysis。**  
   生成 `d_model=128`、`max_pos=512` 的 sinusoidal positional encoding。  
   计算任意两个位置编码的 dot product，验证它主要依赖相对距离 `|p1-p2|`，而不是绝对位置。

6. **STFT spectrogram。**  
   构造一个 chirp 信号，频率随时间升高。计算 STFT，观察 spectrogram 中能量峰如何随时间上移。

---

### 6.5 关键术语

| 术语 | 常见说法 | 真正含义 |
|---|---|---|
| DFT | 离散傅里叶变换 | 把 N 个时间域样本变成 N 个频域复数系数 |
| FFT | 快速傅里叶变换 | `O(N log N)` 计算 DFT 的算法 |
| IDFT | 逆 DFT | 从频域系数恢复时间域信号 |
| Frequency bin | 频率槽 | DFT 输出 index 对应的离散频率 |
| DC component | 直流分量 | `X[0]`，零频率，和均值成比例 |
| Nyquist frequency | 奈奎斯特频率 | `fs/2`，采样率下可表示的最高频率 |
| Power spectrum | 功率谱 | `|X[k]|^2`，每个频率的能量 |
| Phase spectrum | 相位谱 | `angle(X[k])`，每个频率的相位偏移 |
| Spectral leakage | 谱泄漏 | 非周期截断导致能量扩散到其他频率 |
| Window function | 窗函数 | 减少边界不连续的 tapering 函数 |
| Twiddle factor | 旋转因子 | FFT 合并子问题时使用的复指数 |
| Convolution theorem | 卷积定理 | 时域卷积等价于频域逐点乘法 |
| Circular convolution | 循环卷积 | DFT 默认的周期卷积 |
| Linear convolution | 线性卷积 | 普通不绕回的卷积 |
| Parseval theorem | Parseval 定理 | 时域和频域能量守恒 |
| Aliasing | 混叠 | 高频因采样率不足伪装成低频 |
| STFT | 短时傅里叶变换 | 对滑动窗口分别做 FFT |
| Spectrogram | 频谱图 | 时间-频率能量二维表示 |
| Zero-padding | 补零 | 平滑频谱显示、支持 FFT 长度和线性卷积 |

---

### 6.6 自检问题

学完本课后，你应该能回答：

- 傅里叶变换为什么重要？
- 时间域和频率域分别看什么？
- DFT 公式中 `e^(-2πikn/N)` 表示什么？
- `X[k]` 的 magnitude、phase、power 分别表示什么？
- `X[0]` 为什么是 DC component？
- Nyquist frequency 为什么是 `fs/2`？
- 实值信号为什么有共轭对称频谱？
- IDFT 为什么可以完美重构信号？
- FFT 为什么是 `O(N log N)`？
- Cooley-Tukey 如何用 even/odd split 加速？
- 频率分辨率由什么决定？
- 卷积定理为什么能加速卷积？
- circular convolution 和 linear convolution 有什么区别？
- windowing 为什么能减少 spectral leakage？
- STFT 为什么适合音频？
- aliasing 为什么无法事后恢复？
- zero-padding 为什么不增加真实分辨率？
- Transformer sinusoidal PE 和 Fourier 有什么关系？
- CNN 卷积如何从频域理解？

---

## 附录：本课最小代码文件建议

如果要拆成代码文件，可以这样组织：

```txt
code/
├── fourier.py
├── use_numpy_fft.py
└── use_scipy_signal.py
```

### `fourier.py`

包含：

```txt
Complex
dft
idft
fft
power_spectrum
magnitude_spectrum
phase_spectrum
convolve_direct
convolve_fft
hann_window
apply_window
```

### `use_numpy_fft.py`

包含：

```txt
np.fft.fft
np.fft.ifft
np.fft.fftfreq
dominant frequency detection
power spectrum plotting
sinusoidal positional encoding analysis
```

### `use_scipy_signal.py`

包含：

```txt
scipy.signal.windows
scipy.signal.fftconvolve
scipy.signal.stft
spectrogram
windowing comparison
aliasing demo
```
