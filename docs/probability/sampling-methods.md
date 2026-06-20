---
title: 采样方法
description: 面向 AI 算法工程师的采样课程：从均匀采样、逆 CDF、拒绝采样、重要性采样、Monte Carlo、MCMC，到 LLM 解码、重参数化技巧和 diffusion sampling。
---

# 采样方法

> 采样是 AI 探索可能性空间的方式。

**课程类型：** 实现 / 应用  
**所属模块：** 概率统计 / 生成模型 / 推理采样  
**前置知识：** Phase 1 Lesson 06-07：概率与分布、贝叶斯公式  
**预计时间：** 约 120 分钟  
**使用语言：** Python / NumPy / SciPy  

---

## 1. 提出问题：为什么 AI 算法工程师必须理解采样

一个语言模型处理完你的 prompt 后，会输出一个长度为 50,000 的 logits 向量。  
每个 logit 对应词表里的一个 token。

现在问题来了：

```txt
模型到底应该选哪个 token？
```

如果每次都选概率最高的 token，输出就会非常确定、重复、乏味。  
如果完全均匀随机选 token，输出就会变成胡言乱语。

真正有用的生成质量，往往来自二者之间：

```txt
既要保持高概率和连贯性，
又要保留一定随机性和多样性。
```

这就是采样策略的作用。

采样不仅出现在文本生成里：

```txt
SGD 采样 mini-batch
dropout 随机丢弃神经元
data augmentation 采样随机变换
RL 采样 trajectory
VAE 从 latent distribution 中采样
diffusion 从噪声开始逐步采样生成图像
MCMC 从复杂后验分布中采样
Monte Carlo 用样本估计难以解析计算的积分
```

可以说：

```txt
每一个生成式 AI 系统，都是一个采样系统。
```

采样策略会直接影响：

```txt
输出质量
多样性
可控性
训练稳定性
估计方差
探索能力
```

---

### 1.1 数学概念历史出现缘由

采样方法最早来自概率论、统计推断、数值积分和物理模拟。  
当一个分布无法直接计算、无法枚举、无法解析积分时，采样就变成核心工具。

| 数学 / 工程概念 | 出现缘由 |
|---|---|
| 均匀采样 | 为了从最简单、最基础的随机源开始生成随机数 |
| 逆 CDF 采样 | 为了把均匀随机数转换成目标分布样本 |
| 拒绝采样 | 为了在无法求逆 CDF 时，利用 proposal distribution 采样 |
| 重要性采样 | 为了用另一个更容易采样的分布估计目标分布下的期望 |
| Monte Carlo | 为了用随机样本估计高维积分 |
| MCMC | 为了从无法归一化或无法直接采样的复杂分布中采样 |
| Metropolis-Hastings | 为了构造以目标分布为平稳分布的 Markov chain |
| Gibbs sampling | 为了在可采样条件分布时逐维更新 |
| Temperature sampling | 为了控制语言模型输出的确定性和多样性 |
| Top-k sampling | 为了截断低概率长尾 token |
| Top-p / nucleus sampling | 为了动态保留累计概率足够高的 token 集合 |
| Reparameterization trick | 为了让连续随机采样可微，使 VAE 可以反向传播 |
| Gumbel-Softmax | 为了近似可微地从 categorical distribution 中采样 |
| Stratified sampling | 为了减少 Monte Carlo 估计方差 |
| Diffusion sampling | 为了从噪声逐步采样生成数据 |

可以这样理解：

```txt
现实问题：只有均匀随机数，如何生成其他分布？
数学方法：逆 CDF 采样

现实问题：目标分布知道形状但难采样，怎么办？
数学方法：拒绝采样

现实问题：目标分布难采样，但只需要估计期望，怎么办？
数学方法：重要性采样

现实问题：高维积分无法解析计算，怎么办？
数学方法：Monte Carlo

现实问题：后验分布只知道未归一化密度，怎么办？
数学方法：MCMC

现实问题：LLM 既要连贯又要多样，怎么办？
工程方法：temperature / top-k / top-p

现实问题：VAE 采样不可导，怎么办？
数学技巧：reparameterization trick
```

---

### 1.2 原始问题是什么

本课要解决的原始问题是：

```txt
如何从简单均匀随机数生成复杂分布样本？
什么时候可以用逆 CDF 采样？
什么时候需要拒绝采样？
重要性采样为什么能用 q(x) 的样本估计 p(x) 下的期望？
Monte Carlo 为什么可以用随机样本近似积分？
MCMC 如何从复杂后验分布中采样？
Metropolis-Hastings 为什么只需要未归一化概率？
Gibbs sampling 为什么逐维采样也能收敛到联合分布？
语言模型如何从 logits 中采样 token？
temperature、top-k、top-p 分别控制什么？
VAE 为什么需要 reparameterization trick？
Gumbel-Softmax 如何让离散采样近似可微？
Diffusion model 为什么本质上是一个逐步采样过程？
```

换成 AI 语言，就是：

```txt
LLM 的 temperature 到底是什么？
top-p 为什么比 top-k 更自适应？
为什么 greedy decoding 容易无聊？
为什么随机采样可能导致胡言乱语？
VAE 为什么不能直接 z ~ N(mu, sigma^2) 然后 backprop？
为什么 PPO 里会出现 importance ratio？
为什么 MCMC 能用于贝叶斯后验推断？
为什么 diffusion 是从随机噪声一步步采样出图像？
```

---

### 1.3 AI 中的现代问题

| 采样知识点 | AI 中的现代问题 |
|---|---|
| Uniform sampling | 随机初始化、数据增强、基础随机源 |
| Inverse CDF | 从已知 CDF 的分布采样 |
| Rejection sampling | 从未归一化分布或复杂 PDF 中采样 |
| Importance sampling | RL、PPO、off-policy correction |
| Monte Carlo | 期望估计、ELBO、积分近似 |
| MCMC | 贝叶斯后验、Bayesian inference |
| Metropolis-Hastings | 未归一化 target distribution 采样 |
| Gibbs sampling | 图模型、Bayesian networks |
| Temperature | LLM 输出随机性控制 |
| Top-k | 截断固定数量候选 token |
| Top-p | nucleus sampling，自适应候选集合 |
| Reparameterization trick | VAE 可微采样 |
| Gumbel-Softmax | 离散 latent variable 可微近似 |
| Stratified sampling | 降低估计方差 |
| Diffusion sampling | 从噪声到图像的反向生成过程 |
| Thompson sampling | bandit 探索和利用 |
| Dropout sampling | 随机网络子结构 |
| Mini-batch sampling | SGD 训练 |

---

### 1.4 本课要解决的核心问题

学完本课，你应该能够回答：

1. 为什么所有采样方法都可以从 uniform random number 开始？
2. 逆 CDF 采样的数学依据是什么？
3. 拒绝采样为什么需要 proposal distribution 和 envelope constant？
4. 重要性采样中的 weight `p(x)/q(x)` 表示什么？
5. Monte Carlo 误差为什么是 `O(1/sqrt(N))`？
6. Metropolis-Hastings 的 acceptance ratio 如何计算？
7. MCMC 中 burn-in、thinning、proposal scale 分别是什么意思？
8. Gibbs sampling 适合什么条件？
9. temperature 如何改变 softmax 分布？
10. top-k 和 top-p 的区别是什么？
11. 为什么 top-p 更能适应上下文不确定性？
12. reparameterization trick 为什么能让 VAE 反向传播？
13. Gumbel-Softmax 如何近似 categorical sampling？
14. stratified sampling 为什么能降低方差？
15. diffusion model 的采样过程和本课方法有什么联系？
16. 如何从零实现这些采样算法？

---

## 2. 概念定位：这节课学什么

---

### 2.1 所属模块

```txt
所属模块：概率统计 / 生成模型 / 推理采样
前置知识：概率分布、CDF、PDF、softmax、log probability、贝叶斯后验
后续连接：LLM decoding、VAE、Diffusion、MCMC、RL、PPO、Bayesian inference
```

前面的概率课程解决了：

```txt
如何描述分布？
```

本课进一步解决：

```txt
如何从分布中生成样本，以及如何用样本估计难以解析计算的量？
```

---

### 2.2 本课学习目标

完成本课后，你应该能够：

- 从零实现 uniform sampling
- 从零实现 inverse CDF sampling
- 从零实现 rejection sampling
- 从零实现 importance sampling
- 用 Monte Carlo 估计积分和 π
- 实现 Metropolis-Hastings MCMC
- 理解 Gibbs sampling 的条件采样思想
- 实现 temperature sampling
- 实现 top-k 和 top-p sampling
- 解释 reparameterization trick
- 实现 Gumbel-Softmax 的核心逻辑
- 解释 stratified sampling 如何降低方差
- 说明 sampling 如何连接 LLM、VAE、diffusion、RL 和贝叶斯推断

---

### 2.3 本课在整体路线中的位置

```txt
概率与分布
    ↓
贝叶斯公式
    ↓
采样方法
    ↓
Monte Carlo / MCMC
    ↓
LLM decoding / VAE / Diffusion
    ↓
生成式 AI 与概率推断
```

本课的核心能力是：

```txt
看到一个 AI 系统中的随机性时，能判断它是在生成、训练、估计还是探索，并知道对应的采样算法是什么。
```

---

## 3. 理解概念：直觉、公式、算法解释、AI 连接

---

### 3.1 为什么采样重要

**一句话直觉：**  
采样让模型可以在概率分布中探索可能结果，而不是只输出一个确定答案。

采样在 AI 中有四类核心作用：

| 角色 | 例子 |
|---|---|
| 生成 | LLM、diffusion、GAN、VAE |
| 训练 | SGD mini-batch、dropout、data augmentation |
| 估计 | Monte Carlo、ELBO、期望 loss |
| 探索 | RL trajectory、MCMC、Thompson sampling |

**AI 连接：**

| 系统 | 采样作用 |
|---|---|
| LLM | 从 token distribution 中采下一个 token |
| Diffusion | 从噪声逐步采样生成图像 |
| VAE | 从 latent distribution 中采样 |
| RL | 从 policy 中采样 action 和 trajectory |
| Bayesian model | 从 posterior 中采样 |
| Dropout | 采样要关闭的神经元 |
| SGD | 采样 mini-batch |

---

### 3.2 Uniform random sampling：一切采样的起点

**一句话直觉：**  
均匀随机数是最基础的随机源，很多复杂采样方法都从 `U ~ Uniform(0, 1)` 开始。

定义：

```text
U ~ Uniform(0, 1)
```

性质：

```text
P(a <= U <= b) = b - a
E[U] = 0.5
Var(U) = 1/12
```

从 `[a, b]` 采样：

```text
X = a + (b - a) * U
```

从 `n` 个离散项中均匀采样：

```text
index = floor(n * U)
```

**核心洞察：**

```txt
一个 uniform random number 包含足够的随机性，可以通过合适变换生成其他分布样本。
```

---

### 3.3 Inverse CDF sampling：把概率映射回数值

**一句话直觉：**  
CDF 把数值变成概率，inverse CDF 把均匀概率变成目标分布的数值。

CDF：

```text
F(x) = P(X <= x)
```

如果：

```text
U ~ Uniform(0, 1)
```

则：

```text
X = F^{-1}(U)
```

服从目标分布。

为什么成立？

```text
P(X <= x)
= P(F^{-1}(U) <= x)
= P(U <= F(x))
= F(x)
```

#### 指数分布例子

PDF：

```text
f(x) = lambda * exp(-lambda x), x >= 0
```

CDF：

```text
F(x) = 1 - exp(-lambda x)
```

令：

```text
u = F(x)
```

解得：

```text
x = -log(1 - u) / lambda
```

由于 `1-U` 和 `U` 分布相同，也常写成：

```text
x = -log(U) / lambda
```

**适用场景：**

```txt
能写出或数值计算 inverse CDF 的分布。
```

---

### 3.4 Rejection sampling：提出样本，再接受或拒绝

**一句话直觉：**  
拒绝采样先从容易采样的 proposal distribution 中采样，再按 target/proposal 比例决定是否接受。

设：

```txt
target distribution: p(x)
proposal distribution: q(x)
bound: p(x) <= M q(x)
```

算法：

```txt
1. 从 q(x) 采样 x。
2. 从 Uniform(0,1) 采样 u。
3. 如果 u < p(x) / (M q(x))，接受 x。
4. 否则拒绝，重新采样。
```

接受率大约为：

```text
1 / M
```

M 越紧，效率越高。

**局限：**

```txt
低维有效。
高维时接受率可能指数级下降。
```

**AI 连接：**

| 场景 | 作用 |
|---|---|
| 截断分布采样 | truncated normal |
| 未归一化 PDF | 只知道比例 |
| Monte Carlo 几何估计 | 例如估计 π |
| 教学 | 理解 target/proposal |

---

### 3.5 Importance sampling：用另一个分布估计目标期望

**一句话直觉：**  
重要性采样用容易采样的 q(x) 样本，带权重估计 p(x) 下的期望。

目标：

```text
E_p[f(x)] = integral f(x) p(x) dx
```

改写：

```text
E_p[f(x)]
= integral f(x) * p(x)/q(x) * q(x) dx
= E_q[f(x) w(x)]
```

其中：

```text
w(x) = p(x) / q(x)
```

估计器：

```text
E_p[f(x)] ≈ (1/N) * sum_i f(x_i) w(x_i), x_i ~ q(x)
```

self-normalized importance sampling：

```text
E_p[f(x)] ≈ sum_i w_i f(x_i) / sum_i w_i
```

**AI 连接：**

PPO 中的 importance ratio：

```text
r_t = pi_new(a_t|s_t) / pi_old(a_t|s_t)
```

它表示：

```txt
旧策略采样到的 trajectory 如何用于估计新策略目标。
```

如果 `q` 和 `p` 差异太大，少数样本会有巨大权重，估计方差会很高。

---

### 3.6 Monte Carlo estimation：用随机样本估计积分

**一句话直觉：**  
Monte Carlo 用样本平均近似期望和积分。

目标积分：

```text
I = integral_D g(x) dx
```

如果从区域 `D` 均匀采样：

```text
I ≈ Volume(D) / N * sum_i g(x_i)
```

误差：

```text
O(1 / sqrt(N))
```

重要特点：

```txt
误差收敛速度与维度无关。
```

这使得 Monte Carlo 在高维积分中非常重要。

#### 估计 π

```txt
从 [-1,1] x [-1,1] 采样点。
计算落在单位圆内的比例。
pi ≈ 4 * inside / total
```

**AI 连接：**

| 场景 | Monte Carlo 作用 |
|---|---|
| expected loss | 用样本平均估计 |
| Bayesian inference | 估计后验期望 |
| VAE | ELBO Monte Carlo estimate |
| RL | trajectory return estimate |
| Diffusion | 训练目标期望估计 |
| 高维积分 | 无解析解时采样估计 |

---

### 3.7 MCMC 与 Metropolis-Hastings

**一句话直觉：**  
MCMC 构造一个 Markov chain，让它长期停留的分布正好是目标分布。

Metropolis-Hastings 目标：

```txt
从 p(x) 中采样。
但 p(x) 可能只知道未归一化形式。
```

算法：

```txt
1. 从当前状态 x 出发。
2. proposal q(x'|x) 提出新状态 x'。
3. 计算 acceptance ratio。
4. 以 min(1, alpha) 的概率接受。
5. 否则留在原状态。
```

一般 acceptance ratio：

```text
alpha = [p(x') q(x|x')] / [p(x) q(x'|x)]
```

如果 proposal 对称：

```text
q(x'|x) = q(x|x')
```

则：

```text
alpha = p(x') / p(x)
```

常见概念：

| 概念 | 含义 |
|---|---|
| burn-in | 丢弃早期尚未收敛的样本 |
| thinning | 每隔 k 个样本保留一个，降低自相关 |
| proposal scale | 提议步长 |
| acceptance rate | 接受比例 |
| detailed balance | 保证目标分布为平稳分布的条件 |

proposal scale 太小：

```txt
接受率高，但探索慢。
```

proposal scale 太大：

```txt
拒绝率高，链容易卡住。
```

---

### 3.8 Gibbs sampling：逐个变量采样

**一句话直觉：**  
Gibbs sampling 每次只更新一个变量，从该变量的条件分布中采样。

目标：

```text
p(x_1, x_2, ..., x_d)
```

循环：

```txt
sample x_1 ~ p(x_1 | x_2, ..., x_d)
sample x_2 ~ p(x_2 | x_1, x_3, ..., x_d)
...
sample x_d ~ p(x_d | x_1, ..., x_{d-1})
```

特点：

```txt
每一步都从精确条件分布采样。
接受率为 1。
```

局限：

```txt
变量强相关时，逐维更新会混合很慢。
```

**AI 连接：**

| 场景 | Gibbs sampling |
|---|---|
| Bayesian networks | 图模型推断 |
| Gaussian mixture | 条件分布易采样 |
| Ising model | 每个变量依赖邻居 |
| topic model | LDA collapsed Gibbs |
| Bayesian posterior | 条件分布可得时 |

---

### 3.9 Temperature sampling：控制 LLM 输出随机性

**一句话直觉：**  
temperature 通过缩放 logits 控制 softmax 分布的尖锐程度。

公式：

```text
p_i = exp(z_i / T) / sum_j exp(z_j / T)
```

含义：

| Temperature | 效果 |
|---:|---|
| `T -> 0` | 接近 argmax，确定性输出 |
| `T < 1` | 分布更尖锐，更保守 |
| `T = 1` | 原始 softmax |
| `T > 1` | 分布更平坦，更多样 |
| `T -> inf` | 接近 uniform |

常见经验：

| 任务 | temperature |
|---|---:|
| factual QA | 0 或较低 |
| code generation | 0.3 - 0.7 |
| general chat | 0.7 - 1.0 |
| creative writing | 1.0 左右或略高 |
| highly random generation | > 1.5，通常较难控制 |

**AI 连接：**

temperature 不会改变 token 排名，只会改变概率质量分布。

---

### 3.10 Top-k sampling：只保留前 k 个 token

**一句话直觉：**  
Top-k 采样只从概率最高的 k 个 token 中重新归一化并采样。

算法：

```txt
1. 计算所有 token 概率。
2. 按概率从大到小排序。
3. 保留前 k 个。
4. 把这 k 个概率重新归一化。
5. 从中采样。
```

特殊情况：

```txt
k = 1：greedy decoding
k = vocabulary size：不截断
k = 40/50：常见经验值
```

优点：

```txt
排除长尾低概率 token，减少胡言乱语。
```

缺点：

```txt
k 是固定的，不随上下文不确定性变化。
模型很确定时，k 可能太大。
模型很不确定时，k 可能太小。
```

---

### 3.11 Top-p / Nucleus sampling：保留累计概率前 p 的 token

**一句话直觉：**  
Top-p 不固定 token 数，而是保留最小的一组 token，使其累计概率超过 p。

算法：

```txt
1. 计算所有 token 概率。
2. 按概率降序排序。
3. 依次累加概率，直到累计概率 >= p。
4. 保留这些 token。
5. 重新归一化并采样。
```

例子：

```txt
p = 0.9
保留覆盖 90% 概率质量的最小 token 集合。
```

特点：

```txt
模型很确定时，候选 token 很少。
模型不确定时，候选 token 变多。
```

这就是 nucleus sampling 更自适应的原因。

**AI 连接：**

常见组合：

```txt
temperature = 0.7
top_p = 0.9
```

适合很多通用生成任务。

---

### 3.12 Reparameterization trick：把随机性移到外面

**一句话直觉：**  
重参数化技巧把不可导的采样，改写成“参数的可导函数 + 参数无关随机噪声”。

VAE 中希望采样：

```text
z ~ N(mu, sigma^2)
```

直接采样不可导：

```txt
random sampling 会阻断梯度。
```

重参数化：

```text
epsilon ~ N(0, 1)
z = mu + sigma * epsilon
```

此时：

```text
dz/dmu = 1
dz/dsigma = epsilon
```

随机性来自 `epsilon`，它不依赖参数。  
`z` 对 `mu` 和 `sigma` 是可导的。

**AI 连接：**

VAE 训练流程：

```txt
encoder 输出 mu 和 logvar
采样 epsilon
z = mu + sigma * epsilon
decoder 重构输入
loss backward 可以通过 z 传回 encoder
```

没有 reparameterization trick，VAE 无法用标准 backprop 高效训练。

---

### 3.13 Gumbel-Softmax：可微的离散采样近似

**一句话直觉：**  
Gumbel-Softmax 用 Gumbel noise + softmax，把 categorical sampling 近似成可微操作。

Gumbel-Max trick：

```txt
1. 对每个类别采样 g_i ~ Gumbel(0,1)
2. 返回 argmax(log p_i + g_i)
```

这能得到精确 categorical sample，但 `argmax` 不可导。

Gumbel-Softmax 用 softmax 替代 argmax：

```text
y_i = exp((log p_i + g_i) / tau) / sum_j exp((log p_j + g_j) / tau)
```

temperature `tau` 控制近似程度：

| tau | 效果 |
|---:|---|
| `tau -> 0` | 接近 one-hot |
| `tau = 1` | soft approximation |
| `tau -> inf` | 接近 uniform |

**AI 连接：**

| 场景 | Gumbel-Softmax |
|---|---|
| discrete latent variable VAE | 可微近似采样 |
| neural architecture search | 选择离散操作 |
| hard attention | 近似可微选择 |
| RL discrete actions | differentiable relaxation |

---

### 3.14 Stratified sampling：强制覆盖采样空间

**一句话直觉：**  
stratified sampling 把空间分成多个 strata，每个 strata 都采样，从而减少随机空洞和聚集。

一维 `[0, 1]` 中：

```txt
普通 Monte Carlo：
随机采 N 个点，可能某些区域密集、某些区域空缺。

Stratified：
把 [0,1] 分成 N 段，每段采一个点。
```

公式：

```text
x_i = (i + u_i) / N, u_i ~ Uniform(0,1)
```

性质：

```text
Var(stratified) <= Var(standard Monte Carlo)
```

**AI 连接：**

| 场景 | stratified sampling |
|---|---|
| train/test split | 保持类别比例 |
| cross-validation | stratified folds |
| NeRF | 沿 ray 分层采样 |
| numerical integration | 降低估计方差 |
| imbalanced dataset | 保证少数类覆盖 |

---

### 3.15 Diffusion sampling：从噪声到数据的逐步采样

**一句话直觉：**  
Diffusion model 先把数据逐步加噪成纯噪声，再学习反向过程，从噪声一步步采样回数据。

Forward process：

```text
x_t = sqrt(alpha_t) x_{t-1} + sqrt(1 - alpha_t) epsilon
```

其中：

```text
epsilon ~ N(0, I)
```

随着 `t` 增加：

```txt
x_t 越来越接近纯噪声。
```

Reverse process：

```txt
模型学习从 x_t 预测噪声或 denoised x_{t-1}。
从纯噪声开始，逐步采样生成图像。
```

与本课方法的连接：

| 概念 | diffusion 中的对应 |
|---|---|
| Gaussian sampling | forward 加噪和 reverse sampling |
| reparameterization | 噪声 + deterministic transform |
| Monte Carlo | 训练目标期望估计 |
| Markov chain | 每一步只依赖当前状态 |
| temperature / schedule | noise schedule 控制采样过程 |

---

### 3.16 本课核心概念总表

| 概念 | 一句话直觉 | AI 中的对应位置 |
|---|---|---|
| Uniform sampling | 基础随机源 | 所有采样方法起点 |
| Inverse CDF | 均匀数变目标分布 | 指数分布、可逆 CDF |
| Rejection sampling | propose 后接受/拒绝 | 复杂 PDF 采样 |
| Importance sampling | 用 q 样本估计 p 期望 | PPO、off-policy |
| Monte Carlo | 样本平均估计积分 | 期望、ELBO、RL |
| MCMC | 构造目标平稳分布的链 | Bayesian posterior |
| Metropolis-Hastings | 按密度比接受 proposal | 未归一化分布采样 |
| Gibbs sampling | 逐变量条件采样 | 图模型 |
| Temperature | logits 缩放 | LLM decoding |
| Top-k | 保留固定 k 个候选 | token sampling |
| Top-p | 保留累计概率 p 的候选 | nucleus sampling |
| Reparameterization | 随机性外移 | VAE |
| Gumbel-Softmax | 可微离散采样近似 | discrete latent |
| Stratified sampling | 分层覆盖 | 降低方差、数据划分 |
| Diffusion sampling | 从噪声逐步生成 | 图像生成 |

---

## 4. 手写实现：从零实现主要采样方法

这一节只使用 Python 标准库的 `random` 和 `math`，先理解核心算法。

---

### 4.1 Uniform 与 Inverse CDF 采样

```python
import math
import random


def sample_uniform(a, b):
    return a + (b - a) * random.random()


def sample_exponential_inverse_cdf(lam):
    u = random.random()

    return -math.log(u) / lam
```

验证指数分布均值：

```python
samples = [
    sample_exponential_inverse_cdf(lam=2.0)
    for _ in range(10000)
]

print(sum(samples) / len(samples))
print("expected:", 1 / 2.0)
```

---

### 4.2 Rejection sampling

```python
def rejection_sample(target_pdf, proposal_sample, proposal_pdf, M):
    while True:
        x = proposal_sample()
        u = random.random()

        accept_prob = target_pdf(x) / (M * proposal_pdf(x))

        if u < accept_prob:
            return x
```

使用时要保证：

```text
target_pdf(x) <= M * proposal_pdf(x)
```

否则采样结果不正确。

---

### 4.3 Importance sampling

```python
def importance_sampling_estimate(
    f,
    target_pdf,
    proposal_pdf,
    proposal_sample,
    n,
):
    total = 0.0

    for _ in range(n):
        x = proposal_sample()
        w = target_pdf(x) / proposal_pdf(x)

        total += f(x) * w

    return total / n
```

Self-normalized 版本：

```python
def self_normalized_importance_sampling(
    f,
    target_pdf,
    proposal_pdf,
    proposal_sample,
    n,
):
    weighted_sum = 0.0
    weight_sum = 0.0

    for _ in range(n):
        x = proposal_sample()
        w = target_pdf(x) / proposal_pdf(x)

        weighted_sum += w * f(x)
        weight_sum += w

    return weighted_sum / weight_sum
```

---

### 4.4 Monte Carlo 估计 π

```python
def monte_carlo_pi(n):
    inside = 0

    for _ in range(n):
        x = random.uniform(-1, 1)
        y = random.uniform(-1, 1)

        if x * x + y * y <= 1:
            inside += 1

    return 4 * inside / n
```

测试：

```python
for n in [1000, 10000, 100000]:
    print(n, monte_carlo_pi(n))
```

---

### 4.5 Metropolis-Hastings MCMC

```python
def metropolis_hastings(
    target_log_pdf,
    proposal_sample,
    proposal_log_pdf,
    x0,
    n_samples,
    burn_in,
):
    samples = []
    x = x0

    for i in range(n_samples + burn_in):
        x_new = proposal_sample(x)

        log_alpha = (
            target_log_pdf(x_new)
            + proposal_log_pdf(x, x_new)
            - target_log_pdf(x)
            - proposal_log_pdf(x_new, x)
        )

        if math.log(random.random()) < log_alpha:
            x = x_new

        if i >= burn_in:
            samples.append(x)

    return samples
```

如果 proposal 对称，可以简化 `proposal_log_pdf` 项。

---

### 4.6 Gibbs sampling

```python
def gibbs_sampling_2d(
    conditional_x_given_y,
    conditional_y_given_x,
    x0,
    y0,
    n_samples,
    burn_in,
):
    x, y = x0, y0
    samples = []

    for i in range(n_samples + burn_in):
        x = conditional_x_given_y(y)
        y = conditional_y_given_x(x)

        if i >= burn_in:
            samples.append((x, y))

    return samples
```

---

### 4.7 Categorical sampling 基础函数

```python
def softmax(logits):
    max_l = max(logits)
    exps = [math.exp(z - max_l) for z in logits]
    total = sum(exps)

    return [e / total for e in exps]


def sample_from_probs(probs):
    r = random.random()
    cumsum = 0.0

    for i, p in enumerate(probs):
        cumsum += p

        if r <= cumsum:
            return i

    return len(probs) - 1
```

---

### 4.8 Temperature sampling

```python
def temperature_sample(logits, temperature):
    if temperature <= 0:
        return max(range(len(logits)), key=lambda i: logits[i])

    scaled = [z / temperature for z in logits]
    probs = softmax(scaled)

    return sample_from_probs(probs)
```

---

### 4.9 Top-k sampling

```python
def top_k_sample(logits, k):
    indexed = sorted(
        enumerate(logits),
        key=lambda x: -x[1]
    )

    top = indexed[:k]
    top_logits = [logit for _, logit in top]

    probs = softmax(top_logits)
    idx = sample_from_probs(probs)

    return top[idx][0]
```

---

### 4.10 Top-p / Nucleus sampling

```python
def top_p_sample(logits, p):
    probs = softmax(logits)

    indexed = sorted(
        enumerate(probs),
        key=lambda x: -x[1]
    )

    cumsum = 0.0
    selected = []

    for token_idx, prob in indexed:
        cumsum += prob
        selected.append((token_idx, prob))

        if cumsum >= p:
            break

    selected_probs = [prob for _, prob in selected]
    total = sum(selected_probs)
    selected_probs = [prob / total for prob in selected_probs]

    idx = sample_from_probs(selected_probs)

    return selected[idx][0]
```

---

### 4.11 Reparameterization trick

```python
def reparam_sample(mu, sigma):
    epsilon = random.gauss(0, 1)

    return mu + sigma * epsilon


def reparam_gradient(mu, sigma, epsilon):
    dz_dmu = 1.0
    dz_dsigma = epsilon

    return dz_dmu, dz_dsigma
```

---

### 4.12 Gumbel-Softmax

```python
def gumbel_sample():
    u = random.random()

    return -math.log(-math.log(u))


def gumbel_softmax(logits, temperature):
    noisy_logits = [
        logit + gumbel_sample()
        for logit in logits
    ]

    return softmax([
        z / temperature
        for z in noisy_logits
    ])
```

注意：

```txt
这里 logits 是原始分数。
如果你传入的是 probabilities，需要先取 log。
```

---

## 5. 生产使用：NumPy / SciPy 中的采样

---

### 5.1 NumPy 随机数生成器

```python
import numpy as np

rng = np.random.default_rng(42)

uniform_samples = rng.uniform(0, 1, size=10000)
normal_samples = rng.normal(loc=0, scale=1, size=10000)
exponential_samples = rng.exponential(scale=2.0, size=10000)

print(exponential_samples.mean())
```

---

### 5.2 SciPy CDF 与 inverse CDF

```python
from scipy import stats

normal = stats.norm(loc=0, scale=1)

print(f"CDF at 1.96: {normal.cdf(1.96):.4f}")
print(f"Inverse CDF at 0.975: {normal.ppf(0.975):.4f}")
```

---

### 5.3 NumPy token sampling

```python
logits = np.array([2.0, 1.0, 0.5, 0.1, -1.0])

temperature = 0.7
scaled = logits / temperature

probs = np.exp(scaled - scaled.max())
probs = probs / probs.sum()

token = rng.choice(len(logits), p=probs)

print(f"Sampled token index: {token}")
```

---

### 5.4 PyTorch top-k / top-p 实践提示

```python
import torch
import torch.nn.functional as F

logits = torch.tensor([2.0, 1.0, 0.5, 0.1, -1.0])

temperature = 0.7
probs = F.softmax(logits / temperature, dim=-1)

token = torch.multinomial(probs, num_samples=1)

print(token.item())
```

---

### 5.5 生产级 MCMC 库

对于真实 Bayesian inference，不建议手写 MCMC。常见选择：

| 库 | 适用 |
|---|---|
| PyMC | 完整 Bayesian modeling，NUTS |
| emcee | ensemble MCMC |
| NumPyro | JAX 加速 Bayesian inference |
| Stan | 经典概率编程 |
| TensorFlow Probability | TensorFlow 生态概率建模 |

---

### 5.6 手写实现 vs 生产库

| 对比项 | 手写实现 | 生产库 |
|---|---|---|
| 目的 | 理解采样原理 | 真实任务 |
| inverse CDF | 手写公式 | scipy.stats.ppf |
| rejection sampling | 手写 accept/reject | 专用采样器 |
| Monte Carlo | 手写平均 | NumPy 向量化 |
| MCMC | 手写 MH | PyMC / Stan / NumPyro |
| token sampling | 手写 top-k/top-p | transformers generate |
| VAE sampling | 手写 reparam | PyTorch autograd |
| Gumbel-Softmax | 手写公式 | torch.nn.functional.gumbel_softmax |

---

## 6. 交付沉淀：产出物、连接关系、练习、术语表

---

### 6.1 本课产出

```txt
code/sampling.py
code/use_numpy.py
code/use_pytorch.py
outputs/prompt-sampling-methods.md
outputs/exercises-solutions.md
```

| 文件 | 作用 |
|---|---|
| `code/sampling.py` | 从零实现 inverse CDF、rejection、importance、Monte Carlo、MCMC、LLM sampling |
| `code/use_numpy.py` | NumPy / SciPy 版本采样示例 |
| `code/use_pytorch.py` | PyTorch token sampling、Gumbel-Softmax、reparameterization 示例 |
| `outputs/prompt-sampling-methods.md` | 用于选择采样策略的 prompt |
| `outputs/exercises-solutions.md` | 练习题与参考答案 |

---

### 6.2 知识连接关系

| 本课知识点 | 后续连接 |
|---|---|
| Uniform sampling | 所有随机采样基础 |
| Inverse CDF | 基础分布采样 |
| Rejection sampling | 未归一化分布 |
| Importance sampling | PPO、off-policy RL |
| Monte Carlo | 高维积分、期望估计 |
| MCMC | Bayesian inference |
| Gibbs sampling | 图模型 |
| Temperature | LLM decoding |
| Top-k / Top-p | 文本生成策略 |
| Reparameterization | VAE |
| Gumbel-Softmax | 离散 latent variable |
| Stratified sampling | 方差降低、数据划分 |
| Diffusion sampling | 生成模型 |

---

### 6.3 AI 应用连接

| 采样方法 | AI 应用 |
|---|---|
| Uniform | 初始化、augmentation |
| Mini-batch sampling | SGD |
| Monte Carlo | 期望估计、RL return |
| Importance sampling | PPO、off-policy correction |
| MCMC | Bayesian posterior |
| Temperature | 控制 LLM 随机性 |
| Top-k | 截断 token 长尾 |
| Top-p | nucleus decoding |
| Reparameterization | VAE backprop |
| Gumbel-Softmax | 可微离散选择 |
| Stratified | 类别平衡 split |
| Diffusion | 从噪声生成图像 |

---

### 6.4 练习

1. **Cauchy inverse CDF。**  
   Cauchy distribution 的 CDF：

   ```text
   F(x) = 0.5 + arctan(x) / pi
   ```

   推导 inverse CDF，生成 10,000 个样本，并观察 heavy tail。

2. **Beta(2,5) rejection sampling。**  
   用 Uniform(0,1) 作为 proposal，从 Beta(2,5) 中拒绝采样。画出样本直方图并估计接受率。

3. **Monte Carlo 积分。**  
   用 Monte Carlo 估计：

   ```text
   integral_0^pi sin(x) dx
   ```

   分别使用 1,000、10,000、100,000 个样本，观察误差是否接近 `O(1/sqrt(N))`。

4. **二维 Metropolis-Hastings。**  
   从目标分布：

   ```text
   p(x,y) proportional to exp(-(x^2 y^2 + x^2 + y^2 - 8x - 8y) / 2)
   ```

   中采样。尝试不同 proposal standard deviation，观察 acceptance rate 和轨迹。

5. **文本生成采样对比。**  
   给定 10 个词和一组 logits，分别用：

   ```txt
   greedy
   temperature = 0.7
   top-k = 3
   top-p = 0.9
   ```

   生成长度为 20 的序列，每种方法运行 5 次，对比多样性。

6. **Gumbel-Softmax 温度实验。**  
   对同一组 logits，分别设置 temperature 为：

   ```txt
   1.0, 0.5, 0.1
   ```

   观察输出如何逐渐接近 one-hot。

---

### 6.5 关键术语

| 术语 | 常见说法 | 真正含义 |
|---|---|---|
| Sampling | 随机抽样 | 按概率分布生成样本 |
| Uniform distribution | 均匀分布 | 区间内每个位置概率密度相同 |
| Inverse CDF | 反函数采样 | 用 `F^{-1}(U)` 把均匀样本变成目标分布样本 |
| Rejection sampling | 拒绝采样 | 从 proposal 采样，再按 target/proposal 比例接受 |
| Importance sampling | 重要性采样 | 用 q 的样本和 p/q 权重估计 p 下期望 |
| Monte Carlo | 蒙特卡洛 | 用随机样本平均近似积分或期望 |
| MCMC | 马尔可夫链蒙特卡洛 | 构造目标分布为平稳分布的 Markov chain |
| Metropolis-Hastings | MH 算法 | 用 acceptance ratio 接受或拒绝 proposal |
| Gibbs sampling | 吉布斯采样 | 逐个变量从条件分布采样 |
| Temperature | 温度 | 缩放 logits，控制分布尖锐程度 |
| Top-k sampling | 前 k 采样 | 只从概率最高的 k 个 token 中采样 |
| Nucleus / Top-p | 核采样 | 保留累计概率达到 p 的最小 token 集合 |
| Reparameterization trick | 重参数化技巧 | 把随机性移出参数路径，使采样可微 |
| Gumbel-Softmax | 可微 categorical 采样 | 用 Gumbel noise 和 softmax 近似离散采样 |
| Stratified sampling | 分层采样 | 分层覆盖样本空间，降低方差 |
| Burn-in | 预热期 | 丢弃 MCMC 早期未收敛样本 |
| Detailed balance | 细致平衡 | 保证目标分布为 Markov chain 平稳分布的条件 |
| Diffusion sampling | 扩散采样 | 从噪声开始，通过逐步去噪生成数据 |

---

### 6.6 自检问题

学完本课后，你应该能回答：

- 为什么采样是生成式 AI 的核心？
- 为什么 uniform random number 可以作为所有采样方法的起点？
- inverse CDF sampling 为什么正确？
- rejection sampling 的 M 如何影响接受率？
- importance sampling 的权重为什么是 `p(x)/q(x)`？
- Monte Carlo 为什么适合高维积分？
- Metropolis-Hastings 为什么只需要未归一化 target density？
- proposal scale 太大或太小分别会怎样？
- Gibbs sampling 为什么接受率为 1？
- temperature 如何改变 LLM token distribution？
- top-k 和 top-p 有什么本质区别？
- 为什么 top-p 更自适应？
- VAE 为什么需要 reparameterization trick？
- Gumbel-Softmax 解决了什么问题？
- stratified sampling 为什么能降低方差？
- diffusion model 的采样过程如何理解？

---

## 附录：本课最小代码文件建议

如果要拆成代码文件，可以这样组织：

```txt
code/
├── sampling.py
├── use_numpy.py
└── use_pytorch.py
```

### `sampling.py`

包含：

```txt
sample_uniform
sample_exponential_inverse_cdf
rejection_sample
importance_sampling_estimate
self_normalized_importance_sampling
monte_carlo_pi
metropolis_hastings
gibbs_sampling_2d
softmax
sample_from_probs
temperature_sample
top_k_sample
top_p_sample
reparam_sample
gumbel_sample
gumbel_softmax
```

### `use_numpy.py`

包含：

```txt
np.random.default_rng
rng.uniform
rng.normal
rng.exponential
rng.choice
scipy.stats.cdf
scipy.stats.ppf
```

### `use_pytorch.py`

包含：

```txt
torch.multinomial
torch.softmax
torch.nn.functional.gumbel_softmax
VAE reparameterization
LLM logits sampling utilities
```
