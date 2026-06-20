---
title: 随机过程
description: 面向 AI 算法工程师的随机过程课程：从随机游走、Markov chain、平稳分布、Brownian motion、Langevin dynamics，到 MCMC、扩散模型和强化学习。
---

# 随机过程

> 随机过程是“有结构的随机性”。它是随机游走、Markov chain、MCMC 和 diffusion model 背后的数学语言。

**课程类型：** 学习 / 实现  
**所属模块：** 概率统计 / 生成模型 / 强化学习  
**前置知识：** Phase 1 Lesson 06-07：概率与分布、贝叶斯公式  
**预计时间：** 约 75 分钟  
**使用语言：** Python / NumPy  

---

## 1. 提出问题：为什么 AI 算法工程师必须理解随机过程

很多 AI 系统不是一次性产生一个随机变量，而是：

```txt
随机性随着时间一步一步演化。
```

这不是静态随机，而是结构化的序列随机。

例如：

```txt
语言模型一个 token 一个 token 地生成文本。
每个 token 依赖前面的上下文。
模型输出一个概率分布，采样一个 token，然后继续。
```

这就是一个随机过程。

再比如 diffusion model：

```txt
前向过程：一步步给图像加噪声，直到变成纯噪声。
反向过程：一步步去噪，从噪声生成图像。
```

这个过程本质上是 Markov chain。

强化学习也是：

```txt
agent 在环境中采取动作。
每个动作以某种概率导致下一个状态。
agent 的 policy 也是随机的。
整个系统是 Markov Decision Process。
```

贝叶斯推断中的 MCMC 也是：

```txt
构造一个 Markov chain。
让它的平稳分布正好是你想采样的 posterior。
```

所以，随机过程不是抽象数学。它是现代 AI 中很多系统的底层机制：

```txt
LLM token generation
diffusion model
reinforcement learning
MCMC posterior sampling
Node2Vec random walk
stochastic gradient Langevin dynamics
simulated annealing
PageRank
```

---

### 1.1 数学概念历史出现缘由

随机过程最初是为了描述“随机变量随时间变化”的系统。  
它连接概率论、统计物理、金融数学、信号处理、贝叶斯推断和生成式 AI。

| 数学 / 工程概念 | 出现缘由 |
|---|---|
| 随机过程 | 为了描述随时间演化的随机系统 |
| 随机游走 | 为了研究一步步随机移动的过程 |
| `sqrt(n)` 位移缩放 | 为了解释独立随机增量累加后的扩散速度 |
| Brownian motion | 为了描述连续时间极限下的随机游走 |
| Markov property | 为了描述“未来只依赖当前，不依赖完整历史”的系统 |
| Markov chain | 为了用转移概率描述离散状态随机演化 |
| 转移矩阵 | 为了用矩阵形式表示状态转移概率 |
| 平稳分布 | 为了描述 Markov chain 长期运行后的状态分布 |
| 吸收态 | 为了描述进入后无法离开的终止状态 |
| mixing time | 为了衡量 Markov chain 多快忘记初始状态 |
| spectral gap | 为了用特征值衡量 mixing 速度 |
| Langevin dynamics | 为了把梯度下降和随机噪声结合，用于采样 |
| MCMC | 为了从复杂后验分布中采样 |
| Metropolis-Hastings | 为了用接受/拒绝机制构造目标平稳分布 |
| detailed balance | 为了保证 Markov chain 的平稳分布正确 |
| diffusion process | 为了描述逐步加噪和逐步去噪的生成过程 |
| SGLD | 为了把 SGD 和 Langevin dynamics 结合，近似贝叶斯采样 |

可以这样理解：

```txt
现实问题：随机系统如何一步步演化？
数学抽象：随机过程

现实问题：每一步随机左/右走，长期会离原点多远？
基础模型：随机游走

现实问题：未来只依赖当前状态，该如何建模？
数学模型：Markov chain

现实问题：Markov chain 长期会停在哪个分布？
核心概念：stationary distribution

现实问题：如何从复杂 posterior 中采样？
算法工具：MCMC / Metropolis-Hastings

现实问题：diffusion model 为什么能从噪声生成图像？
过程解释：学习反转一个逐步加噪 Markov chain
```

---

### 1.2 原始问题是什么

本课要解决的原始问题是：

```txt
什么是随机过程？
随机游走为什么位移标准差按 sqrt(n) 增长？
随机游走和 Brownian motion 有什么关系？
什么是 Markov property？
Markov chain 如何用 transition matrix 表示？
什么是 stationary distribution？
stationary distribution 如何用 eigendecomposition 计算？
Markov chain 什么时候会收敛到唯一平稳分布？
什么是 absorbing state？
mixing time 和 spectral gap 有什么关系？
Langevin dynamics 为什么是 gradient descent + noise？
MCMC 如何用 Markov chain 从目标分布中采样？
Metropolis-Hastings 为什么只需要目标分布的比例？
Diffusion model 的 forward process 为什么是 Markov chain？
Reverse process 为什么是 learned Markov chain？
SGLD 如何把优化和采样结合？
```

换成 AI 语言，就是：

```txt
LLM 生成 token 为什么可以看成随机过程？
temperature 调整的是什么随机性？
diffusion model 为什么从纯噪声一步步生成图像？
MCMC 为什么能采样 Bayesian posterior？
RL 的 MDP 和 Markov chain 有什么关系？
Node2Vec 为什么用 random walk 生成节点序列？
SGLD 为什么能给神经网络提供不确定性估计？
```

---

### 1.3 AI 中的现代问题

| 随机过程知识点 | AI 中的现代问题 |
|---|---|
| Random walk | Node2Vec、RL exploration、图表示学习 |
| `sqrt(n)` scaling | SGD noise、随机误差缩放 |
| Brownian motion | diffusion forward process、SDE model |
| Markov chain | LLM token generation、MCMC、PageRank |
| Transition matrix | 离散状态系统、文本生成、天气模型 |
| Stationary distribution | MCMC 目标分布、PageRank 长期分布 |
| Spectral gap | Markov chain mixing speed |
| Absorbing state | EOS token、RL terminal state |
| Langevin dynamics | score-based model、SGLD |
| Metropolis-Hastings | Bayesian posterior sampling |
| Detailed balance | MCMC 正确性保证 |
| Diffusion process | DDPM、score-based generative model |
| Reverse Markov chain | diffusion generation |
| Temperature | LLM sampling、Boltzmann exploration |
| MDP | reinforcement learning |
| SGLD | Bayesian deep learning、uncertainty |

---

### 1.4 本课要解决的核心问题

学完本课，你应该能够回答：

1. 随机过程和普通随机变量有什么区别？
2. 1D random walk 的期望位置和标准差分别是多少？
3. 为什么随机游走的典型位移按 `sqrt(n)` 增长？
4. Brownian motion 如何作为随机游走的连续极限？
5. Markov property 是什么？
6. Transition matrix 如何描述 Markov chain？
7. Stationary distribution 满足什么方程？
8. 如何用 power method 和 eigenvector 计算平稳分布？
9. Irreducible 和 aperiodic 分别保证什么？
10. Absorbing state 在 AI 中对应哪些终止状态？
11. Mixing time 为什么由 spectral gap 控制？
12. Langevin dynamics 为什么能从能量分布中采样？
13. Metropolis-Hastings 的 acceptance ratio 如何保证正确分布？
14. Diffusion model 的 forward / reverse process 如何用 Markov chain 理解？
15. SGLD 如何从优化过渡到采样？
16. 随机过程如何连接 LLM、diffusion、RL 和 Bayesian inference？

---

## 2. 概念定位：这节课学什么

---

### 2.1 所属模块

```txt
所属模块：概率统计 / 生成模型 / 强化学习
前置知识：概率分布、条件概率、贝叶斯公式、采样方法、特征值与特征向量
后续连接：MCMC、diffusion model、score-based model、RL、MDP、SGLD、PageRank
```

前面的概率和采样课程已经解决了：

```txt
如何描述一个分布？
如何从一个分布中采样？
```

本课进一步解决：

```txt
如果随机变量随时间一步步变化，应该如何建模和分析？
```

---

### 2.2 本课学习目标

完成本课后，你应该能够：

- 模拟 1D 和 2D random walk
- 验证随机游走位移的 `sqrt(n)` 缩放
- 解释 random walk 和 Brownian motion 的关系
- 实现 Markov chain simulator
- 用 power method 计算 stationary distribution
- 用 eigendecomposition 计算 stationary distribution
- 理解 irreducible、aperiodic、absorbing state 和 mixing time
- 实现 Langevin dynamics
- 实现 Metropolis-Hastings MCMC
- 解释 diffusion forward process 与 Brownian motion 的关系
- 解释 diffusion reverse process 为什么是 learned Markov chain
- 说明 stochastic processes 在 LLM、RL、MCMC 和 diffusion 中的应用

---

### 2.3 本课在整体路线中的位置

```txt
概率与分布
    ↓
贝叶斯公式
    ↓
采样方法
    ↓
随机过程
    ↓
MCMC / Langevin dynamics
    ↓
Diffusion models / RL / Bayesian inference
```

本课的核心能力是：

```txt
看到一个随时间演化的随机系统时，能判断它是不是 Markov chain、是否有平稳分布、如何采样、如何收敛，以及它和现代 AI 系统的关系。
```

---

## 3. 理解概念：直觉、公式、动态过程、AI 连接

---

### 3.1 随机过程：随时间演化的随机变量

**一句话直觉：**  
随机过程是一组按时间排序的随机变量。

可以写成：

```text
X_0, X_1, X_2, ..., X_t
```

每个时刻都有一个随机状态。

例子：

| 随机过程 | 状态 |
|---|---|
| random walk | 当前位置 |
| Markov chain | 当前离散状态 |
| LLM generation | 当前 token / context |
| RL trajectory | 当前环境状态 |
| diffusion | 当前噪声程度下的样本 |
| MCMC | 当前采样位置 |

普通随机变量是一次性的。  
随机过程关心的是：

```txt
随机性如何一步步变化。
```

---

### 3.2 Random walk：最简单的随机过程

**一句话直觉：**  
随机游走每一步随机向左或向右走，长期没有方向偏置，但会越来越远离起点。

1D random walk：

```txt
起点：0
每一步：
  以 0.5 概率 +1
  以 0.5 概率 -1
```

第 n 步位置：

```text
S_n = X_1 + X_2 + ... + X_n
```

其中：

```text
X_i ∈ {-1, +1}
```

期望：

```text
E[S_n] = 0
```

方差：

```text
Var(S_n) = n
```

标准差：

```text
Std(S_n) = sqrt(n)
```

所以：

```txt
随机游走虽然没有 drift，但典型距离会按 sqrt(n) 增长。
```

**AI 连接：**

`sqrt(n)` 缩放出现在很多地方：

```txt
SGD noise ~ 1/sqrt(batch_size)
独立随机误差累积 ~ sqrt(n)
embedding 初始化尺度 ~ 1/sqrt(d)
Monte Carlo 标准误差 ~ 1/sqrt(N)
```

---

### 3.3 2D Random Walk

**一句话直觉：**  
二维随机游走每步随机向上下左右移动，轨迹会像不断扩散的粒子路径。

每一步从四个方向中随机选一个：

```txt
right: (+1, 0)
left:  (-1, 0)
up:    (0, +1)
down:  (0, -1)
```

二维位移也有扩散行为：

```txt
距离原点的典型尺度仍然约为 sqrt(n)。
```

**AI 连接：**

二维随机游走可用于理解：

```txt
图上的 random walk
探索策略
Brownian motion
diffusion noise path
```

---

### 3.4 Brownian Motion：随机游走的连续极限

**一句话直觉：**  
Brownian motion 是步长变小、步数变多后的连续时间随机游走。

离散近似：

```text
B(t + dt) = B(t) + sqrt(dt) * z
```

其中：

```text
z ~ N(0,1)
```

Brownian motion 性质：

```txt
B(0)=0
B(t)-B(s) ~ Normal(0, t-s)
不重叠时间区间的增量独立
路径连续但处处不可导
```

为什么有 `sqrt(dt)`？

```txt
因为方差随时间线性增长，标准差是方差开方。
```

**AI 连接：**

Brownian motion 是 diffusion model 和 SDE-based generative model 的基础直觉。

---

### 3.5 Gambler's Ruin：带吸收边界的随机游走

**一句话直觉：**  
随机游走如果有两个终点边界，一旦到达边界就停止。

设：

```txt
起点 k
左边界 0
右边界 N
```

公平随机游走到达 N 之前先到 0 的概率为：

```text
P(reach N before 0) = k / N
```

这很优雅：

```txt
起点越靠近 N，赢的概率越高。
公平游走的期望未来值等于当前值。
```

这连接到 martingale。

**AI 连接：**

Absorbing state 对应：

```txt
游戏终止状态
RL terminal state
语言模型 EOS token
用户流失 churn state
```

---

### 3.6 Markov Property

**一句话直觉：**  
Markov property 表示未来只依赖当前状态，不依赖完整历史。

公式：

```text
P(X_{t+1}=j | X_t=i, X_{t-1}, ..., X_0)
=
P(X_{t+1}=j | X_t=i)
```

也就是：

```txt
给定当前状态，过去不再提供额外信息。
```

这不是说系统没有历史。  
而是说：

```txt
当前状态已经浓缩了预测未来所需的信息。
```

**AI 连接：**

| 系统 | Markov 状态 |
|---|---|
| RL | 当前环境 state |
| MDP | state 包含决策所需信息 |
| LLM | context 作为当前状态 |
| Diffusion | 当前 noisy sample `x_t` |
| MCMC | 当前样本位置 |

---

### 3.7 Markov Chain 与 Transition Matrix

**一句话直觉：**  
Markov chain 是在离散状态之间按固定概率转移的随机过程。

转移矩阵：

```text
P[i][j] = probability of moving from state i to state j
```

每一行和为 1：

```text
sum_j P[i][j] = 1
```

天气例子：

```text
States: Sunny, Rainy, Cloudy

P = [[0.7, 0.1, 0.2],
     [0.3, 0.4, 0.3],
     [0.4, 0.2, 0.4]]
```

含义：

```txt
如果今天 Sunny，明天 70% Sunny、10% Rainy、20% Cloudy。
```

**AI 连接：**

transition matrix 是理解：

```txt
PageRank
MCMC
离散状态 RL
简单语言模型
diffusion forward process
```

的基础。

---

### 3.8 Stationary Distribution

**一句话直觉：**  
平稳分布是 Markov chain 长期运行后不再变化的状态分布。

定义：

```text
π P = π
```

其中：

| 符号 | 含义 |
|---|---|
| `π` | stationary distribution |
| `P` | transition matrix |

这说明：

```txt
如果当前状态分布是 π，再走一步后仍然是 π。
```

矩阵角度：

```txt
π 是 P 的 left eigenvector，对应 eigenvalue = 1。
```

等价地：

```txt
π 是 P^T 的 right eigenvector，对应 eigenvalue = 1。
```

**AI 连接：**

| 场景 | 平稳分布 |
|---|---|
| MCMC | 目标 posterior |
| PageRank | 网页长期访问概率 |
| Random walk on graph | 节点长期访问频率 |
| Diffusion forward | 高斯噪声极限分布 |

---

### 3.9 计算 Stationary Distribution

**一句话直觉：**  
平稳分布可以通过反复乘转移矩阵，也可以通过特征向量求解。

#### Power method

从任意初始分布开始：

```text
dist_{t+1} = dist_t @ P
```

重复很多次：

```text
dist_t -> π
```

#### Eigenvalue method

求解：

```text
P^T π^T = π^T
```

也就是找到 `P^T` 的 eigenvalue 1 对应的 eigenvector。

---

### 3.10 Markov Chain 收敛条件

**一句话直觉：**  
要收敛到唯一平稳分布，链需要能互相到达，且不能陷入固定周期。

两个常见条件：

| 条件 | 含义 |
|---|---|
| Irreducible | 任意状态都能到达任意其他状态 |
| Aperiodic | 不会以固定周期循环 |

如果满足这些条件，Markov chain 通常会收敛到唯一 stationary distribution。

---

### 3.11 Absorbing State

**一句话直觉：**  
吸收态是一旦进入就不会离开的状态。

定义：

```text
P[i][i] = 1
```

并且：

```text
P[i][j] = 0 for j != i
```

例子：

```txt
游戏结束
用户流失
EOS token
任务完成
系统失败状态
```

**AI 连接：**

RL 中 terminal state 就是吸收态。  
语言模型中的 `<eos>` 也可以看成吸收状态。

---

### 3.12 Mixing Time 与 Spectral Gap

**一句话直觉：**  
mixing time 衡量 Markov chain 多快忘记初始状态；spectral gap 越大，mixing 越快。

转移矩阵特征值中：

```text
最大特征值 = 1
```

第二大特征值绝对值为：

```text
|λ_2|
```

spectral gap：

```text
gap = 1 - |λ_2|
```

gap 大：

```txt
收敛快，mixing time 短。
```

gap 小：

```txt
收敛慢，链容易长时间停在某些区域。
```

**AI 连接：**

MCMC 中 mixing slow 会导致样本高度相关，posterior estimate 不可靠。

---

### 3.13 Langevin Dynamics：梯度下降加噪声

**一句话直觉：**  
Langevin dynamics 是带随机噪声的梯度下降，它不是只找最低点，而是在低能量区域附近采样。

目标分布：

```text
p(x) proportional to exp(-U(x)/T)
```

其中：

| 符号 | 含义 |
|---|---|
| `U(x)` | energy function |
| `T` | temperature |

更新：

```text
x_{t+1} =
x_t - dt * grad U(x_t)
+ sqrt(2Tdt) * z_t
```

其中：

```text
z_t ~ N(0, I)
```

两种力量：

| 项 | 作用 |
|---|---|
| `-dt * grad U` | 往低能量方向走 |
| `sqrt(2Tdt) * z` | 随机探索 |

特殊情况：

```txt
T=0：纯 gradient descent。
T 高：更像 random walk。
```

**AI 连接：**

Langevin dynamics 用于：

```txt
score-based generative models
SGLD
Bayesian neural networks
energy-based models
diffusion sampling
```

---

### 3.14 MCMC 与 Metropolis-Hastings

**一句话直觉：**  
MCMC 构造一个 Markov chain，让它的平稳分布正好是你想采样的目标分布。

目标：

```txt
从 p(x) 采样。
```

但可能：

```txt
p(x) 只知道未归一化形式。
无法直接采样。
```

Metropolis-Hastings 算法：

```txt
1. 当前状态 x。
2. 从 proposal Q(x'|x) 提出新状态 x'。
3. 计算 acceptance ratio。
4. 以 min(1, a) 接受 x'。
5. 否则停留在 x。
```

一般接受率：

```text
a = p(x') Q(x|x') / (p(x) Q(x'|x))
```

如果 proposal 对称：

```text
Q(x'|x) = Q(x|x')
```

则：

```text
a = p(x') / p(x)
```

重要点：

```txt
只需要概率比值。
归一化常数会抵消。
```

**AI 连接：**

MCMC 是 Bayesian inference 的核心工具。  
很多 posterior 无法解析归一化，只能通过 MCMC 采样。

---

### 3.15 Detailed Balance

**一句话直觉：**  
Detailed balance 保证从 x 到 y 的长期流量等于从 y 到 x 的长期流量，因此目标分布保持不变。

条件：

```text
p(x) T(x -> y) = p(y) T(y -> x)
```

如果满足 detailed balance，则 `p` 是 Markov chain 的 stationary distribution。

Metropolis-Hastings 的接受率就是为了满足 detailed balance 而设计的。

---

### 3.16 Diffusion Process：从数据到噪声，再从噪声到数据

**一句话直觉：**  
Diffusion model 的前向过程是不断加噪的 Markov chain，反向过程是学习如何一步步去噪。

DDPM forward process：

```text
q(x_t | x_{t-1}) =
N(x_t; sqrt(1-beta_t) x_{t-1}, beta_t I)
```

直觉：

```txt
每一步把当前样本和一点 Gaussian noise 混合。
经过很多步后，x_T 接近标准高斯噪声。
```

Reverse process：

```text
p_theta(x_{t-1} | x_t) =
N(x_{t-1}; mu_theta(x_t,t), sigma_t^2 I)
```

其中神经网络学习：

```txt
如何从 noisy sample 预测去噪方向或噪声。
```

生成过程：

```txt
从 x_T ~ N(0,I) 开始。
逐步采样 x_{T-1}, x_{T-2}, ..., x_0。
最终得到生成数据。
```

**AI 连接：**

Diffusion model 是 Markov chain 和 Brownian-like noise process 在生成式 AI 中的核心应用。

---

### 3.17 SGLD：优化和采样之间的桥梁

**一句话直觉：**  
SGLD 在 stochastic gradient descent 中加入合适噪声，使训练后期从 optimization 过渡到 posterior sampling。

Stochastic Gradient Langevin Dynamics：

```text
theta_{t+1} =
theta_t - eta_t * grad Loss(theta_t)
+ sqrt(2 eta_t) * noise
```

当学习率逐渐衰减时：

```txt
SGLD 不只是找一个最优点，而是近似从 Bayesian posterior 中采样。
```

**AI 连接：**

SGLD 是获得神经网络不确定性估计的一种简单方法。

---

### 3.18 随机过程与 LLM、RL、Diffusion 的关系

| 随机过程 | AI 应用 |
|---|---|
| Random walk | Node2Vec、图嵌入、RL exploration |
| Markov chain | LLM token generation、MCMC |
| Absorbing state | EOS token、RL terminal state |
| Brownian motion | diffusion forward process |
| Langevin dynamics | score-based generative model |
| MCMC | Bayesian posterior sampling |
| Metropolis-Hastings | posterior sampling、simulated annealing |
| MDP | reinforcement learning |
| Stationary distribution | PageRank、MCMC convergence |
| Temperature | LLM sampling、simulated annealing |

---

### 3.19 本课核心概念总表

| 概念 | 一句话直觉 | AI 中的对应位置 |
|---|---|---|
| Stochastic process | 随时间变化的随机变量 | 生成、RL、MCMC |
| Random walk | 随机一步步移动 | Node2Vec、探索 |
| `sqrt(n)` scaling | 独立随机增量的扩散尺度 | SGD noise、MC error |
| Brownian motion | 连续随机游走 | diffusion、SDE |
| Markov property | 未来只依赖当前 | MDP、MCMC |
| Transition matrix | 状态转移概率表 | Markov chain |
| Stationary distribution | 长期平衡分布 | MCMC、PageRank |
| Absorbing state | 进入后不离开 | EOS、terminal |
| Mixing time | 多久接近平稳 | MCMC 诊断 |
| Spectral gap | 收敛速度指标 | Markov chain |
| Langevin dynamics | 梯度下降 + 噪声 | score model、SGLD |
| Metropolis-Hastings | proposal + accept/reject | Bayesian sampling |
| Detailed balance | 流量平衡 | MCMC 正确性 |
| Diffusion process | 逐步加噪/去噪 | DDPM |
| SGLD | SGD + Langevin noise | Bayesian DL |

---

## 4. 手写实现：随机游走、Markov chain、Langevin 和 MCMC

这一节用 NumPy 从零实现核心算法。

---

### 4.1 1D Random Walk

```python
import numpy as np


def random_walk_1d(n_steps, seed=None):
    rng = np.random.RandomState(seed)

    steps = rng.choice([-1, 1], size=n_steps)

    positions = np.concatenate([
        [0],
        np.cumsum(steps)
    ])

    return positions
```

验证 `sqrt(n)`：

```python
n_steps = 10000
walk = random_walk_1d(n_steps, seed=42)

print("final position:", walk[-1])
print("expected scale:", np.sqrt(n_steps))
print("actual distance:", abs(walk[-1]))
```

---

### 4.2 2D Random Walk

```python
def random_walk_2d(n_steps, seed=None):
    rng = np.random.RandomState(seed)

    directions = rng.choice(4, size=n_steps)

    dx = np.zeros(n_steps)
    dy = np.zeros(n_steps)

    dx[directions == 0] = 1
    dx[directions == 1] = -1
    dy[directions == 2] = 1
    dy[directions == 3] = -1

    x = np.concatenate([
        [0],
        np.cumsum(dx)
    ])

    y = np.concatenate([
        [0],
        np.cumsum(dy)
    ])

    return x, y
```

---

### 4.3 Markov Chain 类

```python
class MarkovChain:
    def __init__(self, transition_matrix, state_names=None):
        self.P = np.array(transition_matrix, dtype=float)
        self.n_states = len(self.P)

        self.state_names = state_names or [
            str(i)
            for i in range(self.n_states)
        ]

    def step(self, current_state, rng=None):
        if rng is None:
            rng = np.random.RandomState()

        probs = self.P[current_state]

        return rng.choice(self.n_states, p=probs)

    def simulate(self, start_state, n_steps, seed=None):
        rng = np.random.RandomState(seed)

        states = [start_state]
        current = start_state

        for _ in range(n_steps):
            current = self.step(current, rng)
            states.append(current)

        return states

    def stationary_distribution(self):
        eigenvalues, eigenvectors = np.linalg.eig(self.P.T)

        idx = np.argmin(np.abs(eigenvalues - 1.0))

        stationary = np.real(eigenvectors[:, idx])
        stationary = stationary / stationary.sum()

        return np.abs(stationary)
```

---

### 4.4 Power method 计算平稳分布

```python
def stationary_distribution_power(P, steps=1000):
    n = P.shape[0]

    dist = np.ones(n) / n

    for _ in range(steps):
        dist = dist @ P

    return dist
```

天气例子：

```python
P = np.array([
    [0.7, 0.1, 0.2],
    [0.3, 0.4, 0.3],
    [0.4, 0.2, 0.4],
])

mc = MarkovChain(
    P,
    state_names=["Sunny", "Rainy", "Cloudy"]
)

print(mc.stationary_distribution())
print(stationary_distribution_power(P))
```

---

### 4.5 Spectral gap

```python
def spectral_gap(P):
    eigenvalues = np.linalg.eigvals(P)

    abs_vals = sorted(
        np.abs(eigenvalues),
        reverse=True
    )

    return 1 - abs_vals[1]
```

示例：

```python
P = np.array([
    [0.9, 0.1],
    [0.3, 0.7],
])

gap = spectral_gap(P)

print("spectral gap:", gap)
print("approx mixing scale:", 1 / gap)
```

---

### 4.6 Langevin Dynamics

```python
def langevin_dynamics(
    grad_U,
    x0,
    dt,
    temperature,
    n_steps,
    seed=None,
):
    rng = np.random.RandomState(seed)

    x = np.array(x0, dtype=float)
    trajectory = [x.copy()]

    for _ in range(n_steps):
        noise = rng.randn(*x.shape)

        x = (
            x
            - dt * grad_U(x)
            + np.sqrt(2 * temperature * dt) * noise
        )

        trajectory.append(x.copy())

    return np.array(trajectory)
```

双井势能：

```python
def double_well_U_grad(x):
    # U(x) = (x^2 - 1)^2
    return 4 * x * (x ** 2 - 1)


traj = langevin_dynamics(
    grad_U=double_well_U_grad,
    x0=np.array([0.0]),
    dt=0.01,
    temperature=0.5,
    n_steps=10000,
    seed=42,
)

print(traj[-5:])
```

---

### 4.7 Metropolis-Hastings

```python
def metropolis_hastings(
    target_log_prob,
    proposal_std,
    x0,
    n_samples,
    seed=None,
):
    rng = np.random.RandomState(seed)

    x = np.array(x0, dtype=float)
    samples = [x.copy()]
    accepted = 0

    for _ in range(n_samples - 1):
        x_proposed = x + rng.randn(*x.shape) * proposal_std

        log_ratio = (
            target_log_prob(x_proposed)
            - target_log_prob(x)
        )

        if np.log(rng.rand()) < log_ratio:
            x = x_proposed
            accepted += 1

        samples.append(x.copy())

    acceptance_rate = accepted / (n_samples - 1)

    return np.array(samples), acceptance_rate
```

目标分布例子：

```python
def target_log_prob(x):
    # standard normal up to constant
    return -0.5 * np.sum(x ** 2)


samples, acc = metropolis_hastings(
    target_log_prob=target_log_prob,
    proposal_std=1.0,
    x0=np.array([5.0]),
    n_samples=10000,
    seed=42,
)

print("acceptance rate:", acc)
print("sample mean:", samples.mean())
print("sample std:", samples.std())
```

---

### 4.8 Forward diffusion process

```python
def forward_diffusion_1d(x0, betas, seed=None):
    rng = np.random.RandomState(seed)

    xs = [np.array(x0, dtype=float)]

    x = np.array(x0, dtype=float)

    for beta in betas:
        noise = rng.randn(*x.shape)

        x = (
            np.sqrt(1 - beta) * x
            + np.sqrt(beta) * noise
        )

        xs.append(x.copy())

    return np.array(xs)
```

使用：

```python
signal = np.sin(np.linspace(0, 2 * np.pi, 128))

betas = np.linspace(1e-4, 0.02, 100)

noisy_path = forward_diffusion_1d(
    signal,
    betas,
    seed=42
)

print(noisy_path.shape)
```

---

## 5. 生产使用：NumPy、PyMC、diffusers 和 RL 框架

---

### 5.1 NumPy 模拟 Markov chain

```python
import numpy as np

P = np.array([
    [0.7, 0.1, 0.2],
    [0.3, 0.4, 0.3],
    [0.4, 0.2, 0.4],
])

distribution = np.array([1.0, 0.0, 0.0])

for _ in range(100):
    distribution = distribution @ P

print(np.round(distribution, 4))
```

---

### 5.2 MCMC 生产库

真实 Bayesian inference 不建议手写 MCMC。常见工具：

| 库 | 用途 |
|---|---|
| PyMC | Bayesian modeling，NUTS sampler |
| NumPyro | JAX 加速 MCMC |
| Stan | 概率编程和 HMC/NUTS |
| emcee | ensemble MCMC |
| TensorFlow Probability | TensorFlow 生态概率建模 |

---

### 5.3 Diffusion 框架中的 Markov chain

Hugging Face diffusers 中的 scheduler 负责：

```txt
forward noise schedule
reverse denoising schedule
alpha / beta 系列参数
采样 step 更新
```

DDPM 前向过程：

```text
q(x_t | x_{t-1})
=
N(x_t; sqrt(1-beta_t) x_{t-1}, beta_t I)
```

这正是本课的 Markov chain 思想。

---

### 5.4 RL 中的 Markov Decision Process

MDP 在 Markov chain 上加入 action 和 reward：

```text
P(s_{t+1} | s_t, a_t)
```

组成：

| 元素 | 含义 |
|---|---|
| `S` | state space |
| `A` | action space |
| `P` | transition dynamics |
| `R` | reward |
| `π(a|s)` | policy |

RL 的核心就是：

```txt
在随机环境中选择动作，使长期回报最大。
```

---

### 5.5 手写实现 vs 生产库

| 对比项 | 手写实现 | 生产库 |
|---|---|---|
| Random walk | NumPy cumsum | 自定义即可 |
| Markov chain | transition matrix | NumPy / scipy |
| Stationary distribution | eigen / power method | scipy eigensolver |
| MCMC | 手写 MH | PyMC / Stan / NumPyro |
| Langevin | 手写 update | score-based model 框架 |
| Diffusion | 手写 forward noising | diffusers scheduler |
| RL MDP | 概念建模 | Gymnasium / RLlib |
| SGLD | 手写 optimizer | PyTorch optimizer extension |

---

## 6. 交付沉淀：产出物、连接关系、练习、术语表

---

### 6.1 本课产出

```txt
code/stochastic_processes.py
code/use_numpy_markov.py
code/diffusion_forward.py
outputs/prompt-stochastic-process-advisor.md
outputs/exercises-solutions.md
```

| 文件 | 作用 |
|---|---|
| `code/stochastic_processes.py` | 从零实现 random walk、MarkovChain、Langevin、Metropolis-Hastings |
| `code/use_numpy_markov.py` | NumPy transition matrix、stationary distribution、spectral gap |
| `code/diffusion_forward.py` | 1D forward diffusion process 示例 |
| `outputs/prompt-stochastic-process-advisor.md` | 判断问题适合哪类 stochastic process 的 prompt |
| `outputs/exercises-solutions.md` | 练习题与参考答案 |

---

### 6.2 知识连接关系

| 本课知识点 | 后续连接 |
|---|---|
| Random walk | Brownian motion、Node2Vec |
| Markov property | MDP、MCMC、diffusion |
| Transition matrix | Markov chain、PageRank |
| Stationary distribution | MCMC convergence、PageRank |
| Spectral gap | mixing time |
| Langevin dynamics | score-based generative model |
| Metropolis-Hastings | Bayesian inference |
| Detailed balance | MCMC 正确性 |
| Brownian motion | diffusion SDE |
| Forward diffusion | DDPM |
| Reverse process | generative sampling |
| SGLD | Bayesian deep learning |
| Absorbing state | RL terminal、EOS token |

---

### 6.3 AI 应用连接

| 随机过程 | AI 应用 |
|---|---|
| Random walk | Node2Vec、图嵌入、RL 探索 |
| Markov chain | LLM token generation、PageRank、MCMC |
| Brownian motion | diffusion forward noising |
| Langevin dynamics | score-based model、SGLD |
| MCMC | Bayesian posterior sampling |
| Metropolis-Hastings | posterior sampling、simulated annealing |
| Stationary distribution | MCMC 目标、PageRank |
| Mixing time | MCMC 诊断 |
| Absorbing state | EOS token、terminal state |
| MDP | reinforcement learning |
| Diffusion process | DDPM、图像生成 |
| Temperature | LLM sampling、Boltzmann exploration |

---

### 6.4 练习

1. **模拟 1000 条随机游走。**  
   每条 10,000 步，记录最终位置。  
   画出最终位置分布，验证它近似：

   ```text
   Normal(0, 100^2)
   ```

   因为 `sqrt(10000)=100`。

2. **用 Markov chain 做文本生成。**  
   给定一个小语料，对每个词统计下一个词的转移概率。  
   构建 transition matrix，用采样方式生成新句子。

3. **实现 simulated annealing。**  
   用 Metropolis-Hastings，在高温时接受很多移动，逐渐降温。  
   用它寻找一个多局部最小值函数的全局低点。

4. **比较不同温度下的 Langevin dynamics。**  
   目标势能：

   ```text
   U(x) = (x^2 - 1)^2
   ```

   低温时样本可能卡在一个 well。  
   高温时样本可以跨越两个 well。  
   找到开始频繁跨井的温度范围。

5. **实现 forward diffusion。**  
   从一个 1D sine wave 开始，用 100 步线性 noise schedule 逐步加噪。  
   展示信号如何逐渐变成纯噪声。  
   尝试写一个简单 denoiser 反向恢复信号。

6. **Markov chain mixing time。**  
   构造两个 2-state Markov chain，一个 spectral gap 大，一个 spectral gap 小。  
   从同一初始状态出发，比较它们收敛到 stationary distribution 的速度。

---

### 6.5 关键术语

| 术语 | 常见说法 | 真正含义 |
|---|---|---|
| Stochastic process | 随机过程 | 随时间演化的一组随机变量 |
| Random walk | 随机游走 | 每步随机移动的过程 |
| Markov property | 无记忆性 | 未来只依赖当前状态，不依赖完整历史 |
| Transition matrix | 转移矩阵 | `P[i][j]` 表示从 i 到 j 的转移概率 |
| Stationary distribution | 平稳分布 | 满足 `πP=π` 的长期分布 |
| Brownian motion | 布朗运动 | 随机游走的连续时间极限 |
| Langevin dynamics | Langevin 动力学 | 梯度下降加随机噪声 |
| MCMC | 马尔可夫链蒙特卡洛 | 构造目标平稳分布的 Markov chain |
| Metropolis-Hastings | MH 算法 | proposal + accept/reject 的 MCMC |
| Temperature | 温度 | 控制探索和随机性 |
| Absorbing state | 吸收态 | 进入后不会离开的状态 |
| Mixing time | 混合时间 | 接近平稳分布所需步数 |
| Spectral gap | 谱间隙 | `1-|λ2|`，控制混合速度 |
| Detailed balance | 细致平衡 | 保证目标分布为平稳分布的条件 |
| Diffusion process | 扩散过程 | forward 加噪、reverse 去噪 |
| SGLD | 随机梯度 Langevin | SGD 加噪，用于近似后验采样 |
| MDP | 马尔可夫决策过程 | 带 action 和 reward 的 Markov process |

---

### 6.6 自检问题

学完本课后，你应该能回答：

- 随机过程和普通随机变量有什么区别？
- 随机游走为什么期望位置为 0，但典型距离增长为 `sqrt(n)`？
- Brownian motion 如何从 random walk 极限得到？
- Markov property 为什么重要？
- Transition matrix 每一行为什么要和为 1？
- Stationary distribution 满足什么方程？
- 如何用 power method 和 eigenvector 求平稳分布？
- Irreducible 和 aperiodic 分别有什么作用？
- Absorbing state 在 LLM 和 RL 中有什么对应？
- Spectral gap 如何影响 mixing time？
- Langevin dynamics 如何结合梯度和噪声？
- Metropolis-Hastings 为什么只需要概率比？
- Detailed balance 如何保证 MCMC 正确性？
- Diffusion forward process 为什么是 Markov chain？
- Reverse diffusion 为什么是 learned Markov chain？
- SGLD 如何连接 SGD 和 Bayesian posterior sampling？
- 随机过程如何贯穿 LLM、diffusion、RL 和 MCMC？

---

## 附录：本课最小代码文件建议

如果要拆成代码文件，可以这样组织：

```txt
code/
├── stochastic_processes.py
├── use_numpy_markov.py
└── diffusion_forward.py
```

### `stochastic_processes.py`

包含：

```txt
random_walk_1d
random_walk_2d
MarkovChain
stationary_distribution_power
spectral_gap
langevin_dynamics
metropolis_hastings
forward_diffusion_1d
```

### `use_numpy_markov.py`

包含：

```txt
transition matrix simulation
stationary distribution via power method
stationary distribution via eigendecomposition
spectral gap and mixing scale
absorbing state examples
```

### `diffusion_forward.py`

包含：

```txt
linear beta schedule
forward diffusion on 1D signal
noise progression visualization
simple denoising baseline
```
