# Research Keywords — Mar 22

现在可以 research 的方向，按项目 Part 分类。每个 keyword 附带搜索建议和预期产出。

---

## 1. Implicit FD / Crank-Nicolson for HJB (Part B 升级, P2)

当前 explicit FD 对 nonlinear impact 不稳定。需要 implicit scheme。

| Keyword | 搜索什么 |
|---------|---------|
| `Crank-Nicolson HJB optimal execution` | 如何把 CN 应用到含 min_v 的 HJB |
| `implicit finite difference nonlinear PDE Python` | 实现参考 |
| `policy iteration HJB` | 替代 grid search 的迭代方法 |
| `Howard algorithm HJB` | 经典 HJB policy iteration 算法 |
| `penalty method terminal condition PDE` | 处理大 terminal penalty 的技巧 |

**预期产出**: 一个稳定的 nonlinear HJB solver，能处理 alpha ≠ 1

---

## 2. Heston Stochastic Vol + Execution (Part D, All)

将 constant sigma 替换为 Heston process，观察 stochastic vol 如何改变 optimal trajectory。

| Keyword | 搜索什么 |
|---------|---------|
| `optimal execution stochastic volatility Heston` | 核心论文——有没有 closed-form 或 known results |
| `2D HJB PDE finite difference inventory variance` | 2D PDE 网格方法 (x × v state space) |
| `Heston SDE Euler-Maruyama correlated Brownian` | 如何离散化 2D correlated SDE |
| `Heston calibration crypto options Deribit` | crypto 市场 Heston 参数的典型范围 |
| `ADI scheme 2D PDE` | Alternating Direction Implicit — 2D PDE 标准方法 |
| `Feller condition Heston xi kappa theta` | 参数约束: 2κθ > ξ² 保证 variance 不为零 |

**预期产出**: 理解 2D PDE 的数值方法选择，Heston MC 的实现路径

---

## 3. Regime Detection / HMM (Part E, All)

用 Hidden Markov Model 把市场分成 risk-on / risk-off，per-regime 求解不同的 optimal trajectory。

| Keyword | 搜索什么 |
|---------|---------|
| `hmmlearn GaussianHMM Python tutorial` | 最主流的 HMM 库，如何 fit 2-state model |
| `regime switching optimal execution` | 有没有已有的 regime-aware execution 理论 |
| `HMM volatility regime crypto Bitcoin` | crypto 上的 regime detection 实例 |
| `Baum-Welch algorithm implementation` | 如果不用 hmmlearn，手写 EM |
| `PELT changepoint detection` | 替代 HMM 的 changepoint 方法 (Plan 提过) |
| `regime dependent market impact` | risk-off 时 spread 和 impact 如何变化 |

**预期产出**: 选定 HMM vs PELT，确定 feature set (returns? vol? spread?)

---

## 4. Market Impact Calibration from Real Data (Part A, P1)

从 Binance order book 提取真实 market impact 参数。

| Keyword | 搜索什么 |
|---------|---------|
| `Binance historical data download free` | data.binance.vision aggTrades + depth |
| `Kyle lambda estimation tick data` | 从 trade data 估计 permanent impact |
| `power law market impact crypto` | temporary impact 的 power-law 拟合方法 |
| `Almgren Chriss calibration real data` | 有没有人用真实数据 calibrate 过 A-C 参数 |
| `order book depth slippage estimation` | walk-the-book 方法论 |
| `Tardis.dev order book data` | 另一个 data source（如果 Binance 不够） |
| `realized volatility 5-minute crypto` | 用 tick data 计算 sigma 的标准做法 |

**预期产出**: 数据下载 pipeline，参数估计结果 (gamma, eta, alpha, sigma)

---

## 5. Variance Reduction 进阶 (Part C 加分, P3)

当前有 antithetic + control variate。可以再加。

| Keyword | 搜索什么 |
|---------|---------|
| `importance sampling optimal execution` | 针对 tail risk 的 IS |
| `stratified sampling Monte Carlo` | 另一种方差减少技术 |
| `quasi Monte Carlo Sobol sequence` | 低差异序列替代伪随机 |
| `Milstein scheme SDE higher order` | 比 Euler-Maruyama 更精确的 SDE 离散化 |

**预期产出**: 判断是否值得加，如果加哪个 ROI 最高

---

## 6. 报告 & Presentation (写作准备)

| Keyword | 搜索什么 |
|---------|---------|
| `Almgren Chriss execution trajectory visualization` | 别人是怎么画 trajectory 对比图的 |
| `execution cost heatmap lambda sigma` | cost 关于 risk aversion × volatility 的 heatmap |
| `optimal execution animated trajectory` | 动画展示执行过程 |
| `MF796 computational finance final project examples` | 看看同课程之前的项目长什么样 |

**预期产出**: presentation 视觉设计灵感

---

## 7. Nonlinear Impact Theory (Part B 理论, P2)

当前只做了 alpha=1 (linear)。Almgren (2003) 扩展到 nonlinear。

| Keyword | 搜索什么 |
|---------|---------|
| `Almgren 2003 nonlinear impact execution` | 原始论文 |
| `concave market impact optimal execution` | alpha < 1 时的最优执行特性 |
| `Gatheral no dynamic arbitrage market impact` | impact 函数的理论约束 |
| `transient market impact Bouchaud` | 暂态冲击模型（更现实但更复杂） |

**预期产出**: 理解 nonlinear impact 对 trajectory shape 的影响，为 FD solver 提供 test cases

---

## 优先级排序

| 紧急度 | 方向 | 理由 |
|--------|------|------|
| **现在就做** | #4 Market Impact Calibration | P1 blocking — 没有真实数据其他都是 synthetic |
| **现在就做** | #1 Implicit FD | P2 核心 deliverable — explicit 不稳定 |
| **本周内** | #2 Heston + Execution | Part D 的理论基础，需要先读论文再写代码 |
| **本周内** | #3 Regime / HMM | Part E 相对简单，但需要选方法 |
| **下周** | #6 报告 & Presentation | 4 月中旬开始写不迟 |
| **可选** | #5 Variance Reduction 进阶 | 锦上添花，不影响核心交付 |
| **可选** | #7 Nonlinear Impact | 理论深度加分，但 linear case 已经足够 |
