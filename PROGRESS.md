# MF796 项目进度

## 已完成

### 共享基础设施 (`shared/`)
- `ACParams` dataclass — 全队参数契约，包含 market impact、risk aversion、volume profile
- `almgren_chriss_closed_form()` — Almgren-Chriss closed-form optimal trajectory + expected cost
- `execution_cost()` / `execution_risk()` / `objective()` — permanent + temporary impact cost model
- `DEFAULT_PARAMS` — synthetic 参数 (kappa*T=1.5)，P1 calibration 完成后一行替换
- Plotting utilities — trajectory 对比图 + MC cost distribution 图

### Part B: HJB PDE Solver (`pde/`)
- **Riccati ODE solver** (linear impact, alpha=1) — 用 `scipy.integrate.solve_ivp` 求解，与 analytical solution 误差 <5%
- **Explicit FD solver** (nonlinear impact, alpha≠1) — grid search 优化 v*，已知 explicit scheme 稳定性限制
- `extract_optimal_trajectory()` — 从 value function 提取最优 inventory path
- `analytical_value_function()` — V(x,t) = eta·kappa·coth(kappa·(T-t))·x² 用于 validation

### Part C: Monte Carlo Engine (`montecarlo/`)
- **Euler-Maruyama SDE simulation** — exact log-normal discretization（保证正价格）
- **Implementation shortfall** 计算 — IS = Σ n_k·(S₀ - S_k) + h_k·n_k
- **Antithetic variates** — 方差减少 ~30-50%
- **Control variate** — 用 TWAP cost 作为 control，进一步降低 estimator variance
- **TWAP / VWAP / Optimal** 三种 strategy trajectory 生成器
- `CostMetrics` — VaR₉₅, CVaR₉₅, mean, std 统计

### Part D/E 基础 (`extensions/`)
- `heston_cf()` — Heston characteristic function（从 HW2 移植，已验证）
- `fft_call_price()` — Carr-Madan FFT 定价（Part D calibration 基础）
- `HestonParams` dataclass + `RegimeParams` dataclass — 接口已定义
- Skeleton: `calibrate_heston()`, `fit_hmm()`, `regime_aware_params()`

### Part A 基础 (`calibration/`)
- `estimate_kyle_lambda()` — Welford online 协方差算法（从 QC_Trade_Platform 移植）
- Skeleton: `load_trades()`, `load_orderbook_snapshots()`, `walk_the_book_slippage()`, `calibrated_params()`

### 测试 & 质量
- **53 tests** (52 passed, 1 skipped) — closed-form、PDE vs analytical、MC vs deterministic 三角互验
- 3-agent 系统级审计已完成：docstring 修正、log-normal GBM、control variate bug 修复
- GitHub: https://github.com/Eva-Qi/HJB-PDE-TEAM-PROJECT

## 待做

| 优先级 | 任务 | 负责 |
|--------|------|------|
| 高 | 下载 Binance 数据 + 实现 `calibrated_params()` | P1 |
| 高 | FD solver 改 implicit scheme（支持 nonlinear impact 稳定求解） | P2 |
| 中 | Heston stochastic vol 扩展（2D SDE + calibration） | All |
| 中 | HMM regime detection + per-regime optimal trajectory | All |
| 低 | Demo notebook + presentation 可视化 | All |
