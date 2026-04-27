# Heston Stochastic Volatility in Optimal Execution

Research for MF796 Course Project — Part D Extension

---

## 1. How Stochastic Volatility Changes the Execution Problem

### 1.1 Intuition

In the standard Almgren-Chriss framework, volatility `sigma` is a constant scalar baked into the risk cost:

```
Risk cost = lambda * sigma^2 * integral_0^T x(t)^2 dt
```

The trader knows the volatility environment in advance and the optimal trajectory is deterministic — a fixed sinh-shaped schedule.

With Heston stochastic volatility, `sigma^2` is replaced by `v_t`, a random process. The key consequences are:

1. **Variance is a second state variable.** The optimal trading rate now depends not only on remaining inventory `x` but also on current variance `v`. When variance spikes, the agent faces two competing pressures: trade faster to reduce risk exposure, but also recognize that high variance may mean-revert. The optimal policy is a surface `v*(x, v, t)` rather than a curve `v*(x, t)`.

2. **The risk penalty becomes stochastic.** The market risk cost at time `t` for holding `x_t` shares is `lambda * v_t * x_t^2`. Under stochastic vol, this is itself uncertain, so the agent must hedge against variance fluctuations in the value function.

3. **Leverage effect (rho < 0) matters.** Negative correlation between price and variance means price drops tend to coincide with rising volatility — a "double hit" for a liquidation strategy. This creates an asymmetric hedging motive: the agent wants to trade faster when both inventory AND volatility are high.

4. **No closed-form trajectory.** The sinh-shaped closed form of Almgren-Chriss relied on constant coefficients in the HJB. With stochastic `v_t`, the PDE has variable coefficients that depend on both `(x, v)`, and a numerical 2D PDE or simulation is required.

### 1.2 Modified Dynamics

The price and variance processes under Heston are:

```
dS_t = mu * S_t * dt + sqrt(v_t) * S_t * dW_S
dv_t = kappa * (theta - v_t) * dt + xi * sqrt(v_t) * dW_v
corr(dW_S, dW_v) = rho * dt
```

For the **execution problem**, the relevant dynamics are those of log-returns (drift is typically set to zero over the short execution horizon). The market impact enters as a separate term:

```
dS_t = sqrt(v_t) * S_t * dW_S - gamma * v_t_dt   (permanent impact)
```

The inventory evolves deterministically given a control:

```
dx_t = -v_t_control * dt
```

where `v_t_control` (trading rate, not variance) is the control variable. Note: we use `u_t` below for the control (trading rate) to avoid confusion with variance `v`.

### 1.3 Parameter Roles in Execution Context

| Parameter | Role in Heston | Role in Execution |
|-----------|---------------|-------------------|
| `kappa`   | Mean-reversion speed of variance | Fast kappa → variance returns to theta quickly; agent can rely on "average vol" for longer time horizons |
| `theta`   | Long-run variance | Determines the baseline risk level; equivalent to `sigma^2` in Almgren-Chriss |
| `xi`      | Vol-of-vol | Measures uncertainty about future variance; higher xi → more hedging motive in HJB |
| `rho`     | Price-vol correlation | Negative rho (leverage) amplifies the urgency to trade when variance is high |
| `v0`      | Initial variance | Starting point of the 2D state space |

---

## 2. 2D HJB Formulation

### 2.1 Value Function Definition

Define the value function:

```
V(x, v, t) = inf_{u_s, t <= s <= T} E_t[ integral_t^T (eta * u_s^2 + lambda * v_s * x_s^2) ds + phi * x_T^2 ]
```

where:
- `x_s` = remaining inventory at time `s`
- `v_s` = variance at time `s` (Heston process)
- `u_s` = trading rate (control), with `dx_s = -u_s * ds`
- `eta` = temporary impact coefficient
- `lambda` = risk aversion
- `phi` = terminal penalty for leftover inventory

The state space is `(x, v, t) in [0, X0] x [v_min, v_max] x [0, T]`.

### 2.2 The 2D HJB PDE

Applying Ito's lemma to `V(x_t, v_t, t)` and the dynamic programming principle, the HJB equation is:

```
-V_t + min_{u >= 0} { eta * u^2 - u * V_x } + lambda * v * x^2
    + kappa * (theta - v) * V_v
    + (1/2) * xi^2 * v * V_vv
    + rho * xi * sqrt(v) * [sqrt(v) * S * (...)] * V_xv  ... (see note)
= 0
```

**More precisely**, since the inventory `x` evolves deterministically (`dx = -u dt`), there is **no diffusion term in x** and hence no `V_xx` term and no `V_xv` cross term directly from Ito's lemma on `x`. The full 2D HJB is:

```
V_t(x, v, t) + min_{u >= 0} { eta * u^2 - u * V_x(x, v, t) }
             + lambda * v * x^2
             + kappa * (theta - v) * V_v(x, v, t)
             + (1/2) * xi^2 * v * V_vv(x, v, t)
             = 0
```

**with terminal condition:** `V(x, v, T) = phi * x^2` for all `v`.

**Boundary conditions:**
- `V(0, v, t) = 0` for all `v, t` (no inventory, no cost)
- `V(X0, v, t)`: treated as the "worst case" — use large penalty or Neumann BC `V_x = 0` at `x = X0`
- `V(x, v_min, t)`: for `v_min = 0`, use Feller boundary (see Section 5)
- `V(x, v_max, t)`: Dirichlet or Neumann depending on truncation choice

### 2.3 Optimal Control from the HJB

The minimization over `u` inside the HJB is a simple quadratic problem:

```
min_{u >= 0} { eta * u^2 - u * V_x }
```

Setting the first-order condition: `2 * eta * u - V_x = 0`, so:

```
u*(x, v, t) = max(0, V_x(x, v, t) / (2 * eta))
```

The optimal trading rate is the positive part of `V_x / (2 * eta)`. For liquidation, `V_x > 0` everywhere (marginal cost of holding inventory is positive), so the constraint `u >= 0` is never binding in the interior.

Substituting back:

```
min_{u >= 0} { eta * u^2 - u * V_x } = -(V_x)^2 / (4 * eta)    [when V_x > 0]
```

So the PDE simplifies to the **reduced HJB**:

```
V_t - (V_x)^2 / (4 * eta)
    + lambda * v * x^2
    + kappa * (theta - v) * V_v
    + (1/2) * xi^2 * v * V_vv
    = 0
```

This is a nonlinear PDE (due to the `(V_x)^2` term). The nonlinearity means standard linear PDE solvers cannot be applied directly.

### 2.4 Separable Ansatz (Approximate)

If we guess the form `V(x, v, t) = A(v, t) * x^2` (consistent with quadratic value function in `x`), then:

```
V_x = 2 * A(v, t) * x
V_vv = A_vv(v, t) * x^2
V_v = A_v(v, t) * x^2
V_t = A_t(v, t) * x^2
```

Substituting into the reduced HJB:

```
A_t * x^2 - (2 * A * x)^2 / (4 * eta) + lambda * v * x^2 + kappa * (theta - v) * A_v * x^2 + (1/2) * xi^2 * v * A_vv * x^2 = 0
```

Dividing by `x^2`:

```
A_t - A^2 / eta + lambda * v + kappa * (theta - v) * A_v + (1/2) * xi^2 * v * A_vv = 0
```

This is a **1D nonlinear PDE in `(v, t)`** for the scalar function `A(v, t)`, which is much cheaper to solve than the full 2D problem. The optimal control becomes:

```
u*(x, v, t) = A(v, t) * x / eta
```

This has the same structure as the Almgren-Chriss solution `u*(x, t) = alpha(t) * x / eta`, but now `A` depends on variance `v` as well as time `t`.

### 2.5 Full 2D vs Separable Ansatz

| Approach | Grid | Cost | Accuracy |
|----------|------|------|----------|
| Full 2D HJB on `(x, v, t)` | `M_x * M_v * N` | O(M_x * M_v * N) | Exact (within discretization) |
| Separable ansatz `A(v, t) * x^2` | `M_v * N` | O(M_v * N) | Exact if quadratic in `x` holds |
| Constant-vol AC closed form | Scalar ODE | O(N) | Exact only for constant `sigma` |

For linear impact (`eta * u^2`), the quadratic ansatz is exact. For nonlinear impact (`eta * |u|^(alpha+1)` with `alpha != 1`), the full 2D grid is needed.

---

## 3. Numerical Methods for 2D PDE

### 3.1 Why Standard Explicit Euler Fails

The reduced HJB for `A(v, t)` (separable ansatz) is a 1D PDE in variance `v`. It has:
- A diffusion term: `(1/2) * xi^2 * v * A_vv`
- A drift term: `kappa * (theta - v) * A_v`
- A nonlinear reaction term: `-A^2 / eta`
- A forcing term: `lambda * v`

For the full 2D problem `V(x, v, t)`, the `V_vv` diffusion term in `v` introduces a CFL constraint:

```
dt <= (dv)^2 / (xi^2 * v_max)
```

With `xi ~ 0.5` (crypto), `v_max ~ 4 * theta`, and `dv ~ 0.01`, this gives `dt_max ~ 0.01^2 / (0.25 * 4 * theta)`. For `theta = 0.09` (30% vol), `dt_max ~ 0.00028`, meaning thousands of time steps for a 1-day execution horizon. Explicit Euler is impractical.

### 3.2 Crank-Nicolson for 1D Separable Problem

For the `A(v, t)` PDE, a standard Crank-Nicolson scheme works well:

**Discretize** `v` on a grid `v_0 < v_1 < ... < v_{M_v}` and time backward from `A(v, T) = phi` (terminal penalty coefficient).

At each time step, the linear diffusion and drift terms can be handled implicitly (tridiagonal system), while the nonlinear `A^2 / eta` term is handled explicitly or via Newton iteration.

**Procedure (semi-implicit):**
1. Compute the nonlinear term explicitly: `NL_j = A_j^{n+1} * A_j^n / eta` (Picard linearization)
2. Form the tridiagonal system for the implicit diffusion-drift terms
3. Solve the tridiagonal system
4. Iterate until convergence (usually 2-3 Picard iterations)

The tridiagonal coefficients for interior point `j` using central differences:

```
lower[j] = -0.5 * dt * [ kappa*(theta - v_j)/(2*dv) - (1/2)*xi^2*v_j/(dv^2) ]
diag[j]  = 1 + 0.5 * dt * xi^2 * v_j / (dv^2)
upper[j] = -0.5 * dt * [ -kappa*(theta - v_j)/(2*dv) - (1/2)*xi^2*v_j/(dv^2) ]
```

### 3.3 Alternating Direction Implicit (ADI) for Full 2D Problem

For the full `V(x, v, t)` PDE, the ADI method splits the time step into two half-steps, each implicit in one direction only. The most common schemes are:

#### Douglas-Rachford (DR) Splitting

**Half-step 1** (implicit in `v`, explicit in `x`):
```
(V^{n+1/2} - V^n) / (dt/2) = L_v V^{n+1/2} + L_x V^n + F^n
```

**Half-step 2** (implicit in `x`, explicit in `v`):
```
(V^{n+1} - V^{n+1/2}) / (dt/2) = L_x V^{n+1} + L_v V^{n+1/2}
```

where `L_v` is the differential operator acting on `v` (diffusion + drift) and `L_x` is the operator acting on `x` (the `V_x` gradient and control term).

Each half-step requires solving a **tridiagonal** system (not a 2D system), which is O(M_x * M_v) per step.

**Unconditionally stable** (no CFL constraint).

#### Craig-Sneyd Splitting (recommended for Heston)

The Craig-Sneyd scheme handles the cross-derivative term `rho * xi * sqrt(v) * V_{xv}` that appears if the inventory had diffusion. Since the inventory has **no diffusion** in our problem (only drift from the control), there is no `V_{xv}` term, and the standard Douglas-Rachford suffices. This simplifies the implementation significantly.

#### Hundsdorfer-Verwer (HV) Scheme

More accurate than Douglas-Rachford for problems with mixed derivatives, but overkill here given the absence of the cross term.

### 3.4 Operator Splitting for the Nonlinear Term

The nonlinear `(V_x)^2 / (4 * eta)` term (or `-A^2 / eta` in the separable case) can be handled via **Strang splitting**:

1. **Half-step for diffusion** (linear, implicit, tridiagonal)
2. **Full step for nonlinear reaction** (pointwise ODE, solved exactly or with explicit Euler)
3. **Half-step for diffusion** (same as step 1)

For the reaction ODE `dA/dt = A^2 / eta - lambda * v`:
```
A^{n+1} = A^n / (1 - dt * A^n / eta)   (implicit Euler, stable for A > 0)
```

### 3.5 Recommended Approach for This Project

Given the separable ansatz reduces the problem to 1D in `v`:

**Recommended:** Solve the `A(v, t)` PDE using Crank-Nicolson with semi-implicit nonlinear treatment.

- Grid: `M_v = 100` points in `v` from `0` to `4 * theta`
- Time steps: `N_t = 1000` backward from `T` to `0`
- Boundary at `v = 0`: use Feller condition analysis (see Section 5)
- Boundary at `v_max`: `A_v = 0` (Neumann, no flux at truncation boundary)

**Only use full 2D if:**
- Nonlinear impact (`alpha != 1`)
- Need the full trajectory policy rather than the quadratic approximation

---

## 4. 2D SDE Simulation

### 4.1 Cholesky Decomposition for Correlated Brownians

Given `dW_S` and `dW_v` with `corr = rho`, generate correlated increments via:

```
dW_S = Z_1 * sqrt(dt)
dW_v = (rho * Z_1 + sqrt(1 - rho^2) * Z_2) * sqrt(dt)
```

where `Z_1, Z_2 ~ iid N(0, 1)`. This is the Cholesky decomposition of the 2x2 correlation matrix `[[1, rho], [rho, 1]]`.

### 4.2 Euler-Maruyama for Heston

The standard Euler-Maruyama discretization for the Heston SDE is:

```
S_{k+1} = S_k + mu * S_k * dt + sqrt(v_k) * S_k * dW_S_k
v_{k+1} = v_k + kappa * (theta - v_k) * dt + xi * sqrt(v_k) * dW_v_k
```

**Critical issue: Negative variance.** The Euler scheme can produce `v_{k+1} < 0` because the Brownian term `xi * sqrt(v_k) * dW_v_k` can dominate. Two common fixes:

**Full truncation (recommended):**
```
v_{k+1} = max(0, v_k + kappa * (theta - v_k) * dt + xi * sqrt(max(0, v_k)) * dW_v_k)
```

**Reflection:**
```
v_{k+1} = |v_k + kappa * (theta - v_k) * dt + xi * sqrt(max(0, v_k)) * dW_v_k|
```

Full truncation is preferred (Lord et al. 2010) because it preserves the sign constraint without distorting the drift.

### 4.3 Milstein Correction

The Milstein scheme adds a second-order correction to reduce discretization error:

```
v_{k+1} = v_k + kappa*(theta - v_k)*dt + xi*sqrt(v_k)*dW_v_k + (1/4)*xi^2*(dW_v_k^2 - dt)
```

The correction term `(1/4) * xi^2 * (dW_v_k^2 - dt)` comes from Ito's formula applied to `sqrt(v)`. Combined with full truncation:

```
v_pos_k = max(0, v_k)
v_{k+1} = max(0, v_k + kappa*(theta - v_pos_k)*dt + xi*sqrt(v_pos_k)*dW_v_k + (1/4)*xi^2*(dW_v_k^2 - dt))
```

### 4.4 QE (Quadratic Exponential) Scheme

For better accuracy when the Feller condition `2*kappa*theta > xi^2` is NOT satisfied (common in crypto), the QE scheme (Andersen 2007) approximates the non-central chi-squared distribution of `v`:

- When `v` is large: approximate with a normal distribution (Gaussian regime)
- When `v` is small: approximate with a point mass at 0 + exponential tail (boundary regime)

QE is significantly more accurate than Milstein near `v = 0` but requires more code. For a course project, full-truncation Euler-Maruyama is sufficient for `kappa*theta/xi^2 > 1`.

### 4.5 Implementation Pattern for Execution Simulation

The Heston MC path simulation for execution cost estimation extends `simulate_execution` in `sde_engine.py`:

```python
def simulate_heston_execution(
    params: ACParams,
    heston: HestonParams,
    trajectory_x: np.ndarray,
    n_paths: int = 10_000,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns: (price_paths, variance_paths, costs)
    Shape: (n_paths, N+1), (n_paths, N+1), (n_paths,)
    """
    rng = np.random.default_rng(seed)
    N = params.N
    dt = params.dt
    sqrt_dt = np.sqrt(dt)

    # Pre-generate correlated normals
    # Z shape: (n_paths, N, 2)
    Z = rng.standard_normal((n_paths, N, 2))
    dW_S = Z[:, :, 0] * sqrt_dt
    dW_v = (heston.rho * Z[:, :, 0] + np.sqrt(1 - heston.rho**2) * Z[:, :, 1]) * sqrt_dt

    S = np.zeros((n_paths, N + 1))
    v = np.zeros((n_paths, N + 1))
    S[:, 0] = params.S0
    v[:, 0] = heston.v0

    n_k = trajectory_x[:-1] - trajectory_x[1:]  # shares sold each step

    costs = np.zeros(n_paths)

    for k in range(N):
        v_pos = np.maximum(0.0, v[:, k])
        sqrt_v = np.sqrt(v_pos)

        # Cost accumulation: lambda * v_k * x_k^2 * dt + eta * u_k^2 * dt
        # (the deterministic part; market cost captured via implementation shortfall)
        costs += n_k[k] * (params.S0 - S[:, k]) + params.eta * (n_k[k] / dt)**2 * dt

        # Price evolution
        S[:, k+1] = S[:, k] * np.exp(
            -0.5 * v_pos * dt + sqrt_v * dW_S[:, k]
        ) - params.gamma * n_k[k]

        # Variance evolution (full truncation Euler-Maruyama)
        v[:, k+1] = np.maximum(0.0,
            v[:, k]
            + heston.kappa * (heston.theta - v_pos) * dt
            + heston.xi * sqrt_v * dW_v[:, k]
        )

    return S, v, costs
```

### 4.6 Antithetic Variates Under Heston

For antithetic variates, negate **both** `Z_1` and `Z_2`:

```python
Z_half = rng.standard_normal((n_paths // 2, N, 2))
Z = np.concatenate([Z_half, -Z_half], axis=0)
```

This preserves the correlation structure `rho` while providing variance reduction.

---

## 5. Heston Calibration for Crypto

### 5.1 The Feller Condition

The Feller condition ensures the variance process never reaches zero:

```
2 * kappa * theta > xi^2
```

**Interpretation:** The drift force `kappa * theta` must be strong enough relative to the diffusion `xi^2 / 2` to prevent the process from reaching the zero boundary. When violated, `v_t` can touch zero and the Euler scheme becomes degenerate.

**In practice for crypto:**
- Crypto options (e.g., BTC on Deribit) often imply Heston parameters that **violate** the Feller condition
- This is because vol-of-vol `xi` is very high for crypto (~0.6-1.5)
- Mean reversion `kappa` tends to be moderate (~2-6) and `theta` moderate (~0.04-0.16)
- `2 * 3 * 0.09 = 0.54 < xi^2 = 0.64` for `xi = 0.8` — Feller violated

When Feller is violated, use full-truncation Euler-Maruyama or the QE scheme. The PDE boundary condition at `v = 0` changes from Dirichlet to the "degenerate" boundary (a natural boundary, no BC needed if the process reflects).

### 5.2 Typical Heston Parameter Ranges for Crypto

Based on calibrations to BTC and ETH options on Deribit (literature and market observations as of 2024-2025):

| Parameter | Equities (S&P 500) | BTC (Deribit) | ETH (Deribit) |
|-----------|-------------------|---------------|---------------|
| `v0`      | 0.03-0.06         | 0.08-0.25     | 0.10-0.35     |
| `theta`   | 0.03-0.05         | 0.07-0.18     | 0.09-0.22     |
| `kappa`   | 1-4               | 2-8           | 2-10          |
| `xi`      | 0.2-0.5           | 0.5-1.5       | 0.6-1.8       |
| `rho`     | -0.8 to -0.5      | -0.3 to 0.1   | -0.4 to 0.2   |
| Feller?   | Usually satisfied | Often violated | Often violated |

**Key differences from equities:**
- Much higher `v0` and `theta` (crypto is 3-5x more volatile than equities)
- Higher `xi` (vol-of-vol) — crypto variance itself is highly uncertain
- Weaker leverage effect (less negative `rho`) — crypto often shows near-zero or even positive `rho`

### 5.3 Data Sources for Crypto Calibration

**Primary source: Deribit**
- Deribit is the dominant BTC/ETH options exchange by volume
- API endpoint: `https://www.deribit.com/api/v2/public/get_instruments`
- Provides implied volatility surface (strikes, maturities, implied vols)
- Data format: mark IV (mark implied volatility), bid IV, ask IV

**Calibration workflow:**
1. Fetch current options chain: strikes `K_i`, maturities `T_j`, mid implied vols `sigma_imp(K_i, T_j)`
2. Convert to option prices: `C_market = BS_price(S0, K, T, r, sigma_imp)` using Black-Scholes as a pricing convention
3. Minimize sum of squared pricing errors (SSE) over Heston parameters:
   ```
   min_{kappa, theta, xi, rho, v0} sum_i (heston_price(K_i, T_i) - C_market_i)^2 / C_market_i^2
   ```
   Note: relative errors (divide by `C_market_i^2`) prevent near-ATM options from dominating
4. Use `fft_call_price()` (already implemented) for Heston model prices

**Alternative: Calibrate to implied vol, not price:**
```
min_{params} sum_i (sigma_heston(K_i, T_i) - sigma_imp(K_i, T_i))^2 * w_i
```

where `w_i` are weights (e.g., vega-weighted: `w_i = vega(K_i, T_i)`). Vega-weighting focuses the calibration on the liquid ATM region.

### 5.4 Parameter Bounds for calibrate_heston()

The `calibrate_heston` stub in `extensions/heston.py` should use these bounds for `scipy.optimize.minimize`:

```python
bounds = [
    (0.1, 20.0),    # kappa: mean-reversion speed
    (0.001, 1.0),   # theta: long-run variance (0.001 = 3% vol, 1.0 = 100% vol)
    (0.01, 3.0),    # xi: vol-of-vol
    (-0.999, 0.999),# rho: correlation
    (0.001, 2.0),   # v0: initial variance
]
```

**Feller soft constraint:** Add a penalty term:
```python
feller_penalty = max(0, xi**2 - 2*kappa*theta) * 1000
```

This discourages but does not enforce the Feller condition, which is appropriate since crypto calibrations often produce Feller-violating parameters.

### 5.5 Representative Calibrated Params for Project (Synthetic BTC)

For the Part D implementation, use these representative BTC parameters (consistent with Deribit mid-market calibrations, 30-day horizon, 2024):

```python
HestonParams(
    kappa = 3.5,    # moderate mean-reversion
    theta = 0.10,   # long-run var = 10% → 31.6% annualized vol
    xi    = 0.75,   # vol-of-vol
    rho   = -0.10,  # weak negative correlation (typical BTC)
    v0    = 0.12,   # initial var = 12% → 34.6% current vol
)
# Feller: 2 * 3.5 * 0.10 = 0.70 < xi^2 = 0.5625 → barely satisfied
```

---

## 6. Key References

### Optimal Execution with Stochastic Volatility

1. **Almgren & Chriss (2001)** — "Optimal Execution of Portfolio Transactions." *Journal of Risk*, 3(2). The baseline constant-vol framework that this project extends. Establishes the sinh-trajectory result.

2. **Cartea, Jaimungal & Penalva (2015)** — *Algorithmic and High-Frequency Trading* (Cambridge). Chapters 6-7 provide the HJB formulation for optimal execution. Chapter 9 discusses stochastic volatility extensions.

3. **Almgren (2012)** — "Optimal Trading with Stochastic Liquidity and Volatility." *SIAM Journal on Financial Mathematics*, 3(1):163-181. Extends Almgren-Chriss to stochastic volatility (not Heston-specific, but directly relevant). Shows the value function ansatz approach.

4. **Fouque, Papanicolaou & Sircar (2000)** — *Derivatives in Financial Markets with Stochastic Volatility* (Cambridge). Background on stochastic vol in derivatives pricing; relevant for understanding the PDE structure.

5. **Gatheral (2006)** — *The Volatility Surface* (Wiley). Chapter 2 covers the Heston model including characteristic function, calibration, and the role of the Feller condition.

### Numerical Methods for 2D PDEs

6. **Heston (1993)** — "A Closed-Form Solution for Options with Stochastic Volatility." *Review of Financial Studies*, 6(2):327-343. Original Heston paper; establishes the characteristic function used in `heston_cf()`.

7. **in 't Hout & Foulon (2010)** — "ADI Finite Difference Schemes for Option Pricing in the Heston Model with Correlation." *International Journal of Numerical Analysis and Modeling*, 7(2):303-320. Systematic comparison of Douglas-Rachford, Craig-Sneyd, and Hundsdorfer-Verwer for the Heston PDE. Key recommendation: Craig-Sneyd is most accurate for problems with cross-derivatives (not needed here, but good reference).

8. **Peaceman & Rachford (1955)** — The original ADI paper. For historical reference on the splitting methodology.

### SDE Simulation

9. **Lord, Koekkoek & Van Dijk (2010)** — "A Comparison of Biased Simulation Schemes for Stochastic Volatility Models." *Quantitative Finance*, 10(2):177-194. Comprehensive comparison of truncation schemes for Heston SDE. Recommends full truncation. Must-read for implementing `simulate_heston_execution`.

10. **Andersen (2007)** — "Efficient Simulation of the Heston Stochastic Volatility Model." *Journal of Computational Finance*, 11(3):1-42. Introduces the QE (Quadratic Exponential) scheme. Relevant if Feller condition is violated.

11. **Glasserman (2004)** — *Monte Carlo Methods in Financial Engineering* (Springer). Chapter 3 covers correlated Brownian simulation; Chapter 6 covers variance reduction techniques (antithetic variates).

### Heston Calibration to Crypto

12. **Alexander & Rauch (2021)** — "Model-Free Discretisation-Invariant Swaps and S&P 500 Higher-Moment Risk Premia." Discusses volatility surface fitting for crypto derivatives.

13. **Matic, Packham & Schoutens (2021)** — "Hedging Cryptocurrency Options." *Review of Derivatives Research*. Empirically analyzes Heston calibration to BTC options on Deribit; finds Feller condition typically violated, recommends SABR or 3/2 model as alternatives.

14. **Deribit API Documentation** — `https://docs.deribit.com/`. Primary data source for live BTC/ETH implied volatility surfaces.

---

## 7. Implementation Roadmap

The following order minimizes dependencies and allows incremental testing.

### Step 1: Complete `calibrate_heston()` (already stubbed)

File: `/extensions/heston.py`

**What to do:**
- Implement `calibrate_heston()` using `scipy.optimize.minimize` with `method='L-BFGS-B'`
- Use `fft_call_price()` (already implemented) for model prices
- Add Feller penalty to objective function
- Test with synthetic market prices generated from known Heston params

**Validation:** Recover known parameters from synthetic data (round-trip test).

### Step 2: Implement Heston SDE Simulation

File: `/montecarlo/sde_engine.py` (new function `simulate_heston_execution`)

**What to do:**
- Add `simulate_heston_paths(heston: HestonParams, params: ACParams, ...)` for pure price/variance path generation
- Add `simulate_heston_execution(...)` for execution cost simulation
- Use full-truncation Euler-Maruyama (Section 4.2)
- Use Cholesky decomposition for correlated Brownians (Section 4.1)
- Antithetic variates: negate both `Z_1` and `Z_2` (Section 4.6)

**Validation:** Check that simulated variance `E[v_t]` matches the theoretical mean `theta + (v0 - theta) * exp(-kappa * t)`.

### Step 3: Solve the 1D Reduced HJB for `A(v, t)`

File: `/pde/hjb_solver.py` (new function `solve_hjb_heston`)

**What to do:**
- Implement the separable ansatz `V(x, v, t) = A(v, t) * x^2`
- Solve `A_t - A^2/eta + lambda*v + kappa*(theta-v)*A_v + (1/2)*xi^2*v*A_vv = 0`
- Use Crank-Nicolson with semi-implicit treatment of nonlinear term
- Grid: `M_v = 100` in `v in [0, 4*theta]`
- Terminal condition: `A(v, T) = terminal_penalty`
- Boundary at `v = 0`: `A_v = 0` or one-sided differencing
- Boundary at `v_max`: `A_v = 0` (Neumann)

**Optimal control:** `u*(x, v, t) = A(v, t) * x / eta`

**Validation:** Check that as `xi -> 0`, `A(v, t)` converges to the Almgren-Chriss `alpha(t)` (evaluated at `v = theta`).

### Step 4: Extract Heston-Optimal Trajectory

File: `/pde/hjb_solver.py` (new function `extract_heston_trajectory`)

**What to do:**
- Given `A(v, t)` surface and a simulated variance path `v_t`, compute the adapted optimal control:
  ```
  u_k = A(v_k, t_k) * x_k / eta
  x_{k+1} = x_k - u_k * dt
  ```
- This creates a **random** trajectory (adapts to variance realizations)

**Compare with:** The constant-vol AC trajectory — shows the benefit of stochastic vol adaptation.

### Step 5: Monte Carlo Cost Comparison

File: `/montecarlo/cost_analysis.py`

**What to do:**
- Run `simulate_heston_execution` for three strategies:
  1. AC constant-vol (uses fixed sigma = sqrt(theta))
  2. Heston-adapted (uses `A(v, t)` surface from Step 3)
  3. TWAP (benchmark)
- Report: E[cost], Var[cost], Mean-Variance frontier under Heston dynamics

**Key insight to demonstrate:** The Heston-adapted strategy has lower variance than the constant-vol strategy because it speeds up trading when variance is high and slows down when variance is low.

### Step 6 (Optional): Calibrate to Real Deribit Data

File: `/calibration/data_loader.py`

**What to do:**
- Fetch BTC options from Deribit API (or use saved CSV)
- Run `calibrate_heston()` to get BTC Heston params
- Re-run Steps 3-5 with calibrated params
- Compare with synthetic params from Section 5.5

### Summary Table

| Step | File | Function | Priority |
|------|------|----------|----------|
| 1 | `extensions/heston.py` | `calibrate_heston()` | High |
| 2 | `montecarlo/sde_engine.py` | `simulate_heston_execution()` | High |
| 3 | `pde/hjb_solver.py` | `solve_hjb_heston()` | High |
| 4 | `pde/hjb_solver.py` | `extract_heston_trajectory()` | Medium |
| 5 | `montecarlo/cost_analysis.py` | MC cost comparison | Medium |
| 6 | `calibration/data_loader.py` | Deribit fetch + calibrate | Low |

### Notes on the Existing Grid Structure

The existing `ExecutionGrid` in `pde/grid.py` is 1D in `x`. For the Heston extension, you need a 2D grid `(x, v)`. Rather than modifying `ExecutionGrid`, add a new `HestonExecutionGrid` dataclass:

```python
@dataclass
class HestonExecutionGrid:
    x_grid: np.ndarray   # shape (M_x + 1,)
    v_grid: np.ndarray   # shape (M_v + 1,)
    t_grid: np.ndarray   # shape (N + 1,)
    M_x: int
    M_v: int
    N: int
    dx: float
    dv: float
    dt: float
```

The existing `check_cfl` function in `pde/grid.py` already has a placeholder note for this extension. The CFL condition for the 2D explicit scheme would be:

```
dt <= min(dv^2 / (xi^2 * v_max), ...)
```

This CFL is the motivation for using Crank-Nicolson or ADI (unconditionally stable) rather than explicit Euler in the `v` direction.
