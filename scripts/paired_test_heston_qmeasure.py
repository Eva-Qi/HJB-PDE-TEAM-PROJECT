"""Paired statistical test: Heston Q-measure vs P-measure vs constant-vol.

Re-runs the Heston-vs-const-vol paired test using RELIABLE Q-measure Heston
parameters (κ=9.09, ρ=-0.385, ξ=2.04, θ=0.229, v₀=0.162) calibrated to the
Deribit BTC option IV surface, instead of the structurally broken P-measure
params from spot-based calibration.

Three comparisons:
  A: const-vol vs Heston-P  (reproduces earlier p≈0.33 on mean)
  B: const-vol vs Heston-Q  (KEY TEST — Q-measure with real leverage ρ=-0.385)
  C: Heston-P vs Heston-Q   (how much does calibration quality matter?)

All simulations use common random numbers (seed=42) throughout so per-path
cost differences are clean paired observations.
"""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from shared.params import ACParams
from montecarlo.strategies import twap_trajectory
from montecarlo.sde_engine import simulate_execution, simulate_heston_execution
from montecarlo.cost_analysis import paired_strategy_test
from extensions.heston import HestonParams


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
N_PATHS = 10_000
SEED = 42
BOOTSTRAP_REPS = 5_000
LAM = 1e-6

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
PARAMS_JSON = DATA_DIR / "heston_pmeasure_vs_qmeasure.json"
OUT_JSON = DATA_DIR / "paired_heston_qmeasure_results.json"


# ---------------------------------------------------------------------------
# Base ACParams (fee_bps=0 to isolate vol impact)
# ---------------------------------------------------------------------------
def _make_base_params(sigma: float) -> ACParams:
    """ACParams matching paired_test_heston_vs_const.py with fee_bps=0."""
    return ACParams(
        S0=69000.0,
        sigma=sigma,
        mu=0.0,
        X0=10.0,
        T=1 / 24,
        N=50,
        gamma=1.48,
        eta=1.58e-4,
        alpha=1.0,
        lam=LAM,
        fee_bps=0.0,
    )


# ---------------------------------------------------------------------------
# Bootstrap helpers (from paired_test_regime_aware_v2.py template)
# ---------------------------------------------------------------------------
def _var_95(arr: np.ndarray) -> float:
    return float(np.percentile(arr, 95))


def _cvar_95(arr: np.ndarray) -> float:
    v = np.percentile(arr, 95)
    tail = arr[arr >= v]
    return float(np.mean(tail)) if len(tail) > 0 else float(v)


def _objective(arr: np.ndarray, lam: float = LAM) -> float:
    return float(np.mean(arr) + lam * np.var(arr))


def paired_bootstrap_stat_test(
    costs_a: np.ndarray,
    costs_b: np.ndarray,
    stat_fn,
    n_bootstrap: int = 5_000,
    seed: int = 42,
) -> tuple[float, float, float]:
    """Paired bootstrap test for a scalar population statistic.

    Jointly resamples both arrays (preserving pairing), computes the
    statistic on each resample, and tests H_0: stat(A) = stat(B).

    Returns (stat_a, stat_b, p_value).
    """
    assert costs_a.shape == costs_b.shape
    n = len(costs_a)
    rng = np.random.default_rng(seed)

    stat_a = stat_fn(costs_a)
    stat_b = stat_fn(costs_b)
    observed_diff = stat_a - stat_b

    idx = rng.integers(0, n, size=(n_bootstrap, n))
    boot_a = costs_a[idx]
    boot_b = costs_b[idx]

    boot_diff = np.array([
        stat_fn(boot_a[b]) - stat_fn(boot_b[b])
        for b in range(n_bootstrap)
    ])

    # Shift null to zero
    boot_diff_shifted = boot_diff - np.mean(boot_diff)
    p_value = float(np.mean(np.abs(boot_diff_shifted) >= abs(observed_diff)))

    return stat_a, stat_b, p_value


# ---------------------------------------------------------------------------
# One comparison block
# ---------------------------------------------------------------------------
def run_comparison(
    label_a: str,
    label_b: str,
    costs_a: np.ndarray,
    costs_b: np.ndarray,
    bootstrap_seed: int = SEED,
) -> dict:
    """Run all four metric tests for one (A vs B) pair."""
    # (1) Mean cost — t-test + bootstrap via paired_strategy_test
    res_mean = paired_strategy_test(
        costs_a=costs_a,
        costs_b=costs_b,
        label_a=label_a,
        label_b=label_b,
        test="both",
        n_bootstrap=BOOTSTRAP_REPS,
        seed=bootstrap_seed,
    )
    mean_a = float(np.mean(costs_a))
    mean_b = float(np.mean(costs_b))
    p_mean = res_mean.bootstrap_pvalue

    # (2) Objective = E[cost] + lam * Var[cost]
    obj_a, obj_b, p_obj = paired_bootstrap_stat_test(
        costs_a, costs_b,
        stat_fn=lambda c: _objective(c, lam=LAM),
        n_bootstrap=BOOTSTRAP_REPS,
        seed=bootstrap_seed + 1,
    )

    # (3) VaR_95
    var_a, var_b, p_var = paired_bootstrap_stat_test(
        costs_a, costs_b,
        stat_fn=_var_95,
        n_bootstrap=BOOTSTRAP_REPS,
        seed=bootstrap_seed + 2,
    )

    # (4) CVaR_95
    cvar_a, cvar_b, p_cvar = paired_bootstrap_stat_test(
        costs_a, costs_b,
        stat_fn=_cvar_95,
        n_bootstrap=BOOTSTRAP_REPS,
        seed=bootstrap_seed + 3,
    )

    return {
        "label_a": label_a,
        "label_b": label_b,
        "metrics": {
            "mean_cost":  {"a": mean_a,  "b": mean_b,  "diff": mean_a - mean_b,  "p_value": p_mean},
            "objective":  {"a": obj_a,   "b": obj_b,   "diff": obj_a  - obj_b,   "p_value": p_obj},
            "var_95":     {"a": var_a,   "b": var_b,   "diff": var_a  - var_b,   "p_value": p_var},
            "cvar_95":    {"a": cvar_a,  "b": cvar_b,  "diff": cvar_a - cvar_b,  "p_value": p_cvar},
        },
    }


# ---------------------------------------------------------------------------
# Print one comparison table
# ---------------------------------------------------------------------------
ALPHA = 0.05


def _sig(p: float) -> str:
    return "YES *" if p < ALPHA else "no"


def print_comparison(comp: dict) -> None:
    la = comp["label_a"]
    lb = comp["label_b"]
    print()
    print(f"Comparison: {la} vs {lb}")
    header = (
        f"  {'Metric':<14} {'mean(A)':>14} {'mean(B)':>14} "
        f"{'diff(A-B)':>12} {'p-value':>10} {'sig@0.05':>10}"
    )
    print(header)
    print("  " + "-" * (len(header) - 2))

    metric_labels = {
        "mean_cost": "mean cost",
        "objective": "objective",
        "var_95":    "VaR_95",
        "cvar_95":   "CVaR_95",
    }
    for key, label in metric_labels.items():
        m = comp["metrics"][key]
        print(
            f"  {label:<14} {m['a']:>14.4f} {m['b']:>14.4f} "
            f"{m['diff']:>+12.4f} {m['p_value']:>10.4f} {_sig(m['p_value']):>10}"
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    print()
    print("=" * 80)
    print("Paired Heston Q-measure vs P-measure vs Constant-Vol Test")
    print("=" * 80)

    # ── Load Q / P measure params ──────────────────────────────────────────
    print(f"\n[1/4] Loading Heston params from {PARAMS_JSON.name}...")
    with open(PARAMS_JSON) as f:
        heston_data = json.load(f)

    qm = heston_data["q_measure"]
    pm = heston_data["p_measure"]

    q_params = HestonParams(
        kappa=qm["kappa"],
        theta=qm["theta"],
        xi=qm["xi"],
        rho=qm["rho"],
        v0=qm["v0"],
    )
    p_params = HestonParams(
        kappa=pm["kappa"],
        theta=pm["theta"],
        xi=pm["xi"],
        rho=pm["rho"],
        v0=pm["v0"],
    )

    print(f"  Q-measure: κ={q_params.kappa:.4f} θ={q_params.theta:.4f} "
          f"ξ={q_params.xi:.4f} ρ={q_params.rho:.4f} v₀={q_params.v0:.4f}")
    print(f"  P-measure: κ={p_params.kappa:.4f} θ={p_params.theta:.4f} "
          f"ξ={p_params.xi:.4f} ρ={p_params.rho:.4f} v₀={p_params.v0:.4f}")

    # ── Feller condition checks ────────────────────────────────────────────
    q_feller_lhs = 2 * q_params.kappa * q_params.theta
    q_feller_rhs = q_params.xi ** 2
    p_feller_lhs = 2 * p_params.kappa * p_params.theta
    p_feller_rhs = p_params.xi ** 2

    print(f"\n  Feller check Q: 2κθ={q_feller_lhs:.4f}  ξ²={q_feller_rhs:.4f}  "
          f"margin={(q_feller_lhs - q_feller_rhs):.4f} "
          f"({'OK' if q_feller_lhs >= q_feller_rhs else 'VIOLATED'})")
    print(f"  Feller check P: 2κθ={p_feller_lhs:.4f}  ξ²={p_feller_rhs:.4f}  "
          f"margin={(p_feller_lhs - p_feller_rhs):.4f} "
          f"({'OK' if p_feller_lhs >= p_feller_rhs else 'VIOLATED'})")

    if q_feller_lhs < q_feller_rhs:
        warnings.warn(
            "Q-measure Feller condition violated — variance may hit zero during sim. "
            "Full truncation scheme will clamp at zero, expect some artifactual paths."
        )
    if q_feller_lhs >= q_feller_rhs and (q_feller_lhs - q_feller_rhs) < 0.2:
        print(
            "  WARNING: Q-measure Feller margin is very narrow "
            f"({q_feller_lhs - q_feller_rhs:.4f} < 0.2). "
            "Variance may approach zero frequently; monitor for collapse artifacts."
        )

    # ── Base sigma from Q-measure theta (vol scale comparability) ──────────
    sigma_base = float(np.sqrt(q_params.theta))
    print(f"\n  Base sigma = sqrt(theta_Q) = {sigma_base:.6f} "
          f"(annualized vol ≈ {sigma_base:.1%})")

    base_params = _make_base_params(sigma=sigma_base)

    # ── TWAP trajectory ────────────────────────────────────────────────────
    x_twap = twap_trajectory(base_params)

    # ── Simulate — COMMON RANDOM NUMBERS ──────────────────────────────────
    print(f"\n[2/4] Simulating {N_PATHS:,} paths (common random numbers, seed={SEED})...")

    # Constant vol
    print("  (a) Constant-vol simulation...")
    _, costs_const = simulate_execution(
        base_params,
        x_twap,
        n_paths=N_PATHS,
        seed=SEED,
        antithetic=False,
        scheme="exact",
    )

    # Heston P-measure
    print("  (b) Heston P-measure simulation...")
    with warnings.catch_warnings(record=True) as p_warns:
        warnings.simplefilter("always")
        _, _, costs_heston_P = simulate_heston_execution(
            base_params,
            p_params,
            x_twap,
            n_paths=N_PATHS,
            seed=SEED,
        )
    if p_warns:
        print(f"      [P-measure warnings: {len(p_warns)} — e.g. Feller violation]")

    # Heston Q-measure
    print("  (c) Heston Q-measure simulation...")
    with warnings.catch_warnings(record=True) as q_warns:
        warnings.simplefilter("always")
        _, var_paths_Q, costs_heston_Q = simulate_heston_execution(
            base_params,
            q_params,
            x_twap,
            n_paths=N_PATHS,
            seed=SEED,
        )
    if q_warns:
        print(f"      [Q-measure warnings: {len(q_warns)} — e.g. Feller violation]")

    # Check for variance collapse in Q-measure paths
    neg_var_frac = float(np.mean(var_paths_Q < 0))
    near_zero_frac = float(np.mean(var_paths_Q < 1e-4))
    if neg_var_frac > 0.01:
        print(
            f"  WARNING: {neg_var_frac:.1%} of Q-measure variance path values are negative "
            "(full truncation artifact). Inspect CVaR estimates carefully."
        )
    elif near_zero_frac > 0.05:
        print(
            f"  NOTE: {near_zero_frac:.1%} of Q-measure variance values are near-zero "
            "(v < 1e-4) — Feller margin tight but full truncation contained collapse."
        )
    else:
        print(
            f"  Variance collapse check: neg={neg_var_frac:.2%}, near-zero={near_zero_frac:.2%} — OK"
        )

    print(f"\n  mean(const-vol)  = {np.mean(costs_const):.4f}")
    print(f"  mean(Heston-P)   = {np.mean(costs_heston_P):.4f}")
    print(f"  mean(Heston-Q)   = {np.mean(costs_heston_Q):.4f}")

    # ── Run three comparisons ──────────────────────────────────────────────
    print(f"\n[3/4] Running paired tests (bootstrap reps={BOOTSTRAP_REPS:,})...")

    print("  Comparison A: const-vol vs Heston-P...")
    comp_A = run_comparison(
        "const-vol", "Heston-P",
        costs_const, costs_heston_P,
        bootstrap_seed=SEED,
    )

    print("  Comparison B: const-vol vs Heston-Q...")
    comp_B = run_comparison(
        "const-vol", "Heston-Q",
        costs_const, costs_heston_Q,
        bootstrap_seed=SEED + 10,
    )

    print("  Comparison C: Heston-P vs Heston-Q...")
    comp_C = run_comparison(
        "Heston-P", "Heston-Q",
        costs_heston_P, costs_heston_Q,
        bootstrap_seed=SEED + 20,
    )

    # ── Print results tables ───────────────────────────────────────────────
    print()
    print("=" * 80)
    print("RESULTS")
    print("=" * 80)
    print(f"  N paths = {N_PATHS:,}   bootstrap reps = {BOOTSTRAP_REPS:,}   lambda = {LAM:.1e}")
    print(f"  sigma_base = sqrt(theta_Q) = {sigma_base:.6f}")
    print(f"  Q-params: κ={q_params.kappa:.3f} θ={q_params.theta:.4f} "
          f"ξ={q_params.xi:.4f} ρ={q_params.rho:.4f} v₀={q_params.v0:.4f}")
    print(f"  P-params: κ={p_params.kappa:.3f} θ={p_params.theta:.4f} "
          f"ξ={p_params.xi:.4f} ρ={p_params.rho:.4f} v₀={p_params.v0:.4f}")

    print_comparison(comp_A)
    print()
    print("  (A) — reproduces earlier result; expect p>0.1 on mean cost")

    print_comparison(comp_B)
    print()
    print("  (B) — KEY TEST: does real leverage ρ=-0.385 shift risk metrics?")

    print_comparison(comp_C)
    print()
    print("  (C) — calibration quality audit: how wrong was P-measure?")

    # ── Interpretation ─────────────────────────────────────────────────────
    print()
    print("=" * 80)
    print("INTERPRETATION")
    print("=" * 80)

    def _sig_metrics(comp: dict) -> list[str]:
        return [
            label
            for key, label in [
                ("mean_cost", "mean cost"),
                ("objective", "objective"),
                ("var_95", "VaR_95"),
                ("cvar_95", "CVaR_95"),
            ]
            if comp["metrics"][key]["p_value"] < ALPHA
        ]

    sig_A = _sig_metrics(comp_A)
    sig_B = _sig_metrics(comp_B)
    sig_C = _sig_metrics(comp_C)

    print()
    print("1. Does Q-measure Heston reject H_0 on risk metrics where P-measure didn't?")
    if sig_B and not all(m in sig_A for m in sig_B):
        new_sigs = [m for m in sig_B if m not in sig_A]
        print(f"   YES — Q-measure newly rejects: {', '.join(new_sigs)}")
    elif sig_B:
        print(f"   YES (same metrics as P-measure but with stronger signal): {', '.join(sig_B)}")
    else:
        print("   NO — Q-measure also fails to reject H_0 on all metrics.")
        print("   This may indicate that the cost distribution is insensitive to")
        print("   vol model choice at this execution horizon (T=1/24, N=50).")

    print()
    print("2. Q-measure vs P-measure Heston: tail distribution difference")
    # In comp_A: A=const-vol, B=Heston-P; in comp_B: A=const-vol, B=Heston-Q
    cvar_const = comp_A["metrics"]["cvar_95"]["a"]  # const-vol CVaR
    cvar_P = comp_A["metrics"]["cvar_95"]["b"]      # Heston-P CVaR
    cvar_Q = comp_B["metrics"]["cvar_95"]["b"]      # Heston-Q CVaR
    cvar_PQ_diff = cvar_Q - cvar_P
    cvar_PQ_pct = (cvar_PQ_diff / abs(cvar_P) * 100) if cvar_P != 0 else float("nan")
    print(f"   CVaR_95:  const-vol={cvar_const:.4f}  Heston-P={cvar_P:.4f}  Heston-Q={cvar_Q:.4f}")
    print(f"   Q vs P CVaR_95 diff: {cvar_PQ_diff:+.4f} ({cvar_PQ_pct:+.2f}%)")
    if sig_C:
        print(f"   C comparison significant on: {', '.join(sig_C)}")
        print("   => P-measure params materially mis-estimated the cost distribution.")
    else:
        print("   C comparison not significant — P and Q Heston give similar cost tails.")
        print("   This suggests execution cost is robust to ρ and κ at this horizon.")

    print()
    print("3. Does mean cost test become significant with Q-measure?")
    p_mean_A = comp_A["metrics"]["mean_cost"]["p_value"]
    p_mean_B = comp_B["metrics"]["mean_cost"]["p_value"]
    print(f"   const-vol vs Heston-P  mean p-value: {p_mean_A:.4f} "
          f"({'significant' if p_mean_A < ALPHA else 'not significant'})")
    print(f"   const-vol vs Heston-Q  mean p-value: {p_mean_B:.4f} "
          f"({'significant' if p_mean_B < ALPHA else 'not significant'})")
    if p_mean_B >= ALPHA:
        print("   Mean cost insensitive to vol model — confirms that AC benefit lives")
        print("   in TAIL RISK (VaR/CVaR), not in mean cost reduction.")
    else:
        print("   Mean cost significant with Q-measure — surprising; check for bias.")

    # Headline number
    # diff = const-vol CVaR - Heston-Q CVaR (positive means const-vol has HIGHER tail cost)
    print()
    q_cvar_const_val = comp_B["metrics"]["cvar_95"]["a"]   # const-vol CVaR
    q_cvar_q_val = comp_B["metrics"]["cvar_95"]["b"]       # Heston-Q CVaR
    q_cvar_shift = q_cvar_const_val - q_cvar_q_val         # positive = Heston-Q is LOWER
    q_cvar_pct = (q_cvar_shift / abs(q_cvar_const_val) * 100) \
        if q_cvar_const_val != 0 else float("nan")
    p_cvar_95_B = comp_B["metrics"]["cvar_95"]["p_value"]
    direction = "LOWER" if q_cvar_shift > 0 else "HIGHER"
    print(
        f"Part D HEADLINE: Heston Q-measure CVaR_95 is {direction} than const-vol by "
        f"{abs(q_cvar_shift):.4f} ({abs(q_cvar_pct):.2f}%)  "
        f"[Heston-Q={q_cvar_q_val:.2f} vs const={q_cvar_const_val:.2f}]  "
        f"p={p_cvar_95_B:.4f} ({'significant' if p_cvar_95_B < ALPHA else 'not significant'})"
    )

    print("=" * 80)

    # ── Save JSON ──────────────────────────────────────────────────────────
    print(f"\n[4/4] Saving results to {OUT_JSON.name}...")

    out = {
        "version": "q_measure_v1",
        "n_paths": N_PATHS,
        "bootstrap_reps": BOOTSTRAP_REPS,
        "lambda": LAM,
        "seed": SEED,
        "sigma_base": float(sigma_base),
        "q_params": {
            "kappa": float(q_params.kappa),
            "theta": float(q_params.theta),
            "xi": float(q_params.xi),
            "rho": float(q_params.rho),
            "v0": float(q_params.v0),
            "feller_lhs": float(q_feller_lhs),
            "feller_rhs": float(q_feller_rhs),
            "feller_satisfied": bool(q_feller_lhs >= q_feller_rhs),
        },
        "p_params": {
            "kappa": float(p_params.kappa),
            "theta": float(p_params.theta),
            "xi": float(p_params.xi),
            "rho": float(p_params.rho),
            "v0": float(p_params.v0),
            "feller_lhs": float(p_feller_lhs),
            "feller_rhs": float(p_feller_rhs),
            "feller_satisfied": bool(p_feller_lhs >= p_feller_rhs),
        },
        "variance_collapse_check": {
            "negative_fraction": float(neg_var_frac),
            "near_zero_fraction": float(near_zero_frac),
        },
        "summary_means": {
            "const_vol": float(np.mean(costs_const)),
            "heston_P": float(np.mean(costs_heston_P)),
            "heston_Q": float(np.mean(costs_heston_Q)),
        },
        "comparisons": {
            "A_const_vs_P": comp_A,
            "B_const_vs_Q": comp_B,
            "C_P_vs_Q": comp_C,
        },
        "significance_summary": {
            "A_significant_metrics": sig_A,
            "B_significant_metrics": sig_B,
            "C_significant_metrics": sig_C,
        },
        "headline": {
            "cvar95_const_vol": float(q_cvar_const_val),
            "cvar95_heston_Q": float(q_cvar_q_val),
            "cvar95_shift_const_minus_Q": float(q_cvar_shift),
            "cvar95_pct_shift": float(q_cvar_pct),
            "cvar95_pvalue": float(p_cvar_95_B),
        },
    }

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(out, indent=2))
    print(f"  Saved: {OUT_JSON}")


if __name__ == "__main__":
    main()
