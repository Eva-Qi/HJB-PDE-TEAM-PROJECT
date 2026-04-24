"""Paired statistical test: Heston stochastic-vol vs constant-vol execution cost.

Uses the `paired_strategy_test` from montecarlo.cost_analysis (teammate bp)
to formally test whether Heston stochastic-volatility execution cost is
statistically different from constant-vol execution cost under the same
MC paths (common random numbers).

For each vol regime, both simulations share the same seed so the per-path
cost difference is a clean paired observation. This isolates the impact of
stochastic vol on execution cost from simulation noise.
"""

from __future__ import annotations

import json
import sys
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from shared.params import ACParams
from montecarlo.strategies import twap_trajectory
from montecarlo.sde_engine import simulate_execution, simulate_heston_execution
from montecarlo.cost_analysis import paired_strategy_test
from extensions.heston import HestonParams


# ---------------------------------------------------------------------------
# Vol regimes: constant sigma matched to Heston theta = sigma^2
# ---------------------------------------------------------------------------
VOL_REGIMES = [
    {
        "vol_label": "low_vol",
        "sigma": 0.2,
        "heston_params": HestonParams(
            kappa=2.0, theta=0.04, xi=0.3, rho=-0.3, v0=0.04
        ),
    },
    {
        "vol_label": "mid_vol",
        "sigma": 0.3,
        "heston_params": HestonParams(
            kappa=2.0, theta=0.09, xi=0.5, rho=-0.3, v0=0.09
        ),
    },
    {
        "vol_label": "high_vol",
        "sigma": 0.5,
        "heston_params": HestonParams(
            kappa=2.0, theta=0.25, xi=0.8, rho=-0.3, v0=0.25
        ),
    },
]

N_PATHS = 10_000
SEED = 42


def _make_base_params(sigma: float) -> ACParams:
    """Create ACParams with the given sigma, fee_bps=0."""
    return ACParams(
        S0=69000.0,
        sigma=sigma,
        mu=0.0,
        X0=10.0,
        T=1.0/(365.25*24),
        N=250,
        gamma=1.48,
        eta=1.58e-4,
        alpha=1.0,
        lam=1e-6,
        fee_bps=0.0,
    )


def run_one_regime(regime: dict) -> dict:
    """Run paired test for one vol regime and return result dict."""
    vol_label = regime["vol_label"]
    sigma = regime["sigma"]
    heston_params = regime["heston_params"]

    params = _make_base_params(sigma)

    # TWAP trajectory (same for both simulations)
    x_twap = twap_trajectory(params)

    # Constant-vol simulation
    _, costs_const = simulate_execution(
        params,
        x_twap,
        n_paths=N_PATHS,
        seed=SEED,
        antithetic=False,
        scheme="exact",
    )

    # Heston stochastic-vol simulation
    _, _, costs_heston = simulate_heston_execution(
        params,
        heston_params,
        x_twap,
        n_paths=N_PATHS,
        seed=SEED,
    )

    # Paired test: Heston vs ConstVol
    result = paired_strategy_test(
        costs_a=costs_heston,
        costs_b=costs_const,
        label_a="Heston",
        label_b="ConstVol",
        test="both",
        n_bootstrap=5_000,
        seed=SEED,
    )

    mean_const = float(np.mean(costs_const))
    mean_heston = float(np.mean(costs_heston))

    return {
        "vol_label": vol_label,
        "sigma": sigma,
        "mean_const": mean_const,
        "mean_heston": mean_heston,
        "mean_diff": result.mean_diff,
        "t_pval": result.t_pvalue,
        "boot_pval": result.bootstrap_pvalue,
        "significant_at_0.05": (
            (result.t_pvalue is not None and result.t_pvalue < 0.05)
            and (result.bootstrap_pvalue is not None and result.bootstrap_pvalue < 0.05)
        ),
        "n_paths": result.n_paths,
        "heston_params": {
            "kappa": heston_params.kappa,
            "theta": heston_params.theta,
            "xi": heston_params.xi,
            "rho": heston_params.rho,
            "v0": heston_params.v0,
        },
    }


def main() -> None:
    rows: list[dict] = []

    print("\nRunning paired Heston-vs-ConstVol tests across vol regimes...")
    print("=" * 100)
    for regime in VOL_REGIMES:
        r = run_one_regime(regime)
        rows.append(r)
        print(
            f"{r['vol_label']:>10}  "
            f"sigma={r['sigma']:.2f}  "
            f"mean(ConstVol)={r['mean_const']:>12.2f}  "
            f"mean(Heston)={r['mean_heston']:>12.2f}  "
            f"diff={r['mean_diff']:>+12.2f}"
        )
    print("=" * 100)
    print()

    # Significance table
    print("Significance table (H_0: E[C_Heston - C_ConstVol] = 0)")
    print("-" * 100)
    print(
        f"{'vol_label':>10} {'sigma':>8} "
        f"{'mean_const':>14} {'mean_heston':>14} {'mean_diff':>14} "
        f"{'t_pval':>12} {'boot_pval':>12} {'sig@0.05':>10}"
    )
    print("-" * 100)
    for r in rows:
        tp = f"{r['t_pval']:.4f}" if r["t_pval"] is not None else "—"
        bp = f"{r['boot_pval']:.4f}" if r["boot_pval"] is not None else "—"
        sig = "YES" if r["significant_at_0.05"] else "no"
        print(
            f"{r['vol_label']:>10} {r['sigma']:>8.2f} "
            f"{r['mean_const']:>14.2f} {r['mean_heston']:>14.2f} "
            f"{r['mean_diff']:>+14.2f} "
            f"{tp:>12} {bp:>12} {sig:>10}"
        )
    print("-" * 100)
    print()

    # Interpretation
    all_insignificant = all(not r["significant_at_0.05"] for r in rows)
    any_significant = any(r["significant_at_0.05"] for r in rows)

    if all_insignificant:
        print("Interpretation:")
        print(
            "  Heston doesn't materially shift mean cost — "
            "Part D is about tail risk, not mean cost."
        )
    elif any_significant:
        print("Interpretation:")
        print(
            "  Heston shifts mean cost significantly — "
            "Part D matters for expected cost too."
        )
    print()

    # Save JSON
    out_path = (
        Path(__file__).resolve().parent.parent
        / "data"
        / "paired_heston_vs_const_results.json"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(rows, indent=2))
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()